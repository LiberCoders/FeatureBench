"""Loss-aware GitHub discovery with time partitioning and token-aware concurrency."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import httpx

from codeworld.config import DiscoverySettings
from codeworld.discovery.models import SearchPartition
from codeworld.errors import ContractError, InfrastructureError
from codeworld.events import Event, EventSink, NullEventSink
from codeworld.repositories.models import RepositoryCandidate
from codeworld.sources import SourceChoice, SourceManager


@dataclass(slots=True)
class _TokenState:
    key: str
    value: str | None
    remaining: int | None = None
    available_at: float = 0
    in_flight: int = 0
    last_used_at: float = 0


@dataclass(frozen=True, slots=True)
class _TokenLease:
    state: _TokenState


class TokenScheduler:
    """Rotate credentials, enforce a global cap, and cool down limited tokens."""

    def __init__(self, settings: DiscoverySettings, tokens: Iterable[str] = ()) -> None:
        values = tuple(dict.fromkeys(value.strip() for value in tokens if value.strip()))
        self.settings = settings
        self._states = [
            _TokenState(key=f"token-{index}", value=value)
            for index, value in enumerate(values, start=1)
        ] or [_TokenState(key="anonymous", value=None)]
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._lock = asyncio.Lock()

    async def acquire(self) -> _TokenLease:
        await self._semaphore.acquire()
        while True:
            async with self._lock:
                now = time.time()
                for state in self._states:
                    if state.remaining == 0 and state.available_at <= now:
                        state.remaining = None
                candidates = [
                    state
                    for state in self._states
                    if state.available_at <= now and state.remaining != 0
                ]
                if candidates:
                    state = min(
                        candidates,
                        key=lambda item: (item.in_flight, item.last_used_at, item.key),
                    )
                    state.in_flight += 1
                    state.last_used_at = now
                    return _TokenLease(state)
                delay = max(min(state.available_at for state in self._states) - now, 0.05)
            self._semaphore.release()
            await asyncio.sleep(min(delay, 60))
            await self._semaphore.acquire()

    async def release(
        self,
        lease: _TokenLease,
        *,
        response: httpx.Response | None = None,
        cooldown_seconds: float = 0,
    ) -> None:
        async with self._lock:
            state = lease.state
            state.in_flight = max(0, state.in_flight - 1)
            if response is not None:
                remaining = response.headers.get("x-ratelimit-remaining")
                reset_at = response.headers.get("x-ratelimit-reset")
                if remaining is not None:
                    with suppress(ValueError):
                        state.remaining = max(0, int(remaining))
                if reset_at is not None and state.remaining == 0:
                    with suppress(ValueError):
                        state.available_at = max(state.available_at, float(reset_at) + 1)
            if cooldown_seconds > 0:
                state.available_at = max(state.available_at, time.time() + cooldown_seconds)
        self._semaphore.release()

    def headers(self, lease: _TokenLease) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.github_user_agent,
            "X-GitHub-Api-Version": self.settings.github_api_version,
        }
        if lease.state.value is not None:
            headers["Authorization"] = f"Bearer {lease.state.value}"
        return headers

    def rate_limit_cooldown(self, response: httpx.Response) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(float(retry_after), 1)
            except ValueError:
                pass
        reset_at = response.headers.get("x-ratelimit-reset")
        if reset_at is not None:
            try:
                return max(float(reset_at) - time.time() + 1, 1)
            except ValueError:
                pass
        return self.settings.rate_limit_fallback_seconds


class GitHubDiscovery:
    """Discover repositories without silently truncating GitHub's 1000-result window."""

    def __init__(
        self,
        settings: DiscoverySettings,
        *,
        tokens: Iterable[str] = (),
        client: httpx.AsyncClient | None = None,
        events: EventSink | None = None,
        run_id: str = "discovery",
        sources: SourceManager | None = None,
    ) -> None:
        self.settings = settings
        self.tokens = TokenScheduler(settings, tokens)
        self.events = events or NullEventSink()
        self.run_id = run_id
        self.sources = sources
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.github_api_url,
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )

    async def __aenter__(self) -> GitHubDiscovery:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        del exc_info
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _emit(self, kind: str, message: str, **data: object) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind=kind,
                stage="repository-discovery",
                message=message,
                data=dict(data),
            )
        )

    @staticmethod
    def _rate_limited(response: httpx.Response) -> bool:
        if response.status_code not in (403, 429):
            return False
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        if response.headers.get("retry-after"):
            return True
        try:
            message = str(response.json().get("message", "")).lower()
        except (ValueError, AttributeError):
            message = response.text.lower()
        return "rate limit" in message or "abuse detection" in message

    def _backoff(self, attempt: int) -> float:
        return min(
            self.settings.retry_base_seconds * (2 ** max(attempt - 1, 0)),
            self.settings.retry_max_seconds,
        )

    async def _request(
        self, path: str, *, parameters: dict[str, str | int] | None = None
    ) -> dict[str, Any]:
        last_error = "GitHub request did not run"
        sources = self.sources
        source_service = sources.service("github_api") if sources is not None else None
        endpoint_count = len(source_service.endpoints) if source_service is not None else 1
        total_attempts = max(self.settings.request_attempts, endpoint_count)
        excluded: set[str] = set()
        for attempt in range(1, total_attempts + 1):
            choice: SourceChoice | None = None
            request_target = path
            if sources is not None and source_service is not None:
                choices = await sources.ordered("github_api", excluded=frozenset(excluded))
                if not choices:
                    excluded.clear()
                    choices = await sources.ordered("github_api")
                choice = choices[0]
                canonical_url = f"{self.settings.github_api_url.rstrip('/')}/{path.lstrip('/')}"
                request_target = sources.resolve_url(choice, canonical_url)
            lease: _TokenLease | None = await self.tokens.acquire()
            response: httpx.Response | None = None
            source_started_at = datetime.now(UTC)
            source_started_monotonic = time.monotonic()
            try:
                assert lease is not None
                headers = self.tokens.headers(lease)
                if choice is not None and not choice.endpoint.forward_credentials:
                    headers.pop("Authorization", None)
                response = await self._client.get(
                    request_target,
                    params=parameters,
                    headers=headers,
                )
                if self._rate_limited(response):
                    if choice is not None and sources is not None:
                        await sources.record_success(
                            choice,
                            started_at=source_started_at,
                            started_monotonic=source_started_monotonic,
                        )
                    cooldown = self.tokens.rate_limit_cooldown(response)
                    await self.tokens.release(lease, response=response, cooldown_seconds=cooldown)
                    lease = None
                    last_error = f"GitHub rate limit ({response.status_code})"
                    await self._emit(
                        "rate_limited",
                        f"token cooling down for {cooldown:.1f}s",
                        status=response.status_code,
                        cooldown_seconds=cooldown,
                    )
                elif response.status_code >= 500:
                    source_error = InfrastructureError(
                        f"GitHub server error {response.status_code}"
                    )
                    if choice is not None and sources is not None:
                        excluded.add(choice.endpoint.name)
                        await sources.record_failure(
                            choice,
                            source_error,
                            started_at=source_started_at,
                            started_monotonic=source_started_monotonic,
                        )
                    await self.tokens.release(lease, response=response)
                    lease = None
                    last_error = (
                        f"GitHub server error {response.status_code}: {response.text[:500]}"
                    )
                else:
                    if choice is not None and sources is not None:
                        await sources.record_success(
                            choice,
                            started_at=source_started_at,
                            started_monotonic=source_started_monotonic,
                        )
                    await self.tokens.release(lease, response=response)
                    lease = None
                    if response.is_error:
                        raise InfrastructureError(
                            f"GitHub API failed ({response.status_code}) for {path}: "
                            f"{response.text[:2000]}"
                        )
                    payload: Any = response.json()
                    if not isinstance(payload, dict):
                        raise InfrastructureError(f"GitHub returned a non-object for {path}")
                    return cast(dict[str, Any], payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = f"GitHub transport error: {exc}"
                if choice is not None and sources is not None:
                    excluded.add(choice.endpoint.name)
                    await sources.record_failure(
                        choice,
                        exc,
                        started_at=source_started_at,
                        started_monotonic=source_started_monotonic,
                    )
            finally:
                if lease is not None:
                    await self.tokens.release(lease, response=response)
            if attempt < total_attempts:
                await asyncio.sleep(self._backoff(attempt))
        raise InfrastructureError(
            f"GitHub request failed after {total_attempts} attempts: {last_error}"
        )

    async def _search(
        self,
        query: str,
        *,
        page: int,
        per_page: int,
    ) -> dict[str, Any]:
        parameters: dict[str, str | int] = {
            "q": query,
            "sort": self.settings.sort,
            "order": self.settings.order,
            "page": page,
            "per_page": per_page,
        }
        for incomplete_attempt in range(self.settings.incomplete_results_retries + 1):
            payload = await self._request("/search/repositories", parameters=parameters)
            if not bool(payload.get("incomplete_results", False)):
                return payload
            if incomplete_attempt < self.settings.incomplete_results_retries:
                await asyncio.sleep(self._backoff(incomplete_attempt + 1))
        raise InfrastructureError(
            f"GitHub search repeatedly returned incomplete_results for page {page}: {query}"
        )

    async def _count(self, partition: SearchPartition) -> int:
        payload = await self._search(partition.query(self.settings), page=1, per_page=1)
        return max(0, int(payload.get("total_count", 0)))

    def _root_partitions(self) -> list[SearchPartition]:
        created_end = self.settings.created_before or datetime.now(UTC).replace(microsecond=0)
        languages: tuple[str | None, ...] = self.settings.languages or (None,)
        licenses: tuple[str | None, ...] = self.settings.licenses or (None,)
        return [
            SearchPartition(
                language=language,
                license=license_key,
                created_start=self.settings.created_after,
                created_end=created_end,
            )
            for language in languages
            for license_key in licenses
        ]

    async def partitions(
        self,
        *,
        pending_partitions: Iterable[SearchPartition] | None = None,
        accepted_partitions: Iterable[SearchPartition] = (),
        checkpoint: Callable[
            [tuple[SearchPartition, ...], tuple[SearchPartition, ...]], Awaitable[None]
        ]
        | None = None,
    ) -> list[SearchPartition]:
        """Plan lossless partitions and optionally persist the planning frontier after every probe."""
        root_partitions = self._root_partitions()
        pending = (
            list(pending_partitions)
            if pending_partitions is not None
            else list(reversed(root_partitions))
        )
        result = list(accepted_partitions)
        maximum_span = self.settings.partition_max_span_days * 24 * 60 * 60

        async def emit_progress(current: SearchPartition | None) -> None:
            total_weight = sum(item.span_seconds + 1 for item in root_partitions)
            unresolved_weight = sum(item.span_seconds + 1 for item in pending)
            progress = (
                min(max(1 - (unresolved_weight / total_weight), 0.0), 1.0) if total_weight else 1.0
            )
            current_key = (current.language, current.license) if current is not None else None
            group_index = next(
                (
                    index
                    for index, item in enumerate(root_partitions, start=1)
                    if (item.language, item.license) == current_key
                ),
                len(root_partitions),
            )
            await self._emit(
                "discovery_progress",
                "planning GitHub search coverage",
                phase="planning",
                progress=progress,
                current_language=current.language if current is not None else None,
                current_license=current.license if current is not None else None,
                search_group_index=group_index,
                search_group_total=len(root_partitions),
                accepted_partitions=len(result),
                pending_partitions=len(pending),
            )

        await emit_progress(pending[-1] if pending else None)
        if checkpoint is not None and pending_partitions is None:
            await checkpoint(tuple(pending), tuple(result))
        while pending:
            probe_batch: list[SearchPartition] = []
            while pending and len(probe_batch) < self.settings.partition_concurrency:
                partition = pending.pop()
                if partition.span_seconds > maximum_span:
                    left, right = partition.bisect()
                    pending.extend((right, left))
                else:
                    probe_batch.append(partition)
            if not probe_batch:
                continue
            counts = await asyncio.gather(*(self._count(item) for item in probe_batch))
            for partition, count in zip(probe_batch, counts, strict=True):
                await self._emit(
                    "partition_probed",
                    f"{partition.created_start.date()}..{partition.created_end.date()}: {count}",
                    language=partition.language,
                    license=partition.license,
                    count=count,
                    depth=partition.depth,
                )
                if count == 0:
                    continue
                if count <= self.settings.partition_result_limit:
                    result.append(partition.model_copy(update={"expected_count": count}))
                    continue
                if partition.span_seconds <= 0:
                    raise ContractError(
                        "GitHub search has more than the retrievable limit in a one-second window; "
                        "refine languages, licenses, keywords, stars, or dates. Refusing to truncate: "
                        f"{partition.query(self.settings)} reported {count}."
                    )
                left, right = partition.bisect()
                pending.extend((right, left))
            if checkpoint is not None:
                await checkpoint(tuple(pending), tuple(result))
            await emit_progress(pending[-1] if pending else probe_batch[-1])
        result.sort(
            key=lambda item: (
                item.language or "",
                item.license or "",
                item.created_start,
                item.created_end,
            )
        )
        await self._emit(
            "partitions_planned",
            f"planned {len(result)} lossless GitHub search partitions",
            partition_count=len(result),
            expected_results=sum(item.expected_count for item in result),
        )
        return result

    def _candidate(self, item: dict[str, Any]) -> RepositoryCandidate:
        raw_owner = item.get("owner")
        owner = cast(dict[str, Any], raw_owner) if isinstance(raw_owner, dict) else {}
        raw_license = item.get("license")
        license_data = cast(dict[str, Any], raw_license) if isinstance(raw_license, dict) else {}
        return RepositoryCandidate.model_validate(
            {
                "discovery_run_id": self.run_id,
                "github_id": item["id"],
                "node_id": item.get("node_id") or "",
                "full_name": item["full_name"],
                "name": item["name"],
                "owner": owner.get("login") or str(item["full_name"]).split("/", 1)[0],
                "clone_url": item["clone_url"],
                "html_url": item["html_url"],
                "api_url": item.get("url") or "",
                "default_branch": item["default_branch"],
                "description": item.get("description"),
                "stars": item.get("stargazers_count") or 0,
                "forks": item.get("forks_count") or 0,
                "open_issues": item.get("open_issues_count") or 0,
                "size_kb": item.get("size") or 0,
                "language": item.get("language") or "",
                "license": license_data.get("key"),
                "topics": item.get("topics") or (),
                "visibility": item.get("visibility") or "public",
                "archived": item.get("archived", False),
                "fork": item.get("fork", False),
                "private": item.get("private", False),
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
                "pushed_at": item.get("pushed_at"),
            }
        )

    async def _fetch_partition(
        self, index: int, partition: SearchPartition
    ) -> tuple[int, list[RepositoryCandidate]]:
        query = partition.query(self.settings)
        candidates: list[RepositoryCandidate] = []
        total = 0
        for consistency_attempt in range(self.settings.partition_consistency_retries + 1):
            first = await self._search(query, page=1, per_page=self.settings.requests_per_page)
            total = max(0, int(first.get("total_count", 0)))
            if total > 1000:
                raise ContractError(
                    "GitHub partition grew beyond 1000 results after planning; refusing to truncate: "
                    f"{query} now reports {total}"
                )
            first_items = first.get("items")
            if not isinstance(first_items, list):
                raise InfrastructureError("GitHub search response has a non-list items field")
            total_pages = max(1, math.ceil(total / self.settings.requests_per_page))
            remaining_pages = await asyncio.gather(
                *(
                    self._search(query, page=page, per_page=self.settings.requests_per_page)
                    for page in range(2, total_pages + 1)
                )
            )
            raw_items: list[Any] = list(cast(list[Any], first_items))
            for payload in remaining_pages:
                page_items = payload.get("items")
                if not isinstance(page_items, list):
                    raise InfrastructureError("GitHub search response has a non-list items field")
                raw_items.extend(cast(list[Any], page_items))
            unique: dict[int, RepositoryCandidate] = {}
            for item in cast(list[object], raw_items):
                if not isinstance(item, dict):
                    continue
                candidate = self._candidate(cast(dict[str, Any], item))
                unique.setdefault(candidate.github_id, candidate)
            candidates = list(unique.values())
            if len(candidates) == total:
                break
            if consistency_attempt >= self.settings.partition_consistency_retries:
                raise InfrastructureError(
                    "GitHub search changed during pagination and remained inconsistent after "
                    f"{consistency_attempt + 1} complete fetches: reported={total}, "
                    f"unique={len(candidates)}, query={query}"
                )
            await self._emit(
                "partition_retried",
                "GitHub search changed during pagination; refetching the complete partition",
                partition_index=index,
                reported_total=total,
                unique_results=len(candidates),
                attempt=consistency_attempt + 1,
            )
            await asyncio.sleep(self._backoff(consistency_attempt + 1))
        await self._emit(
            "partition_fetched",
            f"fetched {len(candidates)} repositories from partition {index + 1}",
            partition_index=index,
            fetched=len(candidates),
            reported_total=total,
        )
        return index, candidates

    async def fetch_partition(
        self, index: int, partition: SearchPartition
    ) -> list[RepositoryCandidate]:
        """Fetch one already-checkpointed search partition in full."""
        _, candidates = await self._fetch_partition(index, partition)
        return candidates

    async def _fetch_bounded_group(
        self,
        index: int,
        partition: SearchPartition,
        limit: int,
        group_total: int,
    ) -> tuple[int, list[RepositoryCandidate]]:
        """Fetch only the leading candidates needed from one root search group."""
        query = partition.query(self.settings)
        page_size = min(self.settings.requests_per_page, limit)
        first = await self._search(query, page=1, per_page=page_size)
        total = max(0, int(first.get("total_count", 0)))
        if limit > 1000 and total > 1000:
            raise ContractError(
                "bounded GitHub discovery cannot determine an exact Top-K larger than "
                "GitHub's 1000-result search window for this query; lower "
                "maximum_repositories, refine the filters, or set it to null for lossless "
                f"partitioning: {query} reported {total} repositories"
            )

        requested = min(limit, total, 1000)
        total_pages = max(1, math.ceil(requested / page_size))
        payloads = [first]
        if total_pages > 1:
            payloads.extend(
                await asyncio.gather(
                    *(
                        self._search(query, page=page, per_page=page_size)
                        for page in range(2, total_pages + 1)
                    )
                )
            )

        unique: dict[int, RepositoryCandidate] = {}
        for payload in payloads:
            raw_items = payload.get("items")
            if not isinstance(raw_items, list):
                raise InfrastructureError(
                    "GitHub bounded search response has a non-list items field"
                )
            for item in cast(list[object], raw_items):
                if not isinstance(item, dict):
                    continue
                candidate = self._candidate(cast(dict[str, Any], item))
                unique.setdefault(candidate.github_id, candidate)
                if len(unique) >= limit:
                    break
            if len(unique) >= limit:
                break

        candidates = list(unique.values())
        await self._emit(
            "bounded_group_fetched",
            f"fetched {len(candidates)} bounded candidates from search group {index + 1}",
            search_group_index=index + 1,
            search_group_total=group_total,
            language=partition.language,
            license=partition.license,
            fetched=len(candidates),
            reported_total=total,
        )
        return index, candidates

    async def fetch_bounded(self) -> list[RepositoryCandidate]:
        """Return an exact global Top-K without planning the complete historical corpus."""
        limit = self.settings.maximum_repositories
        if limit is None:
            raise ContractError("bounded discovery requires maximum_repositories")

        partitions = self._root_partitions()
        grouped = await asyncio.gather(
            *(
                self._fetch_bounded_group(index, partition, limit, len(partitions))
                for index, partition in enumerate(partitions)
            )
        )
        unique: dict[int, RepositoryCandidate] = {}
        for _, candidates in sorted(grouped):
            for candidate in candidates:
                unique.setdefault(candidate.github_id, candidate)

        ordered = sorted(
            unique.values(), key=lambda item: (item.full_name.casefold(), item.github_id)
        )
        if self.settings.sort == "stars":
            ordered.sort(key=lambda item: item.stars, reverse=self.settings.order == "desc")
        else:
            ordered.sort(key=lambda item: item.updated_at, reverse=self.settings.order == "desc")
        selected = ordered[:limit]
        await self._emit(
            "bounded_search_completed",
            f"selected {len(selected)} repositories from bounded GitHub search",
            repositories=len(selected),
            search_groups=len(partitions),
            maximum_repositories=limit,
        )
        return selected

    async def _target_candidates(self) -> list[RepositoryCandidate]:
        semaphore = asyncio.Semaphore(self.settings.partition_concurrency)

        async def fetch(full_name: str) -> RepositoryCandidate:
            async with semaphore:
                payload = await self._request(f"/repos/{full_name}")
                return self._candidate(payload)

        return list(
            await asyncio.gather(
                *(fetch(full_name) for full_name in self.settings.target_repositories)
            )
        )

    async def fetch_target(self, full_name: str) -> RepositoryCandidate:
        """Fetch one configured target so each target can be checkpointed independently."""
        payload = await self._request(f"/repos/{full_name}")
        return self._candidate(payload)

    async def iter_repositories(self) -> AsyncIterator[RepositoryCandidate]:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        emitted = 0

        async def emit_candidates(
            candidates: Iterable[RepositoryCandidate],
        ) -> AsyncIterator[RepositoryCandidate]:
            nonlocal emitted
            for candidate in candidates:
                name_key = candidate.full_name.casefold()
                if candidate.github_id in seen_ids or name_key in seen_names:
                    continue
                seen_ids.add(candidate.github_id)
                seen_names.add(name_key)
                emitted += 1
                yield candidate
                if (
                    self.settings.maximum_repositories is not None
                    and emitted >= self.settings.maximum_repositories
                ):
                    return

        if self.settings.target_repositories:
            async for candidate in emit_candidates(await self._target_candidates()):
                yield candidate
            return

        if self.settings.maximum_repositories is not None:
            async for candidate in emit_candidates(await self.fetch_bounded()):
                yield candidate
            return

        partitions = await self.partitions()
        next_index = 0
        emit_index = 0
        pending: dict[asyncio.Task[tuple[int, list[RepositoryCandidate]]], int] = {}
        completed: dict[int, list[RepositoryCandidate]] = {}
        try:
            while next_index < len(partitions) or pending:
                while (
                    next_index < len(partitions)
                    and len(pending) < self.settings.partition_concurrency
                ):
                    task = asyncio.create_task(
                        self._fetch_partition(next_index, partitions[next_index])
                    )
                    pending[task] = next_index
                    next_index += 1
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    pending.pop(task)
                    index, candidates = await task
                    completed[index] = candidates
                while emit_index in completed:
                    candidates = completed.pop(emit_index)
                    async for candidate in emit_candidates(candidates):
                        yield candidate
                    if (
                        self.settings.maximum_repositories is not None
                        and emitted >= self.settings.maximum_repositories
                    ):
                        return
                    emit_index += 1
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self._emit(
                "discovery_completed",
                f"discovered {emitted} unique repositories",
                repositories=emitted,
            )
