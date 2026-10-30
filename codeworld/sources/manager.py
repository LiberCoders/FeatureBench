"""Unified source selection, failover, application, health persistence, and metrics."""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

from pydantic import Field

from codeworld.config import SourceEndpointSettings, SourceServiceSettings, SourceSettings
from codeworld.errors import DockerBuildError, InfrastructureError
from codeworld.models import StrictModel
from codeworld.sources.base import (
    SourceBuildContext,
    SourceFailureKind,
    SourceMatchConfidence,
)
from codeworld.sources.defaults import default_source_adapters
from codeworld.sources.registry import SourceAdapterRegistry
from codeworld.storage.files import atomic_write_bytes
from codeworld.storage.jsonl import append_jsonl_record


class SourceAttemptError(InfrastructureError):
    """A source-specific transient failure that is safe to retry on another endpoint."""


class SourceAttemptMetric(StrictModel):
    attempt_id: str
    run_id: str
    service: str
    endpoint: str
    url: str
    canonical: bool
    status: Literal["succeeded", "failed"]
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(ge=0)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SourceChoice:
    service: str
    endpoint: SourceEndpointSettings


@dataclass(frozen=True, slots=True)
class SourceBuildPlan:
    arguments: dict[str, str]
    choices: dict[str, SourceChoice]
    argument_names: dict[str, tuple[str, ...]]

    def managed_choices(self, overridden_arguments: Mapping[str, str]) -> dict[str, SourceChoice]:
        """Exclude services whose adapter-owned arguments were explicitly overridden."""

        return {
            service: choice
            for service, choice in self.choices.items()
            if not any(name in overridden_arguments for name in self.argument_names[service])
        }


class _EndpointHealth(TypedDict, total=False):
    service: str
    endpoint: str
    url: str
    successes: int
    failures: int
    consecutive_failures: int
    cooldown_until: float
    last_status: str
    last_used_at: str
    last_error: str | None


class _SourceState(TypedDict):
    endpoints: dict[str, _EndpointHealth]


def _empty_state() -> _SourceState:
    return {"endpoints": {}}


def _read_state(path: Path) -> _SourceState:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty_state()
    if not isinstance(raw, dict):
        return _empty_state()
    mapping = cast(dict[str, Any], raw)
    endpoints = mapping.get("endpoints")
    if not isinstance(endpoints, dict):
        return _empty_state()
    sanitized: dict[str, _EndpointHealth] = {}
    for raw_key, raw_value in cast(dict[Any, Any], endpoints).items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, dict):
            continue
        value = cast(dict[str, Any], raw_value)
        try:
            sanitized[raw_key] = {
                "service": str(value.get("service", "")),
                "endpoint": str(value.get("endpoint", "")),
                "url": str(value.get("url", "")),
                "successes": max(int(value.get("successes", 0)), 0),
                "failures": max(int(value.get("failures", 0)), 0),
                "consecutive_failures": max(int(value.get("consecutive_failures", 0)), 0),
                "cooldown_until": max(float(value.get("cooldown_until", 0)), 0),
                "last_status": str(value.get("last_status", "")),
                "last_used_at": str(value.get("last_used_at", "")),
                "last_error": (
                    str(value["last_error"])[-2000:]
                    if value.get("last_error") is not None
                    else None
                ),
            }
        except (TypeError, ValueError):
            continue
    return {
        "endpoints": sanitized,
    }


def _write_state(path: Path, state: _SourceState) -> None:
    payload = (
        json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, payload)


def _locked_state[ResultT](path: Path, mutation: Callable[[_SourceState], ResultT]) -> ResultT:
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _read_state(path)
        result = mutation(state)
        _write_state(path, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return result


class SourceManager:
    """Select, switch, apply, clean up, and record sources through adapters."""

    def __init__(
        self,
        settings: SourceSettings | None = None,
        *,
        run_id: str = "mirror",
        adapters: SourceAdapterRegistry | None = None,
    ) -> None:
        self.settings = settings or SourceSettings()
        self.run_id = run_id
        self.adapters = adapters or default_source_adapters()
        self._lock = asyncio.Lock()

    def service(self, name: str) -> SourceServiceSettings | None:
        return self.settings.services.get(name)

    async def ordered(
        self, service: str, *, excluded: frozenset[str] = frozenset()
    ) -> tuple[SourceChoice, ...]:
        configured = self.service(service)
        if configured is None:
            return ()
        endpoints = (
            configured.endpoints
            if self.settings.enabled
            else tuple(endpoint for endpoint in configured.endpoints if endpoint.canonical)
        )
        state = await asyncio.to_thread(_read_state, self.settings.state_path)
        health = state["endpoints"]
        now = time.time()

        def rank(item: tuple[int, SourceEndpointSettings]) -> tuple[int, float, int]:
            index, endpoint = item
            entry: _EndpointHealth = health.get(f"{service}/{endpoint.name}", {})
            if entry.get("url") != endpoint.url:
                entry = {}
            cooldown_until = float(entry.get("cooldown_until", 0))
            return (1, cooldown_until, index) if cooldown_until > now else (0, 0, index)

        ranked = sorted(enumerate(endpoints), key=rank)
        return tuple(
            SourceChoice(service=service, endpoint=endpoint)
            for _, endpoint in ranked
            if endpoint.name not in excluded
        )

    async def attempt_choices(
        self, service: str, *, excluded: frozenset[str] = frozenset()
    ) -> tuple[SourceChoice, ...]:
        """Return the remaining bounded attempt chain with canonical as its fallback."""

        configured = self.service(service)
        if configured is None:
            return ()
        canonical_name = next(
            endpoint.name for endpoint in configured.endpoints if endpoint.canonical
        )
        if canonical_name in excluded:
            return ()
        choices = await self.ordered(service, excluded=excluded)
        if not choices or not self.settings.enabled:
            return choices
        maximum = configured.maximum_attempts
        if maximum is None:
            return choices
        remaining = maximum - len(excluded)
        if remaining <= 0:
            return ()
        canonical = next(
            (choice for choice in choices if choice.endpoint.canonical),
            None,
        )
        if canonical is None:
            return choices[:remaining]
        canonical_index = choices.index(canonical)
        if canonical_index == 0:
            return (canonical,)
        candidates = tuple(
            choice for choice in choices[:canonical_index] if not choice.endpoint.canonical
        )
        return (*candidates[: max(remaining - 1, 0)], canonical)

    async def _record(
        self,
        choice: SourceChoice,
        *,
        status: Literal["succeeded", "failed"],
        started_at: datetime,
        started_monotonic: float,
        error: BaseException | None,
    ) -> None:
        ended_at = datetime.now(UTC)
        duration = max(time.monotonic() - started_monotonic, 0)
        key = f"{choice.service}/{choice.endpoint.name}"

        def mutate(state: _SourceState) -> None:
            endpoints = state["endpoints"]
            current = endpoints.get(key, {})
            failures = int(current.get("failures", 0))
            successes = int(current.get("successes", 0))
            consecutive = int(current.get("consecutive_failures", 0))
            if status == "succeeded":
                successes += 1
                consecutive = 0
                cooldown_until = 0.0
            else:
                failures += 1
                consecutive += 1
                if consecutive >= self.settings.failure_threshold:
                    exponent = consecutive - self.settings.failure_threshold
                    cooldown = min(
                        self.settings.cooldown_seconds * (2**exponent),
                        self.settings.maximum_cooldown_seconds,
                    )
                    cooldown_until = time.time() + cooldown
                else:
                    cooldown_until = 0.0
            endpoints[key] = {
                "service": choice.service,
                "endpoint": choice.endpoint.name,
                "url": choice.endpoint.url,
                "successes": successes,
                "failures": failures,
                "consecutive_failures": consecutive,
                "cooldown_until": cooldown_until,
                "last_status": status,
                "last_used_at": ended_at.isoformat(),
                "last_error": str(error)[-2000:] if error is not None else None,
            }

        async with self._lock:
            await asyncio.to_thread(_locked_state, self.settings.state_path, mutate)
            if self.settings.metrics.enabled:
                metric = SourceAttemptMetric(
                    attempt_id=uuid4().hex,
                    run_id=self.run_id,
                    service=choice.service,
                    endpoint=choice.endpoint.name,
                    url=choice.endpoint.url,
                    canonical=choice.endpoint.canonical,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_seconds=duration,
                    error=str(error)[-5000:] if error is not None else None,
                )
                await asyncio.to_thread(
                    append_jsonl_record,
                    self.settings.metrics.path,
                    metric,
                    parse=SourceAttemptMetric.model_validate_json,
                    serialize=lambda record: record.model_dump_json(exclude_none=True),
                )

    async def record_success(
        self, choice: SourceChoice, *, started_at: datetime, started_monotonic: float
    ) -> None:
        await self._record(
            choice,
            status="succeeded",
            started_at=started_at,
            started_monotonic=started_monotonic,
            error=None,
        )

    async def record_failure(
        self,
        choice: SourceChoice,
        error: BaseException,
        *,
        started_at: datetime,
        started_monotonic: float,
    ) -> None:
        await self._record(
            choice,
            status="failed",
            started_at=started_at,
            started_monotonic=started_monotonic,
            error=error,
        )

    async def execute[ResultT](
        self,
        service: str,
        operation: Callable[[SourceChoice], Awaitable[ResultT]],
    ) -> ResultT:
        """Try the service's bounded endpoint chain once, ending at canonical."""

        configured = self.service(service)
        if configured is None:
            raise InfrastructureError(f"mirror service is not configured: {service}")
        excluded: set[str] = set()
        failures: list[str] = []
        while True:
            choices = await self.attempt_choices(service, excluded=frozenset(excluded))
            if not choices:
                break
            choice = choices[0]
            excluded.add(choice.endpoint.name)
            started_at = datetime.now(UTC)
            started_monotonic = time.monotonic()
            try:
                result = await operation(choice)
            except SourceAttemptError as exc:
                failures.append(f"{choice.endpoint.name}: {exc}")
                await self._record(
                    choice,
                    status="failed",
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=exc,
                )
                continue
            await self._record(
                choice,
                status="succeeded",
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=None,
            )
            return result
        detail = "; ".join(failures)[-10_000:]
        raise InfrastructureError(f"all sources failed for {service}: {detail}")

    def resolve_url(self, choice: SourceChoice, canonical_url: str) -> str:
        return self.adapters.resolve_url(
            choice.service,
            endpoint_url=choice.endpoint.url,
            canonical=choice.endpoint.canonical,
            canonical_url=canonical_url,
        )

    async def build_arguments(self) -> dict[str, str]:
        """Select build-only package endpoints; ARG values never persist in the final image."""

        return (await self.build_plan()).arguments

    async def build_plan(
        self,
        *,
        excluded: Mapping[str, Collection[str]] | None = None,
    ) -> SourceBuildPlan:
        """Resolve one coherent source snapshot for a Docker build attempt."""

        excluded = excluded or {}
        download = self.settings.download
        arguments: dict[str, str] = {
            "CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS": str(download.connect_timeout_seconds),
            "CW_DOWNLOAD_TIMEOUT_SECONDS": str(download.timeout_seconds),
            "CW_DOWNLOAD_RETRIES": str(download.retries),
        }
        selected: dict[str, SourceChoice] = {}
        argument_names: dict[str, tuple[str, ...]] = {}
        for service in self.settings.services:
            adapters = self.adapters.for_service(service)
            if not adapters:
                continue
            choices = await self.attempt_choices(
                service,
                excluded=frozenset(excluded.get(service, ())),
            )
            if not choices:
                continue
            choice = choices[0]
            values: dict[str, str] = {}
            managed_names: list[str] = []
            for adapter in adapters:
                emitted = dict(
                    adapter.build_arguments(
                        service,
                        url=choice.endpoint.url,
                        canonical=choice.endpoint.canonical,
                    )
                )
                managed_names.extend(emitted)
                download_values = adapter.download_arguments(
                    service,
                    connect_timeout_seconds=download.connect_timeout_seconds,
                    timeout_seconds=download.timeout_seconds,
                    retries=download.retries,
                )
                duplicate = emitted.keys() & download_values.keys()
                if duplicate:
                    names = ", ".join(sorted(duplicate))
                    raise ValueError(
                        f"source adapter {adapter.name!r} emitted duplicate build "
                        f"and download arguments: {names}"
                    )
                emitted.update(download_values)
                duplicate = values.keys() & emitted.keys()
                if duplicate:
                    names = ", ".join(sorted(duplicate))
                    raise ValueError(
                        f"source adapters for {service!r} emitted duplicate build "
                        f"arguments: {names}"
                    )
                values.update(emitted)
            if not values:
                continue
            conflicts = arguments.keys() & values.keys()
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(f"source adapters emitted duplicate build arguments: {names}")
            selected[service] = choice
            argument_names[service] = tuple(managed_names)
            arguments.update(values)
        return SourceBuildPlan(
            arguments=arguments,
            choices=selected,
            argument_names=argument_names,
        )

    def materialize_build_sources(
        self,
        source: str,
        plan: SourceBuildPlan,
        *,
        build_arguments: Mapping[str, str] | None = None,
    ) -> str:
        """Apply registered build-scoped source adapters to one Dockerfile."""

        return self.adapters.render(
            source,
            SourceBuildContext(
                services=frozenset(plan.choices),
                build_argument_names=tuple(build_arguments or plan.arguments),
            ),
        )

    def build_failure_service(
        self,
        failure: DockerBuildError,
        choices: Mapping[str, SourceChoice],
    ) -> str | None:
        """Return one unambiguous retryable source, based only on the failed step."""
        if failure.failed_step is None:
            return None
        step = failure.failed_step
        matches: dict[tuple[str, SourceFailureKind], SourceMatchConfidence] = {}
        for service, choice in choices.items():
            for adapter in self.adapters.for_service(service):
                confidence = adapter.match_build_step(
                    service,
                    instruction=step.instruction,
                    output=step.output,
                    endpoint_url=choice.endpoint.url,
                )
                if confidence is None:
                    continue
                kind = adapter.classify_build_failure(
                    service,
                    detail=step.output,
                    confidence=confidence,
                )
                if kind is None:
                    continue
                key = (service, kind)
                matches[key] = max(matches.get(key, confidence), confidence)
        if not matches:
            return None
        strongest = max(matches.values())
        winners = {key for key, confidence in matches.items() if confidence == strongest}
        services = {service for service, _ in winners}
        kinds = {kind for _, kind in winners}
        if len(services) != 1 or len(kinds) != 1:
            return None
        service = next(iter(services))
        kind = next(iter(kinds))
        choice = choices[service]
        if kind is SourceFailureKind.NOT_FOUND and choice.endpoint.canonical:
            return None
        return service

    async def pull_image(
        self,
        image: str,
        *,
        pull: Callable[[str], Awaitable[None]],
        tag: Callable[[str, str], Awaitable[None]],
    ) -> bool:
        """Pull via an image adapter and restore the canonical tag; return whether handled."""

        request = self.adapters.image_request(image)
        if request is None or self.service(request.service) is None:
            return False

        async def pull_source(choice: SourceChoice) -> str:
            candidate = self.adapters.image_reference(
                request,
                endpoint_url=choice.endpoint.url,
            )
            await pull(candidate)
            return candidate

        pulled = await self.execute(request.service, pull_source)
        if pulled != image:
            await tag(pulled, image)
        return True
