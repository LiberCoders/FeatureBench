"""Same-run checkpoint orchestration for repository discovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from codeworld.discovery.github import GitHubDiscovery
from codeworld.discovery.models import SearchPartition
from codeworld.errors import ContractError
from codeworld.events import Event, EventSink, NullEventSink
from codeworld.repositories.identity import repository_aliases
from codeworld.repositories.jsonl import RepositoryJsonlAppender, index_repository_outcomes
from codeworld.repositories.models import RepositoryCandidate
from codeworld.selection import (
    WorkSelection,
    WorkState,
    default_work_selection_registry,
)
from codeworld.state import JobRecord, JobStatus, StateStore

_MANIFEST_JOB = "discovery-manifest"
_PLAN_STAGE = "repository-discovery-plan"
_FETCH_STAGE = "repository-discovery-fetch"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"discovery checkpoint {label} must be an object")
    return cast(dict[str, Any], value)


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"discovery checkpoint {label} must be an array")
    return tuple(cast(list[object] | tuple[object, ...], value))


class DiscoveryRun:
    """Own a discovery output and resume only checkpoints belonging to the same run ID."""

    def __init__(
        self,
        *,
        discovery: GitHubDiscovery,
        writer: RepositoryJsonlAppender,
        state: StateStore,
        run_id: str,
        output: str | Path,
        resume: bool,
        target_repositories: tuple[str, ...] | None = None,
        events: EventSink | None = None,
        selection: WorkSelection | None = None,
    ) -> None:
        self.discovery = discovery
        self.writer = writer
        self.state = state
        self.run_id = run_id
        self.output = str(Path(output).resolve())
        self.resume = resume
        self.target_repositories = (
            discovery.settings.target_repositories
            if target_repositories is None
            else target_repositories
        )
        self.external_targets = target_repositories is not None
        self.events = events or NullEventSink()
        self.selection = selection or default_work_selection_registry().resolve("unseen")
        self.outcomes = index_repository_outcomes(writer.records)
        self._checkpoint_reset = False

    async def _emit_progress(
        self,
        *,
        phase: str,
        progress: float,
        repositories: int,
        repository_target: int | None,
        completed_jobs: int,
        total_jobs: int,
    ) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="discovery_progress",
                stage="repository-discovery",
                message=f"{phase} repository discovery",
                data={
                    "phase": phase,
                    "progress": min(max(progress, 0.0), 1.0),
                    "repositories": repositories,
                    "repository_target": repository_target,
                    "completed_jobs": completed_jobs,
                    "total_jobs": total_jobs,
                },
            )
        )

    @staticmethod
    def _fetch_progress(
        *,
        repositories: int,
        repository_target: int | None,
        completed_jobs: int,
        total_jobs: int,
    ) -> float:
        repository_fraction = repositories / repository_target if repository_target else 0.0
        job_fraction = completed_jobs / total_jobs if total_jobs else 1.0
        return min(max(repository_fraction, job_fraction), 1.0)

    def _manifest_payload(self) -> dict[str, Any]:
        if self.target_repositories:
            mode = "targets"
        elif self.discovery.settings.maximum_repositories is not None:
            mode = "bounded"
        else:
            mode = "lossless"
        payload: dict[str, Any] = {
            "output": self.output,
            "mode": mode,
            "maximum_repositories": self.discovery.settings.maximum_repositories,
            "settings_digest": hashlib.sha256(
                json.dumps(
                    self.discovery.settings.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        if self.external_targets:
            normalized = "\n".join(self.target_repositories).encode()
            payload["target_count"] = len(self.target_repositories)
            payload["target_digest"] = hashlib.sha256(normalized).hexdigest()
        return payload

    def _save_manifest(self, status: JobStatus, result: dict[str, object] | None) -> None:
        self.state.upsert_job(
            self.run_id,
            _MANIFEST_JOB,
            _PLAN_STAGE,
            status,
            self._manifest_payload(),
            result=result,
        )

    def _existing_manifest(self) -> JobRecord | None:
        manifest = self.state.get_job(self.run_id, _MANIFEST_JOB)
        if manifest is None:
            return None
        if manifest.payload != self._manifest_payload():
            self.state.delete_jobs(
                self.run_id,
                (_PLAN_STAGE, _FETCH_STAGE),
            )
            self._checkpoint_reset = True
            return None
        return manifest

    async def _plan_search(self, manifest: JobRecord) -> list[SearchPartition]:
        if manifest.status == "completed":
            raw_result = manifest.result
            result = _mapping(raw_result if raw_result is not None else {}, "result")
            return [
                SearchPartition.model_validate(item)
                for item in _sequence(result.get("partitions", ()), "partitions")
            ]

        raw_result = manifest.result
        result = _mapping(raw_result if raw_result is not None else {}, "planning result")
        pending_raw = result.get("pending")
        accepted_raw = result.get("accepted", ())
        pending = (
            tuple(
                SearchPartition.model_validate(item) for item in _sequence(pending_raw, "pending")
            )
            if pending_raw is not None
            else None
        )
        accepted = tuple(
            SearchPartition.model_validate(item) for item in _sequence(accepted_raw, "accepted")
        )

        async def checkpoint(
            frontier: tuple[SearchPartition, ...],
            accepted_partitions: tuple[SearchPartition, ...],
        ) -> None:
            self._save_manifest(
                "running",
                {
                    "phase": "planning",
                    "pending": [item.model_dump(mode="json") for item in frontier],
                    "accepted": [item.model_dump(mode="json") for item in accepted_partitions],
                },
            )

        partitions = await self.discovery.partitions(
            pending_partitions=pending,
            accepted_partitions=accepted,
            checkpoint=checkpoint,
        )
        return partitions

    def _create_fetch_jobs(self, partitions: list[SearchPartition]) -> None:
        for index, partition in enumerate(partitions):
            job_id = f"partition:{index:08d}"
            if self.state.get_job(self.run_id, job_id) is None:
                self.state.upsert_job(
                    self.run_id,
                    job_id,
                    _FETCH_STAGE,
                    "queued",
                    {
                        "index": index,
                        "kind": "partition",
                        "partition": partition.model_dump(mode="json"),
                    },
                )

    def _create_target_jobs(self) -> None:
        for index, full_name in enumerate(self.target_repositories):
            job_id = f"target:{index:08d}"
            if self.state.get_job(self.run_id, job_id) is None:
                self.state.upsert_job(
                    self.run_id,
                    job_id,
                    _FETCH_STAGE,
                    "queued",
                    {"index": index, "kind": "target", "full_name": full_name},
                )

    def _create_bounded_job(self) -> None:
        job_id = "bounded:00000000"
        if self.state.get_job(self.run_id, job_id) is None:
            self.state.upsert_job(
                self.run_id,
                job_id,
                _FETCH_STAGE,
                "queued",
                {
                    "index": 0,
                    "kind": "bounded",
                    "maximum_repositories": self.discovery.settings.maximum_repositories,
                },
            )

    async def _prepare(self) -> list[JobRecord]:
        manifest = self._existing_manifest()
        if self.resume:
            if manifest is None and not self._checkpoint_reset:
                raise ContractError("discovery run has no checkpoint manifest to resume")
        elif manifest is not None:
            raise ContractError("new discovery run unexpectedly already has a manifest")
        if manifest is None:
            self._save_manifest(
                "running",
                {"phase": "planning", "pending": None, "accepted": []},
            )
            manifest = self._existing_manifest()
            assert manifest is not None

        if manifest.status != "completed":
            plan_result: dict[str, object]
            if self.target_repositories:
                self._create_target_jobs()
                plan_result = {
                    "mode": "targets",
                    "targets": list(self.target_repositories),
                }
            elif self.discovery.settings.maximum_repositories is not None:
                self._create_bounded_job()
                plan_result = {
                    "mode": "bounded",
                    "maximum_repositories": self.discovery.settings.maximum_repositories,
                }
            else:
                partitions = await self._plan_search(manifest)
                self._create_fetch_jobs(partitions)
                plan_result = {
                    "mode": "lossless",
                    "partitions": [item.model_dump(mode="json") for item in partitions],
                }
            self._save_manifest("completed", plan_result)
        return self.state.jobs(self.run_id, _FETCH_STAGE)

    async def _fetch_job(self, job: JobRecord) -> list[RepositoryCandidate]:
        payload = job.payload
        kind = payload.get("kind")
        if kind == "target":
            full_name = payload.get("full_name")
            if not isinstance(full_name, str) or not full_name:
                raise ContractError(f"discovery checkpoint {job.job_id} has no target name")
            return [await self.discovery.fetch_target(full_name)]
        if kind == "bounded":
            return await self.discovery.fetch_bounded()
        if kind != "partition":
            raise ContractError(f"discovery checkpoint {job.job_id} has invalid kind {kind!r}")
        index = payload.get("index")
        if type(index) is not int or index < 0:
            raise ContractError(f"discovery checkpoint {job.job_id} has invalid index")
        partition = SearchPartition.model_validate(payload.get("partition"))
        return await self.discovery.fetch_partition(index, partition)

    def _complete_remaining_for_limit(self, jobs: list[JobRecord], start: int) -> None:
        for job in jobs[start:]:
            if job.status == "completed":
                continue
            self.state.upsert_job(
                self.run_id,
                job.job_id,
                _FETCH_STAGE,
                "completed",
                job.payload,
                attempt=job.attempt,
                result={"not_fetched_due_to_run_limit": True, "appended": 0},
            )

    async def run(self) -> tuple[int, int]:
        jobs = await self._prepare()
        if not self.selection.accepts(WorkState.UNSEEN):
            return self.writer.count_for_run(self.run_id), self.writer.appended
        checkpointed_records = 0
        completed_jobs = 0
        for job in jobs:
            if job.status != "completed":
                continue
            completed_jobs += 1
            raw_result = job.result
            result = _mapping(
                raw_result if raw_result is not None else {},
                f"fetch result for {job.job_id}",
            )
            checkpointed_records += int(result.get("appended", 0))
        if self.writer.count_for_run(self.run_id) < checkpointed_records:
            raise ContractError(
                "discovery output has fewer records than its completed checkpoints; "
                "restore the original JSONL before resuming"
            )
        maximum = self.discovery.settings.maximum_repositories
        repository_target = (
            min(maximum or len(jobs), len(jobs)) if self.target_repositories else maximum
        )
        concurrency = self.discovery.settings.partition_concurrency
        repositories = self.writer.count_for_run(self.run_id)
        await self._emit_progress(
            phase="fetching",
            progress=self._fetch_progress(
                repositories=repositories,
                repository_target=repository_target,
                completed_jobs=completed_jobs,
                total_jobs=len(jobs),
            ),
            repositories=repositories,
            repository_target=repository_target,
            completed_jobs=completed_jobs,
            total_jobs=len(jobs),
        )
        index = 0
        while index < len(jobs):
            if maximum is not None and self.writer.count_for_run(self.run_id) >= maximum:
                self._complete_remaining_for_limit(jobs, index)
                completed_jobs = len(jobs)
                break
            batch: list[tuple[JobRecord, int]] = []
            while index < len(jobs) and len(batch) < concurrency:
                job = jobs[index]
                index += 1
                if job.status == "completed":
                    continue
                attempt = job.attempt + 1
                self.state.upsert_job(
                    self.run_id,
                    job.job_id,
                    _FETCH_STAGE,
                    "running",
                    job.payload,
                    attempt=attempt,
                )
                batch.append((job, attempt))
            if not batch:
                continue

            tasks = [asyncio.create_task(self._fetch_job(job)) for job, _ in batch]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for (job, attempt), task in zip(batch, tasks, strict=True):
                    if task.done() and not task.cancelled() and task.exception() is not None:
                        self.state.upsert_job(
                            self.run_id,
                            job.job_id,
                            _FETCH_STAGE,
                            "failed",
                            job.payload,
                            attempt=attempt,
                            error=str(task.exception()),
                        )
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            limit_reached = False
            for (job, attempt), candidates in zip(batch, results, strict=True):
                appended_for_job = 0
                for candidate in candidates:
                    if candidate.discovery_run_id != self.run_id:
                        raise ContractError("GitHub discovery returned a candidate for another run")
                    if maximum is not None and self.writer.count_for_run(self.run_id) >= maximum:
                        limit_reached = True
                        break
                    aliases = repository_aliases(
                        candidate.clone_url,
                        github_id=candidate.github_id,
                        full_name=candidate.full_name,
                    )
                    if self.selection.accepts(
                        self.outcomes.state_of(aliases)
                    ) and self.writer.append(candidate):
                        appended_for_job += 1
                        self.outcomes.record(aliases, WorkState.SUCCEEDED)
                self.writer.flush()
                self.state.upsert_job(
                    self.run_id,
                    job.job_id,
                    _FETCH_STAGE,
                    "completed",
                    job.payload,
                    attempt=attempt,
                    result={
                        "fetched": len(candidates),
                        "appended": appended_for_job,
                        "truncated_by_run_limit": limit_reached,
                    },
                )
                completed_jobs += 1
                repositories = self.writer.count_for_run(self.run_id)
                await self._emit_progress(
                    phase="fetching",
                    progress=self._fetch_progress(
                        repositories=repositories,
                        repository_target=repository_target,
                        completed_jobs=completed_jobs,
                        total_jobs=len(jobs),
                    ),
                    repositories=repositories,
                    repository_target=repository_target,
                    completed_jobs=completed_jobs,
                    total_jobs=len(jobs),
                )
                if limit_reached:
                    positions = {current.job_id: position for position, current in enumerate(jobs)}
                    remaining_batch_start = positions[job.job_id] + 1
                    self._complete_remaining_for_limit(jobs, remaining_batch_start)
                    completed_jobs = len(jobs)
                    break
            if limit_reached:
                break
        repositories = self.writer.count_for_run(self.run_id)
        await self._emit_progress(
            phase="completed",
            progress=1.0,
            repositories=repositories,
            repository_target=repository_target,
            completed_jobs=len(jobs),
            total_jobs=len(jobs),
        )
        return repositories, self.writer.appended
