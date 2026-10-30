"""Scheduling and durable persistence for Environment Build jobs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field

from codeworld.agents.factory import AgentFactory
from codeworld.config import AgentSettings, FactorySettings
from codeworld.environment.build_diagnostics import (
    BuildDiagnosticStore,
    RepositoryDiagnostics,
)
from codeworld.environment.bundle import write_environment_bundle
from codeworld.environment.job import (
    EnvironmentBuildJob,
    EnvironmentBuildOutcome,
    EnvironmentBuildTarget,
)
from codeworld.environment.jsonl import (
    EnvironmentFailureJsonlAppender,
    EnvironmentJsonlAppender,
    index_environment_build_outcomes,
)
from codeworld.errors import ConfigurationError
from codeworld.events import Event, EventSink, NullEventSink
from codeworld.models import StrictModel
from codeworld.repositories.jsonl import iter_repositories
from codeworld.repositories.staging import RepositoryStager
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import WorkSelection, default_work_selection_registry
from codeworld.state import JobStatus, StateStore

_TerminalStatus = Literal["built", "skipped", "rejected", "failed"]


class EnvironmentProductionSummary(StrictModel):
    submitted: int = Field(ge=0)
    built: int = Field(ge=0)
    skipped: int = Field(ge=0)
    rejected: int = Field(ge=0)
    failed: int = Field(ge=0)


def _output_directory(path: str | Path) -> Path:
    return Path(path).resolve().parent


class EnvironmentProducer:
    """Schedule Build jobs and commit their terminal outcomes durably."""

    def __init__(
        self,
        *,
        settings: FactorySettings,
        runtime: DockerRuntime,
        agents: AgentFactory[AgentSettings],
        events: EventSink | None = None,
        state: StateStore | None = None,
        run_id: str = "environment-build",
        selection: WorkSelection | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.agents = agents
        self.events = events or NullEventSink()
        self.state = state
        self.run_id = run_id
        self.selection = selection or default_work_selection_registry().resolve("unseen")
        self.stager = RepositoryStager(
            settings.workspace / "staging",
            settings.repository,
            sources=runtime.sources,
        )

    async def _emit(
        self,
        target: EnvironmentBuildTarget,
        kind: str,
        message: str,
        **data: object,
    ) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind=kind,
                stage="environment-build",
                task_id=target.task_id,
                message=message,
                data=dict(data),
            )
        )

    def _state_job(
        self,
        target: EnvironmentBuildTarget,
        status: JobStatus,
        *,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        if self.state is None:
            return
        self.state.upsert_job(
            self.run_id,
            target.job_id,
            "environment-build",
            status,
            target.state_payload,
            result=result,
            error=error,
        )

    async def _preflight(self) -> None:
        if self.agents.settings.model is None:
            raise ConfigurationError(
                "Environment Build requires agent.model in build.yaml and its provider "
                "credential in .env"
            )
        await self.runtime.check_available()
        await self.stager.check_available()

    def _new_job(self, diagnostics: BuildDiagnosticStore) -> EnvironmentBuildJob:
        return EnvironmentBuildJob(
            runtime=self.runtime,
            settings=self.settings.environment,
            agents=self.agents,
            stager=self.stager,
            diagnostics=diagnostics,
            events=self.events,
            run_id=self.run_id,
        )

    async def _execute(
        self,
        job: EnvironmentBuildJob,
        target: EnvironmentBuildTarget,
        existing_environment_ids: frozenset[str],
    ) -> EnvironmentBuildOutcome:
        self._state_job(target, "running")
        return await job.run(
            target,
            existing_environment_ids=existing_environment_ids,
        )

    async def _finalize_success(
        self,
        diagnostics: RepositoryDiagnostics,
        status: Literal["succeeded", "skipped"],
    ) -> None:
        diagnostics.finalize(status)
        trajectory = self.agents.settings.trajectory
        if trajectory.enabled and trajectory.retention == "failures":
            await asyncio.to_thread(
                diagnostics.remove_successful_trajectories,
                trajectory.directory,
            )

    async def _commit_outcome(
        self,
        outcome: EnvironmentBuildOutcome,
        *,
        output_writer: EnvironmentJsonlAppender,
        failure_writer: EnvironmentFailureJsonlAppender,
    ) -> _TerminalStatus:
        """Persist first, then expose the corresponding terminal job state."""
        target = outcome.target
        diagnostics = outcome.diagnostics
        if outcome.record is not None:
            appended = output_writer.append(outcome.record)
            output_writer.flush()
            skipped = not appended
            self._state_job(
                target,
                "completed",
                result={
                    "environment_id": outcome.record.environment_id,
                    "skipped": skipped,
                    "diagnostics_path": diagnostics.relative_path,
                    "attempts": diagnostics.summary.attempt_count,
                },
            )
            if skipped:
                await self._emit(
                    target,
                    "environment_skipped",
                    "environment already exists in output JSONL",
                    environment_id=outcome.record.environment_id,
                )
                await self._finalize_success(diagnostics, "skipped")
                return "skipped"

            await self._emit(
                target,
                "environment_completed",
                "validated environment is durable in output JSONL",
                environment_id=outcome.record.environment_id,
                image=outcome.record.image,
            )
            await self._finalize_success(diagnostics, "succeeded")
            if self.settings.environment.audit.enabled:
                write_environment_bundle(
                    self.settings.environment.audit.directory / outcome.record.environment_id,
                    outcome.record,
                )
            return "built"

        if outcome.failure is not None:
            failure = outcome.failure
            failure_writer.append(failure)
            if failure.disposition == "rejected":
                diagnostics.finalize(
                    "rejected",
                    error=outcome.cause,
                    error_type=failure.error_type,
                    error_text=failure.error,
                    failed_phase=failure.failed_phase,
                )
                self._state_job(
                    target,
                    "rejected",
                    result={
                        "diagnostics_path": failure.diagnostics_path,
                        "attempts": failure.attempts,
                        "failed_phase": failure.failed_phase,
                    },
                    error=failure.error,
                )
                await self._emit(
                    target,
                    "environment_rejected",
                    "repository is outside the configured acceptance policy",
                    category=failure.category,
                    reason=failure.error,
                )
                return "rejected"

            diagnostics.finalize(
                "failed",
                error=outcome.cause,
                error_type=failure.error_type,
                error_text=failure.error,
                failed_phase=failure.failed_phase,
            )
            self._state_job(
                target,
                "failed",
                result={
                    "diagnostics_path": failure.diagnostics_path,
                    "attempts": failure.attempts,
                    "failed_phase": failure.failed_phase,
                },
                error=failure.error,
            )
            await self._emit(
                target,
                "environment_failed",
                "Environment Build failed for repository",
                error_type=failure.error_type,
                error=failure.error,
            )
            return "failed"

        if outcome.skipped:
            environment_id = diagnostics.summary.environment_id
            self._state_job(
                target,
                "completed",
                result={
                    "environment_id": environment_id,
                    "skipped": True,
                    "diagnostics_path": diagnostics.relative_path,
                    "attempts": diagnostics.summary.attempt_count,
                },
            )
            await self._emit(
                target,
                "environment_skipped",
                "environment already exists in output JSONL",
                environment_id=environment_id,
            )
            await self._finalize_success(diagnostics, "skipped")
            return "skipped"

        raise RuntimeError("EnvironmentBuildJob returned no terminal outcome")

    async def _emit_summary(self, summary: EnvironmentProductionSummary) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="environment_production_completed",
                stage="environment-build",
                message=f"completed {summary.submitted} repository build jobs",
                data=summary.model_dump(mode="json"),
            )
        )

    async def produce_one(
        self,
        *,
        repository: str,
        ref: str | None,
        output: str | Path,
        failures: str | Path,
    ) -> EnvironmentBuildOutcome | None:
        """Run the same Build job and durable commit path used by batch production."""
        target = EnvironmentBuildTarget.direct(repository, ref)
        diagnostics = BuildDiagnosticStore(_output_directory(output), run_id=self.run_id)
        with (
            EnvironmentJsonlAppender(
                output,
                fsync_interval=self.settings.jsonl_fsync_interval,
            ) as output_writer,
            EnvironmentFailureJsonlAppender(failures) as failure_writer,
        ):
            existing_ids = frozenset(record.environment_id for record in output_writer.records)
            outcomes = index_environment_build_outcomes(
                output_writer.records,
                failure_writer.records,
            )
            state = outcomes.state_of(target.outcome_aliases)
            if not self.selection.accepts(state):
                summary = EnvironmentProductionSummary(
                    submitted=0,
                    built=0,
                    skipped=1,
                    rejected=0,
                    failed=0,
                )
                await self._emit(
                    target,
                    "environment_skipped",
                    "repository already has a durable terminal output",
                    prior_state=state.value,
                    selection=self.selection.name,
                )
                await self._emit_summary(summary)
                return None
            await self._preflight()
            self._state_job(target, "queued")
            outcome = await self._execute(self._new_job(diagnostics), target, existing_ids)
            status = await self._commit_outcome(
                outcome,
                output_writer=output_writer,
                failure_writer=failure_writer,
            )

        summary = EnvironmentProductionSummary(
            submitted=1,
            built=int(status == "built"),
            skipped=int(status == "skipped"),
            rejected=int(status == "rejected"),
            failed=int(status == "failed"),
        )
        await self._emit_summary(summary)
        if outcome.failure is not None:
            if outcome.cause is None:
                raise RuntimeError(outcome.failure.error)
            raise outcome.cause
        return outcome

    async def produce(
        self,
        *,
        repositories: str | Path,
        output: str | Path,
        failures: str | Path,
        limit: int | None = None,
    ) -> EnvironmentProductionSummary:
        """Build a bounded concurrent stream without one bad repository stopping the batch."""
        if limit is not None and limit < 1:
            raise ValueError("Environment Build limit must be at least one")
        diagnostic_store = BuildDiagnosticStore(_output_directory(output), run_id=self.run_id)
        job = self._new_job(diagnostic_store)

        counts: dict[_TerminalStatus, int] = {
            "built": 0,
            "skipped": 0,
            "rejected": 0,
            "failed": 0,
        }
        submitted = 0
        considered = 0
        preflight_complete = False
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        source = iter(iter_repositories(repositories))
        exhausted = False
        pending: set[asyncio.Task[EnvironmentBuildOutcome]] = set()

        with (
            EnvironmentJsonlAppender(
                output,
                fsync_interval=self.settings.jsonl_fsync_interval,
            ) as output_writer,
            EnvironmentFailureJsonlAppender(failures) as failure_writer,
        ):
            existing_ids = frozenset(record.environment_id for record in output_writer.records)
            outcomes = index_environment_build_outcomes(
                output_writer.records,
                failure_writer.records,
            )
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < self.settings.concurrency:
                        try:
                            candidate = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        name_key = candidate.full_name.casefold()
                        if candidate.github_id in seen_ids or name_key in seen_names:
                            continue
                        seen_ids.add(candidate.github_id)
                        seen_names.add(name_key)
                        if limit is not None and considered >= limit:
                            exhausted = True
                            break
                        considered += 1
                        target = EnvironmentBuildTarget.from_candidate(candidate)
                        state = outcomes.state_of(target.outcome_aliases)
                        if not self.selection.accepts(state):
                            counts["skipped"] += 1
                            await self._emit(
                                target,
                                "environment_skipped",
                                "repository already has a durable terminal output",
                                prior_state=state.value,
                                selection=self.selection.name,
                            )
                            continue
                        if not preflight_complete:
                            await self._preflight()
                            preflight_complete = True
                        submitted += 1
                        self._state_job(target, "queued")
                        pending.add(asyncio.create_task(self._execute(job, target, existing_ids)))
                    if not pending:
                        continue
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        outcome = await task
                        status = await self._commit_outcome(
                            outcome,
                            output_writer=output_writer,
                            failure_writer=failure_writer,
                        )
                        counts[status] += 1
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        summary = EnvironmentProductionSummary(
            submitted=submitted,
            built=counts["built"],
            skipped=counts["skipped"],
            rejected=counts["rejected"],
            failed=counts["failed"],
        )
        await self._emit_summary(summary)
        return summary
