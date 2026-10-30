"""The one complete state transition for building one repository environment."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from codeworld.agents.factory import AgentFactory
from codeworld.config import AgentSettings, EnvironmentSettings
from codeworld.environment.build_diagnostics import (
    BuildDiagnosticStore,
    RepositoryDiagnostics,
)
from codeworld.environment.builder import EnvironmentBuilder, validate_portable_repository_url
from codeworld.environment.models import EnvironmentBuildFailure, EnvironmentRecord
from codeworld.errors import RepositoryMaterializationRejected, RepositoryRejected
from codeworld.events import Event, EventSink, NullEventSink
from codeworld.repositories.identity import repository_aliases
from codeworld.repositories.models import RepositoryCandidate
from codeworld.repositories.staging import RepositoryStager
from codeworld.runtime.container import DockerRuntime


@dataclass(frozen=True, slots=True)
class EnvironmentBuildTarget:
    """Repository identity and provenance required by one Build job."""

    repository_url: str
    ref: str | None
    task_id: str
    state_payload: dict[str, object]
    github_id: int | None = None

    @property
    def job_id(self) -> str:
        return str(self.github_id) if self.github_id is not None else self.task_id

    @property
    def outcome_aliases(self) -> tuple[str, ...]:
        return repository_aliases(
            self.repository_url,
            github_id=self.github_id,
            full_name=self.task_id if self.github_id is not None else None,
        )

    @classmethod
    def from_candidate(cls, candidate: RepositoryCandidate) -> EnvironmentBuildTarget:
        return cls(
            repository_url=candidate.clone_url,
            ref=candidate.default_branch,
            task_id=candidate.full_name,
            github_id=candidate.github_id,
            state_payload=candidate.model_dump(mode="json"),
        )

    @classmethod
    def direct(cls, repository_url: str, ref: str | None) -> EnvironmentBuildTarget:
        return cls(
            repository_url=repository_url,
            ref=ref,
            task_id=repository_url,
            state_payload={"repository": repository_url, "ref": ref},
        )


@dataclass(frozen=True, slots=True)
class EnvironmentBuildOutcome:
    """One terminal Build-job outcome, not yet committed to output JSONL."""

    target: EnvironmentBuildTarget
    diagnostics: RepositoryDiagnostics
    record: EnvironmentRecord | None = None
    failure: EnvironmentBuildFailure | None = None
    skipped: bool = False
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        terminals = sum(
            (
                self.record is not None,
                self.failure is not None,
                self.skipped,
            )
        )
        if terminals != 1:
            raise ValueError("Build outcome must contain exactly one terminal result")
        if self.cause is not None and self.failure is None:
            raise ValueError("Build outcome cause requires a failure record")


class EnvironmentBuildJob:
    """Materialize, build, time-bound, and classify exactly one repository."""

    def __init__(
        self,
        *,
        runtime: DockerRuntime,
        settings: EnvironmentSettings,
        agents: AgentFactory[AgentSettings],
        stager: RepositoryStager,
        diagnostics: BuildDiagnosticStore,
        events: EventSink | None = None,
        run_id: str = "environment-build",
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.agents = agents
        self.stager = stager
        self.diagnostics = diagnostics
        self.events = events or NullEventSink()
        self.run_id = run_id

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

    @staticmethod
    def _failure(
        target: EnvironmentBuildTarget,
        diagnostics: RepositoryDiagnostics,
        error: BaseException,
        *,
        commit: str | None,
        disposition: Literal["failed", "rejected"] = "failed",
        category: str | None = None,
        failed_phase: str | None = None,
        detail: str | None = None,
    ) -> EnvironmentBuildFailure:
        rendered = detail if detail is not None else (str(error)[-50_000:] or repr(error))
        return EnvironmentBuildFailure(
            run_id=diagnostics.summary.run_id,
            task_id=target.task_id,
            github_id=target.github_id,
            repository=target.repository_url,
            ref=target.ref,
            commit=commit,
            disposition=disposition,
            category=category,
            environment_id=diagnostics.summary.environment_id,
            failed_phase=(
                failed_phase
                or diagnostics.summary.failed_phase
                or ("materialization" if commit is None else "environment_build")
            ),
            attempts=diagnostics.summary.attempt_count,
            diagnostics_path=diagnostics.relative_path,
            error_type=type(error).__name__,
            error=rendered,
        )

    async def run(
        self,
        target: EnvironmentBuildTarget,
        *,
        existing_environment_ids: frozenset[str],
    ) -> EnvironmentBuildOutcome:
        """Return one classified outcome; persistence is the producer's commit boundary."""
        commit: str | None = None
        record: EnvironmentRecord | None = None
        diagnostics = self.diagnostics.repository(
            repository=target.repository_url,
            ref=target.ref,
            task_id=target.task_id,
            github_id=target.github_id,
        )
        diagnostics.begin()
        deadline = asyncio.timeout(self.settings.total_timeout_seconds)
        try:
            async with deadline:
                validate_portable_repository_url(target.repository_url)
                await self._emit(target, "materialize_started", "materializing repository")
                async with self.stager.stage(target.repository_url, target.ref) as (
                    repository,
                    resolved_commit,
                ):
                    commit = resolved_commit
                    environment_id = EnvironmentBuilder.environment_id(
                        target.repository_url,
                        commit,
                    )
                    diagnostics.set_commit(commit, environment_id)
                    if environment_id in existing_environment_ids:
                        return EnvironmentBuildOutcome(
                            target=target,
                            diagnostics=diagnostics,
                            skipped=True,
                        )
                    await self._emit(
                        target,
                        "materialize_completed",
                        f"resolved {commit[:12]}",
                        commit=commit,
                    )
                    record = await EnvironmentBuilder(
                        self.runtime,
                        self.settings,
                        agents=self.agents,
                        events=self.events,
                        run_id=self.run_id,
                        task_id=target.task_id,
                        diagnostics=diagnostics,
                    ).build(
                        repository=repository,
                        repository_url=target.repository_url,
                        commit=commit,
                    )
                return EnvironmentBuildOutcome(
                    target=target,
                    diagnostics=diagnostics,
                    record=record,
                )
        except TimeoutError as exc:
            if not deadline.expired():
                return EnvironmentBuildOutcome(
                    target=target,
                    diagnostics=diagnostics,
                    failure=self._failure(target, diagnostics, exc, commit=commit),
                    cause=exc,
                )
            if record is not None and self.settings.cleanup_intermediate_images:
                with suppress(Exception):
                    await self.runtime.remove_image(record.image)
            seconds = self.settings.total_timeout_seconds
            detail = (
                "Environment Build exceeded the per-repository total wall-clock "
                f"deadline of {seconds:g}s"
            )
            await self._emit(
                target,
                "environment_timed_out",
                detail,
                timeout_seconds=seconds,
                image=record.image if record is not None else None,
            )
            return EnvironmentBuildOutcome(
                target=target,
                diagnostics=diagnostics,
                failure=self._failure(
                    target,
                    diagnostics,
                    exc,
                    commit=commit,
                    category="environment_total_timeout",
                    failed_phase=diagnostics.summary.failed_phase or "repository_total_timeout",
                    detail=detail,
                ),
                cause=exc,
            )
        except RepositoryMaterializationRejected as exc:
            return EnvironmentBuildOutcome(
                target=target,
                diagnostics=diagnostics,
                failure=self._failure(
                    target,
                    diagnostics,
                    exc,
                    commit=commit,
                    disposition="rejected",
                    category=exc.category,
                    failed_phase="materialization",
                    detail=exc.reason,
                ),
                cause=exc,
            )
        except RepositoryRejected as exc:
            return EnvironmentBuildOutcome(
                target=target,
                diagnostics=diagnostics,
                failure=self._failure(
                    target,
                    diagnostics,
                    exc,
                    commit=commit,
                    disposition="rejected",
                    category=exc.category,
                    failed_phase=diagnostics.summary.failed_phase or "analysis_policy",
                    detail=exc.reason,
                ),
                cause=exc,
            )
        except Exception as exc:
            return EnvironmentBuildOutcome(
                target=target,
                diagnostics=diagnostics,
                failure=self._failure(target, diagnostics, exc, commit=commit),
                cause=exc,
            )
