"""Durable, run-local diagnostics for Environment Build postmortems."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.planning import EnvironmentAnalysis, EnvironmentArtifacts
from codeworld.models import StrictModel
from codeworld.run_layout import RunLayout
from codeworld.storage.files import atomic_write_bytes
from codeworld.storage.jsonl import append_jsonl_object

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class DiagnosticArtifact(StrictModel):
    kind: Literal["dockerfile", "activation", "runner"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    path: str = Field(min_length=1)
    object_path: str = Field(min_length=1)


class RepositoryDiagnosticSummary(StrictModel):
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    github_id: int | None = None
    repository: str = Field(min_length=1)
    ref: str | None = None
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    environment_id: str | None = None
    status: Literal["queued", "running", "succeeded", "failed", "rejected", "skipped"]
    invocation_count: int = Field(default=0, ge=0)
    attempt_count: int = Field(default=0, ge=0)
    failed_phase: str | None = None
    error_type: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


def _atomic_json(path: Path, value: StrictModel | dict[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


class BuildDiagnosticStore:
    """One Build run's content-addressed diagnostic object store."""

    def __init__(self, run_directory: str | Path, *, run_id: str) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.root = RunLayout(self.run_directory).diagnostics
        self.objects = self.root / "objects"
        self.repositories = self.root / "repositories"
        self.run_id = run_id
        self.objects.mkdir(parents=True, exist_ok=True)
        self.repositories.mkdir(parents=True, exist_ok=True)

    def repository(
        self,
        *,
        repository: str,
        ref: str | None,
        task_id: str | None,
        github_id: int | None,
    ) -> RepositoryDiagnostics:
        identity = task_id or repository
        safe = _SAFE_COMPONENT.sub("_", identity).strip("._-")[:80] or "repository"
        digest = hashlib.sha256(repository.encode("utf-8")).hexdigest()[:12]
        return RepositoryDiagnostics(
            store=self,
            root=self.repositories / f"{safe}-{digest}",
            repository=repository,
            ref=ref,
            task_id=identity,
            github_id=github_id,
        )

    def store_text(self, kind: str, content: str) -> tuple[Path, str, int]:
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        suffix = {"dockerfile": "Dockerfile", "activation": "activate", "runner": "runner"}[kind]
        path = self.objects / f"{digest}.{suffix}"
        if not path.exists():
            atomic_write_bytes(path, encoded, executable=kind in {"activation", "runner"})
        return path, digest, len(encoded)


class RepositoryDiagnostics:
    """Human-readable evidence and retention ownership for one repository."""

    def __init__(
        self,
        *,
        store: BuildDiagnosticStore,
        root: Path,
        repository: str,
        ref: str | None,
        task_id: str,
        github_id: int | None,
    ) -> None:
        self.store = store
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.root / "summary.json"
        self.events_path = self.root / "events.jsonl"
        self.agent_runs_path = self.root / "agent-runs.jsonl"
        self._trajectory_paths = self._load_trajectory_paths()
        now = datetime.now(UTC)
        if self.summary_path.is_file():
            self.summary = RepositoryDiagnosticSummary.model_validate_json(
                self.summary_path.read_text(encoding="utf-8")
            )
            if self.summary.run_id != store.run_id or self.summary.repository != repository:
                raise ValueError(f"diagnostic identity mismatch: {self.summary_path}")
        else:
            self.summary = RepositoryDiagnosticSummary(
                run_id=store.run_id,
                task_id=task_id,
                github_id=github_id,
                repository=repository,
                ref=ref,
                status="queued",
                created_at=now,
                updated_at=now,
            )
            self._save_summary()

    def _load_trajectory_paths(self) -> set[Path]:
        """Recover retention ownership when a run is resumed in the same directory."""
        paths: set[Path] = set()
        if not self.agent_runs_path.is_file():
            return paths
        for line in self.agent_runs_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            trajectory = record.get("trajectory_path")
            if isinstance(trajectory, str) and trajectory:
                paths.add(Path(trajectory).resolve())
        return paths

    @property
    def relative_path(self) -> str:
        return str(self.root.relative_to(self.store.run_directory))

    def _save_summary(self) -> None:
        _atomic_json(self.summary_path, self.summary)

    def _update_summary(self, **changes: object) -> None:
        self.summary = self.summary.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._save_summary()

    def append_event(
        self,
        kind: str,
        *,
        attempt: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        append_jsonl_object(
            self.events_path,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": kind,
                "attempt": attempt,
                "data": data or {},
            },
        )

    def begin(self) -> None:
        self._update_summary(
            status="running",
            invocation_count=self.summary.invocation_count + 1,
            failed_phase=None,
            error_type=None,
            error=None,
        )
        self.append_event("repository_started")

    def set_commit(self, commit: str, environment_id: str) -> None:
        self._update_summary(commit=commit, environment_id=environment_id)
        self.append_event(
            "repository_materialized",
            data={"commit": commit, "environment_id": environment_id},
        )

    def record_analysis(self, analysis: EnvironmentAnalysis) -> None:
        _atomic_json(self.root / "analysis.json", analysis.model_dump(mode="json"))
        self.append_event("analysis_recorded")

    def _link_artifact(
        self,
        *,
        attempt_directory: Path,
        kind: Literal["dockerfile", "activation", "runner"],
        content: str,
    ) -> DiagnosticArtifact:
        object_path, digest, size = self.store.store_text(kind, content)
        name = {"dockerfile": "Dockerfile", "activation": "activate", "runner": "runner"}[kind]
        destination = attempt_directory / name
        if not destination.exists():
            os.link(object_path, destination)
        return DiagnosticArtifact(
            kind=kind,
            sha256=digest,
            bytes=size,
            path=str(destination.relative_to(self.store.run_directory)),
            object_path=str(object_path.relative_to(self.store.run_directory)),
        )

    def record_attempt(
        self,
        *,
        attempt: int,
        dockerfile: str,
        artifacts: EnvironmentArtifacts,
    ) -> None:
        sequence = self.summary.attempt_count + 1
        attempt_directory = self.root / "attempts" / f"{sequence:04d}"
        attempt_directory.mkdir(parents=True, exist_ok=True)
        references = (
            self._link_artifact(
                attempt_directory=attempt_directory,
                kind="dockerfile",
                content=dockerfile,
            ),
            self._link_artifact(
                attempt_directory=attempt_directory,
                kind="activation",
                content=artifacts.activation,
            ),
            self._link_artifact(
                attempt_directory=attempt_directory,
                kind="runner",
                content=artifacts.runner,
            ),
        )
        _atomic_json(
            attempt_directory / "manifest.json",
            {
                "attempt": attempt,
                "attempt_sequence": sequence,
                "artifacts": [item.model_dump(mode="json") for item in references],
            },
        )
        self._update_summary(attempt_count=sequence)
        self.append_event(
            "attempt_started",
            attempt=attempt,
            data={
                "attempt_sequence": sequence,
                "directory": str(attempt_directory.relative_to(self.store.run_directory)),
                "artifacts": [item.model_dump(mode="json") for item in references],
            },
        )

    def record_feedback(
        self,
        *,
        attempt: int,
        feedback: AgentFeedback,
        failed_phase: str,
    ) -> None:
        sequence = self.summary.attempt_count
        attempt_directory = self.root / "attempts" / f"{sequence:04d}"
        _atomic_json(attempt_directory / "feedback.json", feedback.model_dump(mode="json"))
        self._update_summary(failed_phase=failed_phase)
        self.append_event(
            "attempt_failed",
            attempt=attempt,
            data={
                "attempt_sequence": sequence,
                "failed_phase": failed_phase,
                "feedback": feedback.model_dump(mode="json"),
            },
        )

    def record_validation(self, data: dict[str, object]) -> None:
        sequence = self.summary.attempt_count
        attempt_directory = self.root / "attempts" / f"{sequence:04d}"
        _atomic_json(attempt_directory / "validation.json", data)
        failed_phase = data.get("failed_phase")
        if isinstance(failed_phase, str):
            self._update_summary(failed_phase=failed_phase)

    def record_agent_run(
        self,
        *,
        role: str,
        kind: str,
        attempt: int | None,
        data: dict[str, object],
    ) -> None:
        trajectory = data.get("trajectory_path")
        if isinstance(trajectory, str) and trajectory:
            self._trajectory_paths.add(Path(trajectory).resolve())
        append_jsonl_object(
            self.agent_runs_path,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": kind,
                "role": role,
                "attempt": attempt,
                **data,
            },
        )

    def finalize(
        self,
        status: Literal["succeeded", "failed", "rejected", "skipped"],
        *,
        error: BaseException | None = None,
        error_type: str | None = None,
        error_text: str | None = None,
        failed_phase: str | None = None,
    ) -> None:
        detail = error_text
        if detail is None and error is not None:
            detail = str(error)[-50_000:] or repr(error)
        observed_type = error_type or (type(error).__name__ if error is not None else None)
        self._update_summary(
            status=status,
            failed_phase=failed_phase or self.summary.failed_phase,
            error_type=observed_type,
            error=detail,
        )
        self.append_event(
            "repository_completed",
            data={
                "status": status,
                "failed_phase": failed_phase or self.summary.failed_phase,
                "error_type": observed_type,
                "error": detail,
            },
        )

    def remove_successful_trajectories(self, configured_root: str | Path) -> tuple[str, ...]:
        """Remove only exact Agent directories recorded for this successful repository."""
        root = Path(configured_root).resolve()
        removed: list[str] = []
        for path in sorted(self._trajectory_paths):
            if path.parent != root or not path.name:
                raise ValueError(f"refusing to remove trajectory outside configured root: {path}")
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
        if removed:
            self.append_event("successful_trajectories_removed", data={"paths": removed})
        return tuple(removed)
