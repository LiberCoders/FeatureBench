"""Run-local, Seed-addressed diagnostics for FeatureBench generation."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from codeworld.adapters.base import QualityGateResult
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.models import (
    InterfaceDescription,
    IssueDraft,
    MaskPlan,
    QualityReport,
    RepositoryIndexData,
    SourceObject,
    TraceBundle,
)
from codeworld.environment.models import EnvironmentRecord
from codeworld.models import StrictModel, TaskRecord
from codeworld.run_layout import RunLayout
from codeworld.selection import WorkState
from codeworld.storage.files import atomic_write_bytes
from codeworld.storage.jsonl import append_jsonl_object

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_COMPRESSION_THRESHOLD_BYTES = 1_000_000
SeedStatus = Literal[
    "queued",
    "running",
    "awaiting_quality_gate",
    "succeeded",
    "rejected",
    "failed",
    "interrupted",
    "skipped",
]


class FeatureBenchSeedSummary(StrictModel):
    run_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    seed_test_file: str = Field(min_length=1)
    status: SeedStatus = "queued"
    phase: str = "queued"
    invocation_count: int = Field(default=0, ge=0)
    validation_count: int = Field(default=0, ge=0)
    task_id: str | None = None
    error_type: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class FeatureBenchFailure(StrictModel):
    run_id: str
    environment_id: str
    seed_test_file: str
    disposition: Literal["failed", "rejected", "interrupted"]
    phase: str
    invocation: int = Field(ge=0)
    validations: int = Field(ge=0)
    task_id: str | None = None
    diagnostics_path: str
    error_type: str
    error: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _atomic_json(path: Path, value: StrictModel | dict[str, object] | list[object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _append_jsonl(path: Path, value: StrictModel | dict[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    append_jsonl_object(path, payload)


class FeatureBenchDiagnostics:
    """One Generate run's Adapter-owned diagnostics and content-addressed objects."""

    def __init__(
        self,
        run_directory: Path,
        *,
        run_id: str,
        environment: EnvironmentRecord,
        settings: FeatureBenchSettings,
    ) -> None:
        self.run_directory = run_directory.resolve()
        safe_environment = (
            _SAFE_COMPONENT.sub("_", environment.environment_id).strip("._-")[-80:] or "environment"
        )
        environment_digest = hashlib.sha256(environment.environment_id.encode()).hexdigest()[:12]
        self.root = (
            RunLayout(self.run_directory).diagnostics
            / "featurebench"
            / "environments"
            / f"{safe_environment}-{environment_digest}"
        )
        self.objects = self.root / "objects"
        self.seeds_root = self.root / "seeds"
        self.failures_path = self.root / "failures.jsonl"
        self.runner_capacity_path = self.root / "runner-capacity.json"
        self.run_id = run_id
        self.environment = environment
        self.settings = settings
        self._seeds: dict[str, FeatureBenchSeedDiagnostics] = {}
        self._tasks: dict[str, FeatureBenchSeedDiagnostics] = {}
        self.objects.mkdir(parents=True, exist_ok=True)
        self.seeds_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.root / "environment.json", environment)
        _atomic_json(self.root / "settings.json", settings)
        self._write_run_summary("running", None)

    def _write_run_summary(self, status: str, error: BaseException | None) -> None:
        summaries = [seed.summary for seed in self.seeds()]
        _atomic_json(
            self.root / "summary.json",
            {
                "run_id": self.run_id,
                "environment_id": self.environment.environment_id,
                "repository": self.environment.repository,
                "commit": self.environment.commit,
                "status": status,
                "seeds": len(summaries),
                "succeeded": sum(item.status == "succeeded" for item in summaries),
                "rejected": sum(item.status == "rejected" for item in summaries),
                "failed": sum(item.status == "failed" for item in summaries),
                "interrupted": sum(item.status == "interrupted" for item in summaries),
                "skipped": sum(item.status == "skipped" for item in summaries),
                "error_type": type(error).__name__ if error is not None else None,
                "error": (str(error)[-50_000:] or repr(error)) if error is not None else None,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def store_bytes(self, kind: str, content: bytes) -> Path:
        digest = hashlib.sha256(content).hexdigest()
        safe_kind = _SAFE_COMPONENT.sub("_", kind).strip("._-") or "object"
        path = self.objects / f"{digest}.{safe_kind}"
        if not path.exists():
            atomic_write_bytes(path, content)
        return path

    def link_bytes(self, destination: Path, *, kind: str, content: bytes) -> None:
        source = self.store_bytes(kind, content)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        os.link(source, destination)

    def link_json(
        self,
        destination: Path,
        value: StrictModel | dict[str, object] | list[object],
    ) -> Path:
        payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        if len(encoded) >= _COMPRESSION_THRESHOLD_BYTES:
            destination = destination.with_name(destination.name + ".gz")
            encoded = gzip.compress(encoded, compresslevel=6, mtime=0)
            kind = "json.gz"
        else:
            kind = "json"
        self.link_bytes(destination, kind=kind, content=encoded)
        return destination

    def record_shared_inventory(
        self,
        *,
        index: RepositoryIndexData,
        test_ids_by_file: dict[str, tuple[str, ...]],
        test_source_paths: dict[str, str],
    ) -> None:
        shared = self.root / "shared"
        self.link_json(shared / "repository-index.json", index)
        self.link_json(
            shared / "test-inventory.json",
            {
                "test_ids_by_file": test_ids_by_file,
                "test_source_paths": test_source_paths,
            },
        )

    def runner_capacity_signature(
        self,
        test_ids_by_file: Mapping[str, tuple[str, ...]],
    ) -> dict[str, object]:
        inventory = {path: sorted(test_ids_by_file[path]) for path in sorted(test_ids_by_file)}
        inventory_sha256 = hashlib.sha256(
            json.dumps(
                inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "environment_id": self.environment.environment_id,
            "image": self.environment.image,
            "runtime_resources": self.environment.runtime.resources.model_dump(mode="json"),
            "full_test_timeout_seconds": self.settings.full_test_timeout_seconds,
            "maximum_selected_p2p_test_files": (self.settings.maximum_selected_p2p_test_files),
            "environment_concurrency": self.settings.environment_concurrency,
            "seed_concurrency": self.settings.seed_concurrency,
            "test_inventory_sha256": inventory_sha256,
        }

    def load_runner_capacity(self, signature: Mapping[str, object]) -> int | None:
        if not self.runner_capacity_path.is_file():
            return None
        try:
            raw = cast(
                object,
                json.loads(self.runner_capacity_path.read_text(encoding="utf-8")),
            )
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        payload = cast(dict[str, object], raw)
        if payload.get("format") != 1 or payload.get("signature") != dict(signature):
            return None
        capacity = payload.get("test_capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            return None
        return capacity

    def record_runner_capacity(
        self,
        *,
        signature: Mapping[str, object],
        test_capacity: int,
        source_seed_test_file: str,
    ) -> int:
        if test_capacity < 1:
            raise ValueError("Runner test capacity must be positive")
        previous = self.load_runner_capacity(signature)
        if previous is not None and previous <= test_capacity:
            return previous
        _atomic_json(
            self.runner_capacity_path,
            {
                "format": 1,
                "signature": dict(signature),
                "test_capacity": test_capacity,
                "source_seed_test_file": source_seed_test_file,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return test_capacity

    def seed(self, seed_test_file: str) -> FeatureBenchSeedDiagnostics:
        existing = self._seeds.get(seed_test_file)
        if existing is not None:
            return existing
        safe = _SAFE_COMPONENT.sub("_", seed_test_file).strip("._-")[-80:] or "seed"
        digest = hashlib.sha256(seed_test_file.encode()).hexdigest()[:12]
        diagnostic = FeatureBenchSeedDiagnostics(
            store=self,
            root=self.seeds_root / f"{safe}-{digest}",
            seed_test_file=seed_test_file,
        )
        self._seeds[seed_test_file] = diagnostic
        if diagnostic.summary.task_id is not None:
            self._tasks[diagnostic.summary.task_id] = diagnostic
        return diagnostic

    def seeds(self) -> tuple[FeatureBenchSeedDiagnostics, ...]:
        for path in self.seeds_root.glob("*/summary.json"):
            try:
                raw = FeatureBenchSeedSummary.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            self.seed(raw.seed_test_file)
        return tuple(self._seeds[key] for key in sorted(self._seeds))

    def register_task(self, task_id: str, seed: FeatureBenchSeedDiagnostics) -> None:
        self._tasks[task_id] = seed

    def seed_for_task(self, task_id: str) -> FeatureBenchSeedDiagnostics | None:
        return self._tasks.get(task_id)

    def append_failure(self, failure: FeatureBenchFailure) -> None:
        _append_jsonl(self.failures_path, failure)

    def _collect_unreferenced_objects(self) -> None:
        for path in self.objects.iterdir():
            if path.is_file() and path.stat().st_nlink == 1:
                path.unlink()

    def finish(self, error: BaseException | None) -> None:
        for seed in self.seeds():
            if seed.summary.status == "queued":
                seed.finalize("skipped", phase="run_completed")
                continue
            if seed.summary.status not in {"running", "awaiting_quality_gate"}:
                continue
            if error is None:
                seed.finalize("skipped", phase="run_completed")
                trajectory = self.settings.agent.trajectory
                if trajectory.enabled and trajectory.retention == "failures":
                    seed.remove_success_trajectories(trajectory.directory)
                if self.settings.audit.retention == "failures":
                    seed.compact_success_audit()
            else:
                seed.finalize("interrupted", phase=seed.summary.phase, error=error)
        self._collect_unreferenced_objects()
        self._write_run_summary("failed" if error is not None else "succeeded", error)


class FeatureBenchSeedDiagnostics:
    """Exact evidence and retention ownership for one physical Seed test file."""

    def __init__(
        self,
        *,
        store: FeatureBenchDiagnostics,
        root: Path,
        seed_test_file: str,
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
            self.summary = FeatureBenchSeedSummary.model_validate_json(
                self.summary_path.read_text(encoding="utf-8")
            )
            if (
                self.summary.run_id != store.run_id
                or self.summary.environment_id != store.environment.environment_id
                or self.summary.seed_test_file != seed_test_file
            ):
                raise ValueError(f"FeatureBench diagnostic identity mismatch: {self.summary_path}")
        else:
            self.summary = FeatureBenchSeedSummary(
                run_id=store.run_id,
                environment_id=store.environment.environment_id,
                seed_test_file=seed_test_file,
                created_at=now,
                updated_at=now,
            )
            self._save()
        self.invocation_root: Path | None = None

    @property
    def relative_path(self) -> str:
        return str(self.root.relative_to(self.store.run_directory))

    def _save(self) -> None:
        _atomic_json(self.summary_path, self.summary)

    def _update(self, **changes: object) -> None:
        self.summary = self.summary.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._save()

    def _load_trajectory_paths(self) -> set[Path]:
        result: set[Path] = set()
        if not self.agent_runs_path.is_file():
            return result
        for line in self.agent_runs_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = record.get("trajectory_path")
            if isinstance(path, str) and path:
                result.add(Path(path).resolve())
        return result

    def append_event(self, kind: str, **data: object) -> None:
        _append_jsonl(
            self.events_path,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": kind,
                "phase": self.summary.phase,
                "seed_invocation": self.summary.invocation_count,
                "data": data,
            },
        )

    def begin(self) -> None:
        invocation = self.summary.invocation_count + 1
        self.invocation_root = self.root / "invocations" / f"{invocation:04d}"
        self.invocation_root.mkdir(parents=True, exist_ok=False)
        self._update(
            status="running",
            phase="trace",
            invocation_count=invocation,
            validation_count=0,
            task_id=None,
            error_type=None,
            error=None,
        )
        self.append_event("seed_started", invocation=invocation)

    def is_durable_success(self, task_ids: frozenset[str]) -> bool:
        return self.summary.status == "succeeded" and self.summary.task_id in task_ids

    def work_state(self, task_ids: frozenset[str]) -> WorkState:
        """Map Adapter-owned durable evidence into the core selection vocabulary."""

        if self.is_durable_success(task_ids):
            return WorkState.SUCCEEDED
        if self.summary.status == "failed":
            return WorkState.FAILED
        if self.summary.status == "rejected":
            return WorkState.REJECTED
        if self.summary.status in {"running", "awaiting_quality_gate", "interrupted"}:
            return WorkState.INTERRUPTED
        return WorkState.UNSEEN

    def _invocation(self) -> Path:
        if self.invocation_root is None:
            self.invocation_root = (
                self.root / "invocations" / f"{self.summary.invocation_count:04d}"
            )
        self.invocation_root.mkdir(parents=True, exist_ok=True)
        return self.invocation_root

    def set_phase(self, phase: str) -> None:
        self._update(phase=phase)

    def record_trace(self, result: dict[str, object], bundle: TraceBundle | None) -> None:
        target = self._invocation() / "trace"
        self.store.link_json(target / "result.json", result)
        if bundle is not None:
            self.store.link_json(target / "bundle.json", bundle)
        self.append_event("trace_completed", **result)

    def record_top(self, objects: tuple[SourceObject, ...]) -> None:
        self.set_phase("top_selection")
        self.store.link_json(
            self._invocation() / "top-selection.json",
            [item.model_dump(mode="json") for item in objects],
        )
        self.append_event("top_selection_completed", top_objects=len(objects))

    def record_initial_plan(self, plan: MaskPlan) -> None:
        self.set_phase("mask_planning")
        self.store.link_json(self._invocation() / "initial-mask-plan.json", plan)
        self.append_event(
            "mask_plan_completed",
            gold_patch_tier=plan.gold_patch_tier,
            removed_lines=plan.removed_lines,
            modified_files=len(plan.patch_files),
        )

    def record_clean_baseline_backoff(
        self,
        *,
        phase: Literal[
            "started",
            "replan_failed",
            "replanned",
            "retry_completed",
            "retry_failed",
        ],
        previous_p2p_test_points: int,
        maximum_p2p_test_points: int,
        plan: MaskPlan | None = None,
        error: BaseException | None = None,
    ) -> None:
        if plan is not None and phase == "replanned":
            self.store.link_json(self._invocation() / "clean-baseline-replan.json", plan)
        detail = None
        if error is not None:
            detail = str(error)[-50_000:] or repr(error)
        self.append_event(
            (
                "clean_baseline_p2p_backoff"
                if phase == "replanned"
                else f"clean_baseline_p2p_backoff_{phase}"
            ),
            previous_p2p_test_points=previous_p2p_test_points,
            maximum_p2p_test_points=maximum_p2p_test_points,
            replanned_p2p_test_points=(len(plan.p2p_test_ids) if plan is not None else None),
            gold_patch_tier=(plan.gold_patch_tier if plan is not None else None),
            removed_lines=(plan.removed_lines if plan is not None else None),
            error_type=type(error).__name__ if error is not None else None,
            error=detail,
        )

    def record_clean_baseline_capacity_replan(
        self,
        plan: MaskPlan,
        *,
        previous_p2p_test_points: int,
        maximum_p2p_test_points: int,
        test_capacity: int,
    ) -> None:
        self.store.link_json(self._invocation() / "clean-baseline-capacity-replan.json", plan)
        self.append_event(
            "clean_baseline_capacity_replan",
            previous_p2p_test_points=previous_p2p_test_points,
            maximum_p2p_test_points=maximum_p2p_test_points,
            replanned_p2p_test_points=len(plan.p2p_test_ids),
            test_capacity=test_capacity,
        )

    def record_validation_started(
        self,
        *,
        attempt: int,
        phase: str,
        plan: MaskPlan,
    ) -> None:
        sequence = self.summary.validation_count + 1
        destination = self._invocation() / "attempts" / f"{sequence:04d}"
        destination.mkdir(parents=True, exist_ok=False)
        self.store.link_json(destination / "plan.json", plan)
        _atomic_json(
            destination / "manifest.json",
            {"sequence": sequence, "attempt": attempt, "phase": phase},
        )
        self._update(
            phase="mask_validation",
            validation_count=sequence,
        )
        self.append_event(
            "validation_started",
            sequence=sequence,
            attempt=attempt,
            validation_phase=phase,
            masked_objects=len(plan.targets),
        )

    def record_candidate_patch(self, candidate_patch: str) -> None:
        destination = self._invocation() / "attempts" / f"{self.summary.validation_count:04d}"
        self.store.link_bytes(
            destination / "corruption.patch",
            kind="patch",
            content=candidate_patch.encode(),
        )

    def record_validation_finished(self, quality: QualityReport) -> None:
        destination = self._invocation() / "attempts" / f"{self.summary.validation_count:04d}"
        compact = quality.model_copy(update={"test_output": ""})
        self.store.link_json(destination / "quality.json", compact)
        if quality.test_output:
            self.store.link_bytes(
                destination / "test-output.log",
                kind="log",
                content=quality.test_output.encode(),
            )
        self.append_event(
            "validation_completed",
            accepted=quality.accepted,
            f2p=len(quality.fail_to_pass),
            p2p=len(quality.pass_to_pass),
            gold_patch_lines=quality.gold_patch_lines,
            gold_patch_files=quality.gold_patch_files,
            errors=quality.errors,
        )

    def record_validation_error(self, error: BaseException) -> None:
        destination = self._invocation() / "attempts" / f"{self.summary.validation_count:04d}"
        _atomic_json(
            destination / "error.json",
            {
                "error_type": type(error).__name__,
                "error": str(error)[-50_000:] or repr(error),
            },
        )
        self.append_event(
            "validation_failed",
            error_type=type(error).__name__,
            error=str(error)[-50_000:] or repr(error),
        )

    def record_final_mask_selected(
        self,
        *,
        plan: MaskPlan,
        quality: QualityReport,
        repair_strategy: str | None,
    ) -> None:
        """Persist the one accepted Mask that advances to candidate authoring."""
        if not quality.accepted:
            raise ValueError("the final selected Mask must have accepted quality")
        plan_payload = plan.model_dump(mode="json")
        plan_sha256 = hashlib.sha256(
            json.dumps(
                plan_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        selection = {
            "plan_sha256": plan_sha256,
            "validation_count": self.summary.validation_count,
            "repair_strategy": repair_strategy,
            "gold_patch_tier": plan.gold_patch_tier,
            "gold_patch_lines": quality.gold_patch_lines,
            "gold_patch_files": quality.gold_patch_files,
            "f2p": len(quality.fail_to_pass),
            "p2p": len(quality.pass_to_pass),
        }
        self.store.link_json(
            self._invocation() / "final-mask-selection.json",
            {**selection, "plan": plan_payload},
        )
        self.append_event("final_mask_selected", **selection)

    def record_agent_run(self, role: str, kind: str, data: dict[str, object]) -> None:
        path = data.get("trajectory_path")
        if isinstance(path, str) and path:
            self._trajectory_paths.add(Path(path).resolve())
        _append_jsonl(
            self.agent_runs_path,
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "role": role,
                "kind": kind,
                **data,
            },
        )

    def new_workspace(self, kind: str) -> Path:
        root = self._invocation() / "workspaces"
        root.mkdir(parents=True, exist_ok=True)
        index = sum(path.is_dir() and path.name.startswith(f"{kind}-") for path in root.iterdir())
        destination = root / f"{kind}-{index + 1:04d}"
        destination.mkdir()
        return destination

    def record_candidate(
        self,
        *,
        task: TaskRecord,
        plan: MaskPlan,
        quality: QualityReport,
        corruption_patch: str,
        interfaces: tuple[InterfaceDescription, ...],
        issue: IssueDraft,
    ) -> None:
        final = self._invocation() / "candidate"
        self.store.link_json(final / "task.json", task)
        self.store.link_json(final / "mask-plan.json", plan)
        self.store.link_json(
            final / "quality.json",
            quality.model_copy(update={"test_output": ""}),
        )
        self.store.link_json(
            final / "interfaces.json",
            [item.model_dump(mode="json") for item in interfaces],
        )
        self.store.link_json(final / "issue.json", issue)
        self.store.link_bytes(
            final / "corruption.patch",
            kind="patch",
            content=corruption_patch.encode(),
        )
        self._update(
            status="awaiting_quality_gate",
            phase="gold_closure",
            task_id=task.task_id,
        )
        self.store.register_task(task.task_id, self)
        self.append_event("candidate_assembled", task_id=task.task_id)

    def record_quality_gate(self, result: QualityGateResult) -> None:
        _atomic_json(
            self.root / "final-quality-gate.json",
            {
                "passed": result.passed,
                "message": result.message,
                "metrics": dict(result.metrics),
            },
        )
        self.append_event(
            "gold_closure_completed",
            passed=result.passed,
            message=result.message,
            metrics=dict(result.metrics),
        )

    def record_quality_gate_error(self, error: BaseException) -> None:
        detail = str(error)[-50_000:] or repr(error)
        _atomic_json(
            self.root / "final-quality-gate-error.json",
            {
                "error_type": type(error).__name__,
                "error": detail,
            },
        )
        self.append_event(
            "gold_closure_failed",
            error_type=type(error).__name__,
            error=detail,
        )

    def finalize(
        self,
        status: Literal["succeeded", "rejected", "failed", "interrupted", "skipped"],
        *,
        phase: str,
        error: BaseException | None = None,
        error_type: str | None = None,
        error_text: str | None = None,
    ) -> None:
        detail = error_text
        if detail is None and error is not None:
            detail = str(error)[-50_000:] or repr(error)
        observed_type = error_type or (type(error).__name__ if error is not None else None)
        self._update(
            status=status,
            phase=phase,
            error_type=observed_type,
            error=detail,
        )
        self.append_event(
            "seed_completed",
            status=status,
            phase=phase,
            error_type=observed_type,
            error=detail,
        )
        if status in {"failed", "rejected", "interrupted"}:
            disposition: Literal["failed", "rejected", "interrupted"]
            if status == "failed":
                disposition = "failed"
            elif status == "rejected":
                disposition = "rejected"
            else:
                disposition = "interrupted"
            self.store.append_failure(
                FeatureBenchFailure(
                    run_id=self.store.run_id,
                    environment_id=self.store.environment.environment_id,
                    seed_test_file=self.summary.seed_test_file,
                    disposition=disposition,
                    phase=phase,
                    invocation=self.summary.invocation_count,
                    validations=self.summary.validation_count,
                    task_id=self.summary.task_id,
                    diagnostics_path=self.relative_path,
                    error_type=observed_type or "CandidateRejected",
                    error=detail or f"candidate {status} without an error message",
                )
            )

    def retain_success_summary(
        self,
        task: TaskRecord,
        metrics: Mapping[str, object],
    ) -> None:
        _atomic_json(
            self.root / "final-summary.json",
            {
                "task_id": task.task_id,
                "seed_test_file": self.summary.seed_test_file,
                "metrics": dict(metrics),
                "validation_count": self.summary.validation_count,
            },
        )

    def remove_success_trajectories(self, trajectory_root: Path) -> None:
        configured_root = trajectory_root.resolve()
        removed_trajectories: list[str] = []
        for path in sorted(self._trajectory_paths):
            if path.parent != configured_root or not path.name:
                raise ValueError(f"refusing to remove trajectory outside configured root: {path}")
            if path.is_dir():
                shutil.rmtree(path)
                removed_trajectories.append(str(path))
        if removed_trajectories:
            self.append_event(
                "successful_trajectories_removed",
                paths=removed_trajectories,
            )

    def compact_success_audit(self) -> None:
        invocations = self.root / "invocations"
        if invocations.is_dir():
            shutil.rmtree(invocations)
        self.append_event("successful_audit_compacted")
