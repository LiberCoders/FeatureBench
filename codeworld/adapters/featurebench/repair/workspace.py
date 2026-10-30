"""File-backed, Agent-readable feedback for FeatureBench Mask repair."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.models import MaskPlan, QualityReport
from codeworld.adapters.featurebench.p2p import TracePointIndex
from codeworld.adapters.featurebench.repair.models import (
    MaskRepairProposal,
    RepairResultFile,
    RepairViolation,
)

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"


def _jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(body, encoding="utf-8")


def _quality_violations(
    quality: QualityReport,
    settings: FeatureBenchSettings,
    plan: MaskPlan,
) -> tuple[RepairViolation, ...]:
    """Render every deterministic quality gate as concise Agent feedback."""
    violations: list[RepairViolation] = []
    known_messages: set[str] = set()

    def add(
        code: str,
        message: str,
        correction: str,
        *,
        diagnostic_file: str | None = None,
    ) -> None:
        known_messages.add(message)
        violations.append(
            RepairViolation(
                code=code,
                message=message,
                correction=correction,
                diagnostic_file=diagnostic_file,
            )
        )

    if quality.collection_errors or quality.test_exit_code not in (0, 1):
        add(
            "test.runtime_broken",
            "mask broke test collection or the test runtime",
            "Restore the smallest shared object set responsible for the import or runtime failure, "
            "and mask feature-specific replacements when scale would otherwise shrink.",
            diagnostic_file=(
                "diagnostics/runner-errors.jsonl"
                if quality.collection_errors
                else "diagnostics/test-output.log"
            ),
        )
    f2p = len(quality.fail_to_pass)
    p2p = len(quality.pass_to_pass)
    if f2p < settings.minimum_f2p_test_points:
        add(
            "f2p.below_minimum",
            f"F2P count {f2p} is below {settings.minimum_f2p_test_points}",
            "Mask additional feature-specific objects while preserving the working runtime.",
        )
    if settings.maximum_f2p_test_points is not None and f2p > settings.maximum_f2p_test_points:
        add(
            "f2p.above_maximum",
            f"F2P count {f2p} exceeds {settings.maximum_f2p_test_points}",
            "Restore high-impact shared objects and compensate with narrower feature objects.",
        )
    if quality.f2p_ratio > settings.maximum_f2p_ratio:
        add(
            "f2p.ratio_too_high",
            f"F2P ratio {quality.f2p_ratio:.3f} exceeds {settings.maximum_f2p_ratio:.3f}",
            "Restore objects that damage broad P2P behavior, preserving scale with narrower masks.",
        )
    if quality.seed_pass_rate > settings.maximum_seed_pass_rate:
        add(
            "seed.pass_rate_too_high",
            f"seed pass rate {quality.seed_pass_rate:.3f} exceeds "
            f"{settings.maximum_seed_pass_rate:.3f}",
            "Mask feature-specific objects reached by the seed tests.",
        )
    if p2p < settings.minimum_remaining_p2p_test_points:
        add(
            "p2p.below_minimum",
            f"P2P count {p2p} is below {settings.minimum_remaining_p2p_test_points}",
            "Restore shared objects implicated by failing non-seed tests.",
        )
    if settings.maximum_p2p_test_points is not None and p2p > settings.maximum_p2p_test_points:
        add(
            "p2p.above_maximum",
            f"P2P count {p2p} exceeds {settings.maximum_p2p_test_points}",
            "Mask a coherent feature-specific slice without changing the fixed P2P inventory.",
        )
    minimum_gold_lines = settings.gold_patch_minimum_for_tier(plan.gold_patch_tier)
    if quality.gold_patch_lines < minimum_gold_lines:
        add(
            "gold_patch.below_minimum",
            f"Gold Patch lines {quality.gold_patch_lines} are below {minimum_gold_lines}",
            "Mask feature-specific replacement objects to recover task scale.",
        )
    if quality.gold_patch_lines > settings.gold_patch_line_maximum:
        add(
            "gold_patch.above_maximum",
            f"Gold Patch lines {quality.gold_patch_lines} exceed "
            f"{settings.gold_patch_line_maximum}",
            "Restore the least feature-essential objects while keeping tests healthy.",
        )
    for message in quality.errors:
        if message in known_messages:
            continue
        violations.append(
            RepairViolation(
                code="quality.contract_failed",
                message=message,
                correction="Inspect the masked repository and apply the smallest relevant increment.",
            )
        )
    return tuple(violations)


def _traceback_mask_violations(
    quality: QualityReport,
    plan: MaskPlan,
) -> tuple[RepairViolation, ...]:
    """Link real source locations in failed output to the current sparse Mask Plan."""
    if not quality.test_output or (
        not quality.collection_errors and quality.test_exit_code in (0, 1)
    ):
        return ()
    normalized_output = quality.test_output.replace("\\", "/")
    implicated_by_path: dict[str, list[str]] = {}
    for item in plan.targets:
        path = item.path.replace("\\", "/").lstrip("./")
        if f"{path}:" not in normalized_output:
            continue
        implicated_by_path.setdefault(path, []).append(item.object_id)
    return tuple(
        RepairViolation(
            code="test.traceback_references_masked_file",
            message=(
                "The real failing output references a source file containing current Mask "
                f"objects: {path}"
            ),
            correction=(
                "Inspect these objects and the referenced traceback first; restore only the "
                "causal objects, then use narrower replacement masks if needed."
            ),
            object_ids=tuple(object_ids),
            diagnostic_file="diagnostics/test-output.log",
        )
        for path, object_ids in implicated_by_path.items()
    )


@dataclass(slots=True)
class MaskRepairWorkspace:
    """Host directory bind-mounted into the Repair Agent container."""

    root: Path
    settings: FeatureBenchSettings
    container_root: str = "/featurebench-repair"

    @property
    def outbox_host_path(self) -> Path:
        return self.root / "outbox" / "proposal.json"

    @property
    def proposal_container_path(self) -> str:
        return str(PurePosixPath(self.container_root) / "outbox" / "proposal.json")

    @property
    def catalog_container_path(self) -> str:
        return str(PurePosixPath(self.container_root) / "catalog" / "objects.jsonl")

    def feedback_container_path(self, attempt_id: str) -> str:
        return str(PurePosixPath(self.container_root) / "attempts" / attempt_id / "feedback")

    def prepare(
        self,
        *,
        index: RepositoryIndex,
        initial_plan: MaskPlan,
        point_index: TracePointIndex,
    ) -> None:
        """Create immutable protocol/catalog inputs and the writable outbox."""
        protocol = self.root / "protocol"
        catalog = self.root / "catalog"
        outbox = self.root / "outbox"
        protocol.mkdir(parents=True, exist_ok=True)
        catalog.mkdir(parents=True, exist_ok=True)
        outbox.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ASSET_ROOT / "repair" / "protocol.md", protocol / "README.md")
        (protocol / "proposal.example.json").write_text(
            MaskRepairProposal(
                based_on_attempt="attempt-0000",
                must_restore_object_ids=(),
                must_mask_object_ids=("package/module.py::Feature::1",),
                rationale="Replace one shared mask with one feature-specific object.",
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
        eligible = set(initial_plan.candidate_object_ids) | {
            item.object_id for item in initial_plan.targets
        }
        protected = set(initial_plan.protected_objects)
        seed_mask = point_index.file_test_masks.get(initial_plan.seed_test_file, 0)
        p2p_mask = point_index.test_mask_for_ids(initial_plan.p2p_test_ids)
        records = [
            {
                "object_id": item.object_id,
                "path": item.path,
                "qualname": item.qualname,
                "kind": item.kind,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "removed_lines": item.removed_lines,
                "eligible_for_mask": item.object_id in eligible,
                "protected": item.object_id in protected,
                "estimated_seed_affected_points": (
                    point_index.object_test_masks.get(item.object_id, 0) & seed_mask
                ).bit_count(),
                "estimated_p2p_affected_points": (
                    point_index.object_test_masks.get(item.object_id, 0) & p2p_mask
                ).bit_count(),
            }
            for item in index.data.definitions
        ]
        _jsonl(catalog / "objects.jsonl", records)

    def proposal_host_path(self, submitted_path: str) -> Path:
        """Map one container-visible outbox path onto the host bind source."""
        raw = PurePosixPath(submitted_path)
        container_root = PurePosixPath(self.container_root)
        if raw.is_absolute():
            try:
                relative = raw.relative_to(container_root)
            except ValueError as exc:
                raise ValueError(
                    f"proposal path must be under {self.container_root}/outbox"
                ) from exc
        else:
            relative = raw
        if ".." in relative.parts or not relative.parts or relative.parts[0] != "outbox":
            raise ValueError("proposal path must be a relative outbox path")
        resolved = (self.root / Path(*relative.parts)).resolve()
        outbox = (self.root / "outbox").resolve()
        if not resolved.is_relative_to(outbox):
            raise ValueError("proposal path resolves outside the repair outbox")
        return resolved

    def write_feedback(
        self,
        *,
        attempt_id: str,
        based_on_attempt: str | None,
        plan: MaskPlan,
        quality: QualityReport,
        previous_plan: MaskPlan | None,
        previous_quality: QualityReport | None,
        proposal: MaskRepairProposal | None,
        learned_protected_ids: frozenset[str] = frozenset(),
        extra_violations: tuple[RepairViolation, ...] = (),
    ) -> str:
        """Write one concise immutable feedback directory and refresh the draft proposal."""
        destination = self.root / "attempts" / attempt_id / "feedback"
        destination.mkdir(parents=True, exist_ok=False)
        violations = (
            *_quality_violations(quality, self.settings, plan),
            *_traceback_mask_violations(quality, plan),
            *extra_violations,
        )
        result = RepairResultFile(
            attempt_id=attempt_id,
            based_on_attempt=based_on_attempt,
            accepted=quality.accepted and not extra_violations,
            collection_ok=not quality.collection_errors,
            runtime_ok=not quality.collection_errors and quality.test_exit_code in (0, 1),
            test_exit_code=quality.test_exit_code,
            f2p=len(quality.fail_to_pass),
            p2p=len(quality.pass_to_pass),
            f2p_ratio=quality.f2p_ratio,
            gold_patch_lines=quality.gold_patch_lines,
            gold_patch_files=quality.gold_patch_files,
            masked_objects=len(plan.targets),
            delta_f2p=(
                len(quality.fail_to_pass) - len(previous_quality.fail_to_pass)
                if previous_quality is not None
                else None
            ),
            delta_p2p=(
                len(quality.pass_to_pass) - len(previous_quality.pass_to_pass)
                if previous_quality is not None
                else None
            ),
            delta_gold_patch_lines=(
                quality.gold_patch_lines - previous_quality.gold_patch_lines
                if previous_quality is not None
                else None
            ),
            delta_gold_patch_files=(
                quality.gold_patch_files - previous_quality.gold_patch_files
                if previous_quality is not None
                else None
            ),
            delta_masked_objects=(
                len(plan.targets) - len(previous_plan.targets)
                if previous_plan is not None
                else None
            ),
            violation_codes=tuple(item.code for item in violations),
        )
        (destination / "result.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        _jsonl(
            destination / "violations.jsonl",
            [item.model_dump(mode="json", exclude_none=True) for item in violations],
        )
        _jsonl(
            destination / "mask-objects.jsonl",
            [
                {
                    "object_id": item.object_id,
                    "classification": item.classification,
                    "path": item.path,
                    "qualname": item.qualname,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "removed_lines": item.removed_lines,
                    "dynamic_calls": item.calls,
                }
                for item in plan.targets
            ],
        )
        if proposal is not None:
            (destination / "submitted-proposal.json").write_text(
                proposal.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        if (quality.test_output or quality.collection_errors) and not result.accepted:
            diagnostics = destination / "diagnostics"
            diagnostics.mkdir()
            if quality.test_output:
                (diagnostics / "test-output.log").write_text(quality.test_output, encoding="utf-8")
            if quality.collection_errors:
                _jsonl(
                    diagnostics / "runner-errors.jsonl",
                    [{"error": error} for error in quality.collection_errors],
                )
        environment = Environment(
            loader=FileSystemLoader(_ASSET_ROOT / "repair"),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,
        )
        readme = environment.get_template("feedback.md.j2").render(
            result=result,
            catalog_path=self.catalog_container_path,
            proposal_path=self.proposal_container_path,
            learned_protection=True,
            impact_catalog=True,
        )
        (destination / "README.md").write_text(readme, encoding="utf-8")
        self.outbox_host_path.write_text(
            json.dumps(
                {
                    "based_on_attempt": attempt_id,
                    "must_restore_object_ids": [],
                    "must_mask_object_ids": [],
                    "rationale": "",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _jsonl(
            destination / "learned-protected.jsonl",
            [
                {
                    "object_id": object_id,
                    "reason": "explicitly restored by an earlier validated repair increment",
                }
                for object_id in sorted(learned_protected_ids)
            ],
        )
        return self.feedback_container_path(attempt_id)
