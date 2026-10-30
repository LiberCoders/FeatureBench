"""FeatureBench post-verification quality gates."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping

from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.masking import hide_test_points
from codeworld.adapters.featurebench.models import (
    FeatureBenchMetrics,
    GoldPatchMetrics,
    GoldPatchTier,
    MaskPlan,
    QualityReport,
    TestPointMetrics,
    TestReport,
    TopObjectMetrics,
)
from codeworld.errors import CandidateRejected, ContractError

_HUNK_TARGET = re.compile(r"\+(?P<line>\d+)(?:,(?P<count>\d+))?")
_GOLD_PATCH_LINE_ERROR = re.compile(r"Gold Patch lines \d+ (?:are below|exceed) \d+")
_F2P_HIDEABILITY_ERROR_PREFIX = "F2P hideability failed: "


def _git_path(value: str) -> str:
    """Decode the first Git diff path token and remove its side prefix."""
    try:
        path = shlex.split(value)[0]
    except (IndexError, ValueError) as exc:
        raise ContractError(f"invalid Git diff path: {value!r}") from exc
    return path[2:] if path.startswith(("a/", "b/")) else path


def _added_line_locations(patch: str) -> tuple[tuple[str, int], ...]:
    """Return repository path and target line for every added Gold Patch line."""
    path: str | None = None
    target_line: int | None = None
    locations: list[tuple[str, int]] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path = None
            target_line = None
        elif line.startswith("+++ "):
            path = _git_path(line[4:])
        elif line.startswith("@@ "):
            match = _HUNK_TARGET.search(line)
            if match is None:
                raise ContractError(f"invalid Git diff hunk header: {line!r}")
            target_line = int(match.group("line"))
        elif target_line is not None and path is not None:
            if line.startswith("+"):
                locations.append((path, target_line))
                target_line += 1
            elif line.startswith(("-", "\\")):
                continue
            elif line.startswith(" "):
                target_line += 1
    return tuple(locations)


def build_featurebench_metrics(
    *,
    gold_patch: str,
    quality: QualityReport,
    plan: MaskPlan,
) -> FeatureBenchMetrics:
    """Derive final JSONL metrics from the accepted patch, tests, and Mask Plan."""
    locations = _added_line_locations(gold_patch)
    if len(locations) != quality.gold_patch_lines:
        raise ContractError("Gold Patch metric does not match its actual added-line count")
    top_spans: dict[str, list[tuple[int, int]]] = {}
    for item in plan.top_objects:
        top_spans.setdefault(item.path, []).append((item.start_line, item.end_line))
    top_lines = sum(
        any(start <= line <= end for start, end in top_spans.get(path, ()))
        for path, line in locations
    )
    return FeatureBenchMetrics(
        gold_patch=GoldPatchMetrics(
            lines=quality.gold_patch_lines,
            files=quality.gold_patch_files,
        ),
        tests=TestPointMetrics(
            f2p=len(set(quality.fail_to_pass)),
            p2p=len(set(quality.pass_to_pass)),
        ),
        top=TopObjectMetrics(
            objects=len({item.object_id for item in plan.top_objects}),
            gold_patch_lines=top_lines,
            gold_patch_ratio=top_lines / quality.gold_patch_lines,
        ),
    )


def reapply_gold_patch_tier_gates(
    quality: QualityReport,
    *,
    settings: FeatureBenchSettings,
    tier: GoldPatchTier = "preferred",
) -> QualityReport:
    """Re-evaluate only the tier gate while retaining every behavioral error."""
    errors = [error for error in quality.errors if _GOLD_PATCH_LINE_ERROR.fullmatch(error) is None]
    minimum = settings.gold_patch_minimum_for_tier(tier)
    maximum = settings.gold_patch_maximum_for_tier(tier)
    if quality.gold_patch_lines < minimum:
        errors.append(f"Gold Patch lines {quality.gold_patch_lines} are below {minimum}")
    if quality.gold_patch_lines > maximum:
        errors.append(f"Gold Patch lines {quality.gold_patch_lines} exceed {maximum}")
    return quality.model_copy(
        update={
            "accepted": not errors,
            "errors": tuple(errors),
        }
    )


def apply_gold_patch_gates(
    quality: QualityReport,
    *,
    gold_patch: str,
    settings: FeatureBenchSettings,
    tier: GoldPatchTier = "preferred",
) -> QualityReport:
    """Measure the real reverse diff and enforce its configured line interval."""
    measured = quality.model_copy(
        update={
            "gold_patch_lines": sum(
                line.startswith("+") and not line.startswith("+++")
                for line in gold_patch.splitlines()
            ),
            "gold_patch_files": sum(
                line.startswith("diff --git ") for line in gold_patch.splitlines()
            ),
        }
    )
    return reapply_gold_patch_tier_gates(measured, settings=settings, tier=tier)


def apply_f2p_hideability_gate(
    quality: QualityReport,
    *,
    index: RepositoryIndex,
    test_paths: Mapping[str, str],
    test_source_paths: Mapping[str, str],
) -> QualityReport:
    """Require every declared F2P point to have a reversible hidden-test transform."""
    if any(_GOLD_PATCH_LINE_ERROR.fullmatch(error) is None for error in quality.errors):
        return quality
    try:
        hidden_sources, hidden_test_files = hide_test_points(
            index.sources,
            index=index,
            test_ids=quality.fail_to_pass,
            test_paths=test_paths,
            test_source_paths=test_source_paths,
        )
        if not any(hidden_sources[path] != index.sources[path] for path in hidden_test_files):
            raise CandidateRejected("F2P test declarations could not be hidden reversibly")
    except CandidateRejected as exc:
        error = _F2P_HIDEABILITY_ERROR_PREFIX + str(exc)
        errors = tuple(dict.fromkeys((*quality.errors, error)))
        return quality.model_copy(update={"accepted": False, "errors": errors})
    return quality


def has_f2p_hideability_error(quality: QualityReport) -> bool:
    """Return whether acceptance failed its hidden-test representation contract."""
    return any(error.startswith(_F2P_HIDEABILITY_ERROR_PREFIX) for error in quality.errors)


def assess_quality(
    *,
    baseline: tuple[str, ...],
    report: TestReport,
    plan: MaskPlan,
    test_paths: Mapping[str, str],
    settings: FeatureBenchSettings,
) -> QualityReport:
    """Classify real post-mask outcomes strictly at pytest node-ID granularity."""
    baseline_set = set(baseline)
    observed = set(report.outcomes)
    missing = baseline_set - observed
    passed_set = baseline_set & set(report.passed)
    # A baseline-passing point that now fails, skips, errors, or disappears is F2P.
    f2p_set = baseline_set - passed_set
    configured_p2p = set(plan.p2p_test_ids)
    selected_p2p = configured_p2p & passed_set
    seed_nodes = {node for node in baseline_set if test_paths.get(node) == plan.seed_test_file}
    seed_passed = seed_nodes & passed_set
    seed_pass_rate = len(seed_passed) / len(seed_nodes) if seed_nodes else 1.0
    f2p = tuple(sorted(f2p_set))
    p2p = tuple(sorted(selected_p2p))
    declared_test_points = len(f2p) + len(p2p)
    f2p_ratio = len(f2p) / declared_test_points if declared_test_points else 1.0
    errors: list[str] = []
    if report.collection_errors or report.exit_code not in (0, 1):
        errors.append("mask broke test collection or the test runtime")
    if missing:
        errors.append(f"isolated runs omitted {len(missing)} selected baseline test points")
    if len(f2p) < settings.minimum_f2p_test_points:
        errors.append(f"F2P count {len(f2p)} is below {settings.minimum_f2p_test_points}")
    if settings.maximum_f2p_test_points is not None and len(f2p) > settings.maximum_f2p_test_points:
        errors.append(f"F2P count {len(f2p)} exceeds {settings.maximum_f2p_test_points}")
    if f2p_ratio > settings.maximum_f2p_ratio:
        errors.append(f"F2P ratio {f2p_ratio:.3f} exceeds {settings.maximum_f2p_ratio:.3f}")
    if not seed_nodes:
        errors.append("seed test file has no baseline test points")
    elif seed_pass_rate > settings.maximum_seed_pass_rate:
        errors.append(
            f"seed pass rate {seed_pass_rate:.3f} exceeds {settings.maximum_seed_pass_rate:.3f}"
        )
    if len(p2p) < settings.minimum_remaining_p2p_test_points:
        errors.append(f"P2P count {len(p2p)} is below {settings.minimum_remaining_p2p_test_points}")
    if settings.maximum_p2p_test_points is not None and len(p2p) > settings.maximum_p2p_test_points:
        errors.append(f"P2P count {len(p2p)} exceeds {settings.maximum_p2p_test_points}")
    return QualityReport(
        accepted=not errors,
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        seed_pass_rate=seed_pass_rate,
        f2p_ratio=f2p_ratio,
        errors=tuple(errors),
        test_exit_code=report.exit_code,
        collection_errors=report.collection_errors,
        test_output=report.output,
    )
