from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
from pydantic import ValidationError

from codeworld.adapters.featurebench.config import (
    FeatureBenchRepairSettings,
    FeatureBenchSettings,
)
from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.masking import FeatureSlicePlanner, MaskExecutor
from codeworld.adapters.featurebench.models import MaskPlan, QualityReport
from codeworld.adapters.featurebench.p2p import TracePointIndex
from codeworld.adapters.featurebench.repair.capability import MaskRepairSubmission
from codeworld.adapters.featurebench.repair.models import MaskRepairProposal
from codeworld.adapters.featurebench.repair.rollback import DeterministicMaskRollback
from codeworld.adapters.featurebench.repair.service import MaskRepairService
from codeworld.adapters.featurebench.repair.workspace import MaskRepairWorkspace
from codeworld.agents.workspace import ContainerWorkspace
from codeworld.errors import CandidateRejected
from codeworld.runtime.container import ContainerSession


class RepairSession:
    def __init__(self, sources: dict[str, str]) -> None:
        self.clean = dict(sources)
        self.files = dict(sources)
        self.clean_count = 0

    async def clean_workspace(self) -> None:
        self.clean_count += 1
        self.files = dict(self.clean)

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content.decode("utf-8")


def _fixture() -> tuple[RepositoryIndex, MaskPlan]:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": """def feature():
    return 1

def replacement():
    return 2
"""
        }
    )
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(top,),
        candidate_object_ids=(replacement.object_id,),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    return index, plan


def _settings() -> FeatureBenchSettings:
    return FeatureBenchSettings(
        gold_patch_line_minimum=1,
        gold_patch_line_preferred_minimum=1,
        gold_patch_line_maximum=100,
        minimum_f2p_test_points=1,
        minimum_remaining_p2p_test_points=1,
    )


def _point_index(index: RepositoryIndex, plan: MaskPlan) -> TracePointIndex:
    seed_test_id = "tests/test_core.py::test_feature"
    p2p_test_id = plan.p2p_test_ids[0]
    seed_mask = 1
    p2p_mask = 2
    feature = index.by_qualname[("pkg/core.py", "feature")]
    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    object_masks = MappingProxyType(
        {
            feature.object_id: seed_mask,
            replacement.object_id: seed_mask | p2p_mask,
        }
    )
    return TracePointIndex(
        test_ids=(seed_test_id, p2p_test_id),
        object_test_masks=object_masks,
        observed_object_test_masks=object_masks,
        file_test_masks=MappingProxyType(
            {
                "tests/test_core.py": seed_mask,
                "tests/test_other.py": p2p_mask,
            }
        ),
        test_paths=("tests/test_core.py", "tests/test_other.py"),
        all_test_mask=seed_mask | p2p_mask,
        traced_associations=3,
    )


def _planner() -> FeatureSlicePlanner:
    return FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=1,
        maximum_code_lines=5_000,
        random_seed=0,
        critical_magic_methods=frozenset({"__init__"}),
    )


def _rejected_quality() -> QualityReport:
    return QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        seed_pass_rate=1,
        f2p_ratio=0,
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("ImportError: feature is unavailable",),
        test_output="short real test diagnostic",
    )


def _accepted_quality() -> QualityReport:
    return QualityReport(
        accepted=True,
        fail_to_pass=("tests/test_core.py::test_feature",),
        pass_to_pass=("tests/test_other.py::test_other",),
        seed_pass_rate=0,
        f2p_ratio=0.5,
        gold_patch_lines=4,
        gold_patch_files=1,
        test_exit_code=1,
    )


@pytest.mark.asyncio
async def test_deterministic_rollback_downgrades_real_preferred_scale_to_fallback() -> None:
    index, plan = _fixture()
    plan = plan.model_copy(
        update={
            "gold_patch_tier": "preferred",
            "candidate_object_ids": (),
        }
    )
    initial = QualityReport(
        accepted=False,
        fail_to_pass=("tests/test_core.py::test_feature",),
        pass_to_pass=("tests/test_other.py::test_other",),
        seed_pass_rate=0,
        f2p_ratio=0.5,
        gold_patch_lines=400,
        gold_patch_files=1,
        errors=("Gold Patch lines 400 are below 600",),
        test_exit_code=1,
    )

    async def validate(_candidate: MaskPlan, _attempt: int) -> QualityReport:
        raise AssertionError("there are no replacement candidates to validate")

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=_point_index(index, plan),
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=200,
            gold_patch_line_preferred_minimum=600,
            gold_patch_line_maximum=1_000,
            minimum_f2p_test_points=1,
            minimum_remaining_p2p_test_points=1,
        ),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.plan.gold_patch_tier == "fallback"
    assert result.quality.accepted
    assert result.quality.errors == ()
    assert result.quality.gold_patch_lines == 400
    assert result.validations == 0


@pytest.mark.asyncio
async def test_deterministic_rollback_restores_only_the_runtime_culprit() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/impl.py": ("def culprit():\n    return 2\n\ndef unrelated():\n    return 3\n"),
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    culprit = index.by_qualname[("pkg/impl.py", "culprit")].model_copy(
        update={"classification": "specific"}
    )
    unrelated = index.by_qualname[("pkg/impl.py", "unrelated")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(culprit, unrelated),
        candidate_object_ids=(culprit.object_id, unrelated.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    initial = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("pkg/impl.py: ImportError while importing module",),
        test_output="pkg/impl.py failed during collection",
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )
    validated_keys: list[tuple[str, ...]] = []

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        target_ids = tuple(sorted(item.object_id for item in candidate.targets))
        validated_keys.append(target_ids)
        if culprit.object_id in target_ids:
            return initial
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    settings = _settings().model_copy(
        update={"repair": FeatureBenchRepairSettings(optimization_target="interval_center")}
    )
    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=point_index,
        settings=settings,
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.accepted
    assert result.restored_object_ids == (culprit.object_id,)
    assert unrelated.object_id in {item.object_id for item in result.plan.targets}
    assert len(validated_keys) == len(set(validated_keys))


@pytest.mark.asyncio
async def test_deterministic_rollback_restores_an_unhideable_f2p_culprit() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/impl.py": "def culprit():\n    return 2\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    culprit = index.by_qualname[("pkg/impl.py", "culprit")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(culprit,),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    unhideable = QualityReport(
        accepted=False,
        fail_to_pass=("tests/test_api.py::test_generated",),
        pass_to_pass=("tests/test_other.py::test_other",),
        seed_pass_rate=0,
        f2p_ratio=0.5,
        gold_patch_lines=plan.removed_lines,
        gold_patch_files=len(plan.patch_files),
        errors=(
            "F2P hideability failed: F2P node IDs could not be mapped to removable "
            "test declarations",
        ),
        test_exit_code=1,
    )
    object_masks = MappingProxyType({top.object_id: 1, culprit.object_id: 1})
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_generated", "tests/test_other.py::test_other"),
        object_test_masks=object_masks,
        observed_object_test_masks=object_masks,
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=2,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        if culprit.object_id in {item.object_id for item in candidate.targets}:
            return unhideable
        return _accepted_quality().model_copy(
            update={
                "gold_patch_lines": candidate.removed_lines,
                "gold_patch_files": len(candidate.patch_files),
            }
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=unhideable,
        point_index=point_index,
        settings=_settings(),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.accepted
    assert result.restored_object_ids == (culprit.object_id,)
    assert result.validations == 1


@pytest.mark.asyncio
async def test_deterministic_rollback_localizes_large_batches_logarithmically() -> None:
    sources = {"pkg/api.py": "def feature():\n    return 1\n"}
    sources.update(
        {
            f"pkg/impl_{ordinal:02d}.py": f"def object_{ordinal:02d}():\n    return {ordinal}\n"
            for ordinal in range(64)
        }
    )
    index = RepositoryIndex.from_sources(sources)
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    specifics = tuple(
        index.by_qualname[(f"pkg/impl_{ordinal:02d}.py", f"object_{ordinal:02d}")].model_copy(
            update={"classification": "specific"}
        )
        for ordinal in range(64)
    )
    culprit = specifics[37]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=specifics,
        candidate_object_ids=tuple(item.object_id for item in specifics),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    initial = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("repository import failed",),
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )
    validation_count = 0

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        nonlocal validation_count
        validation_count += 1
        if culprit.object_id in {item.object_id for item in candidate.targets}:
            return initial
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    settings = _settings().model_copy(
        update={
            "gold_patch_line_maximum": 5_000,
            "repair": FeatureBenchRepairSettings(optimization_target="interval_center"),
        }
    )
    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=point_index,
        settings=settings,
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.restored_object_ids == (culprit.object_id,)
    assert validation_count <= 15


@pytest.mark.asyncio
async def test_deterministic_rollback_refills_scale_without_remasking_restored_object() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/broken.py": "def culprit():\n    return 2\n",
            "pkg/replacement.py": "def replacement():\n    return 3\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    culprit = index.by_qualname[("pkg/broken.py", "culprit")].model_copy(
        update={"classification": "specific"}
    )
    replacement = index.by_qualname[("pkg/replacement.py", "replacement")]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(culprit,),
        candidate_object_ids=(culprit.object_id, replacement.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    runtime_broken = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("pkg/broken.py: culprit is unavailable",),
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({replacement.object_id: 1}),
        observed_object_test_masks=MappingProxyType({replacement.object_id: 1}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=1,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        targets = {item.object_id for item in candidate.targets}
        if culprit.object_id in targets:
            return runtime_broken
        accepted = replacement.object_id in targets
        return QualityReport(
            accepted=accepted,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            errors=() if accepted else ("Gold Patch lines 2 are below 3",),
            test_exit_code=1,
        )

    settings = FeatureBenchSettings(
        gold_patch_line_minimum=3,
        gold_patch_line_preferred_minimum=3,
        gold_patch_line_maximum=10,
        minimum_remaining_p2p_test_points=1,
    )
    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=runtime_broken,
        point_index=point_index,
        settings=settings,
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    final_targets = {item.object_id for item in result.plan.targets}
    assert result.quality.accepted
    assert culprit.object_id not in final_targets
    assert replacement.object_id in final_targets
    assert result.restored_object_ids == (culprit.object_id,)
    assert result.replacement_object_ids == (replacement.object_id,)


@pytest.mark.asyncio
async def test_deterministic_rollback_defaults_to_the_first_accepted_plan() -> None:
    index, plan = _fixture()
    initial = _accepted_quality()

    async def validate(_candidate: MaskPlan, _attempt: int) -> QualityReport:
        raise AssertionError("an accepted initial plan must not be optimized further")

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=_point_index(index, plan),
        settings=_settings(),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.plan == plan
    assert result.quality == initial
    assert result.validations == 0


@pytest.mark.asyncio
async def test_deterministic_rollback_refills_an_accepted_plan_to_the_interval_center() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/broken.py": "def culprit():\n    return 2\n",
            "pkg/replacement.py": "def replacement():\n    return 3\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    culprit = index.by_qualname[("pkg/broken.py", "culprit")].model_copy(
        update={"classification": "specific"}
    )
    replacement = index.by_qualname[("pkg/replacement.py", "replacement")]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(culprit,),
        candidate_object_ids=(culprit.object_id, replacement.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    runtime_broken = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("pkg/broken.py: culprit is unavailable",),
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        targets = {item.object_id for item in candidate.targets}
        if culprit.object_id in targets:
            return runtime_broken
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    settings = FeatureBenchSettings(
        gold_patch_line_minimum=1,
        gold_patch_line_preferred_minimum=1,
        gold_patch_line_maximum=7,
        minimum_remaining_p2p_test_points=1,
        repair=FeatureBenchRepairSettings(optimization_target="interval_center"),
    )
    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=runtime_broken,
        point_index=point_index,
        settings=settings,
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    final_targets = {item.object_id for item in result.plan.targets}
    assert result.quality.accepted
    assert result.quality.gold_patch_lines == 4
    assert culprit.object_id not in final_targets
    assert replacement.object_id in final_targets
    assert result.validations == 2


@pytest.mark.asyncio
async def test_deterministic_rollback_keeps_a_better_plan_after_refill_overshoots() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/broken.py": "def culprit():\n    return 2\n",
            "pkg/replacement.py": (
                "def replacement():\n"
                "    value = 1\n"
                "    value += 1\n"
                "    value += 1\n"
                "    return value\n"
            ),
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    culprit = index.by_qualname[("pkg/broken.py", "culprit")].model_copy(
        update={"classification": "specific"}
    )
    replacement = index.by_qualname[("pkg/replacement.py", "replacement")]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(culprit,),
        candidate_object_ids=(culprit.object_id, replacement.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    runtime_broken = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("pkg/broken.py: culprit is unavailable",),
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        targets = {item.object_id for item in candidate.targets}
        if culprit.object_id in targets:
            return runtime_broken
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=runtime_broken,
        point_index=point_index,
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=1,
            gold_patch_line_preferred_minimum=1,
            gold_patch_line_maximum=7,
            minimum_remaining_p2p_test_points=1,
            repair=FeatureBenchRepairSettings(optimization_target="interval_center"),
        ),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    final_targets = {item.object_id for item in result.plan.targets}
    assert result.quality.gold_patch_lines == 2
    assert replacement.object_id not in final_targets
    assert result.validations == 2


@pytest.mark.asyncio
async def test_deterministic_rollback_refills_an_initially_accepted_plan() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/replacement.py": "def replacement():\n    return 2\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    replacement = index.by_qualname[("pkg/replacement.py", "replacement")]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        candidate_object_ids=(replacement.object_id,),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    initial = QualityReport(
        accepted=True,
        fail_to_pass=("tests/test_api.py::test_feature",),
        pass_to_pass=("tests/test_other.py::test_other",),
        seed_pass_rate=0,
        f2p_ratio=0.5,
        gold_patch_lines=2,
        gold_patch_files=1,
        test_exit_code=1,
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        return initial.model_copy(
            update={
                "gold_patch_lines": candidate.removed_lines,
                "gold_patch_files": len(candidate.patch_files),
            }
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=point_index,
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=1,
            gold_patch_line_preferred_minimum=1,
            gold_patch_line_maximum=7,
            minimum_remaining_p2p_test_points=1,
            repair=FeatureBenchRepairSettings(optimization_target="interval_center"),
        ),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.gold_patch_lines == 4
    assert result.replacement_object_ids == (replacement.object_id,)
    assert result.validations == 1


@pytest.mark.asyncio
async def test_deterministic_rollback_does_not_refill_above_the_center() -> None:
    index, plan = _fixture()
    initial = _accepted_quality().model_copy(update={"gold_patch_lines": 60})

    async def validate(_candidate: MaskPlan, _attempt: int) -> QualityReport:
        raise AssertionError("a plan above the interval center must not be revalidated")

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=_point_index(index, plan),
        settings=_settings().model_copy(
            update={"repair": FeatureBenchRepairSettings(optimization_target="interval_center")}
        ),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.plan == plan
    assert result.quality == initial
    assert result.validations == 0


@pytest.mark.asyncio
async def test_deterministic_rollback_compounds_partial_behavior_improvements() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/first.py": "def first():\n    return 1\n",
            "pkg/second.py": "def second():\n    return 2\n",
            "pkg/poison.py": "def poison():\n    return 3\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    first = index.by_qualname[("pkg/first.py", "first")]
    second = index.by_qualname[("pkg/second.py", "second")]
    poison = index.by_qualname[("pkg/poison.py", "poison")]
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        candidate_object_ids=(first.object_id, second.object_id, poison.object_id),
        p2p_test_ids=("tests/test_other.py::test_one", "tests/test_other.py::test_two"),
    )
    base_quality = QualityReport(
        accepted=False,
        fail_to_pass=("tests/test_api.py::test_feature",),
        pass_to_pass=(),
        seed_pass_rate=0,
        f2p_ratio=1,
        gold_patch_lines=2,
        gold_patch_files=1,
        errors=("P2P count 0 is below 2",),
        test_exit_code=1,
    )
    point_index = TracePointIndex(
        test_ids=(
            "tests/test_api.py::test_feature",
            "tests/test_other.py::test_one",
            "tests/test_other.py::test_two",
        ),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 6}),
        test_paths=("tests/test_api.py", "tests/test_other.py", "tests/test_other.py"),
        all_test_mask=7,
        traced_associations=0,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        targets = {item.object_id for item in candidate.targets}
        if poison.object_id in targets:
            return base_quality.model_copy(
                update={
                    "errors": ("mask broke test collection or the test runtime",),
                    "test_exit_code": 2,
                }
            )
        selected = len(targets & {first.object_id, second.object_id})
        pass_to_pass = (
            ("tests/test_other.py::test_one", "tests/test_other.py::test_two")
            if selected == 2
            else (("tests/test_other.py::test_one",) if selected == 1 else ())
        )
        accepted = selected == 2
        return QualityReport(
            accepted=accepted,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=pass_to_pass,
            seed_pass_rate=0,
            f2p_ratio=1 / (1 + len(pass_to_pass)),
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            errors=() if accepted else ("P2P count 1 is below 2",),
            test_exit_code=1,
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=base_quality,
        point_index=point_index,
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=1,
            gold_patch_line_preferred_minimum=1,
            gold_patch_line_maximum=15,
            minimum_remaining_p2p_test_points=2,
        ),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    final_targets = {item.object_id for item in result.plan.targets}
    assert result.quality.accepted
    assert first.object_id in final_targets
    assert second.object_id in final_targets
    assert poison.object_id not in final_targets


@pytest.mark.asyncio
async def test_deterministic_rollback_uses_top_restore_only_after_specifics() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/first.py": "def first_interface():\n    return 1\n",
            "pkg/second.py": "def second_interface():\n    return 2\n",
        }
    )
    first = index.by_qualname[("pkg/first.py", "first_interface")].model_copy(
        update={"classification": "top"}
    )
    second = index.by_qualname[("pkg/second.py", "second_interface")].model_copy(
        update={"classification": "top"}
    )
    p2p_id = "tests/test_other.py::test_other"
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(first, second),
        p2p_test_ids=(p2p_id,),
    )
    rejected = QualityReport(
        accepted=False,
        fail_to_pass=("tests/test_api.py::test_first",),
        pass_to_pass=(),
        seed_pass_rate=0,
        f2p_ratio=1,
        errors=("P2P count 0 is below 1",),
        test_exit_code=1,
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_first", p2p_id),
        object_test_masks=MappingProxyType({first.object_id: 2}),
        observed_object_test_masks=MappingProxyType({first.object_id: 2}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=1,
    )

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        if first.object_id in {item.object_id for item in candidate.targets}:
            return rejected
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_first",),
            pass_to_pass=(p2p_id,),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=rejected,
        point_index=point_index,
        settings=_settings(),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.accepted
    assert result.restored_object_ids == (first.object_id,)
    assert [item.object_id for item in result.plan.top_objects] == [second.object_id]


@pytest.mark.asyncio
async def test_deterministic_rollback_follows_newly_revealed_diagnostic_chain() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/first.py": "def first_dependency():\n    return 2\n",
            "pkg/second.py": "def second_dependency():\n    return 3\n",
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    first = index.by_qualname[("pkg/first.py", "first_dependency")].model_copy(
        update={"classification": "specific"}
    )
    second = index.by_qualname[("pkg/second.py", "second_dependency")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(first, second),
        candidate_object_ids=(first.object_id, second.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )

    def broken(path: str, symbol: str) -> QualityReport:
        return QualityReport(
            accepted=False,
            fail_to_pass=(),
            pass_to_pass=("tests/test_other.py::test_other",),
            errors=("mask broke test collection or the test runtime",),
            test_exit_code=2,
            collection_errors=(f"{path}: NameError: {symbol} is unavailable",),
        )

    initial = broken("pkg/first.py", "first_dependency")
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )
    validation_count = 0

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        nonlocal validation_count
        validation_count += 1
        targets = {item.object_id for item in candidate.targets}
        if first.object_id in targets:
            return initial
        if second.object_id in targets:
            return broken("pkg/second.py", "second_dependency")
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=point_index,
        settings=_settings(),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.accepted
    assert result.restored_object_ids == tuple(sorted((first.object_id, second.object_id)))
    assert validation_count <= 4


@pytest.mark.asyncio
async def test_deterministic_rollback_restores_definition_time_dependency_closure() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "def feature():\n    return 1\n",
            "pkg/dependent.py": (
                "def dependency():\n"
                "    return 2\n\n"
                "def owner(value=dependency()):\n"
                "    return value\n\n"
                "def unrelated():\n"
                "    return 3\n"
            ),
        }
    )
    top = index.by_qualname[("pkg/api.py", "feature")].model_copy(update={"classification": "top"})
    dependency = index.by_qualname[("pkg/dependent.py", "dependency")].model_copy(
        update={"classification": "specific"}
    )
    owner = index.by_qualname[("pkg/dependent.py", "owner")].model_copy(
        update={"classification": "specific"}
    )
    unrelated = index.by_qualname[("pkg/dependent.py", "unrelated")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(top,),
        specific_objects=(dependency, owner, unrelated),
        candidate_object_ids=(dependency.object_id, owner.object_id, unrelated.object_id),
        p2p_test_ids=("tests/test_other.py::test_other",),
    )
    rejected = QualityReport(
        accepted=False,
        fail_to_pass=(),
        pass_to_pass=("tests/test_other.py::test_other",),
        errors=("mask broke test collection or the test runtime",),
        test_exit_code=2,
        collection_errors=("pkg/dependent.py: owner failed during import",),
    )
    point_index = TracePointIndex(
        test_ids=("tests/test_api.py::test_feature", "tests/test_other.py::test_other"),
        object_test_masks=MappingProxyType({}),
        observed_object_test_masks=MappingProxyType({}),
        file_test_masks=MappingProxyType({"tests/test_api.py": 1, "tests/test_other.py": 2}),
        test_paths=("tests/test_api.py", "tests/test_other.py"),
        all_test_mask=3,
        traced_associations=0,
    )
    validated: list[frozenset[str]] = []

    async def validate(candidate: MaskPlan, _attempt: int) -> QualityReport:
        targets = frozenset(item.object_id for item in candidate.targets)
        validated.append(targets)
        assert owner.object_id not in targets
        assert dependency.object_id not in targets
        return QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_api.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            seed_pass_rate=0,
            f2p_ratio=0.5,
            gold_patch_lines=candidate.removed_lines,
            gold_patch_files=len(candidate.patch_files),
            test_exit_code=1,
        )

    result = await DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=rejected,
        point_index=point_index,
        settings=_settings(),
        validate_plan=validate,
        timeout_seconds=30,
    ).run()

    assert result.quality.accepted
    assert result.restored_object_ids == tuple(sorted((dependency.object_id, owner.object_id)))
    assert unrelated.object_id in {item.object_id for item in result.plan.targets}
    assert len(validated) == 1


@pytest.mark.asyncio
async def test_deterministic_rollback_reports_failed_search_costs() -> None:
    index, plan = _fixture()
    rejected = QualityReport(
        accepted=False,
        fail_to_pass=("tests/test_core.py::test_feature",),
        pass_to_pass=(),
        seed_pass_rate=0,
        f2p_ratio=1,
        gold_patch_lines=2,
        gold_patch_files=1,
        errors=("P2P count 0 is below 1",),
        test_exit_code=1,
    )
    events: list[tuple[str, dict[str, object]]] = []

    async def validate(_candidate: MaskPlan, _attempt: int) -> QualityReport:
        return rejected

    async def progress(kind: str, data: dict[str, object]) -> None:
        events.append((kind, data))

    with pytest.raises(CandidateRejected, match="exhausted scale-preserving candidates"):
        await DeterministicMaskRollback(
            index=index,
            planner=_planner(),
            traces={},
            initial_plan=plan,
            initial_quality=rejected,
            point_index=_point_index(index, plan),
            settings=_settings(),
            validate_plan=validate,
            timeout_seconds=30,
            progress=progress,
        ).run()

    kind, data = events[-1]
    assert kind == "failed"
    assert data["validations"] == 1
    assert isinstance(data["duration_seconds"], float)
    assert isinstance(data["validation_seconds"], float)
    assert isinstance(data["engine_seconds"], float)
    assert "exhausted scale-preserving candidates" in str(data["reason"])


@pytest.mark.asyncio
async def test_deterministic_rollback_bounds_unique_real_validations() -> None:
    index, plan = _fixture()
    rejected = _rejected_quality()
    attempts: list[int] = []
    events: list[tuple[str, dict[str, object]]] = []

    async def validate(_candidate: MaskPlan, attempt: int) -> QualityReport:
        attempts.append(attempt)
        return rejected

    async def progress(kind: str, data: dict[str, object]) -> None:
        events.append((kind, data))

    rollback = DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=rejected,
        point_index=_point_index(index, plan),
        settings=_settings().model_copy(
            update={"repair": FeatureBenchRepairSettings(maximum_validations=2)}
        ),
        validate_plan=validate,
        timeout_seconds=30,
        progress=progress,
    )
    candidates = tuple(
        plan.model_copy(update={"p2p_test_ids": (f"tests/test_other.py::test_{index}",)})
        for index in range(3)
    )

    async def exceed_validation_limit() -> tuple[MaskPlan, QualityReport]:
        await rollback._validate(candidates[0])  # pyright: ignore[reportPrivateUsage]
        # Repeating the same executable plan must hit the cache without spending the budget.
        await rollback._validate(candidates[0])  # pyright: ignore[reportPrivateUsage]
        await rollback._validate(candidates[1])  # pyright: ignore[reportPrivateUsage]
        await rollback._validate(candidates[2])  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("the validation limit must stop search")

    rollback._search = exceed_validation_limit  # type: ignore[method-assign]
    with pytest.raises(CandidateRejected, match="limit of 2 unique real validations"):
        await rollback.run()

    assert attempts == [1, 2]
    kind, data = events[-1]
    assert kind == "failed"
    assert data["validations"] == 2
    assert data["cache_hits"] == 1


@pytest.mark.asyncio
async def test_deterministic_rollback_keeps_an_accepted_plan_at_validation_limit() -> None:
    index, plan = _fixture()
    initial = _accepted_quality()

    async def validate(_candidate: MaskPlan, _attempt: int) -> QualityReport:
        return _rejected_quality()

    rollback = DeterministicMaskRollback(
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=initial,
        point_index=_point_index(index, plan),
        settings=_settings().model_copy(
            update={
                "repair": FeatureBenchRepairSettings(
                    optimization_target="interval_center",
                    maximum_validations=1,
                )
            }
        ),
        validate_plan=validate,
        timeout_seconds=30,
    )
    candidates = tuple(
        plan.model_copy(update={"p2p_test_ids": (f"tests/test_other.py::test_{index}",)})
        for index in range(2)
    )

    async def exceed_validation_limit() -> tuple[MaskPlan, QualityReport]:
        await rollback._validate(candidates[0])  # pyright: ignore[reportPrivateUsage]
        await rollback._validate(candidates[1])  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("the validation limit must stop search")

    rollback._search = exceed_validation_limit  # type: ignore[method-assign]
    result = await rollback.run()

    assert result.plan == plan
    assert result.quality == initial
    assert result.validations == 1


def test_repair_proposal_has_only_the_increment_contract() -> None:
    proposal = MaskRepairProposal(
        based_on_attempt="attempt-0000",
        must_restore_object_ids=("pkg/core.py::feature::1",),
        must_mask_object_ids=("pkg/core.py::replacement::4",),
        rationale="Restore the shared failure and preserve scale with a narrow replacement.",
    )

    assert set(proposal.model_dump()) == {
        "based_on_attempt",
        "must_restore_object_ids",
        "must_mask_object_ids",
        "rationale",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MaskRepairProposal.model_validate(
            {
                **proposal.model_dump(),
                "decision": "accept",
            }
        )
    with pytest.raises(ValidationError, match="restore or mask at least one"):
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            rationale="A no-op must not consume a real validation attempt.",
        )

    empty_rationale = proposal.model_copy(update={"rationale": ""})
    unbounded_rationale = proposal.model_copy(update={"rationale": "x" * 10_000})
    assert empty_rationale.rationale == ""
    assert len(unbounded_rationale.rationale) == 10_000


def test_repair_workspace_is_complete_searchable_and_feedback_is_bounded(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    workspace.prepare(index=index, initial_plan=plan, point_index=_point_index(index, plan))

    feedback = workspace.write_feedback(
        attempt_id="attempt-0000",
        based_on_attempt=None,
        plan=plan,
        quality=_rejected_quality(),
        previous_plan=None,
        previous_quality=None,
        proposal=None,
    )

    catalog = (tmp_path / "repair/catalog/objects.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in catalog.splitlines()]
    assert {record["qualname"] for record in records} == {"feature", "replacement"}
    replacement_record = next(record for record in records if record["qualname"] == "replacement")
    assert replacement_record["estimated_seed_affected_points"] == 1
    assert replacement_record["estimated_p2p_affected_points"] == 1
    assert feedback == "/featurebench-repair/attempts/attempt-0000/feedback"
    feedback_root = tmp_path / "repair/attempts/attempt-0000/feedback"
    result = json.loads((feedback_root / "result.json").read_text(encoding="utf-8"))
    assert result["p2p"] == 1
    assert "pass_to_pass" not in result
    assert (feedback_root / "diagnostics/test-output.log").read_text(
        encoding="utf-8"
    ) == "short real test diagnostic"
    assert [
        json.loads(line)
        for line in (feedback_root / "diagnostics/runner-errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ] == [{"error": "ImportError: feature is unavailable"}]
    runtime_violation = next(
        json.loads(line)
        for line in (feedback_root / "violations.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["code"] == "test.runtime_broken"
    )
    assert runtime_violation["diagnostic_file"] == "diagnostics/runner-errors.jsonl"
    assert not list(feedback_root.rglob("*.patch"))
    assert json.loads(workspace.outbox_host_path.read_text(encoding="utf-8")) == {
        "based_on_attempt": "attempt-0000",
        "must_restore_object_ids": [],
        "must_mask_object_ids": [],
        "rationale": "",
    }


def test_repair_feedback_links_runtime_traceback_to_current_mask_object(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    workspace.prepare(index=index, initial_plan=plan, point_index=_point_index(index, plan))
    quality = _rejected_quality().model_copy(
        update={
            "test_output": (
                "pkg/core.py:1: in feature\n"
                "    return missing_name\n"
                "E   NameError: name 'missing_name' is not defined\n"
            )
        }
    )

    workspace.write_feedback(
        attempt_id="attempt-0000",
        based_on_attempt=None,
        plan=plan,
        quality=quality,
        previous_plan=None,
        previous_quality=None,
        proposal=None,
    )

    violation_records = [
        json.loads(line)
        for line in (tmp_path / "repair/attempts/attempt-0000/feedback/violations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    implicated = [
        record
        for record in violation_records
        if record["code"] == "test.traceback_references_masked_file"
    ]
    assert implicated == [
        {
            "code": "test.traceback_references_masked_file",
            "message": (
                "The real failing output references a source file containing current Mask "
                "objects: pkg/core.py"
            ),
            "correction": (
                "Inspect these objects and the referenced traceback first; restore only the "
                "causal objects, then use narrower replacement masks if needed."
            ),
            "object_ids": [plan.top_objects[0].object_id],
            "diagnostic_file": "diagnostics/test-output.log",
        }
    ]


@pytest.mark.asyncio
async def test_submit_applies_increment_and_runs_authoritative_validation(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)
    validated_revisions: list[int] = []

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        assert attempt_index == 1
        validated_revisions.append(revised.revision)
        await MaskExecutor(index).apply(cast(ContainerSession, session), revised)
        return _accepted_quality()

    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    service = MaskRepairService(
        workspace=workspace,
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=_rejected_quality(),
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            must_restore_object_ids=(plan.top_objects[0].object_id,),
            must_mask_object_ids=(replacement.object_id,),
            rationale="Restore the broad breakage and replace it with the narrow feature object.",
        ).model_dump_json(),
        encoding="utf-8",
    )

    result = await service.submit_file(workspace.proposal_container_path)

    assert result.accepted
    assert result.attempt_id == "attempt-0001"
    assert service.accepted_attempt_id == "attempt-0001"
    assert validated_revisions == [1]
    assert [item.object_id for item in service.current_plan.top_objects] == [replacement.object_id]
    assert session.clean_count == 1
    assert "def feature" in session.files["pkg/core.py"]
    assert "def replacement" not in session.files["pkg/core.py"]


@pytest.mark.asyncio
async def test_submit_rejects_stale_base_without_running_tests(tmp_path: Path) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)
    validations = 0

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        del revised, attempt_index
        nonlocal validations
        validations += 1
        return _accepted_quality()

    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    service = MaskRepairService(
        workspace=workspace,
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=_rejected_quality(),
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-9999",
            must_restore_object_ids=(plan.top_objects[0].object_id,),
            rationale="Deliberately stale proposal for protocol validation.",
        ).model_dump_json(),
        encoding="utf-8",
    )

    result = await service.submit_file("outbox/proposal.json")

    assert not result.accepted
    assert result.attempt_id == "attempt-0001"
    assert validations == 0
    assert service.current_plan == plan
    violations = (tmp_path / "repair/attempts/attempt-0001/feedback/violations.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"code": "repair.stale_attempt"' in violations
    assert (
        json.loads(workspace.outbox_host_path.read_text(encoding="utf-8"))["based_on_attempt"]
        == "attempt-0001"
    )


@pytest.mark.asyncio
async def test_submit_accepts_older_attempt_when_rejected_submits_did_not_change_plan(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)
    validations = 0

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        del revised, attempt_index
        nonlocal validations
        validations += 1
        return _accepted_quality()

    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    service = MaskRepairService(
        workspace=workspace,
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=_rejected_quality(),
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            must_mask_object_ids=("pkg/core.py::missing::999",),
            rationale="Trigger a rejected protocol submit without changing the plan.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    rejected = await service.submit_file(workspace.proposal_container_path)
    assert rejected.attempt_id == "attempt-0001"
    assert validations == 0

    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            must_restore_object_ids=(plan.top_objects[0].object_id,),
            must_mask_object_ids=(replacement.object_id,),
            rationale="The old attempt still names the identical current Mask Plan.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    accepted = await service.submit_file(workspace.proposal_container_path)

    assert accepted.accepted
    assert accepted.attempt_id == "attempt-0002"
    assert validations == 1


@pytest.mark.asyncio
async def test_validated_restores_become_monotonic_protection(tmp_path: Path) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)
    validations = 0

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        del revised, attempt_index
        nonlocal validations
        validations += 1
        return _accepted_quality().model_copy(
            update={
                "accepted": False,
                "errors": ("Gold Patch lines remain below the configured minimum",),
            }
        )

    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    service = MaskRepairService(
        workspace=workspace,
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=_rejected_quality(),
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    feature = plan.top_objects[0]
    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            must_restore_object_ids=(feature.object_id,),
            must_mask_object_ids=(replacement.object_id,),
            rationale="Repair the broken runtime while preserving one mask.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    repaired = await service.submit_file(workspace.proposal_container_path)
    assert not repaired.accepted
    assert feature.object_id in service.learned_protected_ids
    learned = [
        json.loads(line)
        for line in (tmp_path / "repair/attempts/attempt-0001/feedback/learned-protected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["object_id"] for record in learned] == [feature.object_id]

    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0001",
            must_restore_object_ids=(replacement.object_id,),
            must_mask_object_ids=(feature.object_id,),
            rationale="Attempt to re-mask the object proven necessary for runtime health.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    blocked = await service.submit_file(workspace.proposal_container_path)

    assert not blocked.accepted
    assert validations == 1
    violations = (tmp_path / "repair/attempts/attempt-0002/feedback/violations.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"code": "repair.mask_protected"' in violations
    assert feature.object_id in violations


@pytest.mark.asyncio
async def test_restore_remains_protected_without_runtime_health_transition(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)
    validations = 0

    runtime_healthy_quality = _accepted_quality().model_copy(
        update={
            "accepted": False,
            "errors": ("Gold Patch lines remain below the configured minimum",),
        }
    )

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        del revised, attempt_index
        nonlocal validations
        validations += 1
        return runtime_healthy_quality

    workspace = MaskRepairWorkspace(tmp_path / "repair", _settings())
    service = MaskRepairService(
        workspace=workspace,
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=runtime_healthy_quality,
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    feature = plan.top_objects[0]
    replacement = index.by_qualname[("pkg/core.py", "replacement")]
    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0000",
            must_restore_object_ids=(feature.object_id,),
            must_mask_object_ids=(replacement.object_id,),
            rationale="Restore an object while the runtime is already healthy.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    restored = await service.submit_file(workspace.proposal_container_path)
    assert not restored.accepted
    assert feature.object_id in service.learned_protected_ids

    workspace.outbox_host_path.write_text(
        MaskRepairProposal(
            based_on_attempt="attempt-0001",
            must_restore_object_ids=(replacement.object_id,),
            must_mask_object_ids=(feature.object_id,),
            rationale="Try to reverse an earlier validated restore.",
        ).model_dump_json(),
        encoding="utf-8",
    )
    blocked = await service.submit_file(workspace.proposal_container_path)

    assert not blocked.accepted
    assert validations == 1
    violations = (tmp_path / "repair/attempts/attempt-0002/feedback/violations.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"code": "repair.mask_protected"' in violations
    assert feature.object_id in violations


def test_adapter_submit_tool_is_registered_outside_the_common_agent_layer(
    tmp_path: Path,
) -> None:
    index, plan = _fixture()
    session = RepairSession(index.sources)

    async def validate(revised: MaskPlan, attempt_index: int) -> QualityReport:
        del revised, attempt_index
        return _accepted_quality()

    service = MaskRepairService(
        workspace=MaskRepairWorkspace(tmp_path / "repair", _settings()),
        session=cast(ContainerSession, session),
        index=index,
        planner=_planner(),
        traces={},
        initial_plan=plan,
        initial_quality=_rejected_quality(),
        point_index=_point_index(index, plan),
        validate_plan=validate,
    )
    adapter_tools = MaskRepairSubmission(service=service, timeout_seconds=60).get_toolset().tools
    common_tools = (
        ContainerWorkspace(session=cast(ContainerSession, session), writable=True)
        .get_toolset()
        .tools
    )

    assert set(adapter_tools) == {"submit_repair_file"}
    assert "submit_repair_file" not in common_tools
    assert {"run_command", "read_file", "write_file", "edit_file"} <= set(common_tools)
    assert common_tools["run_command"].sequential is True
    assert common_tools["write_file"].sequential is True
    assert common_tools["edit_file"].sequential is True
    assert common_tools["read_file"].sequential is False
