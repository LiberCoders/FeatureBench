import random
from hashlib import sha256
from types import MappingProxyType

import pytest

from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.masking import FeatureSlicePlanner, JointOptimizationOptions
from codeworld.adapters.featurebench.models import (
    P2PSelection,
    TraceBundle,
    TraceEdge,
    TraceImport,
    TraceRecord,
)
from codeworld.adapters.featurebench.optimization import (
    DeletionOptimizationInfeasibleError,
    DeletionUnit,
    optimize_deletions,
)
from codeworld.adapters.featurebench.p2p import TracePointIndex, seed_import_object_ids
from codeworld.errors import CandidateRejected


def _optimize(
    units: tuple[DeletionUnit, ...],
    *,
    required_p2p: int,
    minimum_lines: int = 1,
    maximum_lines: int = 101,
    overlap_pairs: tuple[tuple[str, str], ...] = (),
    seed_test_mask: int = 0,
    minimum_f2p_test_points: int = 0,
    maximum_f2p_ratio: float | None = None,
    maximum_seed_pass_rate: float | None = None,
):
    return optimize_deletions(
        units=units,
        overlap_pairs=overlap_pairs,
        mandatory_unit_keys=frozenset(),
        base_eligible_test_mask=0b1111,
        fixed_removed_lines=1,
        fixed_removed_objects=1,
        minimum_removed_lines=minimum_lines,
        maximum_removed_lines=maximum_lines,
        seed_test_mask=seed_test_mask,
        minimum_f2p_test_points=minimum_f2p_test_points,
        maximum_f2p_ratio=maximum_f2p_ratio,
        maximum_seed_pass_rate=maximum_seed_pass_rate,
        minimum_p2p_test_points=required_p2p,
        timeout_seconds=5,
        workers=1,
        random_seed=0,
    )


def test_optimizer_requires_interval_and_prefers_its_symmetric_center() -> None:
    units = (
        DeletionUnit(key="near", removed_lines=60, object_count=1, test_mask=0),
        DeletionUnit(key="far", removed_lines=30, object_count=1, test_mask=0),
    )

    result = _optimize(units, required_p2p=1)

    assert result.selected_unit_keys == ("near",)
    assert result.removed_lines == 61
    assert result.center_distance == 10
    with pytest.raises(
        DeletionOptimizationInfeasibleError, match="no feasible solution"
    ) as failure:
        _optimize(units, required_p2p=1, minimum_lines=95, maximum_lines=101)
    diagnostics = failure.value.diagnostics
    assert "gold_patch_minimum" in diagnostics.sufficient_constraint_core
    assert diagnostics.fixed_removed_lines == 1
    assert diagnostics.candidate_removed_lines_upper_bound == 91
    assert diagnostics.as_dict()["removed_lines"] == {
        "fixed": 1,
        "candidate_upper_bound": 91,
        "minimum": 95,
        "maximum": 101,
    }


def test_optimizer_diagnoses_a_joint_line_and_p2p_conflict() -> None:
    units = (
        DeletionUnit(key="required-scale", removed_lines=50, object_count=1, test_mask=0b1110),
    )

    with pytest.raises(DeletionOptimizationInfeasibleError) as failure:
        _optimize(
            units,
            required_p2p=2,
            minimum_lines=51,
            maximum_lines=51,
        )

    diagnostics = failure.value.diagnostics
    assert {"gold_patch_minimum", "minimum_p2p"}.issubset(diagnostics.sufficient_constraint_core)
    assert diagnostics.base_eligible_p2p_test_points == 4
    assert diagnostics.minimum_p2p_test_points == 2


def test_optimizer_maximizes_files_after_equal_center_preference() -> None:
    units = (
        DeletionUnit(
            key="one-file",
            removed_lines=50,
            object_count=1,
            test_mask=0,
            files=("pkg/a.py",),
        ),
        DeletionUnit(
            key="two-files",
            removed_lines=50,
            object_count=1,
            test_mask=0,
            files=("pkg/b.py", "pkg/c.py"),
        ),
    )

    result = _optimize(
        units,
        required_p2p=1,
        overlap_pairs=(("one-file", "two-files"),),
    )

    assert result.selected_unit_keys == ("two-files",)
    assert result.modified_files == 2


def test_optimizer_prefers_center_before_more_files() -> None:
    units = (
        DeletionUnit(
            key="near-center",
            removed_lines=60,
            object_count=1,
            test_mask=0,
            files=("pkg/a.py",),
        ),
        DeletionUnit(
            key="more-files",
            removed_lines=30,
            object_count=1,
            test_mask=0,
            files=("pkg/b.py", "pkg/c.py"),
        ),
    )

    result = _optimize(
        units,
        required_p2p=1,
        overlap_pairs=(("near-center", "more-files"),),
    )

    assert result.selected_unit_keys == ("near-center",)
    assert result.modified_files == 1
    assert result.center_distance == 10


def test_optimizer_enforces_estimated_f2p_ratio_then_maximizes_p2p() -> None:
    units = (
        DeletionUnit(
            key="keep-few",
            removed_lines=50,
            object_count=1,
            test_mask=0b0111,
            seed_test_mask=0b010000,
        ),
        DeletionUnit(
            key="keep-many",
            removed_lines=50,
            object_count=1,
            test_mask=0b0001,
            seed_test_mask=0b100000,
        ),
    )

    result = _optimize(
        units,
        required_p2p=1,
        minimum_lines=51,
        maximum_lines=51,
        overlap_pairs=(("keep-few", "keep-many"),),
        seed_test_mask=0b110000,
        minimum_f2p_test_points=1,
        maximum_f2p_ratio=0.3,
        maximum_seed_pass_rate=0.5,
    )

    assert result.selected_unit_keys == ("keep-many",)
    assert result.estimated_f2p_test_points == 1
    assert result.selected_p2p_test_points == 3


def test_optimizer_enforces_dependency_closure_and_overlap() -> None:
    dependency_units = (
        DeletionUnit(
            key="dependent",
            removed_lines=40,
            object_count=1,
            test_mask=0b0001,
            dependencies=("owner",),
        ),
        DeletionUnit(key="owner", removed_lines=50, object_count=1, test_mask=0b0010),
    )
    dependency_result = _optimize(
        dependency_units,
        required_p2p=2,
        minimum_lines=91,
        maximum_lines=91,
    )
    assert dependency_result.selected_unit_keys == ("dependent", "owner")

    overlap_units = (
        DeletionUnit(key="class", removed_lines=60, object_count=1, test_mask=0),
        DeletionUnit(key="method", removed_lines=50, object_count=1, test_mask=0),
    )
    overlap_result = _optimize(
        overlap_units,
        required_p2p=4,
        minimum_lines=60,
        maximum_lines=70,
        overlap_pairs=(("class", "method"),),
    )
    assert overlap_result.selected_unit_keys == ("class",)
    assert overlap_result.removed_lines == 61


def test_optimizer_compresses_equal_test_conflict_signatures() -> None:
    result = _optimize(
        (
            DeletionUnit(key="a", removed_lines=30, object_count=1, test_mask=0b0011),
            DeletionUnit(key="b", removed_lines=20, object_count=1, test_mask=0b0100),
        ),
        required_p2p=1,
        minimum_lines=51,
        maximum_lines=51,
    )

    assert result.status == "OPTIMAL"
    assert result.signature_groups == 2
    assert result.solve_seconds < 5


def test_representative_p2p_maximizes_coverage_with_file_bound() -> None:
    point_index = TracePointIndex(
        test_ids=("a::one", "a::two", "b::one", "b::two"),
        object_test_masks=MappingProxyType(
            {
                "object-a": 0b0011,
                "object-b": 0b0010,
                "object-c": 0b1100,
            }
        ),
        observed_object_test_masks=MappingProxyType(
            {"object-a": 0b0011, "object-b": 0b0010, "object-c": 0b1100}
        ),
        file_test_masks=MappingProxyType({"a": 0b0011, "b": 0b1100}),
        test_paths=("a", "a", "b", "b"),
        all_test_mask=0b1111,
        traced_associations=5,
    )

    diverse = point_index.representative_test_ids(
        eligible_test_mask=0b1111,
        count=2,
        deleted_object_ids=(),
        maximum_files=2,
        random_seed=0,
        seed_test_file="seed",
    )
    concentrated = point_index.representative_test_ids(
        eligible_test_mask=0b1111,
        count=2,
        deleted_object_ids=(),
        maximum_files=1,
        random_seed=0,
        seed_test_file="seed",
    )

    assert "a::two" in diverse
    assert any(test_id.startswith("b::") for test_id in diverse)
    assert concentrated == ("a::one", "a::two")


def test_representative_p2p_lazy_queue_matches_full_rescoring() -> None:
    def full_rescoring(
        point_index: TracePointIndex,
        *,
        count: int,
        maximum_files: int | None,
        random_seed: int,
    ) -> tuple[str, ...]:
        coverage_by_test: dict[int, int] = {}
        for object_index, object_id in enumerate(sorted(point_index.object_test_masks)):
            mask = point_index.object_test_masks[object_id]
            while mask:
                bit = mask & -mask
                test_index = bit.bit_length() - 1
                coverage_by_test[test_index] = coverage_by_test.get(test_index, 0) | (
                    1 << object_index
                )
                mask ^= bit
        remaining = set(range(len(point_index.test_ids)))
        selected: list[int] = []
        selected_files: set[str] = set()
        covered_objects = 0
        while len(selected) < count:
            allowed = [
                test_index
                for test_index in remaining
                if point_index.test_paths[test_index] in selected_files
                or maximum_files is None
                or len(selected_files) < maximum_files
            ]
            if not allowed:
                raise CandidateRejected("file bound is infeasible")
            current_coverage = covered_objects
            current_files = frozenset(selected_files)

            def score(
                test_index: int,
                current_coverage: int = current_coverage,
                current_files: frozenset[str] = current_files,
            ) -> tuple[int, int, int, int]:
                coverage = coverage_by_test.get(test_index, 0)
                digest = sha256(
                    f"{random_seed}:seed:{point_index.test_ids[test_index]}".encode()
                ).digest()
                return (
                    (coverage & ~current_coverage).bit_count(),
                    int(point_index.test_paths[test_index] in current_files),
                    coverage.bit_count(),
                    -int.from_bytes(digest[:8], "big"),
                )

            chosen = max(allowed, key=score)
            selected.append(chosen)
            remaining.remove(chosen)
            selected_files.add(point_index.test_paths[chosen])
            covered_objects |= coverage_by_test.get(chosen, 0)
        return tuple(sorted(point_index.test_ids[test_index] for test_index in selected))

    for random_seed in range(50):
        generator = random.Random(random_seed)
        test_count = generator.randint(8, 24)
        file_count = generator.randint(1, 5)
        test_ids = tuple(f"test-{index}" for index in range(test_count))
        test_paths = tuple(
            f"tests/file-{generator.randrange(file_count)}.py" for _ in range(test_count)
        )
        object_masks = {
            f"object-{index}": generator.getrandbits(test_count)
            for index in range(generator.randint(1, 16))
        }
        file_masks: dict[str, int] = {}
        for test_index, path in enumerate(test_paths):
            file_masks[path] = file_masks.get(path, 0) | (1 << test_index)
        point_index = TracePointIndex(
            test_ids=test_ids,
            object_test_masks=MappingProxyType(object_masks),
            observed_object_test_masks=MappingProxyType(object_masks),
            file_test_masks=MappingProxyType(file_masks),
            test_paths=test_paths,
            all_test_mask=(1 << test_count) - 1,
            traced_associations=sum(mask.bit_count() for mask in object_masks.values()),
        )
        count = generator.randint(1, test_count - 1)
        maximum_files = generator.choice((None, 1, 2, 3))
        try:
            expected = full_rescoring(
                point_index,
                count=count,
                maximum_files=maximum_files,
                random_seed=random_seed,
            )
        except CandidateRejected:
            with pytest.raises(CandidateRejected):
                point_index.representative_test_ids(
                    eligible_test_mask=point_index.all_test_mask,
                    count=count,
                    deleted_object_ids=(),
                    maximum_files=maximum_files,
                    random_seed=random_seed,
                    seed_test_file="seed",
                )
        else:
            assert (
                point_index.representative_test_ids(
                    eligible_test_mask=point_index.all_test_mask,
                    count=count,
                    deleted_object_ids=(),
                    maximum_files=maximum_files,
                    random_seed=random_seed,
                    seed_test_file="seed",
                )
                == expected
            )


def test_joint_planner_optimizes_specific_deletion_before_selecting_p2p() -> None:
    sources = {
        "pkg/core.py": (
            "def feature():\n    return candidate_a() + candidate_b()\n\n"
            "def candidate_a():\n    return 1\n\n"
            "def candidate_b():\n    return 2\n"
        ),
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_one.py": "def test_one():\n    assert True\n",
        "tests/test_two.py": "def test_two():\n    assert True\n",
        "tests/test_three.py": "def test_three():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    seed_test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    feature = index.by_qualname[("pkg/core.py", "feature")]
    candidate_a = index.by_qualname[("pkg/core.py", "candidate_a")]
    candidate_b = index.by_qualname[("pkg/core.py", "candidate_b")]
    seed_id = "tests/test_seed.py::test_seed"
    external_ids = (
        "tests/test_one.py::test_one",
        "tests/test_two.py::test_two",
        "tests/test_three.py::test_three",
    )
    seed = TraceBundle(
        test_file="tests/test_seed.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(seed_id,),
            )
            for item in (seed_test, feature, candidate_a, candidate_b)
        ),
        edges=(
            TraceEdge(caller=seed_test.object_id, callee=feature.object_id, calls=1),
            TraceEdge(caller=feature.object_id, callee=candidate_a.object_id, calls=1),
            TraceEdge(caller=feature.object_id, callee=candidate_b.object_id, calls=1),
        ),
    )
    external = tuple(
        TraceBundle(
            test_file=f"tests/test_{name}.py",
            records=(
                TraceRecord(
                    path=(candidate_a if name == "one" else candidate_b).path,
                    qualname=(candidate_a if name == "one" else candidate_b).qualname,
                    line=(candidate_a if name == "one" else candidate_b).line,
                    calls=1,
                    test_ids=(test_id,),
                ),
            )
            if name != "three"
            else (),
        )
        for name, test_id in zip(("one", "two", "three"), external_ids, strict=True)
    )
    traces = {bundle.test_file: bundle for bundle in (seed, *external)}
    test_paths = {
        seed_id: seed.test_file,
        **{
            test_id: bundle.test_file
            for test_id, bundle in zip(external_ids, external, strict=True)
        },
    }
    point_index = TracePointIndex.build(
        index=index,
        traces=traces.values(),
        candidate_test_ids=(seed_id, *external_ids),
        test_paths=test_paths,
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=4,
        maximum_code_lines=4,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces=traces,
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=(feature,),
        p2p=P2PSelection(test_ids=(), top_object_ids=(feature.object_id,)),
        joint_optimization=JointOptimizationOptions(
            point_index=point_index,
            minimum_p2p_test_points=2,
            maximum_p2p_test_points=2,
            maximum_selected_p2p_test_files=2,
            minimum_f2p_test_points=1,
            maximum_f2p_test_points=None,
            maximum_f2p_ratio=0.7,
            maximum_seed_pass_rate=0.3,
            timeout_seconds=5,
            workers=1,
        ),
    )

    assert len(plan.specific_objects) == 1
    assert len(plan.p2p_test_ids) == 2
    assert external_ids[2] in plan.p2p_test_ids
    assert plan.optimization is not None
    assert plan.optimization.status == "OPTIMAL"
    assert plan.optimization.eligible_p2p_test_points == 2

    unaffected_seed_id = "tests/test_seed.py::test_unaffected"
    weak_point_index = TracePointIndex.build(
        index=index,
        traces=traces.values(),
        candidate_test_ids=(seed_id, unaffected_seed_id, *external_ids),
        test_paths={
            **test_paths,
            unaffected_seed_id: seed.test_file,
        },
    )
    with pytest.raises(CandidateRejected, match="no feasible solution"):
        planner.create_plan(
            index=index,
            seed_trace=seed,
            all_traces=traces,
            collection_trace=TraceBundle(test_file="<collection>", records=()),
            top_objects=(feature,),
            p2p=P2PSelection(test_ids=(), top_object_ids=(feature.object_id,)),
            joint_optimization=JointOptimizationOptions(
                point_index=weak_point_index,
                minimum_p2p_test_points=2,
                maximum_p2p_test_points=2,
                maximum_selected_p2p_test_files=2,
                minimum_f2p_test_points=1,
                maximum_f2p_test_points=None,
                maximum_f2p_ratio=0.7,
                maximum_seed_pass_rate=0.3,
                timeout_seconds=5,
                workers=1,
            ),
        )


def test_collection_time_production_imports_remain_maskable_with_import_edits() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/api.py": "from pkg.core import helper\n\ndef feature():\n    return helper()\n",
            "pkg/core.py": "def helper():\n    return 1\n",
            "tests/test_seed.py": "def test_seed():\n    assert True\n",
            "tests/test_safe.py": "def test_safe():\n    assert True\n",
        }
    )
    seed_test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    helper = index.by_qualname[("pkg/core.py", "helper")]
    seed_id = "tests/test_seed.py::test_seed"
    safe_id = "tests/test_safe.py::test_safe"
    seed = TraceBundle(
        test_file="tests/test_seed.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(seed_id,),
            )
            for item in (seed_test, feature, helper)
        ),
        edges=(
            TraceEdge(
                caller=seed_test.object_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(seed_id,),
            ),
            TraceEdge(
                caller=feature.object_id,
                callee=helper.object_id,
                calls=1,
                test_ids=(seed_id,),
            ),
        ),
    )
    safe = TraceBundle(test_file="tests/test_safe.py", records=())
    collection = TraceBundle(
        test_file="<collection>",
        records=(),
        imports=(
            TraceImport(
                importer_path="pkg/api.py",
                importer_line=1,
                imported_name="helper",
                defined_path="pkg/core.py",
                defined_name="helper",
                defined_line=helper.line,
            ),
        ),
    )
    traces = {seed.test_file: seed, safe.test_file: safe}
    point_index = TracePointIndex.build(
        index=index,
        traces=traces.values(),
        candidate_test_ids=(seed_id, safe_id),
        test_paths={seed_id: seed.test_file, safe_id: safe.test_file},
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=4,
        maximum_code_lines=4,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces=traces,
        collection_trace=collection,
        top_objects=(feature,),
        p2p=P2PSelection(test_ids=(), top_object_ids=(feature.object_id,)),
        joint_optimization=JointOptimizationOptions(
            point_index=point_index,
            minimum_p2p_test_points=1,
            maximum_p2p_test_points=1,
            maximum_selected_p2p_test_files=1,
            minimum_f2p_test_points=1,
            maximum_f2p_test_points=None,
            maximum_f2p_ratio=0.7,
            maximum_seed_pass_rate=0.3,
            timeout_seconds=5,
            workers=1,
        ),
    )

    assert [item.object_id for item in plan.specific_objects] == [helper.object_id]
    assert plan.import_edits[0].importer_path == "pkg/api.py"
    assert plan.import_edits[0].remove_names == ("helper",)


def test_p2p_index_projects_test_imports_to_every_point_in_the_file() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": "def feature():\n    return 1\n",
            "tests/test_core.py": (
                "from pkg.core import feature\n\n"
                "def test_one():\n    assert feature\n\n"
                "def test_two():\n    assert feature\n"
            ),
        }
    )
    test_ids = ("tests/test_core.py::test_one", "tests/test_core.py::test_two")
    point_index = TracePointIndex.build(
        index=index,
        traces=(TraceBundle(test_file="tests/test_core.py", records=()),),
        candidate_test_ids=test_ids,
        test_paths=dict.fromkeys(test_ids, "tests/test_core.py"),
    )
    feature = index.by_qualname[("pkg/core.py", "feature")]

    assert point_index.object_test_masks[feature.object_id] == 0b11
    assert seed_import_object_ids(
        index=index,
        test_file="tests/test_core.py",
    ) == frozenset({feature.object_id})


def test_p2p_selection_excludes_every_point_in_a_file_with_a_masked_import() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": "def feature():\n    return 1\n",
            "tests/test_seed.py": "def test_seed():\n    assert True\n",
            "tests/test_importing.py": (
                "from pkg.core import feature\n\n"
                "def test_one():\n    assert feature\n\n"
                "def test_two():\n    assert feature\n"
            ),
            "tests/test_safe.py": "def test_safe():\n    assert True\n",
        }
    )
    seed_id = "tests/test_seed.py::test_seed"
    importing_ids = (
        "tests/test_importing.py::test_one",
        "tests/test_importing.py::test_two",
    )
    safe_id = "tests/test_safe.py::test_safe"
    paths = {
        seed_id: "tests/test_seed.py",
        importing_ids[0]: "tests/test_importing.py",
        importing_ids[1]: "tests/test_importing.py",
        safe_id: "tests/test_safe.py",
    }
    point_index = TracePointIndex.build(
        index=index,
        traces=tuple(
            TraceBundle(test_file=path, records=()) for path in sorted(set(paths.values()))
        ),
        candidate_test_ids=tuple(paths),
        test_paths=paths,
    )
    feature = index.by_qualname[("pkg/core.py", "feature")]

    selection = point_index.select_p2p(
        seed_test_file="tests/test_seed.py",
        top_object_ids=(feature.object_id,),
        minimum_test_points=1,
        maximum_test_points=None,
        random_seed=0,
    )

    assert selection.test_ids == (safe_id,)


def test_p2p_index_projects_batched_trace_dependencies_to_cold_singletons() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": (
                "from functools import lru_cache\n\n"
                "def cold_dependency():\n    return 1\n\n"
                "@lru_cache\n"
                "def cached_dependency():\n    return cold_dependency()\n"
            ),
            "tests/test_core.py": (
                "def test_warms_cache():\n    pass\n\ndef test_reuses_cache():\n    pass\n"
            ),
        }
    )
    cached = index.by_qualname[("pkg/core.py", "cached_dependency")]
    dependency = index.by_qualname[("pkg/core.py", "cold_dependency")]
    test_ids = (
        "tests/test_core.py::test_warms_cache",
        "tests/test_core.py::test_reuses_cache",
    )
    point_index = TracePointIndex.build(
        index=index,
        traces=(
            TraceBundle(
                test_file="tests/test_core.py",
                records=(
                    TraceRecord(
                        path=cached.path,
                        qualname=cached.qualname,
                        line=cached.line,
                        calls=1,
                        test_ids=(test_ids[0],),
                    ),
                    TraceRecord(
                        path=dependency.path,
                        qualname=dependency.qualname,
                        line=dependency.line,
                        calls=1,
                        test_ids=(test_ids[0],),
                    ),
                ),
                edges=(
                    TraceEdge(
                        caller=cached.object_id,
                        callee=dependency.object_id,
                        calls=1,
                        test_ids=(test_ids[0],),
                    ),
                ),
            ),
        ),
        candidate_test_ids=test_ids,
        test_paths=dict.fromkeys(test_ids, "tests/test_core.py"),
    )

    assert point_index.object_test_masks[dependency.object_id] == 0b11


def test_p2p_index_uses_point_attribution_from_runtime_edges() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": ("def feature():\n    return 1\n\ndef untraced():\n    return 2\n"),
            "tests/test_seed.py": "def test_seed():\n    assert True\n",
            "tests/test_safe.py": "def test_safe():\n    assert True\n",
        }
    )
    seed = index.by_qualname[("tests/test_seed.py", "test_seed")]
    feature = index.by_qualname[("pkg/core.py", "feature")]
    untraced = index.by_qualname[("pkg/core.py", "untraced")]
    seed_id = "tests/test_seed.py::test_seed"
    safe_id = "tests/test_safe.py::test_safe"
    point_index = TracePointIndex.build(
        index=index,
        traces=(
            TraceBundle(
                test_file="tests/test_seed.py",
                records=(
                    TraceRecord(
                        path=feature.path,
                        qualname=feature.qualname,
                        line=feature.line,
                        calls=1,
                    ),
                ),
                edges=(
                    TraceEdge(
                        caller=seed.object_id,
                        callee=feature.object_id,
                        calls=1,
                        test_ids=(seed_id,),
                    ),
                ),
            ),
            TraceBundle(test_file="tests/test_safe.py", records=()),
        ),
        candidate_test_ids=(seed_id, safe_id),
        test_paths={
            seed_id: "tests/test_seed.py",
            safe_id: "tests/test_safe.py",
        },
    )

    assert point_index.object_test_masks[feature.object_id].bit_count() == 1
    selection = point_index.select_p2p(
        seed_test_file="tests/test_seed.py",
        top_object_ids=(feature.object_id,),
        minimum_test_points=1,
        maximum_test_points=None,
        random_seed=0,
    )
    assert selection.test_ids == (safe_id,)
    with pytest.raises(CandidateRejected, match="without point-grained P2P evidence"):
        point_index.select_p2p(
            seed_test_file="tests/test_seed.py",
            top_object_ids=(untraced.object_id,),
            minimum_test_points=1,
            maximum_test_points=None,
            random_seed=0,
        )
