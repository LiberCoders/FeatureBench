from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai_harness.filesystem import FileSystem

import codeworld.adapters.featurebench.generator as generator_module
import codeworld.adapters.featurebench.indexing as indexing_module
import codeworld.adapters.featurebench.quality_gate as quality_gate_module
from codeworld.adapters.base import GeneratorContext
from codeworld.adapters.catalog import default_adapter_catalog
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.definition import featurebench_definition
from codeworld.adapters.featurebench.finalization import assemble_task_record
from codeworld.adapters.featurebench.generator import FeatureBenchGenerator
from codeworld.adapters.featurebench.indexing import RepositoryIndex, index_sources
from codeworld.adapters.featurebench.issue.agent import author_issue
from codeworld.adapters.featurebench.issue.capability import IssueAuthorSubmission
from codeworld.adapters.featurebench.issue.models import IssueAuthorCompletion
from codeworld.adapters.featurebench.issue.validation import (
    issue_validation_issues,
)
from codeworld.adapters.featurebench.masking import (
    FeatureSlicePlanner,
    JointOptimizationOptions,
    MaskExecutor,
    MaskIntervalInfeasible,
    SeedFeasibilityProfile,
    hide_test_points,
    transform_sources,
)
from codeworld.adapters.featurebench.models import (
    FeatureBenchMetrics,
    FeatureBenchPayload,
    GoldPatchMetrics,
    GoldPatchTier,
    InterfaceContract,
    IssueDraft,
    MaskPlan,
    P2PSelection,
    QualityReport,
    TestPointMetrics,
    TestReport,
    TopBatchClassification,
    TopCandidateDecision,
    TopObjectMetrics,
    TraceBundle,
    TraceEdge,
    TraceImport,
    TraceRecord,
)
from codeworld.adapters.featurebench.p2p import (
    TracePointIndex,
    seed_import_object_ids,
    trace_import_object_ids,
)
from codeworld.adapters.featurebench.prompting import build_interfaces, render_prompt
from codeworld.adapters.featurebench.quality import (
    apply_f2p_hideability_gate,
    apply_gold_patch_gates,
    assess_quality,
    build_featurebench_metrics,
    reapply_gold_patch_tier_gates,
)
from codeworld.adapters.featurebench.quality_gate import FeatureBenchQualityGate
from codeworld.adapters.featurebench.repair.models import (
    MaskRepairProposal,
    MaskRepairProposalError,
)
from codeworld.adapters.featurebench.runtime import FeatureBenchTaskRuntime
from codeworld.adapters.featurebench.testing import (
    PythonTracer,
    TraceStatistics,
    collection_trace_from_bundles,
    decode_trace_test_id_ranges,
    merge_trace_bundles,
)
from codeworld.adapters.featurebench.top_agent import FeatureGraphTraversal
from codeworld.agents.factory import AgentFactory
from codeworld.config import AgentFactorySettings
from codeworld.environment.frameworks import pytest as pytest_contract
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner import (
    EnvironmentRunner,
)
from codeworld.environment.runner_contract import (
    CollectedTestPoint,
    RunnerResponse,
)
from codeworld.environment.runner_contract import (
    TestPointExecution as RunnerTestPointExecution,
)
from codeworld.errors import (
    CandidateRejected,
    CommandTimeoutError,
    ConfigurationError,
    ContractError,
    RunnerTimeoutError,
)
from codeworld.events import NullEventSink
from codeworld.models import RuntimeResources, RuntimeSpec, TaskRecord
from codeworld.progress import AdapterPhase, AdapterProgressReporter
from codeworld.runtime.base import ContainerSessionProtocol
from codeworld.runtime.container import ContainerSession
from codeworld.selection import WorkSelection, WorkState


class FakeSession:
    def __init__(self, files: dict[str, str]) -> None:
        self.clean = files.copy()
        self.files = files.copy()
        self.clean_count = 0

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content.decode()

    async def clean_workspace(self) -> None:
        self.clean_count += 1
        self.files = self.clean.copy()


def _payload_gold_patch() -> str:
    return (
        "diff --git b/pkg/core.py a/pkg/core.py\n"
        "--- b/pkg/core.py\n"
        "+++ a/pkg/core.py\n"
        "@@ -1 +1 @@\n"
        "-\n"
        "+return 1\n"
    )


def _payload_metrics(*, f2p: int = 1, p2p: int = 1) -> FeatureBenchMetrics:
    return FeatureBenchMetrics(
        gold_patch=GoldPatchMetrics(lines=1, files=1),
        tests=TestPointMetrics(f2p=f2p, p2p=p2p),
        top=TopObjectMetrics(objects=1, gold_patch_lines=1, gold_patch_ratio=1.0),
    )


class FakeDisposableRuntime:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.closed: list[FakeSession] = []

    @asynccontextmanager
    async def session(self, image: str, *, runtime_spec: RuntimeSpec):
        del image, runtime_spec
        session = FakeSession({})
        self.sessions.append(session)
        try:
            yield session
        finally:
            self.closed.append(session)


class FakeTracer:
    def __init__(self, failing_test_id: str) -> None:
        self.session = FakeSession({})
        self.failing_test_id = failing_test_id
        self.calls: list[tuple[str, ...]] = []
        self.statistics: dict[tuple[str, tuple[str, ...]], TraceStatistics] = {}

    async def trace(
        self,
        *,
        test_command: tuple[str, ...],
        test_file: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> TraceBundle:
        del test_command, timeout_seconds
        self.calls.append(test_ids)
        if self.failing_test_id in test_ids:
            if len(test_ids) > 1:
                raise ContractError("batch contains failing point")
            raise ContractError("point failed")
        self.statistics[(test_file, test_ids)] = TraceStatistics(
            duration_seconds=1,
            report_bytes=100,
            test_ids=len(test_ids),
            records=1,
            edges=0,
            imports=0,
            process_count=1,
        )
        return TraceBundle(
            test_file=test_file,
            records=(
                TraceRecord(
                    path="pkg/core.py",
                    qualname="feature",
                    line=1,
                    calls=1,
                    test_ids=test_ids,
                ),
            ),
            passed_test_ids=test_ids,
        )


class NoopFeatureBenchRuntime(FeatureBenchTaskRuntime):
    async def prepare(
        self,
        session: ContainerSessionProtocol,
        task: TaskRecord,
        payload: BaseModel,
    ) -> None:
        del session, task, payload


class FakeIsolatedRunner:
    def __init__(
        self,
        *,
        failing_path: str,
        test_paths: dict[str, str],
    ) -> None:
        self.failing_path = failing_path
        self.test_paths = test_paths
        self.calls: list[tuple[str, ...]] = []

    async def run(self, test_ids: tuple[str, ...], *, timeout_seconds: float) -> RunnerResponse:
        del timeout_seconds
        self.calls.append(test_ids)
        if any(self.test_paths[test_id] == self.failing_path for test_id in test_ids):
            raise ContractError("pytest collection failed")
        results = tuple(
            RunnerTestPointExecution(
                id=node,
                outcome=(
                    "failed"
                    if self.test_paths[node].endswith("two.py")
                    and index
                    == next(
                        item_index
                        for item_index, item in enumerate(test_ids)
                        if self.test_paths[item].endswith("two.py")
                    )
                    else "passed"
                ),
            )
            for index, node in enumerate(test_ids)
        )
        failed = any(result.outcome == "failed" for result in results)
        return RunnerResponse(
            action="run",
            exit_code=1 if failed else 0,
            results=results,
            output="one selected test failed" if failed else "",
        )


class FeatureBenchGeneratorHarness(FeatureBenchGenerator):
    def __init__(self) -> None:
        super().__init__(runtime=FeatureBenchTaskRuntime())

    @staticmethod
    def ensure_supported_environment(environment: EnvironmentRecord) -> None:
        FeatureBenchGenerator._ensure_supported_environment(environment)

    async def isolated_candidate_report(
        self,
        runner: EnvironmentRunner,
        plan: MaskPlan,
        seed_test_ids: tuple[str, ...],
        test_paths: dict[str, str],
        *,
        validation_timeout_seconds: float = 30,
        full_test_timeout_seconds: float = 30,
        diagnostic_fallback: bool = True,
    ) -> tuple[tuple[str, ...], TestReport]:
        async def run_batch(
            test_ids: tuple[str, ...],
            remaining_seconds: float,
        ) -> RunnerResponse:
            return await runner.run(
                test_ids,
                timeout_seconds=min(full_test_timeout_seconds, remaining_seconds),
            )

        return await self._isolated_candidate_report(
            plan,
            seed_test_ids,
            test_paths,
            run_batch=run_batch,
            timeout_seconds=validation_timeout_seconds,
            diagnostic_fallback=diagnostic_fallback,
        )

    def normalize_clean_baseline(
        self,
        *,
        plan: MaskPlan,
        seed_test_ids: tuple[str, ...],
        selected_test_ids: tuple[str, ...],
        report: TestReport,
    ) -> tuple[MaskPlan, tuple[str, ...]]:
        return self._normalize_clean_baseline(
            plan=plan,
            seed_test_ids=seed_test_ids,
            selected_test_ids=selected_test_ids,
            report=report,
        )

    async def trace_file(
        self,
        tracer: PythonTracer,
        test_file: str,
        test_ids: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        async def trace_selection(
            selection: tuple[str, ...],
            remaining_seconds: float,
        ) -> tuple[TraceBundle, TraceStatistics]:
            await tracer.session.clean_workspace()
            bundle = await tracer.trace(
                test_command=self.settings.test_command,
                test_file=test_file,
                test_ids=selection,
                timeout_seconds=remaining_seconds,
            )
            return bundle, tracer.statistics[(test_file, selection)]

        return await self._trace_file(
            test_file=test_file,
            test_ids=test_ids,
            trace_selection=trace_selection,
            timeout_seconds=(
                self.settings.trace_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
        )

    async def trace_selection_in_fresh_session(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        test_file: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float | None = None,
    ) -> tuple[TraceBundle, TraceStatistics]:
        return await self._trace_selection_in_fresh_session(
            environment=environment,
            context=context,
            test_file=test_file,
            test_ids=test_ids,
            timeout_seconds=(
                self.settings.trace_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
        )

    async def run_candidate_batch_in_fresh_session(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        candidate_patch: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> RunnerResponse:
        return await self._run_candidate_batch_in_fresh_session(
            environment=environment,
            context=context,
            candidate_patch=candidate_patch,
            test_ids=test_ids,
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.asyncio
async def test_offline_trace_stage_finishes_all_files_with_bounded_concurrency() -> None:
    trace_files = tuple(f"tests/test_{index}.py" for index in range(7))
    active = 0
    maximum_active = 0
    completed: list[str] = []

    async def trace_file(test_file: str) -> Any:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        completed.append(test_file)
        active -= 1
        return test_file

    results = await FeatureBenchGenerator._run_trace_inventory(  # type: ignore[arg-type]
        trace_files,
        concurrency=3,
        trace_file=trace_file,
    )

    assert results == trace_files
    assert set(completed) == set(trace_files)
    assert maximum_active == 3
    assert active == 0


@pytest.mark.asyncio
async def test_offline_trace_capacity_is_shared_across_environments() -> None:
    semaphore = asyncio.Semaphore(3)
    active = 0
    maximum_active = 0

    async def trace_file(test_file: str) -> Any:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return test_file

    first = tuple(f"tests/first_{index}.py" for index in range(5))
    second = tuple(f"tests/second_{index}.py" for index in range(5))
    first_results, second_results = await asyncio.gather(
        FeatureBenchGenerator._run_trace_inventory(  # type: ignore[arg-type]
            first,
            concurrency=5,
            trace_file=trace_file,
            semaphore=semaphore,
        ),
        FeatureBenchGenerator._run_trace_inventory(  # type: ignore[arg-type]
            second,
            concurrency=5,
            trace_file=trace_file,
            semaphore=semaphore,
        ),
    )

    assert first_results == first
    assert second_results == second
    assert maximum_active == 3
    assert active == 0


@pytest.mark.asyncio
async def test_environment_compatibility_uses_the_trace_launch_path_without_running_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    traced_commands: list[tuple[str, ...]] = []

    class ProbeSession:
        async def exec(self, command: tuple[str, ...], **options: object) -> object:
            del options
            commands.append(command)
            return SimpleNamespace(
                exit_code=0,
                stdout="9.0.1\n",
                combined_output="9.0.1\n",
            )

    async def trace(
        self: PythonTracer,
        *,
        test_command: tuple[str, ...],
        test_file: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> TraceBundle:
        del self, timeout_seconds
        traced_commands.append(test_command)
        return TraceBundle(test_file=test_file, records=(), passed_test_ids=())

    monkeypatch.setattr(PythonTracer, "trace", trace)
    generator = FeatureBenchGenerator()
    result = await generator._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ProbeSession()),
        test_file="tests/test_small.py",
        test_ids=("tests/test_small.py::test_feature",),
    )

    assert result.status == "passed"
    assert result.pytest_version == "9.0.1"
    assert commands[0][:2] == ("python", "-c")
    assert traced_commands == [("pytest", "-q", "--collect-only")]


@pytest.mark.asyncio
async def test_environment_compatibility_rejects_missing_pytest_before_trace() -> None:
    class ProbeSession:
        async def exec(self, command: tuple[str, ...], **options: object) -> object:
            del command, options
            return SimpleNamespace(
                exit_code=1,
                stdout="",
                combined_output="ModuleNotFoundError: No module named 'pytest'",
            )

    generator = FeatureBenchGenerator()
    with pytest.raises(CandidateRejected, match="cannot import pytest"):
        await generator._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
            session=cast(ContainerSessionProtocol, ProbeSession()),
            test_file="tests/test_small.py",
            test_ids=("tests/test_small.py::test_feature",),
        )


@pytest.mark.asyncio
async def test_environment_compatibility_import_timeout_uses_conservative_capacity() -> None:
    class ProbeSession:
        async def exec(self, command: tuple[str, ...], **options: object) -> object:
            del command, options
            raise CommandTimeoutError("activated pytest import timed out", timeout_seconds=120)

    result = await FeatureBenchGenerator()._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ProbeSession()),
        test_file="tests/test_small.py",
        test_ids=("tests/test_small.py::test_feature",),
    )

    assert result.status == "inconclusive"
    assert result.runner_probe_status == "not_run_compatibility_timeout"
    assert result.estimated_test_capacity == 0


@pytest.mark.asyncio
async def test_environment_compatibility_treats_collection_failures_as_local_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProbeSession:
        async def exec(self, command: tuple[str, ...], **options: object) -> object:
            del command, options
            return SimpleNamespace(
                exit_code=0,
                stdout="9.0.1\n",
                combined_output="9.0.1\n",
            )

    async def local_collection_failure(self: PythonTracer, **options: object) -> TraceBundle:
        del self, options
        raise ContractError("test module imports an unavailable optional dependency")

    generator = FeatureBenchGenerator()
    monkeypatch.setattr(PythonTracer, "trace", local_collection_failure)
    result = await generator._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ProbeSession()),
        test_file="tests/test_small.py",
        test_ids=("tests/test_small.py::test_feature",),
    )
    assert result.status == "inconclusive"
    assert result.estimated_test_capacity is None

    async def coverage_failure(self: PythonTracer, **options: object) -> TraceBundle:
        del self, options
        raise ContractError("FAIL Required test coverage of 90% not reached")

    monkeypatch.setattr(PythonTracer, "trace", coverage_failure)
    result = await generator._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ProbeSession()),
        test_file="tests/test_small.py",
        test_ids=("tests/test_small.py::test_feature",),
    )
    assert result.status == "inconclusive"


@pytest.mark.asyncio
async def test_environment_compatibility_import_and_collection_share_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_trace_timeouts: list[float] = []

    class ProbeSession:
        async def exec(self, command: tuple[str, ...], **options: object) -> object:
            del command, options
            await asyncio.sleep(0.01)
            return SimpleNamespace(
                exit_code=0,
                stdout="9.0.1\n",
                combined_output="9.0.1\n",
            )

    async def stalled_collection(
        self: PythonTracer,
        *,
        timeout_seconds: float,
        **options: object,
    ) -> TraceBundle:
        del self, options
        observed_trace_timeouts.append(timeout_seconds)
        await asyncio.sleep(1)
        raise AssertionError("the shared compatibility deadline must cancel collection")

    monkeypatch.setattr(PythonTracer, "trace", stalled_collection)
    generator = FeatureBenchGenerator(FeatureBenchSettings(compatibility_timeout_seconds=0.05))
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await generator._check_environment_compatibility(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ProbeSession()),
        test_file="tests/test_small.py",
        test_ids=("tests/test_small.py::test_feature",),
    )

    assert result.status == "inconclusive"
    assert loop.time() - started < 0.5
    assert result.runner_probe_status == "not_run_compatibility_timeout"
    assert result.estimated_test_capacity == 0
    assert len(observed_trace_timeouts) == 1
    assert 0 < observed_trace_timeouts[0] < 0.05


def test_runner_capacity_probe_selection_is_deterministic_and_file_stratified() -> None:
    test_ids_by_file = {
        f"tests/test_{file_index:02d}.py": tuple(
            f"tests/test_{file_index:02d}.py::test_{point_index:02d}" for point_index in range(5)
        )
        for file_index in range(4)
    }

    selected = FeatureBenchGenerator._runner_probe_test_ids(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file,
        maximum_test_points=8,
        maximum_test_files=3,
    )

    assert selected == (
        "tests/test_00.py::test_00",
        "tests/test_01.py::test_00",
        "tests/test_02.py::test_00",
        "tests/test_00.py::test_01",
        "tests/test_01.py::test_01",
        "tests/test_02.py::test_01",
        "tests/test_00.py::test_02",
        "tests/test_01.py::test_02",
    )


def test_effective_p2p_limit_subtracts_seed_points_from_total_capacity() -> None:
    generator = FeatureBenchGenerator()

    assert (
        generator._effective_p2p_limit(  # pyright: ignore[reportPrivateUsage]
            seed_test_points=200,
            test_capacity=1400,
        )
        == 1200
    )
    assert (
        generator._effective_p2p_limit(  # pyright: ignore[reportPrivateUsage]
            seed_test_points=200,
            test_capacity=0,
        )
        == 10
    )
    assert (
        generator._effective_p2p_limit(  # pyright: ignore[reportPrivateUsage]
            seed_test_points=200,
            test_capacity=950,
        )
        == 750
    )


@pytest.mark.asyncio
async def test_stale_plan_is_replanned_against_the_latest_total_capacity() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    p2p_ids = tuple(f"tests/test_other.py::test_{index}" for index in range(20))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    limits: list[int] = []

    async def replan(maximum_p2p: int) -> MaskPlan:
        limits.append(maximum_p2p)
        return plan.model_copy(update={"p2p_test_ids": p2p_ids[:maximum_p2p]})

    (
        refreshed,
        previous_p2p,
        maximum_p2p,
    ) = await FeatureBenchGenerator()._refresh_plan_for_test_capacity(  # pyright: ignore[reportPrivateUsage]
        plan=plan,
        seed_test_points=200,
        test_capacity=210,
        replan_with_p2p_limit=replan,
    )

    assert limits == [10]
    assert previous_p2p == 20
    assert maximum_p2p == 10
    assert refreshed.p2p_test_ids == p2p_ids[:10]

    (
        unchanged,
        previous_p2p,
        maximum_p2p,
    ) = await FeatureBenchGenerator()._refresh_plan_for_test_capacity(  # pyright: ignore[reportPrivateUsage]
        plan=plan,
        seed_test_points=200,
        test_capacity=500,
        replan_with_p2p_limit=replan,
    )
    assert unchanged is plan
    assert previous_p2p is None
    assert maximum_p2p is None
    assert limits == [10]


@pytest.mark.asyncio
async def test_runner_capacity_probe_uses_two_bounded_stages() -> None:
    test_ids_by_file = {
        f"tests/test_{file_index:02d}.py": tuple(
            f"tests/test_{file_index:02d}.py::test_{point_index:02d}" for point_index in range(20)
        )
        for file_index in range(64)
    }
    calls: list[tuple[tuple[str, ...], float]] = []

    async def run_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        calls.append((test_ids, timeout_seconds))
        return RunnerResponse(
            action="run",
            exit_code=0,
            results=tuple(
                RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
            ),
        )

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file=test_ids_by_file,
        run_batch=run_batch,
        timeout_seconds=120,
    )

    assert result.status == "passed"
    assert result.estimated_test_capacity == 1280
    assert result.test_points == 128
    assert result.test_files == 64
    assert [len(test_ids) for test_ids, _ in calls] == [32, 128]
    assert calls[0][1] <= 30
    assert calls[1][1] <= 90


@pytest.mark.asyncio
async def test_runner_capacity_probe_runner_timeout_uses_the_minimum_p2p_signal() -> None:
    test_ids_by_file = {
        "tests/test_slow.py": tuple(
            f"tests/test_slow.py::test_{point_index}" for point_index in range(32)
        )
    }
    calls = 0

    async def timed_out_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        nonlocal calls
        del test_ids
        calls += 1
        raise RunnerTimeoutError(
            "Runner run timed out",
            timeout_seconds=timeout_seconds,
        )

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file=test_ids_by_file,
        run_batch=timed_out_batch,
        timeout_seconds=120,
    )

    assert result.status == "timed_out"
    assert result.estimated_test_capacity == 0
    assert result.test_points == 32
    assert calls == 1


@pytest.mark.asyncio
async def test_runner_capacity_probe_deadline_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator_module, "_P2P_PROBE_INITIAL_TIMEOUT_SECONDS", 0.01)
    test_ids_by_file = {
        "tests/test_slow.py": tuple(
            f"tests/test_slow.py::test_{point_index}" for point_index in range(32)
        )
    }
    calls = 0

    async def stalled_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        nonlocal calls
        del test_ids
        calls += 1
        await asyncio.sleep(timeout_seconds)
        raise TimeoutError("Runner probe setup exhausted its stage deadline")

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file=test_ids_by_file,
        run_batch=stalled_batch,
        timeout_seconds=1,
    )

    assert result.status == "inconclusive"
    assert result.estimated_test_capacity is None
    assert result.test_points == 32
    assert calls == 1


@pytest.mark.asyncio
async def test_runner_capacity_probe_does_not_treat_infrastructure_timeout_as_throughput() -> None:
    async def infrastructure_timeout(
        test_ids: tuple[str, ...], timeout_seconds: float
    ) -> RunnerResponse:
        del test_ids, timeout_seconds
        raise CommandTimeoutError("docker exec setup timed out", timeout_seconds=10)

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file={
            "tests/test_api.py": tuple(f"tests/test_api.py::test_{index}" for index in range(32))
        },
        run_batch=infrastructure_timeout,
        timeout_seconds=120,
    )

    assert result.status == "inconclusive"
    assert result.estimated_test_capacity is None


@pytest.mark.asyncio
async def test_runner_capacity_probe_does_not_extrapolate_a_timed_out_expanded_sample() -> None:
    test_ids_by_file = {
        f"tests/test_{file_index:02d}.py": tuple(
            f"tests/test_{file_index:02d}.py::test_{point_index:02d}" for point_index in range(20)
        )
        for file_index in range(64)
    }
    calls = 0

    async def run_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        nonlocal calls
        del timeout_seconds
        calls += 1
        if calls == 2:
            raise RunnerTimeoutError("expanded Runner sample timed out", timeout_seconds=90)
        return RunnerResponse(
            action="run",
            exit_code=0,
            results=tuple(
                RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
            ),
        )

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file=test_ids_by_file,
        run_batch=run_batch,
        timeout_seconds=1,
    )

    assert result.status == "partial_timeout"
    assert result.estimated_test_capacity == 128
    assert result.test_points == 128
    assert calls == 2


@pytest.mark.asyncio
async def test_runner_capacity_probe_expanded_deadline_keeps_the_initial_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator_module, "_P2P_PROBE_TIMEOUT_SECONDS", 0.01)
    test_ids_by_file = {
        f"tests/test_{file_index:02d}.py": tuple(
            f"tests/test_{file_index:02d}.py::test_{point_index:02d}" for point_index in range(20)
        )
        for file_index in range(64)
    }
    calls = 0

    async def run_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.sleep(timeout_seconds)
            raise TimeoutError("Runner probe setup exhausted its stage deadline")
        return RunnerResponse(
            action="run",
            exit_code=0,
            results=tuple(
                RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
            ),
        )

    result = await FeatureBenchGenerator()._probe_runner_capacity(  # pyright: ignore[reportPrivateUsage]
        test_ids_by_file=test_ids_by_file,
        run_batch=run_batch,
        timeout_seconds=1,
    )

    assert result.status == "partial"
    assert result.estimated_test_capacity == 1280
    assert result.test_points == 32
    assert calls == 2


@pytest.mark.asyncio
async def test_seed_pipeline_releases_index_session_and_attributes_unexpected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_files = ("tests/test_a.py", "tests/test_b.py")
    trace_rejected_file = "tests/test_trace_rejected.py"
    test_files = (*pipeline_files, trace_rejected_file)
    test_points = tuple(
        CollectedTestPoint(id=f"{test_file}::test_feature", path=test_file)
        for test_file in test_files
    )
    environment = _environment().model_copy(update={"verified_test_points": test_points})
    repository_index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": "def feature():\n    return 1\n",
            "tests/test_a.py": "def test_feature():\n    assert True\n",
            "tests/test_b.py": "def test_feature():\n    assert True\n",
            trace_rejected_file: "def test_feature():\n    assert True\n",
        }
    )
    runtime = FakeDisposableRuntime()

    async def build_index(
        cls: type[RepositoryIndex],
        session: ContainerSessionProtocol,
        **options: object,
    ) -> RepositoryIndex:
        del cls, options
        assert cast(FakeSession, session).clean_count == 1
        return repository_index

    async def resolve_test_sources(
        *,
        session: ContainerSessionProtocol,
        runner_paths: tuple[str, ...],
        indexed_paths: frozenset[str],
        timeout_seconds: float,
    ) -> dict[str, str]:
        del session, indexed_paths, timeout_seconds
        return {path: path for path in runner_paths}

    async def trace_file(
        *,
        test_file: str,
        test_ids: tuple[str, ...],
        **options: object,
    ) -> object:
        del options
        if test_file == trace_rejected_file:
            return generator_module._TraceFileResult(  # pyright: ignore[reportPrivateUsage]
                test_file=test_file,
                requested_test_ids=test_ids,
                successful_test_ids=(),
                failed_test_ids=tuple((test_id, "trace rejected") for test_id in test_ids),
                bundle=None,
                batch_error="trace rejected",
                statistics=(),
            )
        return generator_module._TraceFileResult(  # pyright: ignore[reportPrivateUsage]
            test_file=test_file,
            requested_test_ids=test_ids,
            successful_test_ids=test_ids,
            failed_test_ids=(),
            bundle=TraceBundle(
                test_file=test_file,
                records=(
                    TraceRecord(
                        path="pkg/core.py",
                        qualname="feature",
                        line=1,
                        calls=1,
                        test_ids=test_ids,
                    ),
                ),
                passed_test_ids=test_ids,
            ),
            batch_error=None,
            statistics=(),
        )

    both_seeds_started = asyncio.Event()
    started_seeds: set[str] = set()

    async def select_top(*, trace: TraceBundle, **options: object) -> tuple[()]:
        assert options["candidate_count"] == 1
        assert options["protected_object_ids"] == frozenset()
        assert runtime.sessions[0] in runtime.closed
        started_seeds.add(trace.test_file)
        if len(started_seeds) == len(pipeline_files):
            both_seeds_started.set()
        await asyncio.wait_for(both_seeds_started.wait(), timeout=1)
        if trace.test_file == pipeline_files[0]:
            raise RuntimeError("unexpected Top failure")
        await asyncio.Event().wait()
        return ()

    monkeypatch.setattr(RepositoryIndex, "build", classmethod(build_index))
    generator = FeatureBenchGenerator(FeatureBenchSettings(seed_concurrency=2))

    async def check_compatibility(**options: object) -> object:
        del options
        return generator_module._EnvironmentCompatibilityResult(  # pyright: ignore[reportPrivateUsage]
            status="passed",
            pytest_version="test",
            smoke_test_file=test_files[0],
        )

    def profile_seed(*, trace: TraceBundle, **options: object) -> SeedFeasibilityProfile:
        del options
        return SeedFeasibilityProfile(
            test_file=trace.test_file,
            top_candidate_count=1,
            estimated_top_batches=1,
            optimistic_removed_lines=200,
            available_p2p_test_points=10,
            minimum_p2p_test_points=10,
            estimated_validation_files=1,
            estimated_validation_points=11,
            preferred_code_scale=True,
            capped_line_capacity=200,
        )

    monkeypatch.setattr(generator, "_resolve_test_source_paths", resolve_test_sources)
    monkeypatch.setattr(generator, "_check_environment_compatibility", check_compatibility)
    monkeypatch.setattr(generator, "_trace_file", trace_file)
    monkeypatch.setattr(generator.planner, "seed_feasibility_profile", profile_seed)
    monkeypatch.setattr(generator.planner, "select_top", select_top)
    progress = AdapterProgressReporter(
        events=NullEventSink(),
        run_id="generate-run",
        adapter="featurebench",
        adapter_name="FeatureBench",
        phases=(
            AdapterPhase(id="inspect_repository", label="Indexing", unit="repository"),
            AdapterPhase(id="analyze_tests", label="Tracing", unit="test files"),
            AdapterPhase(id="create_tasks", label="Generating", unit="seeds"),
        ),
        environment_id=environment.environment_id,
    )
    context = GeneratorContext(
        runtime=cast(Any, runtime),
        events=NullEventSink(),
        progress=progress,
        work_dir=tmp_path / "work",
        run_id="generate-run",
        output_dir=tmp_path,
        selection=WorkSelection("unseen", frozenset((WorkState.UNSEEN,))),
    )
    await generator.run_started(environment, context)

    with pytest.raises(RuntimeError, match="unexpected Top failure") as captured:
        async for _ in generator.generate(environment, context):
            pytest.fail("the failing Seed pipeline must not yield a candidate")
    await generator.run_finished(context, captured.value)

    assert generator.diagnostics is not None
    failed_seed = generator.diagnostics.seed(test_files[0]).summary
    interrupted_seed = generator.diagnostics.seed(test_files[1]).summary
    assert failed_seed.status == "failed"
    assert failed_seed.phase == "top_selection"
    assert failed_seed.error_type == "RuntimeError"
    assert failed_seed.error == "unexpected Top failure"
    assert interrupted_seed.status == "interrupted"
    assert interrupted_seed.phase == "top_selection"
    rejected_seed = generator.diagnostics.seed(trace_rejected_file).summary
    assert rejected_seed.status == "rejected"
    assert rejected_seed.phase == "trace"
    assert progress._snapshot.candidates_rejected == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime.closed == runtime.sessions


class PythonTracerHarness(PythonTracer):
    @classmethod
    def pytest_invocation(
        cls,
        test_command: tuple[str, ...],
        test_file: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return cls._pytest_invocation(test_command, test_file)


def _environment(
    *,
    gpu_requirement: str = "none",
    gpus: str | None = None,
) -> EnvironmentRecord:
    return EnvironmentRecord.model_validate(
        {
            "environment_id": "environment-test",
            "repository": "https://github.com/example/project.git",
            "commit": "a" * 40,
            "language": "python",
            "gpu_requirement": gpu_requirement,
            "image": "example/project@sha256:" + "b" * 64,
            "dockerfile": "FROM python:3.12\nWORKDIR /testbed\nRUN true\n",
            "runtime": RuntimeSpec(
                activation=("#!/usr/bin/env bash\n# Test activation artifact.\ncd /testbed\n"),
                runner=(
                    "#!/usr/bin/env bash\n"
                    "# Test Runner artifact long enough for the strict runtime contract.\n"
                    "# It is not executed by these compatibility tests.\n"
                    "exit 0\n"
                ),
                resources=RuntimeResources(gpus=gpus),
            ),
            "verified_test_points": (
                CollectedTestPoint(
                    id="tests/test_core.py::test_feature", path="tests/test_core.py"
                ),
            ),
        }
    )


def _planner() -> FeatureSlicePlanner:
    return FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=1,
        maximum_code_lines=5000,
        random_seed=0,
        critical_magic_methods=frozenset({"__init__"}),
    )


def test_index_matches_decorated_runtime_lines_and_resolves_imports() -> None:
    sources = {
        "src/pkg/core.py": '''def deco(fn):
    return fn

@deco
def feature(value: int) -> int:
    """Compute the complete feature."""
    return value + 1
''',
        "src/pkg/api.py": "from pkg.core import feature as build\n",
    }
    index = RepositoryIndex.from_sources(sources)
    record = TraceRecord(path="src/pkg/core.py", qualname="feature", line=4, calls=2)
    matched = index.resolve_trace(record)
    assert matched is not None
    assert matched.line == 5
    assert matched.start_line == 4
    binding = next(item for item in index.data.imports if item.importer_path.endswith("api.py"))
    assert binding.resolved_path == "src/pkg/core.py"


def test_index_protects_definition_time_references_but_not_function_body_calls() -> None:
    sources = {
        "pkg/core.py": """def registered():
    return 1

REGISTRY = {"registered": registered}

def helper():
    return 2

def ordinary():
    return helper()

class Service:
    @property
    def value(self):
        return 1

    @value.setter
    def value(self, new_value):
        pass
"""
    }
    index = RepositoryIndex.from_sources(sources)
    registered = index.by_qualname[("pkg/core.py", "registered")]
    helper = index.by_qualname[("pkg/core.py", "helper")]
    value_bindings = tuple(
        item
        for item in index.data.definitions
        if item.path == "pkg/core.py" and item.qualname == "Service.value"
    )
    references = {
        (item.referenced_object_id, item.owner_object_id)
        for item in index.data.definition_time_references
    }

    assert (registered.object_id, None) in references
    assert registered.object_id in index.protected_ids(frozenset())
    assert all(
        item.referenced_object_id != helper.object_id
        for item in index.data.definition_time_references
    )
    assert len(value_bindings) == 2
    setter = value_bindings[-1]
    assert (value_bindings[0].object_id, setter.object_id) in references
    assert index.definition_time_blockers((value_bindings[0].object_id,)) == {
        value_bindings[0].object_id: (setter.object_id,)
    }
    assert not index.definition_time_blockers(item.object_id for item in value_bindings)


def test_index_protects_every_python_data_model_method() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": (
                "class Value:\n"
                "    def __eq__(self, other):\n"
                "        return isinstance(other, Value)\n\n"
                "    def ordinary(self):\n"
                "        return 1\n"
            )
        }
    )

    protected = index.protected_ids(frozenset())

    assert index.by_qualname[("pkg/core.py", "Value.__eq__")].object_id in protected
    assert index.by_qualname[("pkg/core.py", "Value.ordinary")].object_id not in protected
    assert index.protected_ids(frozenset()) is protected


def test_init_and_exported_interfaces_are_evidence_protected_not_blanket_protected() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/__init__.py": ('__all__ = ["feature"]\n\ndef feature():\n    return 1\n'),
            "tests/test_seed.py": (
                "import pkg\n\ndef test_feature():\n    assert pkg.feature() == 1\n"
            ),
            "tests/test_star.py": (
                "from pkg import *\n\ndef test_feature():\n    assert feature() == 1\n"
            ),
        }
    )
    feature = index.by_qualname[("pkg/__init__.py", "feature")]
    seed_test = index.by_qualname[("tests/test_seed.py", "test_feature")]
    seed_id = "tests/test_seed.py::test_feature"
    trace = TraceBundle(
        test_file="tests/test_seed.py",
        records=(
            TraceRecord(
                path=seed_test.path,
                qualname=seed_test.qualname,
                line=seed_test.line,
                calls=1,
                test_ids=(seed_id,),
            ),
            TraceRecord(
                path=feature.path,
                qualname=feature.qualname,
                line=feature.line,
                calls=1,
                test_ids=(seed_id,),
            ),
        ),
        edges=(TraceEdge(caller=seed_test.object_id, callee=feature.object_id, calls=1),),
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=1,
        maximum_code_lines=100,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    assert feature.object_id not in index.protected_ids(frozenset())
    assert feature.object_id in {
        item.object.object_id for item in planner.top_candidates(index, trace)
    }
    assert seed_import_object_ids(
        index=index,
        test_file="tests/test_star.py",
    ) == frozenset({feature.object_id})
    assert trace_import_object_ids(
        index=index,
        dynamic_imports=(
            TraceImport(
                importer_path="pkg/bootstrap.py",
                importer_line=1,
                imported_name="feature",
                defined_path=feature.path,
                defined_name=feature.qualname,
                defined_line=feature.line,
            ),
        ),
    ) == frozenset({feature.object_id})


def test_index_models_implementations_of_abstract_contracts_as_conditional_impacts() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": (
                "from abc import ABC, abstractmethod\n\n"
                "class Contract(ABC):\n"
                "    @abstractmethod\n"
                "    def render(self):\n"
                "        raise NotImplementedError\n\n"
                "class Concrete(Contract):\n"
                "    def render(self):\n"
                "        return 'value'\n\n"
                "    def ordinary(self):\n"
                "        return 1\n"
            )
        }
    )

    protected = index.protected_ids(frozenset())
    abstract_method = index.by_qualname[("pkg/core.py", "Contract.render")]
    concrete_method = index.by_qualname[("pkg/core.py", "Concrete.render")]
    ordinary = index.by_qualname[("pkg/core.py", "Concrete.ordinary")]
    concrete_class = index.by_qualname[("pkg/core.py", "Concrete")]

    assert concrete_method.object_id not in protected
    assert index.structural_test_impact_ids((concrete_method.object_id,)) == frozenset(
        {concrete_class.object_id, concrete_method.object_id, ordinary.object_id}
    )
    assert index.structural_test_impact_ids((abstract_method.object_id,)) == frozenset(
        {abstract_method.object_id}
    )


def test_definition_time_dependency_graph_handles_python_declaration_edges() -> None:
    sources = {
        "pkg/core.py": """def decorator(fn):
    return fn

@decorator
def decorated():
    return 1

def factory():
    return 1

def defaulted(value=factory()):
    return value

class Base:
    pass

class Derived(Base):
    pass

class Service:
    def feature(self):
        return 1

    REGISTRY = {"feature": feature}

def registered():
    return 1

REGISTRY = {"registered": registered}
"""
    }
    index = RepositoryIndex.from_sources(sources)

    for referenced_name, owner_name in (
        ("decorator", "decorated"),
        ("factory", "defaulted"),
        ("Base", "Derived"),
        ("Service.feature", "Service"),
    ):
        referenced = index.by_qualname[("pkg/core.py", referenced_name)]
        owner = index.by_qualname[("pkg/core.py", owner_name)]
        assert index.definition_time_blockers((referenced.object_id,)) == {
            referenced.object_id: (owner.object_id,)
        }
        assert not index.definition_time_blockers((referenced.object_id, owner.object_id))

    registered = index.by_qualname[("pkg/core.py", "registered")]
    assert index.definition_time_blockers((registered.object_id,)) == {
        registered.object_id: (None,)
    }


def test_definition_time_dependency_graph_handles_singledispatch_registration() -> None:
    sources = {
        "pkg/core.py": """from functools import singledispatch

@singledispatch
def normalize(value):
    return value

@normalize.register
def _(value: str):
    return value.lower()
"""
    }
    index = RepositoryIndex.from_sources(sources)
    normalize = index.by_qualname[("pkg/core.py", "normalize")]
    registration = index.by_qualname[("pkg/core.py", "_")]

    assert index.definition_time_blockers((normalize.object_id,)) == {
        normalize.object_id: (registration.object_id,)
    }
    assert not index.definition_time_blockers((normalize.object_id, registration.object_id))


def test_mask_planner_closes_over_declaration_dependency_owners() -> None:
    sources = {
        "pkg/core.py": """def dependency(fn=None):
    if fn is None:
        return 1
    return fn

@dependency
def owner():
    return 2

def entry():
    return dependency() + owner()
""",
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    entry = index.by_qualname[("pkg/core.py", "entry")]
    dependency = index.by_qualname[("pkg/core.py", "dependency")]
    owner = index.by_qualname[("pkg/core.py", "owner")]
    seed_id = "tests/test_seed.py::test_seed"
    seed = TraceBundle(
        test_file="tests/test_seed.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=2 if item is dependency else 1,
                test_ids=(seed_id,),
            )
            for item in (test, entry, dependency, owner)
        ),
        edges=(
            TraceEdge(caller=test.object_id, callee=entry.object_id, calls=1),
            TraceEdge(caller=entry.object_id, callee=dependency.object_id, calls=2),
            TraceEdge(caller=entry.object_id, callee=owner.object_id, calls=1),
        ),
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=100,
        maximum_code_lines=100,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces={seed.test_file: seed},
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=(entry,),
        p2p=P2PSelection(test_ids=(), top_object_ids=(entry.object_id,)),
    )

    assert {item.object_id for item in plan.specific_objects} == {
        dependency.object_id,
        owner.object_id,
    }
    transformed = transform_sources(sources, plan)
    compile(transformed["pkg/core.py"], "pkg/core.py", "exec")


def test_joint_mask_precheck_rejects_an_unreachable_line_minimum_before_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "pkg/core.py": "def feature():\n    return 1\n",
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    feature = index.by_qualname[("pkg/core.py", "feature")]
    seed_test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    other_test = index.by_qualname[("tests/test_other.py", "test_other")]
    seed_id = "tests/test_seed.py::test_seed#0"
    other_id = "tests/test_other.py::test_other#0"
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
            for item in (seed_test, feature)
        ),
        edges=(TraceEdge(caller=seed_test.object_id, callee=feature.object_id, calls=1),),
        passed_test_ids=(seed_id,),
    )
    other = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=other_test.path,
                qualname=other_test.qualname,
                line=other_test.line,
                calls=1,
                test_ids=(other_id,),
            ),
        ),
        passed_test_ids=(other_id,),
    )
    traces = {item.test_file: item for item in (seed, other)}
    point_index = TracePointIndex.build(
        index=index,
        traces=traces.values(),
        candidate_test_ids=(seed_id, other_id),
        test_paths={seed_id: seed.test_file, other_id: other.test_file},
    )

    def unexpected_solver(**_: object) -> None:
        pytest.fail("the CP-SAT solver must not run when the line upper bound is too small")

    monkeypatch.setattr(
        "codeworld.adapters.featurebench.masking.optimize_deletions",
        unexpected_solver,
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=100,
        maximum_code_lines=200,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    with pytest.raises(MaskIntervalInfeasible) as caught:
        planner.create_plan(
            index=index,
            seed_trace=seed,
            all_traces=traces,
            collection_trace=TraceBundle(test_file="<collection>", records=()),
            top_objects=(feature,),
            p2p=P2PSelection(test_ids=(other_id,), top_object_ids=(feature.object_id,)),
            joint_optimization=JointOptimizationOptions(
                point_index=point_index,
                minimum_p2p_test_points=1,
                maximum_p2p_test_points=None,
                maximum_selected_p2p_test_files=None,
                minimum_f2p_test_points=1,
                maximum_f2p_test_points=None,
                maximum_f2p_ratio=0.7,
                maximum_seed_pass_rate=0.9,
                timeout_seconds=5,
                workers=1,
                minimum_code_lines=100,
                maximum_code_lines=200,
            ),
        )

    diagnostics = caught.value.diagnostics[0]
    assert diagnostics.status == "PRECHECK_INFEASIBLE"
    assert diagnostics.candidate_removed_lines_upper_bound == feature.removed_lines
    assert diagnostics.minimum_removed_lines == 100
    assert diagnostics.sufficient_constraint_core == ("gold_patch_minimum_upper_bound",)


def test_seed_feasibility_uses_sound_line_and_p2p_upper_bounds() -> None:
    large_body = "\n".join(f"    value_{index} = {index}" for index in range(120))
    sources = {
        "pkg/core.py": f"def feature():\n{large_body}\n    return value_119\n",
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    feature = index.by_qualname[("pkg/core.py", "feature")]
    seed_test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    other_test = index.by_qualname[("tests/test_other.py", "test_other")]
    seed_id = "tests/test_seed.py::test_seed#0"
    other_id = "tests/test_other.py::test_other#0"
    seed_trace = TraceBundle(
        test_file="tests/test_seed.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(seed_id,),
            )
            for item in (seed_test, feature)
        ),
        edges=(
            TraceEdge(
                caller=seed_test.object_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(seed_id,),
            ),
        ),
        passed_test_ids=(seed_id,),
    )
    other_trace = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=other_test.path,
                qualname=other_test.qualname,
                line=other_test.line,
                calls=1,
                test_ids=(other_id,),
            ),
        ),
        passed_test_ids=(other_id,),
    )
    point_index = TracePointIndex.build(
        index=index,
        traces=(seed_trace, other_trace),
        candidate_test_ids=(seed_id, other_id),
        test_paths={seed_id: seed_trace.test_file, other_id: other_trace.test_file},
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=10,
        minimum_code_lines=1,
        maximum_code_lines=300,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )
    collection = TraceBundle(test_file="<collection>", records=())

    feasible = planner.seed_feasibility_profile(
        index=index,
        trace=seed_trace,
        collection_trace=collection,
        test_ids=(seed_id,),
        point_index=point_index,
        minimum_code_lines=100,
        preferred_minimum_code_lines=200,
        maximum_code_lines=300,
        minimum_p2p_test_points=1,
        maximum_p2p_test_points=None,
        maximum_selected_p2p_test_files=None,
    )
    assert not feasible.rejected
    assert feasible.top_candidate_count == 1
    assert 100 <= feasible.optimistic_removed_lines < 200
    assert not feasible.preferred_code_scale
    assert feasible.available_p2p_test_points == 1
    assert feasible.estimated_validation_files == 1

    infeasible = planner.seed_feasibility_profile(
        index=index,
        trace=seed_trace,
        collection_trace=collection,
        test_ids=(seed_id,),
        point_index=point_index,
        minimum_code_lines=200,
        preferred_minimum_code_lines=250,
        maximum_code_lines=300,
        minimum_p2p_test_points=2,
        maximum_p2p_test_points=None,
        maximum_selected_p2p_test_files=None,
    )
    assert infeasible.rejected
    assert any("line capacity" in reason for reason in infeasible.rejection_reasons)
    assert any("P2P" in reason for reason in infeasible.rejection_reasons)


def test_seed_ordering_is_stable_easy_first_and_has_a_path_fallback() -> None:
    def profile(
        path: str,
        *,
        preferred: bool,
        top_batches: int,
        rejected: bool = False,
    ) -> SeedFeasibilityProfile:
        return SeedFeasibilityProfile(
            test_file=path,
            top_candidate_count=top_batches * 10,
            estimated_top_batches=top_batches,
            optimistic_removed_lines=800 if preferred else 300,
            available_p2p_test_points=20,
            minimum_p2p_test_points=10,
            estimated_validation_files=1,
            estimated_validation_points=11,
            preferred_code_scale=preferred,
            capped_line_capacity=800 if preferred else 300,
            rejection_reasons=("proven infeasible",) if rejected else (),
        )

    profiles = (
        profile("tests/test_z_easy.py", preferred=True, top_batches=1),
        profile("tests/test_a_expensive.py", preferred=True, top_batches=3),
        profile("tests/test_b_fallback.py", preferred=False, top_batches=1),
        profile("tests/test_aa_rejected.py", preferred=True, top_batches=1, rejected=True),
    )

    easy_first = FeatureBenchGenerator._order_seed_profiles(  # pyright: ignore[reportPrivateUsage]
        profiles,
        "easy_first",
    )
    by_path = FeatureBenchGenerator._order_seed_profiles(  # pyright: ignore[reportPrivateUsage]
        profiles,
        "path",
    )
    assert tuple(item.test_file for item in easy_first) == (
        "tests/test_z_easy.py",
        "tests/test_a_expensive.py",
        "tests/test_b_fallback.py",
    )
    assert tuple(item.test_file for item in by_path) == (
        "tests/test_a_expensive.py",
        "tests/test_b_fallback.py",
        "tests/test_z_easy.py",
    )
    assert easy_first[0].as_dict()["capped_line_capacity"] == 800


def test_mask_planner_keeps_dependency_when_its_owner_is_p2p_protected() -> None:
    sources = {
        "pkg/core.py": """def dependency(fn=None):
    if fn is None:
        return 1
    return fn

@dependency
def owner():
    return 2

def entry():
    return dependency() + owner()
""",
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    other_test = index.by_qualname[("tests/test_other.py", "test_other")]
    entry = index.by_qualname[("pkg/core.py", "entry")]
    dependency = index.by_qualname[("pkg/core.py", "dependency")]
    owner = index.by_qualname[("pkg/core.py", "owner")]
    seed_id = "tests/test_seed.py::test_seed"
    other_id = "tests/test_other.py::test_other"
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
            for item in (test, entry, dependency, owner)
        ),
        edges=(
            TraceEdge(caller=test.object_id, callee=entry.object_id, calls=1),
            TraceEdge(caller=entry.object_id, callee=dependency.object_id, calls=1),
            TraceEdge(caller=entry.object_id, callee=owner.object_id, calls=1),
        ),
    )
    other = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=other_test.path,
                qualname=other_test.qualname,
                line=other_test.line,
                calls=1,
                test_ids=(other_id,),
            ),
            TraceRecord(
                path=owner.path,
                qualname=owner.qualname,
                line=owner.line,
                calls=1,
                test_ids=(other_id,),
            ),
        ),
    )

    plan = _planner().create_plan(
        index=index,
        seed_trace=seed,
        all_traces={seed.test_file: seed, other.test_file: other},
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=(entry,),
        p2p=P2PSelection(test_ids=(other_id,), top_object_ids=(entry.object_id,)),
    )

    assert dependency.object_id not in plan.candidate_object_ids
    assert dependency.object_id not in {item.object_id for item in plan.specific_objects}
    assert owner.object_id in plan.protected_objects


def test_mask_repair_rejects_restoring_owner_while_dependency_stays_deleted() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": """def decorator(fn):
    return fn

@decorator
def owner():
    return 1
"""
        }
    )
    decorator = index.by_qualname[("pkg/core.py", "decorator")].model_copy(
        update={"classification": "top"}
    )
    owner = index.by_qualname[("pkg/core.py", "owner")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(decorator,),
        specific_objects=(owner,),
    )

    with pytest.raises(MaskRepairProposalError) as captured:
        _planner().apply_increment(
            index=index,
            plan=plan,
            proposal=MaskRepairProposal(
                based_on_attempt="attempt-0000",
                must_restore_object_ids=(owner.object_id,),
                rationale="Restore the decorated owner while retaining its decorator deletion.",
            ),
        )

    assert captured.value.violations[0].code == "repair.definition_time_blocker"


def test_full_object_deletion_cleans_imports_and_only_repairs_surviving_suite() -> None:
    sources = {
        "src/pkg/core.py": '''def deco(fn):
    return fn

@deco
def feature(value: int) -> int:
    """Compute the complete feature."""
    return value + 1

def unaffected() -> int:
    return 7

class OnlyTarget:
    def removed_method(self) -> int:
        return 1
''',
        "src/pkg/api.py": """from pkg.core import feature as build, unaffected

def lazy_import():
    from pkg.core import feature as lazy_feature
""",
    }
    index = RepositoryIndex.from_sources(sources)
    feature = index.by_qualname[("src/pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    method = index.by_qualname[("src/pkg/core.py", "OnlyTarget.removed_method")].model_copy(
        update={"classification": "specific"}
    )
    edits = _planner().import_edits(index, (feature, method))
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(feature,),
        specific_objects=(method,),
        import_edits=edits,
    )
    transformed = transform_sources(sources, plan)
    core = transformed["src/pkg/core.py"]
    assert "def feature" not in core
    assert "@deco" not in core
    assert "def removed_method" not in core
    assert "class OnlyTarget:\n    pass" in core
    assert "def unaffected" in core
    assert "NotImplementedError" not in core
    assert "# <your code>" not in core
    assert "..." not in core
    assert transformed["src/pkg/api.py"] == (
        "from pkg.core import unaffected\n\ndef lazy_import():\n    pass\n"
    )


def test_hidden_tests_delete_only_f2p_declarations_not_the_test_file() -> None:
    sources = {
        "vendor/pkg/tests/test_api.py": """import pytest

def test_visible():
    assert True

@pytest.mark.parametrize("value", [1, 2])
def test_hidden(value):
    assert value

class TestService:
    def test_hidden_method(self):
        assert True

    def test_visible_method(self):
        assert True
"""
    }
    index = RepositoryIndex.from_sources(sources)
    transformed, paths = hide_test_points(
        sources,
        index=index,
        test_ids=(
            "tests/test_api.py::test_hidden#7",
            "tests/test_api.py::TestService::test_hidden_method#0",
        ),
        test_paths={
            "tests/test_api.py::test_hidden#7": "tests/test_api.py",
            "tests/test_api.py::TestService::test_hidden_method#0": "tests/test_api.py",
        },
        test_source_paths={"tests/test_api.py": "vendor/pkg/tests/test_api.py"},
    )

    assert paths == ("vendor/pkg/tests/test_api.py",)
    hidden = transformed["vendor/pkg/tests/test_api.py"]
    assert "def test_hidden(value)" not in hidden
    assert "def test_hidden_method" not in hidden
    assert "def test_visible()" in hidden
    assert "def test_visible_method" in hidden


def test_hidden_tests_resolve_collected_subclass_ids_to_inherited_methods() -> None:
    sources = {
        "tests/test_api.py": """class CommonTests:
    def test_inherited(self):
        assert True

    def test_stays_visible(self):
        assert True

class TestConcrete(CommonTests):
    def test_direct(self):
        assert True
"""
    }
    index = RepositoryIndex.from_sources(sources)
    transformed, paths = hide_test_points(
        sources,
        index=index,
        test_ids=("tests/test_api.py::TestConcrete::test_inherited#0",),
        test_paths={"tests/test_api.py::TestConcrete::test_inherited#0": "tests/test_api.py"},
        test_source_paths={"tests/test_api.py": "tests/test_api.py"},
    )

    assert paths == ("tests/test_api.py",)
    hidden = transformed["tests/test_api.py"]
    assert "def test_inherited" not in hidden
    assert "def test_stays_visible" in hidden
    assert "def test_direct" in hidden


@pytest.mark.asyncio
async def test_executor_regenerates_every_revision_from_clean_sources() -> None:
    sources = {"pkg/core.py": "def feature():\n    return 1\n\ndef keep():\n    return 2\n"}
    index = RepositoryIndex.from_sources(sources)
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    plan = MaskPlan(seed_test_file="tests/test_core.py", top_objects=(target,))
    fake = FakeSession(sources)
    executor = MaskExecutor(index)
    await executor.apply(cast(ContainerSession, fake), plan)
    assert "def feature" not in fake.files["pkg/core.py"]
    fake.files["pkg/core.py"] = "outside corruption"
    await executor.apply(cast(ContainerSession, fake), plan)
    assert "outside corruption" not in fake.files["pkg/core.py"]


def test_top_specific_and_p2p_follow_dynamic_graph() -> None:
    sources = {
        "pkg/core.py": """def feature():
    return helper()

def helper():
    return 1

def unrelated():
    return 2
""",
        "tests/test_feature.py": "def test_feature():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    seed_id = "tests/test_feature.py::test_feature"
    other_id = "tests/test_other.py::test_other"
    overlapping_other_id = "tests/test_other.py::test_overlapping"
    seed = TraceBundle(
        test_file="tests/test_feature.py",
        records=(
            TraceRecord(
                path="tests/test_feature.py",
                qualname="test_feature",
                line=1,
                calls=1,
                test_ids=(seed_id,),
            ),
            TraceRecord(
                path="pkg/core.py",
                qualname="feature",
                line=1,
                calls=1,
                callers=("tests/test_feature.py::test_feature::1",),
                test_ids=(seed_id,),
            ),
            TraceRecord(
                path="pkg/core.py",
                qualname="helper",
                line=4,
                calls=1,
                callers=("pkg/core.py::feature::1",),
                test_ids=(seed_id,),
            ),
        ),
    )
    other = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path="tests/test_other.py",
                qualname="test_other",
                line=1,
                calls=1,
                test_ids=(other_id,),
            ),
            TraceRecord(
                path="pkg/core.py",
                qualname="unrelated",
                line=7,
                calls=1,
                callers=("tests/test_other.py::test_other::1",),
                test_ids=(other_id,),
            ),
            TraceRecord(
                path="pkg/core.py",
                qualname="feature",
                line=1,
                calls=1,
                callers=("tests/test_other.py::test_overlapping::1",),
                test_ids=(overlapping_other_id,),
            ),
        ),
    )
    planner = _planner()
    candidates = planner.top_candidates(index, seed)
    assert candidates[0].object.qualname == "feature"
    top = (candidates[0].object,)
    point_index = TracePointIndex.build(
        index=index,
        traces=(seed, other),
        candidate_test_ids=(seed_id, other_id, overlapping_other_id),
        test_paths={
            seed_id: seed.test_file,
            other_id: other.test_file,
            overlapping_other_id: other.test_file,
        },
    )
    p2p = point_index.select_p2p(
        seed_test_file=seed.test_file,
        top_object_ids=tuple(item.object_id for item in top),
        minimum_test_points=1,
        maximum_test_points=5,
        random_seed=0,
    )
    assert p2p.test_ids == (other_id,)
    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces={seed.test_file: seed, other.test_file: other},
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=top,
        p2p=P2PSelection(test_ids=p2p.test_ids, top_object_ids=p2p.top_object_ids),
    )
    assert [item.qualname for item in plan.top_objects] == ["feature"]
    assert [item.qualname for item in plan.specific_objects] == ["helper"]


def test_mask_traverses_through_p2p_objects_without_deleting_them() -> None:
    sources = {
        "pkg/core.py": (
            "def feature():\n    return shared()\n\n"
            "def shared():\n    return leaf()\n\n"
            "def leaf():\n    return 1\n"
        ),
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test_seed = index.by_qualname[("tests/test_seed.py", "test_seed")]
    test_other = index.by_qualname[("tests/test_other.py", "test_other")]
    feature = index.by_qualname[("pkg/core.py", "feature")]
    shared = index.by_qualname[("pkg/core.py", "shared")]
    leaf = index.by_qualname[("pkg/core.py", "leaf")]
    seed_id = "tests/test_seed.py::test_seed"
    other_id = "tests/test_other.py::test_other"
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
            for item in (test_seed, feature, shared, leaf)
        ),
        edges=(
            TraceEdge(caller=test_seed.object_id, callee=feature.object_id, calls=1),
            TraceEdge(caller=feature.object_id, callee=shared.object_id, calls=1),
            TraceEdge(caller=shared.object_id, callee=leaf.object_id, calls=1),
        ),
    )
    other = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=test_other.path,
                qualname=test_other.qualname,
                line=test_other.line,
                calls=1,
                test_ids=(other_id,),
            ),
            TraceRecord(
                path=shared.path,
                qualname=shared.qualname,
                line=shared.line,
                calls=1,
                test_ids=(other_id,),
            ),
        ),
    )

    planner = _planner()
    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces={seed.test_file: seed, other.test_file: other},
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=(feature,),
        p2p=P2PSelection(test_ids=(other_id,), top_object_ids=(feature.object_id,)),
    )

    assert shared.object_id in plan.protected_objects
    assert "p2p_trace" in plan.protection_reasons[shared.object_id]
    assert [item.object_id for item in plan.specific_objects] == [leaf.object_id]
    plan = plan.model_copy(
        update={
            "candidate_object_ids": tuple(sorted((*plan.candidate_object_ids, shared.object_id)))
        }
    )
    with pytest.raises(MaskRepairProposalError) as captured:
        planner.apply_increment(
            index=index,
            plan=plan,
            proposal=MaskRepairProposal(
                based_on_attempt="attempt-0000",
                must_mask_object_ids=(shared.object_id,),
                rationale="Try masking an object required by fixed P2P tests.",
            ),
            traces={seed.test_file: seed, other.test_file: other},
        )
    assert captured.value.violations[0].code == "repair.mask_protected"


def test_mask_protects_parent_declaration_of_p2p_method() -> None:
    sources = {
        "pkg/core.py": """class Service:
    def feature(self):
        return 1

    def keep(self):
        return 2
""",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    service = index.by_qualname[("pkg/core.py", "Service")]
    keep = index.by_qualname[("pkg/core.py", "Service.keep")]
    other = index.by_qualname[("tests/test_other.py", "test_other")]
    other_id = "tests/test_other.py::test_other"
    seed_id = "tests/test_seed.py::test_seed"
    safe_id = "tests/test_safe.py::test_safe"
    p2p_trace = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=other.path,
                qualname=other.qualname,
                line=other.line,
                calls=1,
                test_ids=(other_id,),
            ),
            TraceRecord(
                path=keep.path,
                qualname=keep.qualname,
                line=keep.line,
                calls=1,
                test_ids=(other_id,),
            ),
        ),
    )

    with pytest.raises(CandidateRejected, match="structural_ancestor"):
        _planner().create_plan(
            index=index,
            seed_trace=TraceBundle(
                test_file="tests/test_seed.py",
                records=(
                    TraceRecord(
                        path=service.path,
                        qualname=service.qualname,
                        line=service.line,
                        calls=1,
                    ),
                ),
            ),
            all_traces={p2p_trace.test_file: p2p_trace},
            collection_trace=TraceBundle(test_file="<collection>", records=()),
            top_objects=(service,),
            p2p=P2PSelection(test_ids=(other_id,), top_object_ids=(service.object_id,)),
        )

    seed_trace = TraceBundle(
        test_file="tests/test_seed.py",
        records=(
            TraceRecord(
                path=service.path,
                qualname=service.qualname,
                line=service.line,
                calls=1,
                test_ids=(seed_id,),
            ),
        ),
    )
    point_index = TracePointIndex.build(
        index=index,
        traces=(seed_trace, p2p_trace),
        candidate_test_ids=(seed_id, other_id, safe_id),
        test_paths={
            seed_id: seed_trace.test_file,
            other_id: p2p_trace.test_file,
            safe_id: "tests/test_safe.py",
        },
    )
    selection = point_index.select_p2p(
        seed_test_file=seed_trace.test_file,
        top_object_ids=(service.object_id,),
        top_object_impacts=_planner().top_deletion_impacts(index, (service.object_id,)),
        minimum_test_points=1,
        maximum_test_points=None,
        random_seed=0,
    )
    assert selection.test_ids == (safe_id,)


def test_mask_rejects_outer_top_whose_deletion_erases_nested_runtime_contract() -> None:
    index = RepositoryIndex.from_sources(
        {
            "pkg/core.py": """def build_service():
    class Service:
        def __init__(self):
            self.value = 1

        def feature(self):
            return self.value
    return Service
""",
        }
    )
    build_service = index.by_qualname[("pkg/core.py", "build_service")]
    seed_trace = TraceBundle(
        test_file="tests/test_seed.py",
        records=(
            TraceRecord(
                path=build_service.path,
                qualname=build_service.qualname,
                line=build_service.line,
                calls=1,
                test_ids=("tests/test_seed.py::test_seed",),
            ),
        ),
    )

    with pytest.raises(CandidateRejected, match=r"build_service.*deletes.*static_contract"):
        _planner().create_plan(
            index=index,
            seed_trace=seed_trace,
            all_traces={seed_trace.test_file: seed_trace},
            collection_trace=TraceBundle(test_file="<collection>", records=()),
            top_objects=(build_service,),
            p2p=P2PSelection(test_ids=(), top_object_ids=(build_service.object_id,)),
        )


def test_mask_treats_repeated_bindings_as_one_atomic_deletion_unit() -> None:
    sources = {
        "pkg/core.py": """def feature():
    return duplicate()

def duplicate():
    return 1

def duplicate():
    return 2
""",
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test = index.by_qualname[("tests/test_seed.py", "test_seed")]
    feature = index.by_qualname[("pkg/core.py", "feature")]
    duplicates = tuple(
        item
        for item in index.data.definitions
        if item.path == "pkg/core.py" and item.qualname == "duplicate"
    )
    assert len(duplicates) == 2
    traced_duplicate = duplicates[-1]
    seed_id = "tests/test_seed.py::test_seed"
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
            for item in (test, feature, traced_duplicate)
        ),
        edges=(
            TraceEdge(caller=test.object_id, callee=feature.object_id, calls=1),
            TraceEdge(caller=feature.object_id, callee=traced_duplicate.object_id, calls=1),
        ),
    )
    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=5,
        minimum_code_lines=100,
        maximum_code_lines=100,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )

    plan = planner.create_plan(
        index=index,
        seed_trace=seed,
        all_traces={seed.test_file: seed},
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        top_objects=(feature,),
        p2p=P2PSelection(test_ids=(), top_object_ids=(feature.object_id,)),
    )

    assert {item.object_id for item in plan.specific_objects} == {
        item.object_id for item in duplicates
    }


def test_p2p_bitmap_allows_shared_non_top_objects_and_prunes_common_top() -> None:
    sources = {
        "pkg/core.py": (
            "def top_a():\n    return shared()\n\n"
            "def top_b():\n    return shared()\n\n"
            "def shared():\n    return 1\n"
        ),
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": "def test_other():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    top_a = index.by_qualname[("pkg/core.py", "top_a")]
    top_b = index.by_qualname[("pkg/core.py", "top_b")]
    shared = index.by_qualname[("pkg/core.py", "shared")]
    seed_id = "tests/test_seed.py::test_seed"
    candidate_ids = tuple(f"tests/test_other.py::test_{number}" for number in range(3))
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
            for item in (top_a, top_b, shared)
        ),
    )
    other = TraceBundle(
        test_file="tests/test_other.py",
        records=(
            TraceRecord(
                path=top_a.path,
                qualname=top_a.qualname,
                line=top_a.line,
                calls=2,
                test_ids=candidate_ids[:2],
            ),
            TraceRecord(
                path=top_b.path,
                qualname=top_b.qualname,
                line=top_b.line,
                calls=1,
                test_ids=(candidate_ids[2],),
            ),
            TraceRecord(
                path=shared.path,
                qualname=shared.qualname,
                line=shared.line,
                calls=3,
                test_ids=candidate_ids,
            ),
        ),
    )
    point_index = TracePointIndex.build(
        index=index,
        traces=(seed, other),
        candidate_test_ids=(seed_id, *candidate_ids),
        test_paths={
            seed_id: seed.test_file,
            **dict.fromkeys(candidate_ids, other.test_file),
        },
    )

    selection = point_index.select_p2p(
        seed_test_file=seed.test_file,
        top_object_ids=(top_a.object_id, top_b.object_id),
        minimum_test_points=2,
        maximum_test_points=None,
        random_seed=0,
    )

    assert selection.test_ids == candidate_ids[:2]
    assert selection.top_object_ids == (top_b.object_id,)
    assert point_index.object_test_masks[shared.object_id].bit_count() == 4
    assert point_index.object_ids_for_tests(selection.test_ids) == frozenset(
        {top_a.object_id, shared.object_id}
    )
    with pytest.raises(ContractError, match="has no trace"):
        point_index.object_ids_for_tests(("tests/test_missing.py::test_missing",))


def test_p2p_bitmap_matches_complete_class_and_module_object_ids() -> None:
    sources = {
        "pkg/model.py": (
            "class Service:\n"
            "    def feature(self):\n"
            "        return 1\n\n"
            "def normalize():\n"
            "    return 1\n"
        ),
        "tests/test_seed.py": "def test_seed():\n    assert True\n",
        "tests/test_other.py": (
            "def test_overlapping():\n    assert True\n\ndef test_safe():\n    assert True\n"
        ),
    }
    index = RepositoryIndex.from_sources(sources)
    method = index.by_qualname[("pkg/model.py", "Service.feature")]
    function = index.by_qualname[("pkg/model.py", "normalize")]
    seed_id = "tests/test_seed.py::test_seed"
    overlapping_id = "tests/test_other.py::test_overlapping"
    safe_id = "tests/test_other.py::test_safe"
    traces = (
        TraceBundle(
            test_file="tests/test_seed.py",
            records=tuple(
                TraceRecord(
                    path=item.path,
                    qualname=item.qualname,
                    line=item.line,
                    calls=1,
                    test_ids=(seed_id,),
                )
                for item in (method, function)
            ),
        ),
        TraceBundle(
            test_file="tests/test_other.py",
            records=tuple(
                TraceRecord(
                    path=item.path,
                    qualname=item.qualname,
                    line=item.line,
                    calls=1,
                    test_ids=(overlapping_id,),
                )
                for item in (method, function)
            ),
        ),
    )
    point_index = TracePointIndex.build(
        index=index,
        traces=traces,
        candidate_test_ids=(seed_id, overlapping_id, safe_id),
        test_paths={
            seed_id: "tests/test_seed.py",
            overlapping_id: "tests/test_other.py",
            safe_id: "tests/test_other.py",
        },
    )

    selection = point_index.select_p2p(
        seed_test_file="tests/test_seed.py",
        top_object_ids=(method.object_id, function.object_id),
        minimum_test_points=1,
        maximum_test_points=None,
        random_seed=0,
    )

    assert method.object_id == "pkg/model.py::Service.feature::2"
    assert function.object_id == "pkg/model.py::normalize::5"
    assert selection.test_ids == (safe_id,)
    assert selection.top_object_ids == tuple(sorted((method.object_id, function.object_id)))


def test_mask_repair_rejects_masking_an_already_effective_object() -> None:
    index = RepositoryIndex.from_sources(
        {"pkg/core.py": "def feature():\n    return helper()\n\ndef helper():\n    return 1\n"}
    )
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(top,),
        candidate_object_ids=(top.object_id,),
    )
    with pytest.raises(MaskRepairProposalError) as captured:
        _planner().apply_increment(
            index=index,
            plan=plan,
            proposal=MaskRepairProposal(
                based_on_attempt="attempt-0000",
                must_mask_object_ids=(top.object_id,),
                rationale="Try masking the same object again.",
            ),
        )

    assert captured.value.violations[0].code == "repair.mask_already_effective"


def test_quality_classifies_each_p2p_test_point_independently() -> None:
    sources = {"pkg/core.py": "def feature():\n    return 1\n"}
    index = RepositoryIndex.from_sources(sources)
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed = tuple(f"tests/test_feature.py::test_{index}" for index in range(10))
    p2p = tuple(f"tests/test_p2p.py::test_{index}" for index in range(10))
    p2p_two = tuple(f"tests/test_p2p_two.py::test_{index}" for index in range(10))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=(*p2p, *p2p_two),
    )
    baseline = (*seed, *p2p, *p2p_two)
    test_paths = {
        test_id: path
        for path, test_ids in (
            ("tests/test_feature.py", seed),
            ("tests/test_p2p.py", p2p),
            ("tests/test_p2p_two.py", p2p_two),
        )
        for test_id in test_ids
    }
    outcomes = {
        node: "passed" if node in set(seed[:2]) | set(p2p) | set(p2p_two) else "failed"
        for node in baseline
    }
    report = TestReport.model_validate(
        {"exit_code": 1, "collected": baseline, "outcomes": outcomes}
    )
    quality = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(maximum_p2p_test_points=20),
    )
    assert quality.accepted
    assert quality.fail_to_pass == tuple(sorted(seed[2:]))
    assert quality.pass_to_pass == tuple(sorted((*p2p, *p2p_two)))
    too_many_f2p = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(maximum_f2p_test_points=7),
    )
    assert not too_many_f2p.accepted
    assert any("F2P count 8 exceeds 7" in error for error in too_many_f2p.errors)
    too_many_p2p = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(maximum_p2p_test_points=19),
    )
    assert not too_many_p2p.accepted
    assert any("P2P count 20 exceeds 19" in error for error in too_many_p2p.errors)
    damaged = report.model_copy(update={"outcomes": {**outcomes, p2p[0]: "failed"}})
    downgraded = assess_quality(
        baseline=baseline,
        report=damaged,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(maximum_p2p_test_points=20),
    )
    assert downgraded.accepted
    assert p2p[0] in downgraded.fail_to_pass
    assert set(p2p[1:]) <= set(downgraded.pass_to_pass)
    assert set(p2p_two) <= set(downgraded.pass_to_pass)
    ratio_damaged = report.model_copy(
        update={
            "outcomes": {
                node: ("passed" if node in set(seed[:2]) | {p2p[0], p2p_two[0]} else "failed")
                for node in baseline
            }
        }
    )
    ratio_rejected = assess_quality(
        baseline=baseline,
        report=ratio_damaged,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(
            minimum_remaining_p2p_test_points=1,
            maximum_p2p_test_points=20,
        ),
    )
    assert not ratio_rejected.accepted
    assert ratio_rejected.f2p_ratio > 0.7
    assert any("F2P ratio" in error for error in ratio_rejected.errors)
    broadly_damaged = damaged.model_copy(
        update={
            "outcomes": {
                **damaged.outcomes,
                **dict.fromkeys((*p2p[:6], *p2p_two[:5]), "failed"),
            }
        }
    )
    rejected = assess_quality(
        baseline=baseline,
        report=broadly_damaged,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(),
    )
    assert not rejected.accepted
    assert any("P2P count" in error for error in rejected.errors)


@pytest.mark.asyncio
async def test_candidate_acceptance_uses_one_complete_batch_with_task_timeout() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_id = "tests/test_other.py::test_other"
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=(p2p_id,),
    )

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], float]] = []

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            self.calls.append((test_ids, timeout_seconds))
            return RunnerResponse(
                action="run",
                exit_code=1,
                results=(
                    RunnerTestPointExecution(id=seed_id, outcome="failed"),
                    RunnerTestPointExecution(id=p2p_id, outcome="passed"),
                ),
                output="one selected test failed",
            )

    runner = RecordingRunner()
    baseline, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, runner),
        plan,
        (seed_id,),
        {
            seed_id: "tests/test_feature.py",
            p2p_id: "tests/test_other.py",
        },
        validation_timeout_seconds=3600,
        full_test_timeout_seconds=600,
    )

    assert runner.calls == [((seed_id, p2p_id), 600)]
    quality = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths={
            seed_id: "tests/test_feature.py",
            p2p_id: "tests/test_other.py",
        },
        settings=FeatureBenchSettings(minimum_remaining_p2p_test_points=1),
    )
    assert quality.accepted


@pytest.mark.asyncio
async def test_candidate_test_fallback_is_diagnostic_and_cannot_be_accepted() -> None:
    sources = {"pkg/core.py": "def feature():\n    return 1\n"}
    index = RepositoryIndex.from_sources(sources)
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_path = "tests/test_feature.py"
    p2p_path = "tests/test_p2p.py"
    p2p_two_path = "tests/test_p2p_two.py"
    unrelated_path = "tests/test_unrelated.py"
    test_ids_by_file = {
        path: tuple(f"{path}::test_{index}" for index in range(10))
        for path in (seed_path, p2p_path, p2p_two_path, unrelated_path)
    }
    selected_p2p = (*test_ids_by_file[p2p_path][:2], *test_ids_by_file[p2p_two_path][:2])
    plan = MaskPlan(
        seed_test_file=seed_path,
        top_objects=(target,),
        p2p_test_ids=selected_p2p,
    )
    test_paths = {node: path for path, nodes in test_ids_by_file.items() for node in nodes}

    class CombinedOnlyFailureRunner(FakeIsolatedRunner):
        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            if not self.calls:
                self.calls.append(test_ids)
                raise ContractError("combined validation failed")
            return await super().run(test_ids, timeout_seconds=timeout_seconds)

    fake_runner = CombinedOnlyFailureRunner(
        failing_path="tests/never-fails.py",
        test_paths=test_paths,
    )

    baseline, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, fake_runner),
        plan,
        test_ids_by_file[seed_path],
        test_paths,
    )

    selected = (
        *test_ids_by_file[seed_path],
        *test_ids_by_file[p2p_path][:2],
        *test_ids_by_file[p2p_two_path][:2],
    )
    assert fake_runner.calls == [
        selected,
        test_ids_by_file[seed_path],
        (*test_ids_by_file[p2p_path][:2], *test_ids_by_file[p2p_two_path][:2]),
    ]
    assert set(baseline) == (set(test_ids_by_file[seed_path]) | set(selected_p2p))
    assert not set(baseline) & set(test_ids_by_file[unrelated_path])
    assert report.exit_code == 2
    assert report.collection_errors == (
        "complete required-test batch did not finish operationally in one Runner invocation: "
        "ContractError: combined validation failed",
    )
    assert report.output == ""
    assert "one selected test failed" not in report.output
    assert set(report.passed) >= set(test_ids_by_file[p2p_path][:2])

    quality = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(minimum_remaining_p2p_test_points=1),
    )
    assert not quality.accepted
    assert set(test_ids_by_file[p2p_path][:2]) <= set(quality.pass_to_pass)
    assert test_ids_by_file[p2p_two_path][0] in quality.fail_to_pass
    assert test_ids_by_file[p2p_two_path][1] in quality.pass_to_pass
    assert any("collection" in error for error in quality.errors)

    with pytest.raises(
        CandidateRejected,
        match="clean combined baseline did not finish operationally in one Runner invocation",
    ):
        FeatureBenchGeneratorHarness().normalize_clean_baseline(
            plan=plan,
            seed_test_ids=test_ids_by_file[seed_path],
            selected_test_ids=baseline,
            report=report,
        )


@pytest.mark.asyncio
async def test_candidate_test_fallback_preserves_complete_batch_output() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_id = "tests/test_p2p.py::test_p2p"
    test_paths = {
        seed_id: "tests/test_feature.py",
        p2p_id: "tests/test_p2p.py",
    }
    plan = MaskPlan(
        seed_test_file=test_paths[seed_id],
        top_objects=(target,),
        p2p_test_ids=(p2p_id,),
    )

    class CompleteBatchFailureRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            del timeout_seconds
            self.calls.append(test_ids)
            if len(self.calls) == 1:
                return RunnerResponse(
                    action="run",
                    exit_code=2,
                    results=tuple(
                        RunnerTestPointExecution(id=test_id, outcome="error")
                        for test_id in test_ids
                    ),
                    errors=("combined runner failed",),
                    output="raw combined traceback",
                )
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    runner = CompleteBatchFailureRunner()
    baseline, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, runner),
        plan,
        (seed_id,),
        test_paths,
    )

    assert baseline == (seed_id, p2p_id)
    assert runner.calls == [(seed_id, p2p_id), (seed_id,), (p2p_id,)]
    assert report.exit_code == 2
    assert report.output == "[complete required-test batch]\nraw combined traceback"
    assert report.collection_errors == (
        "complete required-test batch did not finish operationally in one Runner invocation: "
        "combined runner failed",
    )


@pytest.mark.asyncio
async def test_clean_baseline_timeout_replans_once_and_retries_one_complete_batch() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_ids = tuple(f"tests/test_other.py::test_{index}" for index in range(20))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    test_paths = {
        seed_id: "tests/test_feature.py",
        **dict.fromkeys(p2p_ids, "tests/test_other.py"),
    }
    calls: list[tuple[str, ...]] = []
    replans: list[int] = []
    backoff_phases: list[str] = []

    async def run_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        del timeout_seconds
        calls.append(test_ids)
        if len(calls) == 1:
            raise RunnerTimeoutError("initial clean batch timed out", timeout_seconds=600)
        return RunnerResponse(
            action="run",
            exit_code=0,
            results=tuple(
                RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
            ),
        )

    async def replan(maximum_p2p: int) -> MaskPlan:
        replans.append(maximum_p2p)
        return plan.model_copy(update={"p2p_test_ids": p2p_ids[:maximum_p2p]})

    async def observe_backoff(event: object) -> None:
        backoff_phases.append(cast(Any, event).phase)

    (
        replanned,
        selected,
        report,
        backoff,
    ) = await FeatureBenchGenerator()._clean_baseline_with_backoff(  # pyright: ignore[reportPrivateUsage]
        plan=plan,
        seed_test_ids=(seed_id,),
        test_paths=test_paths,
        run_batch=run_batch,
        replan_with_p2p_limit=replan,
        observe_backoff=observe_backoff,
    )

    assert replans == [10]
    assert backoff == 10
    assert replanned.p2p_test_ids == p2p_ids[:10]
    assert calls == [(seed_id, *p2p_ids), (seed_id, *p2p_ids[:10])]
    assert selected == (seed_id, *p2p_ids[:10])
    assert report.exit_code == 0
    assert backoff_phases == ["started", "replanned", "retry_completed"]


@pytest.mark.asyncio
async def test_clean_baseline_backoff_rejects_after_the_second_complete_timeout() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_ids = tuple(f"tests/test_other.py::test_{index}" for index in range(40))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    test_paths = {
        seed_id: "tests/test_feature.py",
        **dict.fromkeys(p2p_ids, "tests/test_other.py"),
    }
    calls = 0
    replans: list[int] = []
    backoff_phases: list[str] = []

    async def timed_out_batch(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        nonlocal calls
        del test_ids, timeout_seconds
        calls += 1
        raise RunnerTimeoutError("clean batch timed out", timeout_seconds=600)

    async def replan(maximum_p2p: int) -> MaskPlan:
        replans.append(maximum_p2p)
        return plan.model_copy(update={"p2p_test_ids": p2p_ids[:maximum_p2p]})

    async def observe_backoff(event: object) -> None:
        backoff_phases.append(cast(Any, event).phase)

    with pytest.raises(
        CandidateRejected,
        match="clean combined baseline timed out after one P2P backoff",
    ):
        await FeatureBenchGenerator()._clean_baseline_with_backoff(  # pyright: ignore[reportPrivateUsage]
            plan=plan,
            seed_test_ids=(seed_id,),
            test_paths=test_paths,
            run_batch=timed_out_batch,
            replan_with_p2p_limit=replan,
            observe_backoff=observe_backoff,
        )

    assert replans == [20]
    assert calls == 2
    assert backoff_phases == ["started", "replanned", "retry_failed"]


@pytest.mark.asyncio
async def test_clean_baseline_infrastructure_timeout_does_not_shrink_p2p() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_ids = tuple(f"tests/test_other.py::test_{index}" for index in range(20))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    replans: list[int] = []

    async def infrastructure_timeout(
        test_ids: tuple[str, ...], timeout_seconds: float
    ) -> RunnerResponse:
        del test_ids, timeout_seconds
        raise CommandTimeoutError("docker session startup timed out", timeout_seconds=30)

    async def replan(maximum_p2p: int) -> MaskPlan:
        replans.append(maximum_p2p)
        return plan

    with pytest.raises(CommandTimeoutError, match="docker session startup timed out"):
        await FeatureBenchGenerator()._clean_baseline_with_backoff(  # pyright: ignore[reportPrivateUsage]
            plan=plan,
            seed_test_ids=(seed_id,),
            test_paths={
                seed_id: "tests/test_feature.py",
                **dict.fromkeys(p2p_ids, "tests/test_other.py"),
            },
            run_batch=infrastructure_timeout,
            replan_with_p2p_limit=replan,
        )

    assert replans == []


@pytest.mark.asyncio
async def test_clean_baseline_records_a_failed_replan_before_propagating() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_ids = tuple(f"tests/test_other.py::test_{index}" for index in range(20))
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    backoff_phases: list[str] = []

    async def runner_timeout(test_ids: tuple[str, ...], timeout_seconds: float) -> RunnerResponse:
        del test_ids, timeout_seconds
        raise RunnerTimeoutError("Runner run timed out", timeout_seconds=600)

    async def rejected_replan(maximum_p2p: int) -> MaskPlan:
        del maximum_p2p
        raise CandidateRejected("reduced joint plan is infeasible")

    async def observe_backoff(event: object) -> None:
        backoff_phases.append(cast(Any, event).phase)

    with pytest.raises(CandidateRejected, match="reduced joint plan is infeasible"):
        await FeatureBenchGenerator()._clean_baseline_with_backoff(  # pyright: ignore[reportPrivateUsage]
            plan=plan,
            seed_test_ids=(seed_id,),
            test_paths={
                seed_id: "tests/test_feature.py",
                **dict.fromkeys(p2p_ids, "tests/test_other.py"),
            },
            run_batch=runner_timeout,
            replan_with_p2p_limit=rejected_replan,
            observe_backoff=observe_backoff,
        )

    assert backoff_phases == ["started", "replan_failed"]


@pytest.mark.asyncio
async def test_clean_candidate_batch_failure_does_not_run_diagnostic_fallbacks() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_id = "tests/test_other.py::test_other"
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=(p2p_id,),
    )

    class FailingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            del timeout_seconds
            self.calls.append(test_ids)
            raise ContractError("complete batch failed")

    runner = FailingRunner()
    _, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, runner),
        plan,
        (seed_id,),
        {
            seed_id: "tests/test_feature.py",
            p2p_id: "tests/test_other.py",
        },
        diagnostic_fallback=False,
    )

    assert runner.calls == [(seed_id, p2p_id)]
    assert report.exit_code == 2
    assert report.collection_errors[0].startswith("complete required-test batch")


@pytest.mark.asyncio
async def test_candidate_test_fallbacks_share_one_validation_deadline() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_id = "tests/test_feature.py::test_feature"
    p2p_id = "tests/test_other.py::test_other"
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=(p2p_id,),
    )

    class SlowFallbackRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.timeouts: list[float] = []

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            self.calls.append(test_ids)
            self.timeouts.append(timeout_seconds)
            if len(self.calls) == 1:
                await asyncio.sleep(0.05)
                raise ContractError("combined validation failed")
            await asyncio.sleep(10)
            raise AssertionError("the shared validation deadline should cancel this fallback")

    runner = SlowFallbackRunner()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(
        CommandTimeoutError,
        match=r"validation shared deadline exceeded after 0\.2 seconds",
    ) as raised:
        await FeatureBenchGeneratorHarness().isolated_candidate_report(
            cast(EnvironmentRunner, runner),
            plan,
            (seed_id,),
            {
                seed_id: "tests/test_feature.py",
                p2p_id: "tests/test_other.py",
            },
            validation_timeout_seconds=0.2,
        )

    assert loop.time() - started < 1
    assert runner.calls == [(seed_id, p2p_id), (seed_id,)]
    assert len(runner.timeouts) == 2
    assert 0 < runner.timeouts[1] < runner.timeouts[0] <= 0.2
    assert raised.value.timeout_seconds == 0.2


@pytest.mark.asyncio
async def test_candidate_runner_errors_are_fatal_even_with_test_failure_exit_code() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    test_id = "tests/test_feature.py::test_feature"
    test_path = "tests/test_feature.py"
    plan = MaskPlan(
        seed_test_file=test_path,
        top_objects=(target,),
        p2p_test_ids=(),
    )

    class ErrorRunner:
        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float,
        ) -> RunnerResponse:
            del timeout_seconds
            return RunnerResponse(
                action="run",
                exit_code=1,
                results=tuple(
                    RunnerTestPointExecution(id=selected, outcome="failed") for selected in test_ids
                ),
                errors=("runner execution failed",),
                output="bounded runner failure output",
            )

    baseline, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, ErrorRunner()),
        plan,
        (test_id,),
        {test_id: test_path},
    )

    assert baseline == (test_id,)
    assert report.exit_code == 2
    assert report.collection_errors == (
        "complete required-test batch did not finish operationally in one Runner invocation: "
        "runner execution failed",
        f"{test_path}: runner execution failed",
    )
    assert report.output == f"[{test_path}]\nbounded runner failure output"


@pytest.mark.asyncio
async def test_candidate_batches_use_validated_paths_instead_of_parsing_test_ids() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_ids = ("opaque-selector::seed_one", "opaque-selector::seed_two")
    p2p_ids = ("opaque-selector::p2p_one",)
    seed_path = "specs/feature_cases.py"
    p2p_path = "checks/regression_cases.py"
    test_paths = {
        **dict.fromkeys(seed_ids, seed_path),
        **dict.fromkeys(p2p_ids, p2p_path),
    }
    plan = MaskPlan(
        seed_test_file=seed_path,
        top_objects=(target,),
        p2p_test_ids=p2p_ids,
    )
    runner = FakeIsolatedRunner(failing_path="unrelated.py", test_paths=test_paths)

    baseline, report = await FeatureBenchGeneratorHarness().isolated_candidate_report(
        cast(EnvironmentRunner, runner),
        plan,
        seed_ids,
        test_paths,
    )

    assert runner.calls == [(*seed_ids, *p2p_ids)]
    quality = assess_quality(
        baseline=baseline,
        report=report,
        plan=plan,
        test_paths=test_paths,
        settings=FeatureBenchSettings(minimum_remaining_p2p_test_points=1),
    )
    assert not any("seed test file has no baseline" in error for error in quality.errors)


def test_clean_combined_baseline_excludes_preexisting_skips_from_f2p_inventory() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    target = index.by_qualname[("pkg/core.py", "feature")].model_copy(
        update={"classification": "top"}
    )
    seed_ids = ("tests/test_feature.py::test_feature#0",)
    skipped_p2p = "tests/test_other.py::test_order_dependent#0"
    passing_p2p = tuple(f"tests/test_other.py::test_{index}#0" for index in range(10))
    selected = (*seed_ids, skipped_p2p, *passing_p2p)
    plan = MaskPlan(
        seed_test_file="tests/test_feature.py",
        top_objects=(target,),
        p2p_test_ids=(skipped_p2p, *passing_p2p),
    )
    report = TestReport(
        exit_code=0,
        collected=selected,
        outcomes={
            test_id: "skipped" if test_id == skipped_p2p else "passed" for test_id in selected
        },
    )

    normalized, baseline = FeatureBenchGeneratorHarness().normalize_clean_baseline(
        plan=plan,
        seed_test_ids=seed_ids,
        selected_test_ids=selected,
        report=report,
    )

    assert normalized.p2p_test_ids == passing_p2p
    assert baseline == (*seed_ids, *passing_p2p)
    assert skipped_p2p not in baseline


def test_level_one_prompt_exposes_top_interfaces_but_not_specific_objects() -> None:
    sources = {
        "pkg/core.py": '''class Service:
    """Public service."""
    def feature(self, value: int) -> int:
        """Return the computed feature."""
        return helper(value)

def helper(value: int) -> int:
    return value + 1
'''
    }
    index = RepositoryIndex.from_sources(sources)
    top = index.by_qualname[("pkg/core.py", "Service.feature")].model_copy(
        update={"classification": "top"}
    )
    specific = index.by_qualname[("pkg/core.py", "helper")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_core.py", top_objects=(top,), specific_objects=(specific,)
    )
    draft = IssueDraft(
        task_statement="Implement the public service feature using the existing repository behavior.",
        interface_contracts=(
            InterfaceContract(
                object_id=top.object_id,
                docstring="Return the computed feature.",
            ),
        ),
    )
    interfaces = build_interfaces(index, plan, draft)
    assert "class Service:" in interfaces[0].code
    assert "def feature(self, value: int) -> int:" in interfaces[0].code
    assert "# <your code>" in interfaces[0].code
    assert "helper" not in interfaces[0].code
    assert "Public service." not in interfaces[0].code
    assert "Return the computed feature." in interfaces[0].code
    prompt = render_prompt(
        draft=draft,
        interfaces=interfaces,
        library_name="example",
        black_links=("https://example.invalid",),
    )
    assert "## Task" in prompt
    assert "## Interface Descriptions" in prompt
    assert prompt.count("Path: `/testbed/pkg/core.py`") == 2
    assert "/testbed/" in prompt
    assert "black_links:\n- https://example.invalid" in prompt
    assert "/testbed                   # all your work" in prompt
    assert "Interface Description {n}" in prompt
    assert "Use the environment and dependencies already provided" in prompt
    assert "fetch and install" not in prompt
    assert "fulfill the input/output format described by this interface" in prompt
    assert (
        "The **Interface Description** describes the behavior and requirements of the interfaces "
        "to be implemented."
    ) in prompt
    assert prompt == prompt.strip()

    prompt_without_links = render_prompt(
        draft=draft,
        interfaces=interfaces,
        library_name="example",
        black_links=(),
    )
    assert "You are forbidden to access the following URLs" not in prompt_without_links


def test_level_one_interfaces_reject_non_top_contracts() -> None:
    sources = {
        "pkg/core.py": """class Service:
    def feature(self, value: int) -> int:
        return value + 1

    def unrelated(self) -> None:
        return None
"""
    }
    index = RepositoryIndex.from_sources(sources)
    top = index.by_qualname[("pkg/core.py", "Service.feature")].model_copy(
        update={"classification": "top"}
    )
    unrelated = index.by_qualname[("pkg/core.py", "Service.unrelated")]
    plan = MaskPlan(seed_test_file="tests/test_core.py", top_objects=(top,))
    invalid = IssueDraft(
        task_statement=(
            "Implement the repository-native service feature with its complete behavior and "
            "compatibility requirements."
        ),
        interface_contracts=(
            InterfaceContract(
                object_id=top.object_id,
                docstring="Compute and return the repository-native service result.",
            ),
            InterfaceContract(
                object_id=unrelated.object_id,
                docstring="This non-Top contract must never enter the generated task.",
            ),
        ),
    )
    with pytest.raises(ContractError, match="every and only final Top"):
        build_interfaces(index, plan, invalid)


@pytest.mark.asyncio
async def test_issue_author_writes_and_submits_sandboxed_file_without_container_access(
    tmp_path: Path,
) -> None:
    sources = {
        "pkg/core.py": """def feature(value: int) -> int:
    return helper(value)

def helper(value: int) -> int:
    return value + 1
""",
        "tests/test_core.py": "def test_feature():\n    assert feature(1) == 2\n",
    }
    index = RepositoryIndex.from_sources(sources)
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    specific = index.by_qualname[("pkg/core.py", "helper")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(top,),
        specific_objects=(specific,),
    )

    class FakeIssueAgents:
        settings = SimpleNamespace(model="test:model")

        async def run_structured(self, **kwargs: object) -> IssueAuthorCompletion:
            assert kwargs["container_tools"] is False
            capabilities = cast(tuple[object, ...], kwargs["extra_capabilities"])
            assert len(capabilities) == 2
            filesystem = cast(FileSystem[object], capabilities[0])
            submission = cast(IssueAuthorSubmission, capabilities[1])
            root = Path(filesystem.root_dir)
            prompt = cast(str, kwargs["prompt"])
            assert "return helper(value)" not in prompt
            assert "test_feature" not in prompt
            manifest = [
                json.loads(line)
                for line in (root / "input" / "top-interfaces.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert len(manifest) == 1
            assert manifest[0]["object_id"] == top.object_id
            assert manifest[0]["source_file"] == "sources/pkg/core.py"
            source = (root / "sources" / "pkg" / "core.py").read_text(encoding="utf-8")
            assert "return helper(value)" in source
            assert "def helper" in source
            assert not (root / "sources" / "tests" / "test_core.py").exists()

            rejected = await submission.service.submit_file("outbox/issue.json")
            assert not rejected.accepted
            violations = (root / rejected.feedback_dir / "violations.jsonl").read_text(
                encoding="utf-8"
            )
            assert "featurebench.issue.empty_text" in violations

            issue = IssueDraft(
                task_statement=(
                    "Implement the repository-native feature calculation while preserving its "
                    "established behavior and compatibility."
                ),
                interface_contracts=(
                    InterfaceContract(
                        object_id=top.object_id,
                        docstring=(
                            "Compute the repository-native result for the supplied integer value."
                        ),
                    ),
                ),
            )
            path = "outbox/issue.json"
            (root / path).write_text(issue.model_dump_json(indent=2) + "\n", encoding="utf-8")
            accepted = await submission.service.submit_file(path)
            assert accepted.accepted
            completion = IssueAuthorCompletion(path=path)
            validator = cast(
                Callable[[IssueAuthorCompletion], Awaitable[None]], kwargs["output_validator"]
            )
            await validator(completion)
            return completion

    draft = await author_issue(
        agents=cast(AgentFactory[AgentFactorySettings], FakeIssueAgents()),
        session=cast(ContainerSession, object()),
        index=index,
        plan=plan,
        run_timeout_seconds=60,
        workspace_root=tmp_path / "issue-workspace",
        metadata={"seed_test_file": plan.seed_test_file},
    )
    assert tuple(item.object_id for item in draft.interface_contracts) == (top.object_id,)


def test_issue_text_has_no_character_count_gate() -> None:
    issue = IssueDraft(
        task_statement="Fix it.",
        interface_contracts=(
            InterfaceContract(object_id="pkg/core.py::feature::1", docstring="Return ABI."),
        ),
    )

    assert issue.task_statement == "Fix it."
    assert issue.interface_contracts[0].docstring == "Return ABI."


def test_issue_markup_allows_task_fences_and_triple_quotes_but_not_docstring_fences() -> None:
    sources = {"pkg/core.py": "def feature():\n    return 1\n"}
    index = RepositoryIndex.from_sources(sources)
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    plan = MaskPlan(seed_test_file="tests/test_core.py", top_objects=(top,))
    allowed = IssueDraft(
        task_statement='Use this example:\n```python\nvalue = """text"""\n```',
        interface_contracts=(
            InterfaceContract(
                object_id=top.object_id,
                docstring="Return text containing \"\"\"double\"\"\" and '''single''' quotes.",
            ),
        ),
    )

    assert not issue_validation_issues(allowed, plan)
    native_object_terms = allowed.model_copy(
        update={"task_statement": "Handle the repository's top object and specific object APIs."}
    )
    assert not issue_validation_issues(native_object_terms, plan)
    rendered = build_interfaces(index, plan, allowed)[0].code
    assert r"\"\"\"double\"\"\"" in rendered
    assert "'''single'''" in rendered

    fenced_docstring = allowed.model_copy(
        update={
            "interface_contracts": (
                InterfaceContract(
                    object_id=top.object_id,
                    docstring="Example:\n```python\nreturn 1\n```",
                ),
            )
        }
    )
    violations = issue_validation_issues(fenced_docstring, plan)
    assert [item.code for item in violations] == ["featurebench.issue.invalid_doc_markup"]
    assert violations[0].path == ("interface_contracts", "0", "docstring")


def test_top_rationale_and_indexed_docstrings_have_no_character_count_gate() -> None:
    decision = TopCandidateDecision(
        object_id="pkg/core.py::feature::1",
        classification="top",
        rationale="",
    )
    long_docstring = "x" * 10_000
    data = index_sources(
        {"pkg/core.py": f'def feature():\n    """{long_docstring}"""\n    return 1\n'}
    )

    assert decision.rationale == ""
    assert data.definitions[0].docstring == long_docstring


def test_featurebench_payload_and_unlimited_default_are_strict() -> None:
    defaults = FeatureBenchSettings()
    assert defaults.maximum_tasks_per_environment is None
    assert defaults.seed_concurrency == 16
    assert defaults.agent_concurrency == 4
    assert defaults.trace_concurrency == 24
    assert defaults.seed_ordering == "easy_first"
    assert defaults.top_batch_size == 20
    assert defaults.maximum_selected_p2p_test_files == 64
    assert not hasattr(defaults, "minimum_test_points_per_file")
    with pytest.raises(ValidationError, match="minimum_test_points_per_file"):
        FeatureBenchSettings.model_validate({"minimum_test_points_per_file": 1})
    for removed_concurrency in ("top_concurrency", "p2p_concurrency"):
        with pytest.raises(ValidationError, match=removed_concurrency):
            FeatureBenchSettings.model_validate({removed_concurrency: 4})
    assert defaults.full_test_timeout_seconds == 600
    assert defaults.compatibility_timeout_seconds == 120
    assert defaults.validation_timeout_seconds == 3600
    assert defaults.repair.strategy == "rollback"
    assert defaults.repair.optimization_target == "first_accepted"
    assert defaults.repair.maximum_validations == 10
    assert defaults.repair.timeout_seconds == 3600
    with pytest.raises(ValidationError, match="optimization_target"):
        FeatureBenchSettings.model_validate({"repair": {"optimization_target": "largest_patch"}})
    with pytest.raises(ValidationError, match="maximum_validations"):
        FeatureBenchSettings.model_validate({"repair": {"maximum_validations": 0}})
    assert defaults.agent.top.model == "openai:gpt-5.6-terra"
    assert defaults.agent.repair.model == "openai:gpt-5.6-terra"
    assert defaults.agent.issue_author.model == "openai:gpt-5.6-terra"
    assert defaults.agent.top.run_timeout_seconds == 3600
    assert defaults.agent.repair.run_timeout_seconds == 3600
    assert defaults.agent.issue_author.run_timeout_seconds == 3600
    with pytest.raises(ValidationError, match="agent_timeout_seconds"):
        FeatureBenchSettings.model_validate({"agent_timeout_seconds": 3600})
    with pytest.raises(ValidationError, match="top_run_timeout_seconds"):
        FeatureBenchSettings.model_validate({"top_run_timeout_seconds": 3600})
    with pytest.raises(ValidationError, match="enabled"):
        FeatureBenchSettings.model_validate({"agent": {"issue_author": {"enabled": False}}})
    payload = FeatureBenchPayload(
        type="featurebench",
        prompt="Implement the requested FeatureBench behavior.",
        gold_patch=_payload_gold_patch(),
        hidden_tests_patch="hidden",
        fail_to_pass=(CollectedTestPoint(id="selector::test_a", path="tests/test_a.py"),),
        pass_to_pass=(CollectedTestPoint(id="selector::test_b", path="tests/test_b.py"),),
        test_timeout_seconds=600,
        metrics=_payload_metrics(),
    )
    assert set(payload.model_dump()) == {
        "type",
        "prompt",
        "gold_patch",
        "hidden_tests_patch",
        "fail_to_pass",
        "pass_to_pass",
        "test_timeout_seconds",
        "metrics",
    }
    assert payload.fail_to_pass_ids == ("selector::test_a",)
    assert payload.pass_to_pass_ids == ("selector::test_b",)
    assert payload.protected_test_paths == frozenset(("tests/test_a.py", "tests/test_b.py"))
    assert payload.model_dump(mode="json")["fail_to_pass"] == [
        {"id": "selector::test_a", "path": "tests/test_a.py"}
    ]
    unsorted = FeatureBenchPayload(
        type="featurebench",
        prompt="Implement the requested FeatureBench behavior.",
        gold_patch=_payload_gold_patch(),
        hidden_tests_patch="hidden",
        fail_to_pass=(
            CollectedTestPoint(id="selector::test_b", path="tests/test_b.py"),
            CollectedTestPoint(id="selector::test_a", path="tests/test_a.py"),
        ),
        pass_to_pass=(CollectedTestPoint(id="selector::test_c", path="tests/test_c.py"),),
        test_timeout_seconds=600,
        metrics=_payload_metrics(f2p=2),
    )
    assert unsorted.fail_to_pass_ids == ("selector::test_a", "selector::test_b")
    with pytest.raises(ValueError, match="disjoint"):
        FeatureBenchPayload(
            type="featurebench",
            prompt="Implement the requested FeatureBench behavior.",
            gold_patch=_payload_gold_patch(),
            hidden_tests_patch="hidden",
            fail_to_pass=(CollectedTestPoint(id="selector::test_a", path="tests/test_a.py"),),
            pass_to_pass=(CollectedTestPoint(id="selector::test_a", path="other/test_a.py"),),
            test_timeout_seconds=600,
            metrics=_payload_metrics(),
        )
    patch = "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
    assert FeatureBenchQualityGate.candidate_patch_paths(patch) == {"tests/test_a.py"}
    reverse_patch = (
        "diff --git b/tests/test_a.py a/tests/test_a.py\n"
        "--- b/tests/test_a.py\n"
        "+++ a/tests/test_a.py\n"
    )
    assert FeatureBenchQualityGate.candidate_patch_paths(reverse_patch) == {"tests/test_a.py"}


def test_final_task_record_contains_only_portable_runtime_and_verification_state() -> None:
    index = RepositoryIndex.from_sources({"pkg/core.py": "def feature():\n    return 1\n"})
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    f2p = CollectedTestPoint(id="tests/test_core.py::test_feature", path="tests/test_core.py")
    p2p = CollectedTestPoint(id="tests/test_other.py::test_other", path="tests/test_other.py")
    environment = _environment().model_copy(update={"verified_test_points": (f2p, p2p)})
    task = assemble_task_record(
        environment=environment,
        prompt="Restore the documented feature.",
        plan=MaskPlan(seed_test_file=f2p.path, top_objects=(top,)),
        quality=QualityReport(
            accepted=True,
            fail_to_pass=(f2p.id,),
            pass_to_pass=(p2p.id,),
            gold_patch_lines=1,
            gold_patch_files=1,
            test_exit_code=1,
        ),
        verified_points={f2p.id: f2p, p2p.id: p2p},
        gold_patch=_payload_gold_patch(),
        hidden_tests_patch="hidden-test-patch",
        test_timeout_seconds=600,
    )

    serialized = json.loads(task.model_dump_json(exclude_none=True))
    assert set(serialized) == {
        "task_id",
        "environment_id",
        "image",
        "runtime",
        "adapter",
    }
    assert set(serialized["adapter"]) == {
        "type",
        "prompt",
        "gold_patch",
        "hidden_tests_patch",
        "fail_to_pass",
        "pass_to_pass",
        "test_timeout_seconds",
        "metrics",
    }
    assert serialized["environment_id"] == environment.environment_id
    assert "repository" not in serialized
    assert "commit" not in serialized
    assert "mask_plan" not in serialized["adapter"]
    assert TaskRecord.model_validate_json(task.model_dump_json(exclude_none=True)) == task


@pytest.mark.asyncio
async def test_quality_gate_uses_a_fresh_container_and_enforces_runner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = FeatureBenchPayload(
        type="featurebench",
        prompt="Restore the documented feature.",
        gold_patch=_payload_gold_patch(),
        hidden_tests_patch="hidden-test-patch",
        fail_to_pass=(
            CollectedTestPoint(id="tests/test_core.py::test_feature", path="tests/test_core.py"),
        ),
        pass_to_pass=(
            CollectedTestPoint(id="tests/test_other.py::test_other", path="tests/test_other.py"),
        ),
        test_timeout_seconds=600,
        metrics=_payload_metrics(),
    )
    task = TaskRecord(
        task_id="featurebench-gold-closure",
        environment_id=_environment().environment_id,
        image=_environment().image,
        runtime=_environment().runtime,
        adapter=payload.model_dump(mode="json"),
    )
    expected_payload = payload

    class RecordingRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, RuntimeSpec]] = []
            self.closed = False

        @asynccontextmanager
        async def session(self, image: str, *, runtime_spec: RuntimeSpec):
            self.calls.append((image, runtime_spec))
            try:
                yield cast(ContainerSession, object())
            finally:
                self.closed = True

    class AcceptingTaskRuntime(FeatureBenchTaskRuntime):
        def __init__(self) -> None:
            self.prepared: list[tuple[TaskRecord, BaseModel]] = []

        async def prepare(
            self,
            session: ContainerSessionProtocol,
            task: TaskRecord,
            payload: BaseModel,
        ) -> None:
            del session
            self.prepared.append((task, payload))

        @staticmethod
        async def apply_patch(
            session: ContainerSessionProtocol,
            patch: str,
            *,
            reverse: bool = False,
        ) -> bool:
            del session
            assert patch == expected_payload.gold_patch
            assert not reverse
            return True

        async def restore_verification_assets(
            self,
            session: ContainerSessionProtocol,
            payload: FeatureBenchPayload,
        ) -> bool:
            del session
            assert payload == expected_payload
            return True

    passing_calls: list[tuple[tuple[str, ...], float | None]] = []

    class PassingRunner:
        def __init__(self, session: ContainerSessionProtocol) -> None:
            del session

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float | None = None,
        ) -> RunnerResponse:
            passing_calls.append((test_ids, timeout_seconds))
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    container_runtime = RecordingRuntime()
    task_runtime = AcceptingTaskRuntime()
    monkeypatch.setattr(quality_gate_module, "EnvironmentRunner", PassingRunner)
    context = cast(GeneratorContext, SimpleNamespace(runtime=container_runtime))
    result = await FeatureBenchQualityGate(task_runtime).check(
        task,
        payload,
        context,
    )

    assert result.passed
    assert container_runtime.calls == [(task.image, task.runtime)]
    assert container_runtime.closed
    assert task_runtime.prepared == [(task, payload)]
    assert passing_calls == [(payload.required_test_ids, payload.test_timeout_seconds)]

    class ErrorReportingRunner(PassingRunner):
        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float | None = None,
        ) -> RunnerResponse:
            response = await super().run(test_ids, timeout_seconds=timeout_seconds)
            return response.model_copy(update={"errors": ("runner reported an internal error",)})

    monkeypatch.setattr(quality_gate_module, "EnvironmentRunner", ErrorReportingRunner)
    rejected = await FeatureBenchQualityGate(task_runtime).check(task, payload, context)

    assert not rejected.passed
    assert rejected.message == "runner reported an internal error"
    assert rejected.metrics["runner_errors"] == 1


@pytest.mark.asyncio
async def test_quality_gate_protects_validated_path_for_opaque_test_id() -> None:
    payload = FeatureBenchPayload(
        type="featurebench",
        prompt="Implement the missing feature.",
        gold_patch=_payload_gold_patch(),
        hidden_tests_patch="hidden",
        fail_to_pass=(
            CollectedTestPoint(id="opaque-selector::case", path="specs/feature_cases.py"),
        ),
        pass_to_pass=(
            CollectedTestPoint(id="opaque-selector::regression", path="checks/regression.py"),
        ),
        test_timeout_seconds=600,
        metrics=_payload_metrics(),
    )
    task = TaskRecord(
        task_id="featurebench-test",
        environment_id=_environment().environment_id,
        image="example/project@sha256:" + "b" * 64,
        runtime=_environment().runtime,
        adapter=payload.model_dump(mode="json"),
    )
    candidate_patch = (
        "diff --git b/specs/feature_cases.py a/specs/feature_cases.py\n"
        "--- b/specs/feature_cases.py\n"
        "+++ a/specs/feature_cases.py\n"
    )

    protected_payload = payload.model_copy(update={"gold_patch": candidate_patch})
    context = cast(GeneratorContext, SimpleNamespace(runtime=FakeDisposableRuntime()))
    result = await FeatureBenchQualityGate(NoopFeatureBenchRuntime()).check(
        task,
        protected_payload,
        context,
    )

    assert not result.passed
    assert result.message == "Gold patch modifies protected tests"


def test_featurebench_accepts_only_cpu_execution_paths() -> None:
    generator = FeatureBenchGeneratorHarness()
    generator.ensure_supported_environment(_environment())
    generator.ensure_supported_environment(_environment(gpu_requirement="optional"))

    with pytest.raises(ContractError, match="without GPU resources"):
        generator.ensure_supported_environment(_environment(gpu_requirement="required"))
    with pytest.raises(ContractError, match="without GPU resources"):
        generator.ensure_supported_environment(_environment(gpus="all"))


@pytest.mark.asyncio
async def test_test_source_resolution_excludes_only_unsupported_physical_files() -> None:
    class ResolvingSession:
        async def run_script(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                exit_code=0,
                stdout=(
                    "tests/test_api.py\ttests/test_api.py\n"
                    "testing/freeze/tests/test_doctest.txt\t"
                    "testing/freeze/tests/test_doctest.txt\n"
                ),
            )

    resolved = await FeatureBenchGenerator._resolve_test_source_paths(  # pyright: ignore[reportPrivateUsage]
        session=cast(ContainerSessionProtocol, ResolvingSession()),
        runner_paths=("tests/test_api.py", "testing/freeze/tests/test_doctest.txt"),
        indexed_paths=frozenset({"tests/test_api.py"}),
        timeout_seconds=30,
    )

    assert resolved == {"tests/test_api.py": "tests/test_api.py"}


@pytest.mark.asyncio
async def test_test_source_resolution_rejects_missing_indexed_source() -> None:
    class ResolvingSession:
        async def run_script(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                exit_code=0,
                stdout="tests/test_api.py\ttests/test_api.py\n",
            )

    with pytest.raises(ContractError, match="absent from source index"):
        await FeatureBenchGenerator._resolve_test_source_paths(  # pyright: ignore[reportPrivateUsage]
            session=cast(ContainerSessionProtocol, ResolvingSession()),
            runner_paths=("tests/test_api.py",),
            indexed_paths=frozenset(),
            timeout_seconds=30,
        )


def test_python_tracer_scopes_pytest_to_physical_file() -> None:
    test_file = "tests/test_core.py"
    launcher, arguments = PythonTracerHarness.pytest_invocation(
        ("pytest", "-q"),
        test_file,
    )
    assert launcher == ("python",)
    assert arguments == ("-q", test_file)


def _run_trace_runner(
    tmp_path: Path,
    *,
    repository: Path,
    pytest_args: tuple[str, ...],
    requested_test_ids: tuple[str, ...],
) -> dict[str, Any]:
    runtime_directory = tmp_path / "trace-runtime"
    runtime_directory.mkdir()
    asset_directory = Path(generator_module.__file__).parent / "assets" / "tracing"
    for asset in asset_directory.glob("*.py.txt"):
        (runtime_directory / asset.name.removesuffix(".txt")).write_bytes(asset.read_bytes())
    (runtime_directory / "featurebench_pytest_ids.py").write_bytes(
        Path(pytest_contract.__file__).read_bytes()
    )
    output_path = tmp_path / "trace.json"
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    arguments_path = tmp_path / "arguments.json"
    arguments_path.write_text(
        json.dumps(
            {
                "pytest_args": pytest_args,
                "requested_test_ids": requested_test_ids,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(runtime_directory / "runner.py"),
            str(output_path),
            str(shard_directory),
            str(arguments_path),
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_trace_runner_resolves_stable_ids_to_native_pytest_items(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "helper.py").write_text(
        "def double(value: int) -> int:\n    return value * 2\n",
        encoding="utf-8",
    )
    (repository / "test_example.py").write_text(
        "import pytest\n\n"
        "from helper import double\n\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_value(value: int) -> None:\n"
        "    assert double(value) == value * 2\n",
        encoding="utf-8",
    )
    stable_id = "test_example.py::test_value#1"
    payload = _run_trace_runner(
        tmp_path,
        repository=repository,
        pytest_args=("-q", "test_example.py"),
        requested_test_ids=(stable_id,),
    )

    assert payload["test_ids"] == [stable_id]
    assert payload["passed_test_ids"] == [stable_id]
    helper_records = [item for item in payload["records"] if item["path"] == "helper.py"]
    assert helper_records


def test_trace_runner_collection_smoke_does_not_execute_tests(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_example.py").write_text(
        "def test_must_not_run() -> None:\n"
        "    raise AssertionError('collection smoke executed a test')\n",
        encoding="utf-8",
    )
    requested = ("test_example.py::test_must_not_run#0",)

    payload = _run_trace_runner(
        tmp_path,
        repository=repository,
        pytest_args=("-q", "--collect-only", "test_example.py"),
        requested_test_ids=requested,
    )

    assert payload["passed_test_ids"] == []


def test_trace_runner_disables_repository_xdist_addopts(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-n auto"\n',
        encoding="utf-8",
    )
    (repository / "conftest.py").write_text(
        "def pytest_xdist_auto_num_workers(config):\n    del config\n    return 2\n",
        encoding="utf-8",
    )
    (repository / "helper.py").write_text(
        "def double(value: int) -> int:\n    return value * 2\n",
        encoding="utf-8",
    )
    (repository / "test_example.py").write_text(
        "import os\n\n"
        "from helper import double\n\n"
        "def test_requested() -> None:\n"
        "    assert 'PYTEST_XDIST_WORKER' not in os.environ\n"
        "    assert double(3) == 6\n\n"
        "def test_must_not_run() -> None:\n"
        "    raise AssertionError('unrequested test was executed')\n",
        encoding="utf-8",
    )
    requested = ("test_example.py::test_requested#0",)

    payload = _run_trace_runner(
        tmp_path,
        repository=repository,
        pytest_args=("-q", "test_example.py"),
        requested_test_ids=requested,
    )

    assert payload["test_ids"] == list(requested)
    assert payload["passed_test_ids"] == list(requested)
    helper_records = [item for item in payload["records"] if item["path"] == "helper.py"]
    assert helper_records


def test_trace_runner_resolves_unique_ids_with_a_logical_path_alias(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    physical_directory = repository / "packages" / "core" / "tests"
    physical_directory.mkdir(parents=True)
    (physical_directory / "test_example.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_value(value: int) -> None:\n"
        "    assert value > 0\n",
        encoding="utf-8",
    )
    requested = (
        "tests/logical_core/test_example.py::test_value#0",
        "tests/logical_core/test_example.py::test_value#1",
    )
    payload = _run_trace_runner(
        tmp_path,
        repository=repository,
        pytest_args=("-q", "packages/core/tests/test_example.py"),
        requested_test_ids=requested,
    )

    assert payload["test_ids"] == list(requested)
    assert payload["passed_test_ids"] == list(requested)


def test_trace_runner_reports_only_real_passes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "test_example.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('skip', [False, True])\n"
        "def test_value(skip: bool) -> None:\n"
        "    if skip:\n"
        "        pytest.skip('not runnable here')\n",
        encoding="utf-8",
    )
    requested = (
        "test_example.py::test_value#0",
        "test_example.py::test_value#1",
    )
    payload = _run_trace_runner(
        tmp_path,
        repository=repository,
        pytest_args=("-q", "test_example.py"),
        requested_test_ids=requested,
    )

    assert payload["passed_test_ids"] == [requested[0]]


@pytest.mark.asyncio
async def test_file_trace_excludes_nonpassing_points_without_batch_fallback() -> None:
    test_file = "tests/test_core.py"
    test_ids = tuple(f"{test_file}::test_{index}" for index in range(2))

    class PartialTracer(FakeTracer):
        async def trace(
            self,
            *,
            test_command: tuple[str, ...],
            test_file: str,
            test_ids: tuple[str, ...],
            timeout_seconds: float,
        ) -> TraceBundle:
            bundle = await super().trace(
                test_command=test_command,
                test_file=test_file,
                test_ids=test_ids,
                timeout_seconds=timeout_seconds,
            )
            return bundle.model_copy(update={"passed_test_ids": (test_ids[0],)})

    tracer = PartialTracer("never-fails")
    result = await FeatureBenchGeneratorHarness().trace_file(
        cast(PythonTracer, tracer), test_file, test_ids
    )

    assert tracer.calls == [test_ids]
    assert result.successful_test_ids == (test_ids[0],)
    assert result.failed_test_ids == (
        (test_ids[1], "dynamic trace completed but the test point did not pass"),
    )


@pytest.mark.asyncio
async def test_file_trace_batch_failure_rejects_the_file_without_retrying() -> None:
    test_file = "tests/test_core.py"
    test_ids = tuple(f"{test_file}::test_{index}" for index in range(3))
    tracer = FakeTracer(test_ids[1])

    result = await FeatureBenchGeneratorHarness().trace_file(
        cast(PythonTracer, tracer),
        test_file,
        test_ids,
    )

    assert tracer.calls == [test_ids]
    assert tracer.session.clean_count == 1
    assert result.successful_test_ids == ()
    assert result.failed_test_ids == tuple(
        (test_id, "batch contains failing point") for test_id in test_ids
    )
    assert not result.complete
    assert not result.can_enter_top
    assert result.bundle is None
    assert not result.budget_exhausted


@pytest.mark.asyncio
async def test_file_trace_budget_exhaustion_is_a_local_rejection_result() -> None:
    test_file = "tests/test_core.py"
    test_ids = tuple(f"{test_file}::test_{index}" for index in range(2))
    tracer = FakeTracer("never-fails")

    async def stall_trace(**_: object) -> TraceBundle:
        await asyncio.sleep(1)
        raise AssertionError("the shared deadline must cancel the stalled file")

    tracer.trace = stall_trace  # type: ignore[method-assign]
    result = await FeatureBenchGeneratorHarness().trace_file(
        cast(PythonTracer, tracer),
        test_file,
        test_ids,
        timeout_seconds=0.01,
    )

    assert result.budget_exhausted
    assert result.successful_test_ids == ()
    assert result.failed_test_ids == tuple(
        (test_id, "dynamic trace file budget exhausted after 0.01 seconds") for test_id in test_ids
    )
    assert result.bundle is None
    assert not result.can_enter_top


@pytest.mark.asyncio
async def test_file_without_any_successful_trace_cannot_enter_top() -> None:
    test_file = "tests/test_core.py"
    test_id = f"{test_file}::test_only"
    tracer = FakeTracer(test_id)

    result = await FeatureBenchGeneratorHarness().trace_file(
        cast(PythonTracer, tracer),
        test_file,
        (test_id,),
    )

    assert result.successful_test_ids == ()
    assert result.bundle is None
    assert not result.can_enter_top


@pytest.mark.asyncio
async def test_trace_attempts_use_disposable_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = "tests/test_core.py"
    first = (f"{test_file}::test_one",)
    second = (f"{test_file}::test_two",)
    runtime = FakeDisposableRuntime()

    class FakeConstructedTracer:
        def __init__(self, session: FakeSession, *, maximum_report_bytes: int) -> None:
            del maximum_report_bytes
            self.session = session
            self.statistics: dict[tuple[str, tuple[str, ...]], TraceStatistics] = {}

        async def trace(
            self,
            *,
            test_command: tuple[str, ...],
            test_file: str,
            test_ids: tuple[str, ...],
            timeout_seconds: float,
        ) -> TraceBundle:
            del test_command, timeout_seconds
            self.statistics[(test_file, test_ids)] = TraceStatistics(
                duration_seconds=1,
                report_bytes=100,
                test_ids=len(test_ids),
                records=1,
                edges=0,
                imports=0,
                process_count=1,
            )
            return TraceBundle(
                test_file=test_file,
                records=(
                    TraceRecord(
                        path="pkg/core.py",
                        qualname="feature",
                        line=1,
                        calls=1,
                        test_ids=test_ids,
                    ),
                ),
                passed_test_ids=test_ids,
            )

    monkeypatch.setattr(generator_module, "PythonTracer", FakeConstructedTracer)
    context = cast(GeneratorContext, SimpleNamespace(runtime=runtime))
    generator = FeatureBenchGeneratorHarness()

    await generator.trace_selection_in_fresh_session(
        environment=_environment(),
        context=context,
        test_file=test_file,
        test_ids=first,
    )
    await generator.trace_selection_in_fresh_session(
        environment=_environment(),
        context=context,
        test_file=test_file,
        test_ids=second,
    )

    assert len(runtime.sessions) == 2
    assert runtime.sessions[0] is not runtime.sessions[1]
    assert runtime.closed == runtime.sessions
    assert [session.clean_count for session in runtime.sessions] == [1, 1]


@pytest.mark.asyncio
async def test_candidate_batches_replay_into_disposable_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeDisposableRuntime()
    replayed: list[tuple[FakeSession, str]] = []

    class FakeReplayRuntime(FeatureBenchTaskRuntime):
        @staticmethod
        async def apply_patch(
            session: ContainerSessionProtocol,
            patch: str,
            *,
            reverse: bool = False,
        ) -> bool:
            assert not reverse
            replayed.append((cast(FakeSession, session), patch))
            return True

    class FakeEnvironmentRunner:
        def __init__(self, session: FakeSession) -> None:
            self.session = session

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float | None = None,
        ) -> RunnerResponse:
            del timeout_seconds
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    monkeypatch.setattr(generator_module, "EnvironmentRunner", FakeEnvironmentRunner)
    context = cast(GeneratorContext, SimpleNamespace(runtime=runtime))
    generator = FeatureBenchGeneratorHarness()
    generator.runtime = FakeReplayRuntime()
    first = ("tests/test_core.py::test_one",)
    second = ("tests/test_core.py::test_two",)

    await generator.run_candidate_batch_in_fresh_session(
        environment=_environment(),
        context=context,
        candidate_patch="candidate patch",
        test_ids=first,
        timeout_seconds=600,
    )
    await generator.run_candidate_batch_in_fresh_session(
        environment=_environment(),
        context=context,
        candidate_patch="candidate patch",
        test_ids=second,
        timeout_seconds=600,
    )

    assert len(runtime.sessions) == 2
    assert runtime.sessions[0] is not runtime.sessions[1]
    assert runtime.closed == runtime.sessions
    assert [session.clean_count for session in runtime.sessions] == [1, 1]
    assert replayed == [
        (runtime.sessions[0], "candidate patch"),
        (runtime.sessions[1], "candidate patch"),
    ]


def test_collection_evidence_is_derived_from_normal_test_traces() -> None:
    collection_record = TraceRecord(
        path="pkg/bootstrap.py",
        qualname="register",
        line=1,
        calls=1,
    )
    first = TraceBundle(
        test_file="tests/test_a.py",
        records=(
            collection_record,
            TraceRecord(
                path="pkg/core.py",
                qualname="feature",
                line=2,
                calls=1,
                test_ids=("tests/test_a.py::test_a",),
            ),
        ),
    )
    second = TraceBundle(
        test_file="tests/test_b.py",
        records=(collection_record.model_copy(update={"test_ids": ("tests/test_b.py::test_b",)}),),
    )

    merged = merge_trace_bundles("combined", (first, second))
    collection = collection_trace_from_bundles((first, second))

    assert merged.records[0].calls == 2
    assert tuple(item.object_id for item in collection.records) == (
        "pkg/bootstrap.py::register::1",
    )
    assert collection.records[0].calls == 1
    assert all(not item.test_ids for item in collection.records)


def test_featurebench_definition_composes_data_production_components() -> None:
    settings = FeatureBenchSettings.model_validate(
        {
            "agent": {
                "top": {"model": "test:top"},
                "repair": {"model": "test:repair"},
                "issue_author": {"model": "test:issue"},
            }
        }
    )
    definition = featurebench_definition(settings)

    assert definition.type == "featurebench"
    assert isinstance(definition.generator, FeatureBenchGenerator)
    assert isinstance(definition.quality_gate, FeatureBenchQualityGate)
    assert definition.generator.runtime is definition.quality_gate.runtime
    assert definition.generator.top_agents.settings.model == "test:top"
    assert definition.generator.repair_agents.settings.model == "test:repair"
    assert definition.generator.issue_author_agents.settings.model == "test:issue"
    assert tuple(phase.label for phase in definition.progress_phases) == (
        "Indexing",
        "Tracing",
        "Generating",
    )
    assert settings.seed_concurrency == 16
    assert definition.quality_gate_concurrency == settings.seed_concurrency
    assert definition.generator.resources is not None
    assert definition.generator.top_agents.run_semaphore is (
        definition.generator.resources.agent_slots
    )
    assert definition.generator.repair_agents.run_semaphore is (
        definition.generator.resources.agent_slots
    )
    assert definition.generator.issue_author_agents.run_semaphore is (
        definition.generator.resources.agent_slots
    )
    assert "agents" not in GeneratorContext.__dataclass_fields__


def test_featurebench_invocation_shares_capacity_but_not_generator_state() -> None:
    registration = default_adapter_catalog().get("featurebench")
    settings = registration.validate_settings({"environment_concurrency": 4})
    invocation = registration.begin_validated(settings)

    first = invocation.build_definition()
    second = invocation.build_definition()
    first_generator = cast(FeatureBenchGenerator, first.generator)
    second_generator = cast(FeatureBenchGenerator, second.generator)

    assert first_generator is not second_generator
    assert first_generator.resources is second_generator.resources
    assert first_generator.diagnostics is None
    assert second_generator.diagnostics is None


def test_index_sources_skips_invalid_python_without_losing_valid_files() -> None:
    data = index_sources({"bad.py": "def broken(:\n", "good.py": "def okay():\n    return 1\n"})
    assert [item.qualname for item in data.definitions] == ["okay"]


def test_repository_snapshot_indexes_tracked_directories_named_env(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    tracked = repository / "tests" / "env" / "test_api.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("def test_api():\n    assert True\n", encoding="utf-8")
    untracked = repository / "generated.py"
    untracked.write_text("def generated():\n    return 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "tests/env/test_api.py"), cwd=repository, check=True)
    script = tmp_path / "snapshot.py"
    output = tmp_path / "snapshot.json"
    script.write_text(
        indexing_module._SNAPSHOT_SCRIPT,  # pyright: ignore[reportPrivateUsage]
        encoding="utf-8",
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            str(output),
            "100",
            "100000",
            "1000000",
            *sorted(indexing_module.INDEXED_SOURCE_SUFFIXES),
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    files = json.loads(output.read_text(encoding="utf-8"))["files"]
    assert "tests/env/test_api.py" in files
    assert "generated.py" not in files


def test_feature_graph_traversal_walks_bridged_edges_until_top_boundary() -> None:
    sources = {
        "tests/test_api.py": "from pkg.api import feature\n\ndef test_feature():\n    assert feature(1) == 2\n",
        "pkg/api.py": "from pkg.core import helper\n\ndef feature(value):\n    return helper(value)\n",
        "pkg/core.py": "def helper(value):\n    return value + 1\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test_object = index.by_qualname[("tests/test_api.py", "test_feature")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    helper = index.by_qualname[("pkg/core.py", "helper")]
    node_id = "tests/test_api.py::test_feature#0"
    trace = TraceBundle(
        test_file="tests/test_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(node_id,),
            )
            for item in (test_object, feature, helper)
        ),
        edges=(
            TraceEdge(
                caller=test_object.object_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(node_id,),
                provenance=("third_party_bridge",),
            ),
            TraceEdge(
                caller=feature.object_id,
                callee=helper.object_id,
                calls=1,
                test_ids=(node_id,),
                provenance=("direct",),
            ),
        ),
    )
    traversal = FeatureGraphTraversal(
        index=index,
        trace=trace,
        test_ids=(node_id,),
        max_depth=5,
        batch_size=2,
        critical_magic_methods=frozenset({"__init__"}),
    )
    first_batches = traversal.layer_batches()
    assert len(first_batches) == 1
    first = first_batches[0]
    assert first.candidate_ids == (feature.object_id,)
    first_payload = cast(dict[str, object], first.payload["classification_batch"])
    candidate = cast(list[dict[str, object]], first_payload["candidates"])[0]
    assert candidate["source"] == "def feature(value):\n    return helper(value)"
    assert cast(dict[str, object], candidate["dynamic_evidence"])["provenance"] == [
        "third_party_bridge"
    ]
    wrong_id = TopBatchClassification(
        decisions=(
            TopCandidateDecision(
                object_id=helper.object_id,
                classification="top",
                rationale="This deliberately uses an ID outside the current batch.",
            ),
        )
    )
    assert [issue.code for issue in traversal.validation_issues(first, wrong_id)] == [
        "featurebench.top.batch_mismatch"
    ]
    assert traversal.remaining_candidates == [feature.object_id]
    first_classification = TopBatchClassification(
        decisions=(
            TopCandidateDecision(
                object_id=feature.object_id,
                classification="traverse",
                rationale="This is routing glue toward the behavior-owning helper.",
            ),
        )
    )
    assert not traversal.validation_issues(first, first_classification)
    traversal.apply(first, first_classification)
    second_batches = traversal.layer_batches()
    assert len(second_batches) == 1
    second = second_batches[0]
    assert second.candidate_ids == (helper.object_id,)
    second_classification = TopBatchClassification(
        decisions=(
            TopCandidateDecision(
                object_id=helper.object_id,
                classification="top",
                rationale="This object directly owns the arithmetic behavior under test.",
            ),
        )
    )
    assert not traversal.validation_issues(second, second_classification)
    traversal.apply(second, second_classification)
    assert traversal.done is True
    assert traversal.selected_ids() == (helper.object_id,)


def test_feature_graph_traversal_and_feasibility_share_local_fixture_roots() -> None:
    sources = {
        "tests/test_api.py": (
            "import pkg.api\n\n"
            "def local_fixture():\n"
            "    return pkg.api.feature()\n\n"
            "def test_feature(local_fixture):\n"
            "    assert local_fixture\n"
        ),
        "pkg/api.py": "def feature():\n    return True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    fixture = index.by_qualname[("tests/test_api.py", "local_fixture")]
    test_object = index.by_qualname[("tests/test_api.py", "test_feature")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    test_id = "tests/test_api.py::test_feature#0"
    trace = TraceBundle(
        test_file="tests/test_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(test_id,),
            )
            for item in (fixture, test_object, feature)
        ),
        edges=(
            TraceEdge(
                caller=fixture.object_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(test_id,),
            ),
        ),
        passed_test_ids=(test_id,),
    )
    planner = _planner()
    candidates = planner.eligible_top_candidates(
        index=index,
        trace=trace,
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        test_ids=(test_id,),
    )
    traversal = FeatureGraphTraversal(
        index=index,
        trace=trace,
        test_ids=(test_id,),
        max_depth=5,
        batch_size=10,
        critical_magic_methods=frozenset(),
    )

    assert tuple(candidate.object.object_id for candidate in candidates) == (feature.object_id,)
    assert traversal.roots == frozenset({fixture.object_id, test_object.object_id})
    assert traversal.layer_batches()[0].candidate_ids == (feature.object_id,)


def test_feature_graph_traversal_parses_public_occurrence_before_root_lookup() -> None:
    sources = {
        "checks/spec_api.py": (
            "from pkg.api import feature\n\n"
            "class Specification:\n"
            "    def scenario(self):\n"
            "        assert feature()\n"
        ),
        "pkg/api.py": "def feature():\n    return True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test_object = index.by_qualname[("checks/spec_api.py", "Specification.scenario")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    test_id = "checks/spec_api.py::Specification::scenario#0"
    runtime_test_id = f"checks/spec_api.py::scenario::{test_object.line}"
    trace = TraceBundle(
        test_file="checks/spec_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname="scenario" if item is test_object else item.qualname,
                line=item.line,
                calls=1,
                test_ids=(test_id,),
            )
            for item in (test_object, feature)
        ),
        edges=(
            TraceEdge(
                caller=runtime_test_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(test_id,),
            ),
        ),
    )

    traversal = FeatureGraphTraversal(
        index=index,
        trace=trace,
        test_ids=(test_id,),
        max_depth=5,
        batch_size=10,
        critical_magic_methods=frozenset(),
    )

    assert traversal.layer_batches()[0].candidate_ids == (feature.object_id,)


@pytest.mark.asyncio
async def test_top_selection_uses_one_toolless_structured_call_per_bfs_batch() -> None:
    sources = {
        "tests/test_api.py": "def test_feature():\n    assert True\n",
        "pkg/api.py": "from pkg.core import helper\n\ndef feature(value):\n    return helper(value)\n",
        "pkg/core.py": "def helper(value):\n    return value + 1\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test_object = index.by_qualname[("tests/test_api.py", "test_feature")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    helper = index.by_qualname[("pkg/core.py", "helper")]
    node_id = "tests/test_api.py::test_feature#0"
    trace = TraceBundle(
        test_file="tests/test_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(node_id,),
            )
            for item in (test_object, feature, helper)
        ),
        edges=(
            TraceEdge(
                caller=test_object.object_id,
                callee=feature.object_id,
                calls=1,
                test_ids=(node_id,),
            ),
            TraceEdge(
                caller=feature.object_id,
                callee=helper.object_id,
                calls=1,
                test_ids=(node_id,),
            ),
        ),
    )

    class FakeBatchModel:
        settings = SimpleNamespace(model="test:model")

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def run_structured(self, **kwargs: object) -> TopBatchClassification:
            self.calls.append(kwargs)
            assert kwargs["container_tools"] is False
            assert "extra_capabilities" not in kwargs
            classification = TopBatchClassification(
                decisions=(
                    TopCandidateDecision(
                        object_id=feature.object_id if len(self.calls) == 1 else helper.object_id,
                        classification="traverse" if len(self.calls) == 1 else "top",
                        rationale="This judgment follows the supplied behavior boundary evidence.",
                    ),
                )
            )
            validator = cast(
                Callable[[TopBatchClassification], Awaitable[None]],
                kwargs["output_validator"],
            )
            await validator(classification)
            return classification

    model = FakeBatchModel()
    selected = await _planner().select_top(
        index=index,
        trace=trace,
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        test_ids=(node_id,),
        agents=cast(AgentFactory[AgentFactorySettings], model),
        session=cast(ContainerSession, object()),
        run_timeout_seconds=10,
    )

    assert selected == (helper,)
    assert len(model.calls) == 2
    assert all(call["output_type"] is TopBatchClassification for call in model.calls)

    with pytest.raises(ConfigurationError, match=r"agent\.model is unset"):
        await _planner().select_top(
            index=index,
            trace=trace,
            collection_trace=TraceBundle(test_file="<collection>", records=()),
            test_ids=(node_id,),
            agents=AgentFactory(),
            session=cast(ContainerSession, object()),
            run_timeout_seconds=10,
        )


@pytest.mark.asyncio
async def test_top_selection_runs_same_layer_batches_concurrently() -> None:
    production_source = "\n\n".join(
        (
            f"def feature_{index}(value):\n    return implementation_{index}(value)\n\n"
            f"def implementation_{index}(value):\n    return value + {index}"
        )
        for index in range(6)
    )
    sources = {
        "tests/test_api.py": "def test_feature():\n    assert True\n",
        "pkg/core.py": production_source,
    }
    index = RepositoryIndex.from_sources(sources)
    test_object = index.by_qualname[("tests/test_api.py", "test_feature")]
    features = tuple(index.by_qualname[("pkg/core.py", f"feature_{number}")] for number in range(6))
    implementations = tuple(
        index.by_qualname[("pkg/core.py", f"implementation_{number}")] for number in range(6)
    )
    node_id = "tests/test_api.py::test_feature#0"
    trace = TraceBundle(
        test_file="tests/test_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=(node_id,),
            )
            for item in (test_object, *features, *implementations)
        ),
        edges=tuple(
            edge
            for feature, implementation in zip(features, implementations, strict=True)
            for edge in (
                TraceEdge(
                    caller=test_object.object_id,
                    callee=feature.object_id,
                    calls=1,
                    test_ids=(node_id,),
                ),
                TraceEdge(
                    caller=feature.object_id,
                    callee=implementation.object_id,
                    calls=1,
                    test_ids=(node_id,),
                ),
            )
        ),
    )

    class ConcurrentBatchModel:
        settings = SimpleNamespace(model="test:model")

        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.calls = 0
            self.completed_by_depth = {1: 0, 2: 0}

        async def run_structured(self, **kwargs: object) -> TopBatchClassification:
            self.calls += 1
            metadata = cast(dict[str, str], kwargs["metadata"])
            depth = int(metadata["bfs_depth"])
            if depth == 2:
                assert self.completed_by_depth[1] == 3
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            prompt = cast(str, kwargs["prompt"])
            payload = json.loads(prompt.split("Authoritative evidence JSON:\n\n", 1)[1])
            candidates = cast(
                list[dict[str, object]], payload["classification_batch"]["candidates"]
            )
            await asyncio.sleep(0.01)
            classification = TopBatchClassification(
                decisions=tuple(
                    TopCandidateDecision(
                        object_id=cast(str, candidate["object_id"]),
                        classification="traverse" if depth == 1 else "top",
                        rationale="This function directly owns one tested production behavior.",
                    )
                    for candidate in candidates
                )
            )
            validator = cast(
                Callable[[TopBatchClassification], Awaitable[None]],
                kwargs["output_validator"],
            )
            await validator(classification)
            self.active -= 1
            self.completed_by_depth[depth] += 1
            return classification

    planner = FeatureSlicePlanner(
        top_max_depth=5,
        top_batch_size=2,
        minimum_code_lines=1,
        maximum_code_lines=5000,
        random_seed=0,
        critical_magic_methods=frozenset(),
    )
    model = ConcurrentBatchModel()
    selected = await planner.select_top(
        index=index,
        trace=trace,
        collection_trace=TraceBundle(test_file="<collection>", records=()),
        test_ids=(node_id,),
        agents=cast(AgentFactory[AgentFactorySettings], model),
        session=cast(ContainerSession, object()),
        run_timeout_seconds=10,
        batch_concurrency=2,
    )

    assert {item.object_id for item in selected} == {
        implementation.object_id for implementation in implementations
    }
    assert model.calls == 6
    assert model.maximum_active == 2
    assert model.completed_by_depth == {1: 3, 2: 3}


def test_feature_graph_traversal_bounds_parameterized_test_id_evidence() -> None:
    sources = {
        "tests/test_api.py": (
            "from pkg.api import feature\n\ndef test_feature(value):\n    assert feature(value)\n"
        ),
        "pkg/api.py": "def feature(value):\n    return value\n",
    }
    index = RepositoryIndex.from_sources(sources)
    test_object = index.by_qualname[("tests/test_api.py", "test_feature")]
    feature = index.by_qualname[("pkg/api.py", "feature")]
    test_ids = tuple(f"tests/test_api.py::test_feature#{index}" for index in range(10_000))
    trace = TraceBundle(
        test_file="tests/test_api.py",
        records=tuple(
            TraceRecord(
                path=item.path,
                qualname=item.qualname,
                line=item.line,
                calls=1,
                test_ids=test_ids,
            )
            for item in (test_object, feature)
        ),
        edges=(
            TraceEdge(
                caller=test_object.object_id,
                callee=feature.object_id,
                calls=10_000,
                test_ids=test_ids,
                provenance=("direct",),
            ),
        ),
    )
    traversal = FeatureGraphTraversal(
        index=index,
        trace=trace,
        test_ids=test_ids,
        max_depth=5,
        batch_size=10,
        critical_magic_methods=frozenset(),
    )

    context = traversal.test_context()
    assert context["test_point_count"] == 10_000
    assert set(context) == {"test_file", "test_point_count", "test_source"}
    assert len(json.dumps(context)) < 1_000

    batch_payload = cast(
        dict[str, object], traversal.layer_batches()[0].payload["classification_batch"]
    )
    candidate = cast(list[dict[str, object]], batch_payload["candidates"])[0]
    dynamic_evidence = cast(dict[str, object], candidate["dynamic_evidence"])
    assert dynamic_evidence["test_points"] == 10_000
    assert dynamic_evidence["provenance"] == ["direct"]
    assert len(json.dumps(candidate)) < 1_000


def test_dynamic_import_evidence_complements_unresolved_static_imports() -> None:
    sources = {
        "pkg/core.py": "def feature():\n    return 1\n",
        "pkg/api.py": "from aliaspkg import feature as public_feature\n",
    }
    index = RepositoryIndex.from_sources(sources)
    feature = index.by_qualname[("pkg/core.py", "feature")]
    assert next(iter(index.data.imports)).resolved_path is None
    edits = _planner().import_edits(
        index,
        (feature,),
        (
            TraceImport(
                importer_path="pkg/api.py",
                importer_line=1,
                imported_name="feature",
                defined_path="pkg/core.py",
                defined_name="feature",
                defined_line=1,
            ),
        ),
    )
    assert len(edits) == 1
    assert edits[0].remove_names == ("feature",)


def test_import_rewrites_never_modify_test_sources() -> None:
    sources = {
        "pkg/core.py": "def feature():\n    return 1\n",
        "pkg/api.py": "from pkg.core import feature\n",
        "tests/test_core.py": "from pkg.core import feature\n",
    }
    index = RepositoryIndex.from_sources(sources)
    feature = index.by_qualname[("pkg/core.py", "feature")]
    edits = _planner().import_edits(index, (feature,))
    assert tuple(edit.importer_path for edit in edits) == ("pkg/api.py",)


def test_gold_patch_scale_gate_uses_real_reverse_diff_lines() -> None:
    quality = QualityReport(
        accepted=True,
        fail_to_pass=("tests/test_a.py::test_a",),
        pass_to_pass=("tests/test_b.py::test_b",),
        test_exit_code=1,
    )
    patch = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -0,0 +1,2 @@
+one
+two
"""
    accepted = apply_gold_patch_gates(
        quality,
        gold_patch=patch,
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=2,
            gold_patch_line_preferred_minimum=2,
            gold_patch_line_maximum=4,
        ),
    )
    assert accepted.accepted
    assert accepted.gold_patch_lines == 2
    assert accepted.gold_patch_files == 1
    rejected = apply_gold_patch_gates(
        quality,
        gold_patch=patch,
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=3,
            gold_patch_line_preferred_minimum=3,
            gold_patch_line_maximum=4,
        ),
    )
    assert not rejected.accepted
    assert rejected.errors == ("Gold Patch lines 2 are below 3",)


def test_f2p_hideability_is_part_of_quality_acceptance() -> None:
    sources = {
        "tests/test_api.py": "def test_hidden():\n    assert True\n",
    }
    index = RepositoryIndex.from_sources(sources)
    quality = QualityReport(
        accepted=True,
        fail_to_pass=("tests/test_api.py::test_hidden#0",),
        pass_to_pass=("tests/test_api.py::test_visible#0",),
        test_exit_code=1,
    )
    paths = {
        "tests/test_api.py::test_hidden#0": "tests/test_api.py",
        "tests/test_api.py::test_visible#0": "tests/test_api.py",
    }

    accepted = apply_f2p_hideability_gate(
        quality,
        index=index,
        test_paths=paths,
        test_source_paths={"tests/test_api.py": "tests/test_api.py"},
    )
    assert accepted == quality

    unmappable = quality.model_copy(
        update={"fail_to_pass": ("tests/test_api.py::test_generated#0",)}
    )
    rejected = apply_f2p_hideability_gate(
        unmappable,
        index=index,
        test_paths={"tests/test_api.py::test_generated#0": "tests/test_api.py"},
        test_source_paths={"tests/test_api.py": "tests/test_api.py"},
    )
    assert not rejected.accepted
    assert rejected.errors == (
        "F2P hideability failed: "
        "F2P node IDs could not be mapped to removable test declarations: "
        "['tests/test_api.py::test_generated#0']",
    )

    fallback = reapply_gold_patch_tier_gates(
        rejected.model_copy(
            update={
                "errors": ("Gold Patch lines 2 are below 3", *rejected.errors),
                "gold_patch_lines": 2,
            }
        ),
        settings=FeatureBenchSettings(
            gold_patch_line_minimum=2,
            gold_patch_line_preferred_minimum=3,
            gold_patch_line_maximum=4,
        ),
        tier="fallback",
    )
    assert not fallback.accepted
    assert fallback.errors == rejected.errors

    behaviorally_rejected = unmappable.model_copy(
        update={"accepted": False, "errors": ("P2P count 0 is below 10",)}
    )
    assert (
        apply_f2p_hideability_gate(
            behaviorally_rejected,
            index=index,
            test_paths={"tests/test_api.py::test_generated#0": "tests/test_api.py"},
            test_source_paths={"tests/test_api.py": "tests/test_api.py"},
        )
        == behaviorally_rejected
    )


def test_gold_patch_scale_gate_uses_the_plan_tier_minimum() -> None:
    quality = QualityReport(
        accepted=True,
        fail_to_pass=("tests/test_a.py::test_a",),
        pass_to_pass=("tests/test_b.py::test_b",),
        test_exit_code=1,
    )
    patch = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -0,0 +1,2 @@
+one
+two
"""
    settings = FeatureBenchSettings(
        gold_patch_line_minimum=2,
        gold_patch_line_preferred_minimum=3,
        gold_patch_line_maximum=4,
    )

    preferred = apply_gold_patch_gates(
        quality,
        gold_patch=patch,
        settings=settings,
        tier="preferred",
    )
    fallback = apply_gold_patch_gates(
        quality,
        gold_patch=patch,
        settings=settings,
        tier="fallback",
    )

    assert not preferred.accepted
    assert preferred.errors == ("Gold Patch lines 2 are below 3",)
    assert fallback.accepted
    assert (
        reapply_gold_patch_tier_gates(
            preferred,
            settings=settings,
            tier="fallback",
        )
        == fallback
    )

    behavioral_failure = preferred.model_copy(
        update={
            "errors": (*preferred.errors, "P2P count 0 is below 1"),
        }
    )
    downgraded = reapply_gold_patch_tier_gates(
        behavioral_failure,
        settings=settings,
        tier="fallback",
    )
    assert not downgraded.accepted
    assert downgraded.errors == ("P2P count 0 is below 1",)

    oversized_fallback = apply_gold_patch_gates(
        quality,
        gold_patch=patch + "+three\n",
        settings=settings,
        tier="fallback",
    )
    assert not oversized_fallback.accepted
    assert oversized_fallback.errors == ("Gold Patch lines 3 exceed 2",)


def test_gold_patch_planning_falls_back_only_after_proven_infeasibility() -> None:
    generator = FeatureBenchGeneratorHarness()
    fallback_plan = cast(MaskPlan, object())
    calls: list[tuple[int, int, GoldPatchTier]] = []

    def create_for_interval(
        minimum: int,
        maximum: int,
        tier: GoldPatchTier,
    ) -> MaskPlan:
        calls.append((minimum, maximum, tier))
        if tier == "preferred":
            raise MaskIntervalInfeasible("preferred interval is infeasible")
        return fallback_plan

    result = generator._plan_gold_patch_tiers(  # pyright: ignore[reportPrivateUsage]
        create_for_interval
    )

    assert result is fallback_plan
    assert calls == [(600, 1000, "preferred"), (200, 599, "fallback")]
    calls.clear()

    def fail_without_proof(
        minimum: int,
        maximum: int,
        tier: GoldPatchTier,
    ) -> MaskPlan:
        calls.append((minimum, maximum, tier))
        raise CandidateRejected("optimizer timed out")

    with pytest.raises(CandidateRejected, match="timed out"):
        generator._plan_gold_patch_tiers(  # pyright: ignore[reportPrivateUsage]
            fail_without_proof
        )
    assert calls == [(600, 1000, "preferred")]

    calls.clear()

    def prove_both_infeasible(
        minimum: int,
        maximum: int,
        tier: GoldPatchTier,
    ) -> MaskPlan:
        calls.append((minimum, maximum, tier))
        raise MaskIntervalInfeasible(f"{tier} proof")

    with pytest.raises(
        MaskIntervalInfeasible,
        match="preferred and fallback Gold Patch intervals are infeasible",
    ):
        generator._plan_gold_patch_tiers(  # pyright: ignore[reportPrivateUsage]
            prove_both_infeasible
        )
    assert calls == [(600, 1000, "preferred"), (200, 599, "fallback")]


def test_final_metrics_measure_unique_tests_and_real_top_patch_coverage() -> None:
    index = RepositoryIndex.from_sources(
        {"pkg/core.py": "def feature():\n    return 1\n\ndef helper():\n    return 2\n"}
    )
    top = index.by_qualname[("pkg/core.py", "feature")].model_copy(update={"classification": "top"})
    specific = index.by_qualname[("pkg/core.py", "helper")].model_copy(
        update={"classification": "specific"}
    )
    plan = MaskPlan(
        seed_test_file="tests/test_core.py",
        top_objects=(top,),
        specific_objects=(specific,),
    )
    patch = """diff --git b/pkg/core.py a/pkg/core.py
--- b/pkg/core.py
+++ a/pkg/core.py
@@ -1,5 +1,5 @@
-
-
+def feature():
+    return 1

-
-
+def helper():
+    return 2
"""
    metrics = build_featurebench_metrics(
        gold_patch=patch,
        quality=QualityReport(
            accepted=True,
            fail_to_pass=("tests/test_core.py::test_feature",),
            pass_to_pass=("tests/test_other.py::test_other",),
            gold_patch_lines=4,
            gold_patch_files=1,
            test_exit_code=1,
        ),
        plan=plan,
    )

    assert metrics.model_dump(mode="json") == {
        "gold_patch": {"lines": 4, "files": 1},
        "tests": {"f2p": 1, "p2p": 1},
        "top": {"objects": 1, "gold_patch_lines": 2, "gold_patch_ratio": 0.5},
    }


def test_trace_test_id_ranges_restore_exact_evidence_without_duplication() -> None:
    test_ids = tuple(f"tests/test_api.py::test_case[{index}]" for index in range(8))
    assert decode_trace_test_id_ranges([[0, 3], [5, 8]], test_ids) == (
        *test_ids[:3],
        *test_ids[5:],
    )
    with pytest.raises(ValueError, match="sorted, disjoint, and non-adjacent"):
        decode_trace_test_id_ranges([[0, 3], [3, 5]], test_ids)
    with pytest.raises(ValueError, match="invalid trace test ID range"):
        decode_trace_test_id_ranges([[0, 9]], test_ids)
