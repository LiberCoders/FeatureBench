"""FeatureBench task generation from a complete validated environment record."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from codeworld.adapters.base import GeneratorContext, QualityGateResult
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.diagnostics import FeatureBenchDiagnostics
from codeworld.adapters.featurebench.finalization import assemble_task_record
from codeworld.adapters.featurebench.indexing import INDEXED_SOURCE_SUFFIXES, RepositoryIndex
from codeworld.adapters.featurebench.issue.agent import author_issue
from codeworld.adapters.featurebench.masking import (
    FeatureSlicePlanner,
    JointOptimizationOptions,
    MaskExecutor,
    MaskIntervalInfeasible,
    SeedFeasibilityProfile,
    hide_test_points,
)
from codeworld.adapters.featurebench.models import (
    GoldPatchTier,
    MaskPlan,
    QualityReport,
    SourceObject,
    TestReport,
    TraceBundle,
)
from codeworld.adapters.featurebench.p2p import TracePointIndex
from codeworld.adapters.featurebench.prompting import build_interfaces, render_prompt
from codeworld.adapters.featurebench.quality import (
    apply_f2p_hideability_gate,
    apply_gold_patch_gates,
    assess_quality,
)
from codeworld.adapters.featurebench.repair.agent import run_mask_repair_agent
from codeworld.adapters.featurebench.repair.rollback import DeterministicMaskRollback
from codeworld.adapters.featurebench.repair.service import MaskRepairService
from codeworld.adapters.featurebench.repair.workspace import MaskRepairWorkspace
from codeworld.adapters.featurebench.resources import FeatureBenchRunResources
from codeworld.adapters.featurebench.runtime import FeatureBenchTaskRuntime
from codeworld.adapters.featurebench.testing import (
    PythonTracer,
    TraceStatistics,
    collection_trace_from_bundles,
)
from codeworld.agents.factory import AgentFactory
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner import EnvironmentRunner
from codeworld.environment.runner_contract import CollectedTestPoint, RunnerResponse
from codeworld.errors import (
    CandidateRejected,
    CommandTimeoutError,
    ContractError,
    FactoryError,
    InfrastructureError,
    NoWorkSelected,
    RunnerTimeoutError,
)
from codeworld.events import Event
from codeworld.models import TaskRecord
from codeworld.runtime.base import ContainerSessionProtocol

_SEED_LOCAL_ERRORS = (FactoryError,)

_P2P_PROBE_INITIAL_TEST_POINTS = 32
_P2P_PROBE_INITIAL_TEST_FILES = 8
_P2P_PROBE_INITIAL_TIMEOUT_SECONDS = 30.0
_P2P_PROBE_TEST_POINTS = 128
_P2P_PROBE_TEST_FILES = 64
_P2P_PROBE_TIMEOUT_SECONDS = 90.0
_P2P_RUNTIME_TARGET_RATIO = 0.85


@dataclass(frozen=True, slots=True)
class _TraceFileResult:
    test_file: str
    requested_test_ids: tuple[str, ...]
    successful_test_ids: tuple[str, ...]
    failed_test_ids: tuple[tuple[str, str], ...]
    bundle: TraceBundle | None
    batch_error: str | None
    statistics: tuple[tuple[str, TraceStatistics], ...]
    budget_exhausted: bool = False

    @property
    def complete(self) -> bool:
        return self.successful_test_ids == self.requested_test_ids

    @property
    def can_enter_top(self) -> bool:
        """A file needs one real dynamic trace, not an arbitrary test-count floor."""
        return bool(self.successful_test_ids) and self.bundle is not None


@dataclass(frozen=True, slots=True)
class _SeedPipelineFailure:
    seed_test_file: str
    error: Exception


@dataclass(frozen=True, slots=True)
class _EnvironmentCompatibilityResult:
    status: str
    pytest_version: str
    smoke_test_file: str
    detail: str | None = None
    runner_probe_status: str = "not_run"
    estimated_test_capacity: int | None = None
    runner_probe_test_points: int = 0
    runner_probe_test_files: int = 0
    runner_probe_duration_seconds: float | None = None
    runner_probe_detail: str | None = None


@dataclass(frozen=True, slots=True)
class _RunnerCapacityResult:
    status: str
    estimated_test_capacity: int | None
    test_points: int
    test_files: int
    duration_seconds: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _CleanBaselineBackoffEvent:
    phase: Literal[
        "started",
        "replan_failed",
        "replanned",
        "retry_completed",
        "retry_failed",
    ]
    maximum_p2p_test_points: int
    plan: MaskPlan | None = None
    error: BaseException | None = None


_SEED_WORKER_DONE = object()


class FeatureBenchGenerator:
    """Generate self-contained FeatureBench tasks; never prepare a solution rollout."""

    def __init__(
        self,
        settings: FeatureBenchSettings | None = None,
        *,
        runtime: FeatureBenchTaskRuntime | None = None,
        resources: FeatureBenchRunResources | None = None,
    ) -> None:
        self.settings = settings or FeatureBenchSettings()
        self.runtime = runtime or FeatureBenchTaskRuntime()
        self.resources = resources or FeatureBenchRunResources.from_settings(self.settings)
        self.top_agents = AgentFactory(
            self.settings.agent.factory_settings(self.settings.agent.top),
            run_semaphore=self.resources.agent_slots,
        )
        self.repair_agents = AgentFactory(
            self.settings.agent.factory_settings(self.settings.agent.repair),
            run_semaphore=self.resources.agent_slots,
        )
        self.issue_author_agents = AgentFactory(
            self.settings.agent.factory_settings(self.settings.agent.issue_author),
            run_semaphore=self.resources.agent_slots,
        )
        self.planner = FeatureSlicePlanner(
            top_max_depth=self.settings.top_max_depth,
            top_batch_size=self.settings.top_batch_size,
            minimum_code_lines=self.settings.gold_patch_line_minimum,
            maximum_code_lines=self.settings.gold_patch_line_maximum,
            random_seed=self.settings.random_seed,
            critical_magic_methods=frozenset(self.settings.critical_magic_methods),
        )
        self.diagnostics: FeatureBenchDiagnostics | None = None
        self._candidate_completions: dict[str, asyncio.Future[None]] = {}

    async def run_started(
        self,
        environment: EnvironmentRecord,
        context: GeneratorContext,
    ) -> None:
        if context.output_dir is None:
            raise ContractError("FeatureBench diagnostics require the Generate output directory")
        self.diagnostics = FeatureBenchDiagnostics(
            context.output_dir,
            run_id=context.run_id,
            environment=environment,
            settings=self.settings,
        )

    async def candidate_checked(
        self,
        task: TaskRecord,
        result: QualityGateResult,
        context: GeneratorContext,
    ) -> None:
        del context
        try:
            if self.diagnostics is None:
                raise ContractError("FeatureBench diagnostics were not initialized")
            seed = self.diagnostics.seed_for_task(task.task_id)
            if seed is None:
                raise ContractError(f"missing FeatureBench diagnostics for task {task.task_id}")
            seed.record_quality_gate(result)
            if not result.passed:
                seed.finalize(
                    "rejected",
                    phase="gold_closure",
                    error_type="QualityGateRejected",
                    error_text=result.message or "final Gold closure rejected the candidate",
                )
                return
            payload_metrics = task.adapter.get("metrics")
            metrics = (
                cast(dict[str, object], payload_metrics)
                if isinstance(payload_metrics, dict)
                else dict(result.metrics)
            )
            seed.retain_success_summary(task, metrics)
            seed.finalize("succeeded", phase="completed")
            trajectory = self.settings.agent.trajectory
            if trajectory.enabled and trajectory.retention == "failures":
                seed.remove_success_trajectories(trajectory.directory)
            if self.settings.audit.retention == "failures":
                seed.compact_success_audit()
        finally:
            self._complete_candidate(task.task_id)

    async def candidate_failed(
        self,
        task: TaskRecord,
        error: BaseException,
        context: GeneratorContext,
    ) -> None:
        del context
        try:
            if self.diagnostics is None:
                raise ContractError("FeatureBench diagnostics were not initialized")
            seed = self.diagnostics.seed_for_task(task.task_id)
            if seed is None:
                raise ContractError(f"missing FeatureBench diagnostics for task {task.task_id}")
            seed.record_quality_gate_error(error)
            seed.finalize("failed", phase="gold_closure", error=error)
        finally:
            self._complete_candidate(task.task_id)

    async def run_finished(
        self,
        context: GeneratorContext,
        error: BaseException | None,
    ) -> None:
        del context
        if self.diagnostics is not None:
            self.diagnostics.finish(error)

    def _await_candidate(self, task_id: str) -> asyncio.Future[None]:
        if task_id in self._candidate_completions:
            raise ContractError(f"duplicate in-flight FeatureBench task ID: {task_id}")
        completion = asyncio.get_running_loop().create_future()
        self._candidate_completions[task_id] = completion
        return completion

    def _complete_candidate(self, task_id: str) -> None:
        completion = self._candidate_completions.get(task_id)
        if completion is not None and not completion.done():
            completion.set_result(None)

    @staticmethod
    async def _finalize_seed_error(
        *,
        context: GeneratorContext,
        diagnostics: FeatureBenchDiagnostics,
        seed_test_file: str,
        phase: str,
        event_prefix: str,
        error: Exception,
    ) -> None:
        status = "rejected" if isinstance(error, CandidateRejected) else "failed"
        event_data: dict[str, object] = {
            "seed_test_file": seed_test_file,
            "error_type": type(error).__name__,
        }
        if isinstance(error, MaskIntervalInfeasible) and error.diagnostics:
            event_data["constraint_diagnostics"] = [item.as_dict() for item in error.diagnostics]
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind=f"{event_prefix}_{status}",
                stage="featurebench",
                message=str(error) or repr(error),
                data=event_data,
            )
        )
        if status == "rejected":
            await context.progress.candidate_rejected(current=seed_test_file)
        else:
            await context.progress.candidate_failed(current=seed_test_file)
        diagnostics.seed(seed_test_file).finalize(status, phase=phase, error=error)

    @staticmethod
    def _ensure_supported_environment(environment: EnvironmentRecord) -> None:
        """Accept CPU-only execution, including optional-GPU repositories on their CPU path."""
        if environment.language != "python":
            raise ContractError(
                f"FeatureBench adapter requires a Python environment, got {environment.language}"
            )
        if environment.gpu_requirement == "required" or environment.runtime.resources.gpus:
            raise ContractError(
                "FeatureBench adapter only produces tasks whose complete generation and "
                "verification path runs without GPU resources"
            )

    @staticmethod
    def _runner_probe_test_ids(
        test_ids_by_file: Mapping[str, tuple[str, ...]],
        *,
        maximum_test_points: int,
        maximum_test_files: int,
    ) -> tuple[str, ...]:
        """Select a deterministic file-stratified Runner throughput sample."""
        paths = tuple(
            sorted(
                (path for path, test_ids in test_ids_by_file.items() if test_ids),
                key=lambda path: (-len(test_ids_by_file[path]), path),
            )[:maximum_test_files]
        )
        selected: list[str] = []
        ordinal = 0
        while len(selected) < maximum_test_points:
            added = False
            for path in paths:
                test_ids = test_ids_by_file[path]
                if ordinal < len(test_ids):
                    selected.append(test_ids[ordinal])
                    added = True
                    if len(selected) == maximum_test_points:
                        break
            if not added:
                break
            ordinal += 1
        return tuple(selected)

    def _effective_p2p_limit(
        self,
        *,
        seed_test_points: int,
        test_capacity: int | None,
    ) -> int | None:
        """Convert one complete-batch capacity into a Seed-local P2P bound."""
        limit = self.settings.maximum_p2p_test_points
        if test_capacity is not None:
            capacity_limit = max(
                self.settings.minimum_remaining_p2p_test_points,
                test_capacity - seed_test_points,
            )
            limit = capacity_limit if limit is None else min(limit, capacity_limit)
        return limit

    async def _probe_runner_capacity(
        self,
        *,
        test_ids_by_file: Mapping[str, tuple[str, ...]],
        run_batch: Callable[[tuple[str, ...], float], Awaitable[RunnerResponse]],
        timeout_seconds: float,
    ) -> _RunnerCapacityResult:
        """Estimate one Environment's bounded full-batch capacity without rejecting it."""
        if timeout_seconds <= 0:
            return _RunnerCapacityResult(
                status="not_run",
                estimated_test_capacity=None,
                test_points=0,
                test_files=0,
                detail="compatibility deadline was exhausted before the Runner probe",
            )
        total_test_points = sum(len(test_ids) for test_ids in test_ids_by_file.values())
        if total_test_points == 0:
            return _RunnerCapacityResult(
                status="not_run",
                estimated_test_capacity=None,
                test_points=0,
                test_files=0,
                detail="no verified test points were available for the Runner probe",
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        target_seconds = self.settings.full_test_timeout_seconds * _P2P_RUNTIME_TARGET_RATIO

        def estimate(test_points: int, duration_seconds: float) -> int:
            projected = int(test_points * target_seconds / max(duration_seconds, 0.001))
            return max(1, min(total_test_points, projected))

        def file_count(test_ids: tuple[str, ...]) -> int:
            selected = set(test_ids)
            return sum(
                bool(selected.intersection(file_ids)) for file_ids in test_ids_by_file.values()
            )

        probe_file_cap = self.settings.maximum_selected_p2p_test_files or _P2P_PROBE_TEST_FILES

        async def run_stage(
            test_ids: tuple[str, ...],
            maximum_seconds: float,
        ) -> tuple[RunnerResponse, float, float]:
            remaining = deadline - loop.time()
            stage_timeout = min(maximum_seconds, remaining)
            if stage_timeout <= 0:
                raise TimeoutError("Runner capacity probe exhausted its shared deadline")
            started = loop.time()
            response = await run_batch(test_ids, stage_timeout)
            return response, loop.time() - started, stage_timeout

        initial = self._runner_probe_test_ids(
            test_ids_by_file,
            maximum_test_points=_P2P_PROBE_INITIAL_TEST_POINTS,
            maximum_test_files=min(_P2P_PROBE_INITIAL_TEST_FILES, probe_file_cap),
        )
        initial_files = file_count(initial)
        try:
            initial_response, initial_duration, _ = await run_stage(
                initial,
                _P2P_PROBE_INITIAL_TIMEOUT_SECONDS,
            )
        except RunnerTimeoutError as exc:
            return _RunnerCapacityResult(
                status="timed_out",
                estimated_test_capacity=0,
                test_points=len(initial),
                test_files=initial_files,
                detail=str(exc)[-5000:] or repr(exc),
            )
        except TimeoutError as exc:
            return _RunnerCapacityResult(
                status="inconclusive",
                estimated_test_capacity=None,
                test_points=len(initial),
                test_files=initial_files,
                detail=str(exc)[-5000:] or repr(exc),
            )
        except FactoryError as exc:
            return _RunnerCapacityResult(
                status="inconclusive",
                estimated_test_capacity=None,
                test_points=len(initial),
                test_files=initial_files,
                detail=str(exc)[-5000:] or repr(exc),
            )
        if initial_response.errors or initial_response.exit_code not in (0, 1):
            detail = "; ".join(initial_response.errors) or (
                f"Runner exited with code {initial_response.exit_code}"
            )
            return _RunnerCapacityResult(
                status="inconclusive",
                estimated_test_capacity=None,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
                detail=detail[-5000:],
            )
        initial_capacity = estimate(len(initial), initial_duration)

        expanded = self._runner_probe_test_ids(
            test_ids_by_file,
            maximum_test_points=_P2P_PROBE_TEST_POINTS,
            maximum_test_files=min(_P2P_PROBE_TEST_FILES, probe_file_cap),
        )
        expanded_files = file_count(expanded)
        if expanded == initial:
            return _RunnerCapacityResult(
                status="passed",
                estimated_test_capacity=initial_capacity,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
            )
        remaining = deadline - loop.time()
        if remaining <= 0:
            return _RunnerCapacityResult(
                status="partial",
                estimated_test_capacity=initial_capacity,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
                detail="Runner capacity probe exhausted its shared deadline after the first stage",
            )
        try:
            expanded_response, expanded_duration, _ = await run_stage(
                expanded,
                _P2P_PROBE_TIMEOUT_SECONDS,
            )
        except RunnerTimeoutError as exc:
            # The expanded sample is censored: it took longer than the stage
            # deadline, so extrapolating from that deadline would be an
            # optimistic upper bound. Keep the estimate within the attempted
            # sample instead of pretending that unobserved throughput passed.
            censored_capacity = min(initial_capacity, len(expanded))
            return _RunnerCapacityResult(
                status="partial_timeout",
                estimated_test_capacity=censored_capacity,
                test_points=len(expanded),
                test_files=expanded_files,
                duration_seconds=initial_duration,
                detail=str(exc)[-5000:] or repr(exc),
            )
        except TimeoutError as exc:
            return _RunnerCapacityResult(
                status="partial",
                estimated_test_capacity=initial_capacity,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
                detail=str(exc)[-5000:] or repr(exc),
            )
        except FactoryError as exc:
            return _RunnerCapacityResult(
                status="partial",
                estimated_test_capacity=initial_capacity,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
                detail=str(exc)[-5000:] or repr(exc),
            )
        if expanded_response.errors or expanded_response.exit_code not in (0, 1):
            detail = "; ".join(expanded_response.errors) or (
                f"Runner exited with code {expanded_response.exit_code}"
            )
            return _RunnerCapacityResult(
                status="partial",
                estimated_test_capacity=initial_capacity,
                test_points=len(initial),
                test_files=initial_files,
                duration_seconds=initial_duration,
                detail=detail[-5000:],
            )
        return _RunnerCapacityResult(
            status="passed",
            estimated_test_capacity=estimate(len(expanded), expanded_duration),
            test_points=len(expanded),
            test_files=expanded_files,
            duration_seconds=expanded_duration,
        )

    async def _check_environment_compatibility(
        self,
        *,
        session: ContainerSessionProtocol,
        test_file: str,
        test_ids: tuple[str, ...],
        test_ids_by_file: Mapping[str, tuple[str, ...]] | None = None,
    ) -> _EnvironmentCompatibilityResult:
        """Probe the exact pytest/Trace path and bounded Runner throughput."""
        launcher, _configured_args = PythonTracer.trace_command(self.settings.test_command)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.compatibility_timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                import_result = await session.exec(
                    (
                        *launcher,
                        "-c",
                        "import pytest; print(getattr(pytest, '__version__', 'unknown'))",
                    ),
                    activated=True,
                    timeout_seconds=max(0.001, deadline - loop.time()),
                    max_output_chars=20_000,
                    check=False,
                )
        except (CommandTimeoutError, TimeoutError) as exc:
            return _EnvironmentCompatibilityResult(
                status="inconclusive",
                pytest_version="unknown",
                smoke_test_file=test_file,
                detail=str(exc)[-5000:] or repr(exc),
                runner_probe_status="not_run_compatibility_timeout",
                estimated_test_capacity=0,
            )
        if import_result.exit_code != 0:
            detail = import_result.combined_output[-5000:] or "pytest import failed"
            raise CandidateRejected(
                "FeatureBench compatibility precheck cannot import pytest through the "
                f"configured launch path: {detail}"
            )
        pytest_version = import_result.stdout.strip().splitlines()
        version = pytest_version[-1] if pytest_version else "unknown"

        tracer = PythonTracer(
            session,
            maximum_report_bytes=self.settings.maximum_trace_report_bytes,
        )
        remaining_seconds = deadline - loop.time()
        if remaining_seconds <= 0:
            return _EnvironmentCompatibilityResult(
                status="inconclusive",
                pytest_version=version,
                smoke_test_file=test_file,
                detail="FeatureBench compatibility precheck exhausted its shared deadline",
                runner_probe_status="not_run_compatibility_timeout",
                estimated_test_capacity=0,
            )
        try:
            async with asyncio.timeout_at(deadline):
                await tracer.trace(
                    test_command=(*self.settings.test_command, "--collect-only"),
                    test_file=test_file,
                    test_ids=test_ids,
                    timeout_seconds=remaining_seconds,
                )
        except (CommandTimeoutError, TimeoutError) as exc:
            return _EnvironmentCompatibilityResult(
                status="inconclusive",
                pytest_version=version,
                smoke_test_file=test_file,
                detail=str(exc)[-5000:] or repr(exc),
                runner_probe_status="not_run_compatibility_timeout",
                estimated_test_capacity=0,
            )
        except ContractError as exc:
            detail = str(exc)[-5000:] or repr(exc)
            return _EnvironmentCompatibilityResult(
                status="inconclusive",
                pytest_version=version,
                smoke_test_file=test_file,
                detail=detail,
            )
        capacity = _RunnerCapacityResult(
            status="not_run",
            estimated_test_capacity=None,
            test_points=0,
            test_files=0,
        )
        if test_ids_by_file is not None:
            runner = EnvironmentRunner(session)

            async def run_probe(
                selected_test_ids: tuple[str, ...],
                stage_timeout_seconds: float,
            ) -> RunnerResponse:
                stage_deadline = loop.time() + stage_timeout_seconds
                try:
                    async with asyncio.timeout_at(stage_deadline):
                        await session.clean_workspace()
                except TimeoutError as exc:
                    raise CommandTimeoutError(
                        "Runner capacity probe workspace cleanup exhausted its stage deadline",
                        timeout_seconds=stage_timeout_seconds,
                    ) from exc
                return await runner.run(
                    selected_test_ids,
                    timeout_seconds=stage_timeout_seconds,
                    deadline=stage_deadline,
                )

            capacity = await self._probe_runner_capacity(
                test_ids_by_file=test_ids_by_file,
                run_batch=run_probe,
                timeout_seconds=max(0, deadline - loop.time()),
            )
        return _EnvironmentCompatibilityResult(
            status="passed",
            pytest_version=version,
            smoke_test_file=test_file,
            runner_probe_status=capacity.status,
            estimated_test_capacity=capacity.estimated_test_capacity,
            runner_probe_test_points=capacity.test_points,
            runner_probe_test_files=capacity.test_files,
            runner_probe_duration_seconds=capacity.duration_seconds,
            runner_probe_detail=capacity.detail,
        )

    @staticmethod
    def _order_seed_profiles(
        profiles: tuple[SeedFeasibilityProfile, ...],
        ordering: str,
    ) -> tuple[SeedFeasibilityProfile, ...]:
        runnable = tuple(profile for profile in profiles if not profile.rejected)
        if ordering == "easy_first":
            return tuple(sorted(runnable, key=SeedFeasibilityProfile.easy_first_key))
        if ordering == "path":
            return tuple(sorted(runnable, key=lambda profile: profile.test_file))
        raise ValueError(f"unknown FeatureBench Seed ordering: {ordering}")

    def _plan_gold_patch_tiers(
        self,
        create_for_interval: Callable[[int, int, GoldPatchTier], MaskPlan],
    ) -> MaskPlan:
        """Use the fallback interval only after the preferred tier is proven infeasible."""
        try:
            return create_for_interval(
                self.settings.gold_patch_line_preferred_minimum,
                self.settings.gold_patch_line_maximum,
                "preferred",
            )
        except MaskIntervalInfeasible as preferred_error:
            if (
                self.settings.gold_patch_line_minimum
                >= self.settings.gold_patch_line_preferred_minimum
            ):
                raise
            try:
                return create_for_interval(
                    self.settings.gold_patch_line_minimum,
                    self.settings.gold_patch_line_preferred_minimum - 1,
                    "fallback",
                )
            except MaskIntervalInfeasible as fallback_error:
                raise MaskIntervalInfeasible(
                    "preferred and fallback Gold Patch intervals are infeasible; "
                    f"preferred: {preferred_error}; fallback: {fallback_error}",
                    diagnostics=(
                        *preferred_error.diagnostics,
                        *fallback_error.diagnostics,
                    ),
                ) from fallback_error

    @staticmethod
    async def _run_trace_inventory(
        trace_files: tuple[str, ...],
        *,
        concurrency: int,
        trace_file: Callable[[str], Awaitable[_TraceFileResult]],
        semaphore: asyncio.Semaphore | None = None,
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[_TraceFileResult, ...]:
        """Finish one Environment's Trace barrier within invocation-wide capacity."""
        if concurrency < 1:
            raise ValueError("Trace concurrency must be positive")
        source = iter(enumerate(trace_files))
        results: list[_TraceFileResult | None] = [None] * len(trace_files)

        async def worker() -> None:
            while True:
                try:
                    index, test_file = next(source)
                except StopIteration:
                    return
                if semaphore is None:
                    result = await trace_file(test_file)
                else:
                    async with semaphore:
                        result = await trace_file(test_file)
                results[index] = result
                if progress is not None:
                    await progress(test_file)

        workers = tuple(
            asyncio.create_task(worker()) for _ in range(min(concurrency, len(trace_files)))
        )
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        return tuple(result for result in results if result is not None)

    async def _agent_progress(
        self,
        context: GeneratorContext,
        role: str,
        kind: str,
        data: dict[str, object],
    ) -> None:
        seed_test_file = data.get("seed_test_file")
        if (
            self.diagnostics is not None
            and isinstance(seed_test_file, str)
            and kind in {"agent_run_started", "agent_run_metrics"}
        ):
            self.diagnostics.seed(seed_test_file).record_agent_run(role, kind, data)
        if kind == "output_feedback":
            message = f"{role} Agent output retry {data.get('retry')}: {data.get('codes')}"
        elif kind == "context_compaction_retry":
            message = (
                f"{role} Agent forcing context compaction "
                f"{data.get('attempt')}/{data.get('maximum_attempts')} before retry"
            )
        elif kind == "model_request_retry":
            message = f"{role} Agent retrying a transient model request"
        elif kind == "agent_run_metrics":
            message = f"{role} Agent run {data.get('status')}"
        else:
            action = "started" if kind.endswith("_call") else "completed"
            tool = data.get("tool")
            message = f"{role} Agent {action}"
            if tool:
                message += f" tool {tool}"
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind=f"{role}_agent_{kind}",
                stage="featurebench",
                message=message,
                data={"role": role, **data},
            )
        )

    async def _rollback_progress(
        self,
        context: GeneratorContext,
        kind: str,
        data: dict[str, object],
    ) -> None:
        seed_test_file = data.get("seed_test_file")
        if self.diagnostics is not None and isinstance(seed_test_file, str):
            self.diagnostics.seed(seed_test_file).append_event(f"rollback_{kind}", **data)
        if kind == "started":
            message = "deterministic Mask rollback started"
        elif kind == "validation_started":
            message = f"rollback validation {data.get('validation')} started"
        elif kind == "failed":
            message = f"deterministic Mask rollback failed: {data.get('reason')}"
        else:
            state = "accepted" if data.get("accepted") else "rejected"
            message = f"rollback validation {data.get('validation')} {state}"
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind=f"mask_rollback_{kind}",
                stage="featurebench",
                message=message,
                data=data,
            )
        )

    @staticmethod
    async def _resolve_test_source_paths(
        *,
        session: ContainerSessionProtocol,
        runner_paths: tuple[str, ...],
        indexed_paths: frozenset[str],
        timeout_seconds: float,
    ) -> dict[str, str]:
        """Resolve Runner-visible paths through repository-owned symlinks."""
        script = r"""
set -euo pipefail
root=/testbed
for path do
    resolved=$(realpath -e -- "$root/$path") || {
        printf 'cannot resolve Runner test source: %s\n' "$path" >&2
        exit 1
    }
    case "$resolved" in
        "$root"/*) ;;
        *)
            printf 'Runner test source escapes /testbed: %s -> %s\n' "$path" "$resolved" >&2
            exit 1
            ;;
    esac
    printf '%s\t%s\n' "$path" "${resolved#"$root"/}"
done
""".strip()
        result = await session.run_script(
            script,
            arguments=runner_paths,
            timeout_seconds=timeout_seconds,
            max_output_chars=2_000_000,
            check=False,
        )
        if result.exit_code != 0:
            raise ContractError(
                "cannot resolve FeatureBench test source paths: " + result.combined_output[-5000:]
            )
        resolved_paths: dict[str, str] = {}
        for line in result.stdout.splitlines():
            runner_path, separator, source_path = line.partition("\t")
            if not separator or runner_path not in runner_paths:
                raise ContractError("invalid FeatureBench test-source path mapping")
            resolved_paths[runner_path] = source_path
        missing = set(runner_paths) - resolved_paths.keys()
        unsupported = {
            runner_path: source_path
            for runner_path, source_path in resolved_paths.items()
            if PurePosixPath(source_path).suffix not in INDEXED_SOURCE_SUFFIXES
        }
        unindexed = {
            runner_path: source_path
            for runner_path, source_path in resolved_paths.items()
            if runner_path not in unsupported and source_path not in indexed_paths
        }
        if missing or unindexed:
            details: list[str] = []
            if missing:
                details.append(f"missing mappings for {sorted(missing)[:5]}")
            if unindexed:
                details.append(f"resolved paths absent from source index: {unindexed}")
            raise ContractError(
                "FeatureBench test sources are not indexable: " + "; ".join(details)
            )
        return {
            runner_path: source_path
            for runner_path, source_path in resolved_paths.items()
            if runner_path not in unsupported
        }

    async def _isolated_candidate_report(
        self,
        plan: MaskPlan,
        seed_test_ids: tuple[str, ...],
        test_paths: Mapping[str, str],
        *,
        run_batch: Callable[[tuple[str, ...], float], Awaitable[RunnerResponse]],
        timeout_seconds: float,
        diagnostic_fallback: bool = True,
        raise_complete_batch_timeout: bool = False,
    ) -> tuple[tuple[str, ...], TestReport]:
        """Run exact points together, using partitions only as rejected diagnostics."""
        if timeout_seconds <= 0:
            raise ValueError("FeatureBench validation timeout must be positive")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        budget_error = (
            f"FeatureBench validation shared deadline exceeded after {timeout_seconds:g} seconds"
        )
        selected_baseline = tuple(dict.fromkeys((*seed_test_ids, *plan.p2p_test_ids)))
        selected_by_path: dict[str, list[str]] = {}
        for test_id in selected_baseline:
            try:
                path = test_paths[test_id]
            except KeyError as exc:
                raise ContractError(
                    f"selected FeatureBench test point has no validated environment path: {test_id}"
                ) from exc
            selected_by_path.setdefault(path, []).append(test_id)
        outcomes: dict[str, str] = {}
        collection_errors: list[str] = []
        output_sections: list[str] = []
        exit_code = 0
        complete_batch_failure: str | None = None
        complete_batch_operational = False

        def consume(path_label: str, response: RunnerResponse) -> None:
            nonlocal exit_code
            outcomes.update(
                {
                    result.id: "failed" if result.outcome == "error" else result.outcome
                    for result in response.results
                }
            )
            runtime_errors = response.errors
            collection_errors.extend(f"{path_label}: {error}" for error in runtime_errors)
            # Repair diagnostics must retain the signal that made the candidate
            # operationally invalid. Expected test failures (exit 1) define F2P;
            # their raw pytest output and successful P2P output only displace the
            # collection/runtime error that the Agent needs to repair.
            if response.output and (runtime_errors or response.exit_code not in (0, 1)):
                output_sections.append(f"[{path_label}]\n{response.output}")
            if runtime_errors or response.exit_code not in (0, 1):
                exit_code = 2
            elif response.exit_code == 1 and exit_code == 0:
                exit_code = 1

        async def run_partition(
            groups: tuple[tuple[str, tuple[str, ...]], ...],
            *,
            complete_batch: bool = False,
        ) -> None:
            """Run one partition tree without treating diagnostic splits as acceptance."""
            nonlocal complete_batch_failure, complete_batch_operational, exit_code
            test_ids = tuple(test_id for _, group_ids in groups for test_id in group_ids)
            path_label = groups[0][0] if len(groups) == 1 else f"{len(groups)} test files"
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(budget_error)
            try:
                async with asyncio.timeout_at(deadline):
                    response = await run_batch(test_ids, remaining)
            except FactoryError as exc:
                if (
                    complete_batch
                    and raise_complete_batch_timeout
                    and isinstance(exc, InfrastructureError)
                ):
                    raise
                if complete_batch:
                    complete_batch_failure = f"{type(exc).__name__}: {exc}"
                if len(groups) == 1 or not diagnostic_fallback:
                    message = f"{path_label}: {exc}"
                    collection_errors.append(message)
                    output_sections.append(message)
                    exit_code = 2
                    return
            else:
                operational = response.exit_code in (0, 1) and not response.errors
                if complete_batch:
                    complete_batch_operational = operational
                    if not operational:
                        detail = "; ".join(response.errors)
                        complete_batch_failure = detail or f"exit code {response.exit_code}"
                        if diagnostic_fallback and len(groups) > 1 and response.output:
                            output_sections.append(
                                "[complete required-test batch]\n" + response.output
                            )
                if operational or len(groups) == 1 or not diagnostic_fallback:
                    consume(path_label, response)
                    return

            midpoint = len(groups) // 2
            await run_partition(groups[:midpoint])
            await run_partition(groups[midpoint:])

        groups = tuple(
            (path, tuple(selected_ids)) for path, selected_ids in selected_by_path.items()
        )
        try:
            if groups:
                await run_partition(groups, complete_batch=True)
        except TimeoutError as exc:
            raise CommandTimeoutError(
                budget_error,
                timeout_seconds=timeout_seconds,
            ) from exc
        if groups and not complete_batch_operational:
            detail = f": {complete_batch_failure}" if complete_batch_failure else ""
            collection_errors.insert(
                0,
                "complete required-test batch did not finish operationally in one "
                f"Runner invocation{detail}",
            )
            exit_code = 2
        return selected_baseline, TestReport.model_validate(
            {
                "exit_code": exit_code,
                "collected": selected_baseline,
                "outcomes": outcomes,
                "collection_errors": tuple(collection_errors),
                "output": "\n\n".join(output_sections),
            }
        )

    def _normalize_clean_baseline(
        self,
        *,
        plan: MaskPlan,
        seed_test_ids: tuple[str, ...],
        selected_test_ids: tuple[str, ...],
        report: TestReport,
    ) -> tuple[MaskPlan, tuple[str, ...]]:
        """Keep only points that pass together before any production code is masked."""
        if report.collection_errors or report.exit_code not in (0, 1):
            raise CandidateRejected(
                "clean combined baseline did not finish operationally in one Runner invocation"
            )
        passed = set(report.passed)
        seed = tuple(test_id for test_id in seed_test_ids if test_id in passed)
        if not seed:
            raise CandidateRejected(
                f"clean combined baseline has no passing Seed points for {plan.seed_test_file}"
            )
        p2p = tuple(test_id for test_id in plan.p2p_test_ids if test_id in passed)
        if len(p2p) < self.settings.minimum_remaining_p2p_test_points:
            raise CandidateRejected(
                "clean combined baseline leaves only "
                f"{len(p2p)} P2P points, below "
                f"{self.settings.minimum_remaining_p2p_test_points}"
            )
        normalized = plan.model_copy(update={"p2p_test_ids": p2p})
        retained = set(seed) | set(p2p)
        baseline = tuple(test_id for test_id in selected_test_ids if test_id in retained)
        return normalized, baseline

    async def _refresh_plan_for_test_capacity(
        self,
        *,
        plan: MaskPlan,
        seed_test_points: int,
        test_capacity: int | None,
        replan_with_p2p_limit: Callable[[int], Awaitable[MaskPlan]],
    ) -> tuple[MaskPlan, int | None, int | None]:
        """Replan a concurrently stale P2P selection before it reaches Runner."""
        maximum_p2p = self._effective_p2p_limit(
            seed_test_points=seed_test_points,
            test_capacity=test_capacity,
        )
        if maximum_p2p is None or len(plan.p2p_test_ids) <= maximum_p2p:
            return plan, None, None
        previous_p2p = len(plan.p2p_test_ids)
        replanned = await replan_with_p2p_limit(maximum_p2p)
        if len(replanned.p2p_test_ids) > maximum_p2p:
            raise ContractError("capacity replan exceeded its P2P test-point limit")
        return replanned, previous_p2p, maximum_p2p

    async def _clean_baseline_with_backoff(
        self,
        *,
        plan: MaskPlan,
        seed_test_ids: tuple[str, ...],
        test_paths: Mapping[str, str],
        run_batch: Callable[[tuple[str, ...], float], Awaitable[RunnerResponse]],
        replan_with_p2p_limit: Callable[[int], Awaitable[MaskPlan]],
        observe_backoff: Callable[[_CleanBaselineBackoffEvent], Awaitable[None]] | None = None,
    ) -> tuple[MaskPlan, tuple[str, ...], TestReport, int | None]:
        """Run one complete clean batch and at most one complete P2P backoff."""

        async def run(selected_plan: MaskPlan) -> tuple[tuple[str, ...], TestReport]:
            return await self._isolated_candidate_report(
                selected_plan,
                seed_test_ids,
                test_paths,
                run_batch=run_batch,
                timeout_seconds=self.settings.validation_timeout_seconds,
                diagnostic_fallback=False,
                raise_complete_batch_timeout=True,
            )

        try:
            selected, report = await run(plan)
            return plan, selected, report, None
        except RunnerTimeoutError as initial_timeout:
            backoff_limit = max(
                self.settings.minimum_remaining_p2p_test_points,
                len(plan.p2p_test_ids) // 2,
            )
            if backoff_limit >= len(plan.p2p_test_ids):
                raise CandidateRejected(
                    "clean combined baseline timed out at the minimum P2P test-point limit"
                ) from initial_timeout
            if observe_backoff is not None:
                await observe_backoff(
                    _CleanBaselineBackoffEvent(
                        phase="started",
                        maximum_p2p_test_points=backoff_limit,
                        error=initial_timeout,
                    )
                )
            try:
                replanned = await replan_with_p2p_limit(backoff_limit)
            except BaseException as replan_error:
                if observe_backoff is not None:
                    await observe_backoff(
                        _CleanBaselineBackoffEvent(
                            phase="replan_failed",
                            maximum_p2p_test_points=backoff_limit,
                            error=replan_error,
                        )
                    )
                raise
            if observe_backoff is not None:
                await observe_backoff(
                    _CleanBaselineBackoffEvent(
                        phase="replanned",
                        maximum_p2p_test_points=backoff_limit,
                        plan=replanned,
                    )
                )
            try:
                selected, report = await run(replanned)
            except RunnerTimeoutError as retry_timeout:
                if observe_backoff is not None:
                    await observe_backoff(
                        _CleanBaselineBackoffEvent(
                            phase="retry_failed",
                            maximum_p2p_test_points=backoff_limit,
                            plan=replanned,
                            error=retry_timeout,
                        )
                    )
                raise CandidateRejected(
                    "clean combined baseline timed out after one P2P backoff"
                ) from retry_timeout
            except BaseException as retry_error:
                if observe_backoff is not None:
                    await observe_backoff(
                        _CleanBaselineBackoffEvent(
                            phase="retry_failed",
                            maximum_p2p_test_points=backoff_limit,
                            plan=replanned,
                            error=retry_error,
                        )
                    )
                raise
            if observe_backoff is not None:
                await observe_backoff(
                    _CleanBaselineBackoffEvent(
                        phase="retry_completed",
                        maximum_p2p_test_points=backoff_limit,
                        plan=replanned,
                    )
                )
            return replanned, selected, report, backoff_limit

    async def _run_candidate_batch_in_fresh_session(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        candidate_patch: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> RunnerResponse:
        """Replay one candidate, or a clean baseline, in a disposable container."""
        async with context.runtime.session(
            environment.image,
            runtime_spec=environment.runtime,
        ) as batch_session:
            await batch_session.clean_workspace()
            if candidate_patch and not await self.runtime.apply_patch(
                batch_session, candidate_patch
            ):
                raise ContractError("cannot replay FeatureBench candidate for isolated testing")
            return await EnvironmentRunner(batch_session).run(
                test_ids,
                timeout_seconds=timeout_seconds,
            )

    async def _trace_selection_in_fresh_session(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        test_file: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> tuple[TraceBundle, TraceStatistics]:
        """Trace one selection in a disposable container that cannot taint a retry."""
        async with context.runtime.session(
            environment.image,
            runtime_spec=environment.runtime,
        ) as trace_session:
            await trace_session.clean_workspace()
            tracer = PythonTracer(
                trace_session,
                maximum_report_bytes=self.settings.maximum_trace_report_bytes,
            )
            bundle = await tracer.trace(
                test_command=self.settings.test_command,
                test_file=test_file,
                test_ids=test_ids,
                timeout_seconds=timeout_seconds,
            )
            return bundle, tracer.statistics[(test_file, test_ids)]

    async def _trace_file(
        self,
        *,
        test_file: str,
        test_ids: tuple[str, ...],
        trace_selection: Callable[
            [tuple[str, ...], float], Awaitable[tuple[TraceBundle, TraceStatistics]]
        ],
        timeout_seconds: float,
    ) -> _TraceFileResult:
        """Trace one physical test file once within its hard wall-clock budget."""
        if timeout_seconds <= 0:
            raise ValueError("Trace file timeout must be positive")

        budget_error = f"dynamic trace file budget exhausted after {timeout_seconds:g} seconds"
        try:
            async with asyncio.timeout(timeout_seconds):
                bundle, batch_statistics = await trace_selection(test_ids, timeout_seconds)
        except TimeoutError:
            return _TraceFileResult(
                test_file=test_file,
                requested_test_ids=test_ids,
                successful_test_ids=(),
                failed_test_ids=tuple((test_id, budget_error) for test_id in test_ids),
                bundle=None,
                batch_error=budget_error,
                statistics=(),
                budget_exhausted=True,
            )
        except (CommandTimeoutError, ContractError) as batch_error:
            return _TraceFileResult(
                test_file=test_file,
                requested_test_ids=test_ids,
                successful_test_ids=(),
                failed_test_ids=tuple((test_id, str(batch_error)) for test_id in test_ids),
                bundle=None,
                batch_error=str(batch_error),
                statistics=(),
            )
        passed = set(bundle.passed_test_ids)
        return _TraceFileResult(
            test_file=test_file,
            requested_test_ids=test_ids,
            successful_test_ids=tuple(test_id for test_id in test_ids if test_id in passed),
            failed_test_ids=tuple(
                (test_id, "dynamic trace completed but the test point did not pass")
                for test_id in test_ids
                if test_id not in passed
            ),
            bundle=bundle,
            batch_error=None,
            statistics=((test_file, batch_statistics),),
        )

    async def generate(
        self, environment: EnvironmentRecord, context: GeneratorContext
    ) -> AsyncGenerator[TaskRecord, None]:
        self._ensure_supported_environment(environment)
        if self.diagnostics is None:
            raise ContractError("FeatureBench lifecycle did not initialize diagnostics")
        diagnostics = self.diagnostics
        try:
            all_verified_points = {test.id: test for test in environment.verified_test_points}
            all_test_ids_by_file = {
                path: tuple(
                    test_id for test_id, test in all_verified_points.items() if test.path == path
                )
                for path in sorted({test.path for test in all_verified_points.values()})
            }
            if not all_test_ids_by_file:
                raise CandidateRejected("environment has no verified test points to trace")
            await context.progress.start_phase(
                "inspect_repository",
                total=1,
                current=environment.repository,
            )
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="adapter_inventory",
                    stage="featurebench",
                    message=(
                        f"indexing the repository and inspecting {len(all_test_ids_by_file)} "
                        "physical verified-test batches"
                    ),
                    data={
                        "baseline_tests": len(all_verified_points),
                        "verified_test_files": len(all_test_ids_by_file),
                    },
                )
            )
            async with context.runtime.session(
                environment.image, runtime_spec=environment.runtime
            ) as session:
                smoke_test_file = min(
                    all_test_ids_by_file,
                    key=lambda path: (len(all_test_ids_by_file[path]), path),
                )
                try:
                    compatibility = await self._check_environment_compatibility(
                        session=session,
                        test_file=smoke_test_file,
                        test_ids=all_test_ids_by_file[smoke_test_file],
                        test_ids_by_file=all_test_ids_by_file,
                    )
                except CandidateRejected as exc:
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind="environment_compatibility_rejected",
                            stage="featurebench",
                            message=str(exc) or repr(exc),
                            data={
                                "smoke_test_file": smoke_test_file,
                                "verified_test_files": len(all_test_ids_by_file),
                                "error_type": type(exc).__name__,
                            },
                        )
                    )
                    raise
                await session.clean_workspace()
                await context.events.emit(
                    Event(
                        run_id=context.run_id,
                        kind="environment_compatibility_checked",
                        stage="featurebench",
                        message=(
                            "FeatureBench pytest/Trace compatibility precheck "
                            f"{compatibility.status}"
                        ),
                        data={
                            "status": compatibility.status,
                            "pytest_version": compatibility.pytest_version,
                            "smoke_test_file": compatibility.smoke_test_file,
                            "detail": compatibility.detail,
                            "timeout_seconds": self.settings.compatibility_timeout_seconds,
                            "runner_probe_status": compatibility.runner_probe_status,
                            "estimated_test_capacity": (compatibility.estimated_test_capacity),
                            "runner_probe_test_points": (compatibility.runner_probe_test_points),
                            "runner_probe_test_files": (compatibility.runner_probe_test_files),
                            "runner_probe_duration_seconds": (
                                compatibility.runner_probe_duration_seconds
                            ),
                            "runner_probe_detail": compatibility.runner_probe_detail,
                            "runner_target_ratio": _P2P_RUNTIME_TARGET_RATIO,
                        },
                    )
                )
                index = await RepositoryIndex.build(
                    session,
                    max_files=self.settings.maximum_index_files,
                    max_file_bytes=self.settings.maximum_index_file_bytes,
                    max_total_bytes=self.settings.maximum_index_total_bytes,
                    timeout_seconds=self.settings.index_timeout_seconds,
                )
                test_source_paths = await self._resolve_test_source_paths(
                    session=session,
                    runner_paths=tuple(all_test_ids_by_file),
                    indexed_paths=frozenset(index.sources),
                    timeout_seconds=self.settings.index_timeout_seconds,
                )
            unsupported_test_files = set(all_test_ids_by_file) - test_source_paths.keys()
            verified_points = {
                test_id: test
                for test_id, test in all_verified_points.items()
                if test.path not in unsupported_test_files
            }
            baseline = tuple(verified_points)
            test_paths = {test_id: test.path for test_id, test in verified_points.items()}
            test_ids_by_file = {
                path: tuple(
                    test_id for test_id, test_path in test_paths.items() if test_path == path
                )
                for path in sorted(set(test_paths.values()))
            }
            trace_files = tuple(test_ids_by_file)
            if not trace_files:
                raise CandidateRejected(
                    "environment has no Adapter-indexable verified test points to trace"
                )
            diagnostics.record_shared_inventory(
                index=index.data,
                test_ids_by_file=test_ids_by_file,
                test_source_paths=test_source_paths,
            )
            selected_seed_files: set[str] = set()
            for test_file in trace_files:
                seed_diagnostics = diagnostics.seed(test_file)
                if context.selection.accepts(
                    seed_diagnostics.work_state(context.completed_task_ids)
                ):
                    selected_seed_files.add(test_file)
                    seed_diagnostics.begin()
            inactive_seed_files = set(trace_files) - selected_seed_files
            if unsupported_test_files:
                await context.events.emit(
                    Event(
                        run_id=context.run_id,
                        kind="adapter_inventory_filtered",
                        stage="featurebench",
                        message=(
                            "excluded verified tests whose physical files cannot contain "
                            "Python declarations for reversible F2P hiding"
                        ),
                        data={
                            "excluded_test_files": len(unsupported_test_files),
                            "excluded_test_points": sum(
                                len(all_test_ids_by_file[path]) for path in unsupported_test_files
                            ),
                            "remaining_test_files": len(trace_files),
                            "remaining_test_points": len(baseline),
                        },
                    )
                )
            await context.progress.advance(completed=1)
            await context.progress.finish_phase()
            if not selected_seed_files:
                await context.events.emit(
                    Event(
                        run_id=context.run_id,
                        kind="adapter_selection_completed",
                        stage="featurebench",
                        message="current work selection matched no FeatureBench Seeds",
                        data={"selection": context.selection.name},
                    )
                )
                raise NoWorkSelected(
                    f"selection {context.selection.name!r} matched no FeatureBench Seeds"
                )
            await context.progress.start_phase(
                "analyze_tests",
                total=len(trace_files),
            )

            async def trace_isolated(test_file: str) -> _TraceFileResult:
                async def trace_selection(
                    selection: tuple[str, ...],
                    timeout_seconds: float,
                ) -> tuple[TraceBundle, TraceStatistics]:
                    return await self._trace_selection_in_fresh_session(
                        environment=environment,
                        context=context,
                        test_file=test_file,
                        test_ids=selection,
                        timeout_seconds=timeout_seconds,
                    )

                return await self._trace_file(
                    test_file=test_file,
                    test_ids=test_ids_by_file[test_file],
                    trace_selection=trace_selection,
                    timeout_seconds=self.settings.trace_timeout_seconds,
                )

            trace_results = list(
                await self._run_trace_inventory(
                    trace_files,
                    concurrency=self.settings.trace_concurrency,
                    trace_file=trace_isolated,
                    semaphore=self.resources.trace_slots,
                    progress=lambda test_file: context.progress.advance(current=test_file),
                )
            )
            traces: dict[str, TraceBundle] = {}
            traced_test_ids: set[str] = set()
            traced_test_ids_by_file: dict[str, tuple[str, ...]] = {}
            complete_trace_files: set[str] = set()
            for result in trace_results:
                seed_diagnostics = diagnostics.seed(result.test_file)
                trace_result: dict[str, object] = {
                    "test_file": result.test_file,
                    "requested_test_ids": result.requested_test_ids,
                    "successful_test_ids": result.successful_test_ids,
                    "failed_test_ids": result.failed_test_ids,
                    "batch_error": result.batch_error,
                    "complete": result.complete,
                    "can_enter_top": result.can_enter_top,
                    "budget_exhausted": result.budget_exhausted,
                }
                if result.test_file not in inactive_seed_files:
                    seed_diagnostics.record_trace(trace_result, result.bundle)
                if result.can_enter_top:
                    assert result.bundle is not None
                    traces[result.test_file] = result.bundle
                    traced_test_ids_by_file[result.test_file] = result.successful_test_ids
                traced_test_ids.update(result.successful_test_ids)
                if result.complete:
                    complete_trace_files.add(result.test_file)
                if not result.can_enter_top and result.test_file not in inactive_seed_files:
                    seed_diagnostics.finalize(
                        "rejected",
                        phase="trace",
                        error_type=(
                            "TraceBudgetExceeded" if result.budget_exhausted else "TraceRejected"
                        ),
                        error_text=(
                            result.batch_error
                            or "dynamic trace produced no successful Top evidence"
                        ),
                    )
                    await context.progress.candidate_rejected(current=result.test_file)
                if result.batch_error is not None:
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind=(
                                "dynamic_trace_budget_exhausted"
                                if result.budget_exhausted
                                else "dynamic_trace_failed"
                            ),
                            stage="featurebench",
                            message=(
                                f"file trace exhausted its shared wall-clock budget for "
                                f"{result.test_file}"
                                if result.budget_exhausted
                                else f"file trace failed for {result.test_file}"
                            ),
                            data={
                                "test_file": result.test_file,
                                "requested_test_points": len(result.requested_test_ids),
                                "successful_test_points": len(result.successful_test_ids),
                                "failed_test_points": len(result.failed_test_ids),
                                "budget_exhausted": result.budget_exhausted,
                                "batch_error": result.batch_error,
                            },
                        )
                    )
                for label, statistics in result.statistics:
                    await self._emit_trace_statistics(
                        context=context,
                        test_file=label,
                        statistics=statistics,
                    )
                for test_id, error in result.failed_test_ids:
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind="dynamic_trace_point_rejected",
                            stage="featurebench",
                            message=f"trace failed for verified test point {test_id}",
                            data={
                                "test_file": result.test_file,
                                "test_id": test_id,
                                "error": error,
                            },
                        )
                    )
            collection_trace = collection_trace_from_bundles(traces.values())
            # This event is the explicit offline-phase barrier: every physical trace
            # task has completed before any Seed can enter Top selection or masking.
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="dynamic_trace_inventory_completed",
                    stage="featurebench",
                    message=(
                        f"completed offline tracing; {len(traced_test_ids_by_file)}/"
                        f"{len(trace_files)} physical test files produced Top evidence"
                    ),
                    data={
                        "trace_batches": len(trace_files),
                        "top_candidate_files": len(traced_test_ids_by_file),
                        "complete_trace_files": len(complete_trace_files),
                        "budget_exhausted_files": sum(
                            result.budget_exhausted for result in trace_results
                        ),
                        "traced_test_points": len(traced_test_ids),
                        "rejected_trace_test_points": sum(
                            len(result.failed_test_ids) for result in trace_results
                        ),
                    },
                )
            )
            await context.progress.finish_phase()
            if not traced_test_ids_by_file:
                raise CandidateRejected(
                    "no test file produced a successful verified-test dynamic trace"
                )

            point_index = TracePointIndex.build(
                index=index,
                traces=traces.values(),
                candidate_test_ids=traced_test_ids,
                test_paths=test_paths,
            )

            capacity_signature = diagnostics.runner_capacity_signature(traced_test_ids_by_file)
            restored_test_capacity = diagnostics.load_runner_capacity(capacity_signature)
            measured_test_capacity = compatibility.estimated_test_capacity
            if restored_test_capacity is not None:
                shared_test_capacity = (
                    restored_test_capacity
                    if measured_test_capacity is None
                    else min(restored_test_capacity, measured_test_capacity)
                )
                await context.events.emit(
                    Event(
                        run_id=context.run_id,
                        kind="runner_test_capacity_restored",
                        stage="featurebench",
                        message="restored a compatible learned Runner batch capacity",
                        data={
                            "restored_test_capacity": restored_test_capacity,
                            "measured_test_capacity": measured_test_capacity,
                            "effective_test_capacity": shared_test_capacity,
                        },
                    )
                )
            else:
                shared_test_capacity = measured_test_capacity

            def effective_p2p_limit(test_file: str) -> int | None:
                return self._effective_p2p_limit(
                    seed_test_points=len(traced_test_ids_by_file[test_file]),
                    test_capacity=shared_test_capacity,
                )

            profiled_seed_files = tuple(sorted(set(traced_test_ids_by_file) - inactive_seed_files))
            profiles = tuple(
                self.planner.seed_feasibility_profile(
                    index=index,
                    trace=traces[test_file],
                    collection_trace=collection_trace,
                    test_ids=traced_test_ids_by_file[test_file],
                    point_index=point_index,
                    minimum_code_lines=self.settings.gold_patch_line_minimum,
                    preferred_minimum_code_lines=(self.settings.gold_patch_line_preferred_minimum),
                    maximum_code_lines=self.settings.gold_patch_line_maximum,
                    minimum_p2p_test_points=(self.settings.minimum_remaining_p2p_test_points),
                    maximum_p2p_test_points=effective_p2p_limit(test_file),
                    maximum_selected_p2p_test_files=(self.settings.maximum_selected_p2p_test_files),
                )
                for test_file in profiled_seed_files
            )
            for profile in profiles:
                seed_diagnostics = diagnostics.seed(profile.test_file)
                seed_diagnostics.set_phase("feasibility")
                seed_diagnostics.append_event(
                    "seed_feasibility_profiled",
                    **profile.as_dict(),
                )
            runnable_profiles = self._order_seed_profiles(
                profiles,
                self.settings.seed_ordering,
            )
            selected_pipeline_files = tuple(profile.test_file for profile in runnable_profiles)
            profiles_by_file = {profile.test_file: profile for profile in runnable_profiles}
            rejected_profiles = tuple(profile for profile in profiles if profile.rejected)
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="seed_feasibility_inventory_completed",
                    stage="featurebench",
                    message=(
                        f"profiled {len(profiles)} Seeds before Top; "
                        f"rejected {len(rejected_profiles)} proven-infeasible Seeds"
                    ),
                    data={
                        "profiled_seeds": len(profiles),
                        "pre_top_rejected_seeds": len(rejected_profiles),
                        "scheduled_seeds": len(runnable_profiles),
                        "preferred_scale_seeds": sum(
                            profile.preferred_code_scale for profile in runnable_profiles
                        ),
                        "seed_ordering": self.settings.seed_ordering,
                        "queue_head": [profile.as_dict() for profile in runnable_profiles[:20]],
                    },
                )
            )

            async def select_top_isolated(test_file: str) -> tuple[SourceObject, ...]:
                async with context.runtime.session(
                    environment.image,
                    runtime_spec=environment.runtime,
                ) as top_session:
                    await top_session.clean_workspace()

                    async def top_progress(kind: str, data: dict[str, object]) -> None:
                        await self._agent_progress(
                            context,
                            "top",
                            kind,
                            {**data, "seed_test_file": test_file},
                        )

                    return await self.planner.select_top(
                        index=index,
                        trace=traces[test_file],
                        collection_trace=collection_trace,
                        test_ids=traced_test_ids_by_file[test_file],
                        agents=self.top_agents,
                        session=top_session,
                        run_timeout_seconds=self.settings.agent.top.run_timeout_seconds,
                        batch_concurrency=self.settings.top_batch_concurrency,
                        progress=top_progress,
                        protected_object_ids=(profiles_by_file[test_file].protected_object_ids),
                        candidate_count=profiles_by_file[test_file].top_candidate_count,
                    )

            def create_plan_for_seed(
                test_file: str,
                top_objects: tuple[SourceObject, ...],
                maximum_p2p_test_points: int | None,
            ) -> MaskPlan:
                top_object_ids = tuple(item.object_id for item in top_objects)
                minimum_p2p_points = self.settings.minimum_remaining_p2p_test_points
                preliminary_p2p = point_index.select_p2p(
                    seed_test_file=test_file,
                    top_object_ids=top_object_ids,
                    top_object_impacts=self.planner.top_deletion_impacts(
                        index,
                        top_object_ids,
                    ),
                    minimum_test_points=minimum_p2p_points,
                    maximum_test_points=minimum_p2p_points,
                    random_seed=self.settings.random_seed,
                )

                def create_for_interval(
                    minimum_code_lines: int,
                    maximum_code_lines: int,
                    tier: GoldPatchTier,
                ) -> MaskPlan:
                    return self.planner.create_plan(
                        index=index,
                        seed_trace=traces[test_file],
                        all_traces=traces,
                        collection_trace=collection_trace,
                        top_objects=top_objects,
                        p2p=preliminary_p2p,
                        joint_optimization=JointOptimizationOptions(
                            point_index=point_index,
                            minimum_p2p_test_points=minimum_p2p_points,
                            maximum_p2p_test_points=maximum_p2p_test_points,
                            maximum_selected_p2p_test_files=(
                                self.settings.maximum_selected_p2p_test_files
                            ),
                            minimum_f2p_test_points=self.settings.minimum_f2p_test_points,
                            maximum_f2p_test_points=self.settings.maximum_f2p_test_points,
                            maximum_f2p_ratio=self.settings.maximum_f2p_ratio,
                            maximum_seed_pass_rate=self.settings.maximum_seed_pass_rate,
                            timeout_seconds=self.settings.optimizer_timeout_seconds,
                            workers=self.settings.optimizer_workers,
                            minimum_code_lines=minimum_code_lines,
                            maximum_code_lines=maximum_code_lines,
                            gold_patch_tier=tier,
                        ),
                    )

                return self._plan_gold_patch_tiers(create_for_interval)

            await context.progress.start_phase(
                "create_tasks",
                total=len(profiles),
            )
            for profile in rejected_profiles:
                await self._finalize_seed_error(
                    context=context,
                    diagnostics=diagnostics,
                    seed_test_file=profile.test_file,
                    phase="feasibility",
                    event_prefix="seed_feasibility",
                    error=CandidateRejected("; ".join(profile.rejection_reasons)),
                )
                await context.progress.advance(current=profile.test_file)
            source = iter(selected_pipeline_files)
            output_queue: asyncio.Queue[TaskRecord | _SeedPipelineFailure | object] = (
                asyncio.Queue()
            )

            async def run_seed(test_file: str) -> None:
                seed_diagnostics = diagnostics.seed(test_file)
                try:
                    seed_diagnostics.set_phase("top_selection")
                    await context.progress.set_work_item_activity(
                        test_file, "Targeting", current=test_file
                    )
                    top_objects = await select_top_isolated(test_file)
                    seed_diagnostics.record_top(top_objects)
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind="top_selection_completed",
                            stage="featurebench",
                            message=f"selected {len(top_objects)} Top objects for {test_file}",
                            data={
                                "seed_test_file": test_file,
                                "top_object_count": len(top_objects),
                                "top_objects": [
                                    {
                                        "object_id": item.object_id,
                                        "path": item.path,
                                        "qualname": item.qualname,
                                        "signature": item.signature,
                                        "start_line": item.start_line,
                                        "end_line": item.end_line,
                                    }
                                    for item in top_objects
                                ],
                                "traced_test_points": len(traced_test_ids_by_file[test_file]),
                            },
                        )
                    )

                    seed_diagnostics.set_phase("mask_planning")
                    await context.progress.set_work_item_activity(
                        test_file, "Planning", current=test_file
                    )
                    initial_test_capacity = shared_test_capacity
                    initial_p2p_limit = self._effective_p2p_limit(
                        seed_test_points=len(traced_test_ids_by_file[test_file]),
                        test_capacity=initial_test_capacity,
                    )
                    plan = await asyncio.to_thread(
                        create_plan_for_seed,
                        test_file,
                        top_objects,
                        initial_p2p_limit,
                    )
                    seed_diagnostics.record_initial_plan(plan)
                    selected_top_ids = {item.object_id for item in top_objects}
                    effective_top_ids = {item.object_id for item in plan.top_objects}
                    optimization = plan.optimization
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind="mask_plan_completed",
                            stage="featurebench",
                            message=(
                                f"optimized {plan.removed_lines} deleted lines with "
                                f"{len(plan.p2p_test_ids)} P2P points for {test_file}"
                            ),
                            data={
                                "seed_test_file": test_file,
                                "gold_patch_tier": plan.gold_patch_tier,
                                "p2p_test_points": len(plan.p2p_test_ids),
                                "effective_p2p_limit": initial_p2p_limit,
                                "estimated_environment_test_capacity": (
                                    compatibility.estimated_test_capacity
                                ),
                                "effective_environment_test_capacity": initial_test_capacity,
                                "selected_top_objects": len(selected_top_ids),
                                "effective_top_objects": len(effective_top_ids),
                                "pruned_top_object_ids": sorted(
                                    selected_top_ids - effective_top_ids
                                ),
                                "specific_objects": len(plan.specific_objects),
                                "removed_lines": plan.removed_lines,
                                "modified_files": len(plan.patch_files),
                                "optimization": (
                                    optimization.model_dump(mode="json")
                                    if optimization is not None
                                    else None
                                ),
                            },
                        )
                    )

                    await context.progress.set_work_item_activity(
                        test_file, "Baseline", current=test_file
                    )
                    repair_workspace: Path | None = None
                    bind_mounts: tuple[tuple[Path, str, bool], ...] = ()
                    if self.settings.repair.strategy == "agent":
                        repair_workspace = seed_diagnostics.new_workspace("repair")
                        bind_mounts = ((repair_workspace, "/featurebench-repair", False),)

                    async def replan_with_p2p_limit(maximum_p2p: int) -> MaskPlan:
                        await context.progress.set_work_item_activity(
                            test_file,
                            "Replanning baseline",
                            current=test_file,
                        )
                        return await asyncio.to_thread(
                            create_plan_for_seed,
                            test_file,
                            top_objects,
                            maximum_p2p,
                        )

                    def publish_test_capacity(test_capacity: int) -> int:
                        nonlocal shared_test_capacity
                        retained_capacity = diagnostics.record_runner_capacity(
                            signature=capacity_signature,
                            test_capacity=test_capacity,
                            source_seed_test_file=test_file,
                        )
                        shared_test_capacity = (
                            retained_capacity
                            if shared_test_capacity is None or shared_test_capacity <= 0
                            else min(shared_test_capacity, retained_capacity)
                        )
                        return shared_test_capacity

                    async with context.runtime.session(
                        environment.image,
                        runtime_spec=environment.runtime,
                        bind_mounts=bind_mounts,
                    ) as candidate_session:
                        task = await self._generate_candidate(
                            environment=environment,
                            context=context,
                            session=candidate_session,
                            index=index,
                            plan=plan,
                            point_index=point_index,
                            traces=traces,
                            verified_points=verified_points,
                            test_paths=test_paths,
                            test_source_paths=test_source_paths,
                            test_ids_by_file=traced_test_ids_by_file,
                            repair_workspace=repair_workspace,
                            replan_with_p2p_limit=replan_with_p2p_limit,
                            current_test_capacity=lambda: shared_test_capacity,
                            publish_test_capacity=publish_test_capacity,
                        )

                    completion = self._await_candidate(task.task_id)
                    try:
                        await context.progress.set_work_item_activity(
                            test_file, "Quality gate", current=test_file
                        )
                        await output_queue.put(task)
                        await completion
                    finally:
                        self._candidate_completions.pop(task.task_id, None)
                except _SEED_LOCAL_ERRORS as exc:
                    phase = seed_diagnostics.summary.phase
                    event_prefix = {
                        "top_selection": "top_candidate",
                        "mask_planning": "mask_plan_candidate",
                    }.get(phase, "candidate")
                    await self._finalize_seed_error(
                        context=context,
                        diagnostics=diagnostics,
                        seed_test_file=test_file,
                        phase=phase,
                        event_prefix=event_prefix,
                        error=exc,
                    )
                finally:
                    await context.progress.finish_work_item(test_file)
                await context.progress.advance(current=test_file)

            async def seed_worker() -> None:
                try:
                    while True:
                        try:
                            test_file = next(source)
                        except StopIteration:
                            return
                        try:
                            async with self.resources.seed_slots:
                                await run_seed(test_file)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            await output_queue.put(_SeedPipelineFailure(test_file, exc))
                            return
                finally:
                    await output_queue.put(_SEED_WORKER_DONE)

            worker_count = min(self.settings.seed_concurrency, len(selected_pipeline_files))
            workers = tuple(asyncio.create_task(seed_worker()) for _ in range(worker_count))
            finished_workers = 0
            try:
                while finished_workers < worker_count:
                    item = await output_queue.get()
                    if item is _SEED_WORKER_DONE:
                        finished_workers += 1
                    elif isinstance(item, _SeedPipelineFailure):
                        seed_diagnostics = diagnostics.seed(item.seed_test_file)
                        await self._finalize_seed_error(
                            context=context,
                            diagnostics=diagnostics,
                            seed_test_file=item.seed_test_file,
                            phase=seed_diagnostics.summary.phase,
                            event_prefix="seed_pipeline",
                            error=item.error,
                        )
                        raise item.error
                    else:
                        yield cast(TaskRecord, item)
            finally:
                for worker in workers:
                    worker.cancel()
                if workers:
                    await asyncio.gather(*workers, return_exceptions=True)
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="seed_pipeline_completed",
                    stage="featurebench",
                    message=f"completed {len(selected_pipeline_files)} selected Seed pipelines",
                    data={
                        "profiled_seeds": len(profiles),
                        "pre_top_rejected_seeds": len(rejected_profiles),
                        "selected_seeds": len(selected_pipeline_files),
                        "seed_ordering": self.settings.seed_ordering,
                        "seed_concurrency": self.settings.seed_concurrency,
                        "indexed_test_points": len(point_index.test_ids),
                        "indexed_objects": len(point_index.object_test_masks),
                        "traced_object_test_associations": point_index.traced_associations,
                    },
                )
            )
            await context.progress.finish_phase()
        finally:
            self._candidate_completions.clear()

    @staticmethod
    async def _emit_trace_statistics(
        *, context: GeneratorContext, test_file: str, statistics: TraceStatistics
    ) -> None:
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="dynamic_trace",
                stage="featurebench",
                message=f"traced {test_file}",
                data={
                    "test_file": test_file,
                    "duration_seconds": round(statistics.duration_seconds, 6),
                    "report_bytes": statistics.report_bytes,
                    "test_ids": statistics.test_ids,
                    "records": statistics.records,
                    "edges": statistics.edges,
                    "imports": statistics.imports,
                    "process_count": statistics.process_count,
                },
            )
        )

    async def _generate_candidate(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        session: ContainerSessionProtocol,
        index: RepositoryIndex,
        plan: MaskPlan,
        point_index: TracePointIndex,
        traces: dict[str, TraceBundle],
        verified_points: dict[str, CollectedTestPoint],
        test_paths: dict[str, str],
        test_source_paths: dict[str, str],
        test_ids_by_file: dict[str, tuple[str, ...]],
        repair_workspace: Path | None,
        replan_with_p2p_limit: Callable[[int], Awaitable[MaskPlan]],
        current_test_capacity: Callable[[], int | None],
        publish_test_capacity: Callable[[int], int],
    ) -> TaskRecord:
        await session.clean_workspace()
        full_test_timeout_seconds = self.settings.full_test_timeout_seconds
        seed_test_ids = test_ids_by_file.get(plan.seed_test_file, ())

        while True:
            latest_test_capacity = current_test_capacity()
            (
                plan,
                previous_p2p_test_points,
                latest_p2p_limit,
            ) = await self._refresh_plan_for_test_capacity(
                plan=plan,
                seed_test_points=len(seed_test_ids),
                test_capacity=latest_test_capacity,
                replan_with_p2p_limit=replan_with_p2p_limit,
            )
            if previous_p2p_test_points is None:
                break
            assert latest_p2p_limit is not None
            if self.diagnostics is None:
                raise ContractError("FeatureBench diagnostics disappeared during capacity replan")
            self.diagnostics.seed(plan.seed_test_file).record_clean_baseline_capacity_replan(
                plan,
                previous_p2p_test_points=previous_p2p_test_points,
                maximum_p2p_test_points=latest_p2p_limit,
                test_capacity=latest_test_capacity or 0,
            )
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="clean_baseline_capacity_replan",
                    stage="featurebench",
                    message="replanned a stale P2P selection against current Runner capacity",
                    data={
                        "seed_test_file": plan.seed_test_file,
                        "previous_p2p_test_points": previous_p2p_test_points,
                        "maximum_p2p_test_points": latest_p2p_limit,
                        "replanned_p2p_test_points": len(plan.p2p_test_ids),
                        "test_capacity": latest_test_capacity,
                    },
                )
            )

        async def run_clean_batch(
            test_ids: tuple[str, ...],
            remaining_seconds: float,
        ) -> RunnerResponse:
            return await self._run_candidate_batch_in_fresh_session(
                environment=environment,
                context=context,
                candidate_patch="",
                test_ids=test_ids,
                timeout_seconds=min(full_test_timeout_seconds, remaining_seconds),
            )

        original_p2p = len(plan.p2p_test_ids)
        previous_plan = plan

        async def observe_clean_baseline_backoff(
            backoff: _CleanBaselineBackoffEvent,
        ) -> None:
            if self.diagnostics is None:
                raise ContractError("FeatureBench diagnostics disappeared during P2P backoff")
            self.diagnostics.seed(previous_plan.seed_test_file).record_clean_baseline_backoff(
                phase=backoff.phase,
                previous_p2p_test_points=len(previous_plan.p2p_test_ids),
                maximum_p2p_test_points=backoff.maximum_p2p_test_points,
                plan=backoff.plan,
                error=backoff.error,
            )
            kind = (
                "clean_baseline_p2p_backoff"
                if backoff.phase == "replanned"
                else f"clean_baseline_p2p_backoff_{backoff.phase}"
            )
            error_detail = (
                str(backoff.error)[-5000:] or repr(backoff.error)
                if backoff.error is not None
                else None
            )
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind=kind,
                    stage="featurebench",
                    message={
                        "started": "clean baseline timed out; starting one P2P backoff",
                        "replan_failed": "clean baseline P2P backoff could not produce a plan",
                        "replanned": "replanned one timed-out clean baseline",
                        "retry_completed": "completed the reduced clean baseline batch",
                        "retry_failed": "reduced clean baseline batch failed",
                    }[backoff.phase],
                    data={
                        "seed_test_file": previous_plan.seed_test_file,
                        "previous_p2p_test_points": len(previous_plan.p2p_test_ids),
                        "maximum_p2p_test_points": backoff.maximum_p2p_test_points,
                        "replanned_p2p_test_points": (
                            len(backoff.plan.p2p_test_ids) if backoff.plan is not None else None
                        ),
                        "full_test_timeout_seconds": full_test_timeout_seconds,
                        "error_type": (
                            type(backoff.error).__name__ if backoff.error is not None else None
                        ),
                        "error": error_detail,
                    },
                )
            )

        (
            plan,
            clean_selected,
            clean_report,
            backoff_p2p_limit,
        ) = await self._clean_baseline_with_backoff(
            plan=plan,
            seed_test_ids=seed_test_ids,
            test_paths=test_paths,
            run_batch=run_clean_batch,
            replan_with_p2p_limit=replan_with_p2p_limit,
            observe_backoff=observe_clean_baseline_backoff,
        )

        plan, baseline_test_ids = self._normalize_clean_baseline(
            plan=plan,
            seed_test_ids=seed_test_ids,
            selected_test_ids=clean_selected,
            report=clean_report,
        )
        if backoff_p2p_limit is not None:
            completed_test_points = len(clean_selected)
            published_test_capacity = publish_test_capacity(completed_test_points)
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="clean_baseline_test_capacity_published",
                    stage="featurebench",
                    message="published a verified Runner batch capacity for this Environment",
                    data={
                        "seed_test_file": plan.seed_test_file,
                        "backoff_maximum_p2p_test_points": backoff_p2p_limit,
                        "completed_test_points": completed_test_points,
                        "published_test_capacity": published_test_capacity,
                    },
                )
            )
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="mask_clean_baseline",
                stage="featurebench",
                message="established a clean combined baseline for this Mask candidate",
                data={
                    "seed_test_file": plan.seed_test_file,
                    "selected": len(clean_selected),
                    "passed": len(baseline_test_ids),
                    "excluded_nonpassing": len(clean_selected) - len(baseline_test_ids),
                    "p2p_before": original_p2p,
                    "p2p_after": len(plan.p2p_test_ids),
                    "p2p_backoff_limit": backoff_p2p_limit,
                },
            )
        )
        await context.progress.set_work_item_activity(
            plan.seed_test_file, "Validating mask", current=plan.seed_test_file
        )
        quality = await self._validate_mask_plan(
            environment=environment,
            context=context,
            session=session,
            index=index,
            plan=plan,
            test_paths=test_paths,
            test_source_paths=test_source_paths,
            baseline_test_ids=baseline_test_ids,
            attempt_index=0,
            validation_phase="initial",
        )
        repair_strategy_used: str | None = None
        optimize_accepted = (
            self.settings.repair.strategy == "rollback"
            and self.settings.repair.optimization_target == "interval_center"
        )
        if not quality.accepted or optimize_accepted:
            repair_strategy_used = self.settings.repair.strategy

            async def validate_repair_plan(
                repaired_plan: MaskPlan,
                attempt_index: int,
            ) -> QualityReport:
                await context.progress.set_work_item_activity(
                    plan.seed_test_file,
                    f"Repairing mask · validation {attempt_index}",
                    current=plan.seed_test_file,
                )
                try:
                    return await self._validate_mask_plan(
                        environment=environment,
                        context=context,
                        session=session,
                        index=index,
                        plan=repaired_plan,
                        test_paths=test_paths,
                        test_source_paths=test_source_paths,
                        baseline_test_ids=baseline_test_ids,
                        attempt_index=attempt_index,
                        validation_phase=self.settings.repair.strategy,
                    )
                finally:
                    await context.progress.set_work_item_activity(
                        plan.seed_test_file,
                        "Repairing mask",
                        current=plan.seed_test_file,
                    )

            await context.progress.set_work_item_activity(
                plan.seed_test_file, "Repairing mask", current=plan.seed_test_file
            )
            if self.settings.repair.strategy == "rollback":
                planned_gold_patch_tier = plan.gold_patch_tier
                rollback = DeterministicMaskRollback(
                    index=index,
                    planner=self.planner,
                    traces=traces,
                    initial_plan=plan,
                    initial_quality=quality,
                    point_index=point_index,
                    settings=self.settings,
                    validate_plan=validate_repair_plan,
                    timeout_seconds=self.settings.repair.timeout_seconds,
                    progress=lambda kind, data: self._rollback_progress(
                        context,
                        kind,
                        {**data, "seed_test_file": plan.seed_test_file},
                    ),
                )
                result = await rollback.run()
                plan = result.plan
                quality = result.quality
                if plan.gold_patch_tier != planned_gold_patch_tier:
                    await context.events.emit(
                        Event(
                            run_id=context.run_id,
                            kind="mask_tier_downgraded",
                            stage="featurebench",
                            message=(
                                "accepted the real validated Mask in the fallback Gold Patch tier"
                            ),
                            data={
                                "seed_test_file": plan.seed_test_file,
                                "from_tier": planned_gold_patch_tier,
                                "to_tier": plan.gold_patch_tier,
                                "gold_patch_lines": quality.gold_patch_lines,
                                "reason": (
                                    "preferred scale could not be preserved after real "
                                    "validation and deterministic rollback"
                                ),
                            },
                        )
                    )
                await context.events.emit(
                    Event(
                        run_id=context.run_id,
                        kind="mask_rollback_completed",
                        stage="featurebench",
                        message=(
                            f"deterministic rollback accepted after {result.validations} "
                            "new validations"
                        ),
                        data={
                            "seed_test_file": plan.seed_test_file,
                            "optimization_target": self.settings.repair.optimization_target,
                            "gold_patch_tier": plan.gold_patch_tier,
                            "validations": result.validations,
                            "cache_hits": result.cache_hits,
                            "duration_seconds": round(result.duration_seconds, 6),
                            "validation_seconds": round(result.validation_seconds, 6),
                            "engine_seconds": round(result.engine_seconds, 6),
                            "restored_objects": len(result.restored_object_ids),
                            "replacement_objects": len(result.replacement_object_ids),
                            "gold_patch_lines": quality.gold_patch_lines,
                            "gold_patch_files": quality.gold_patch_files,
                            "f2p": len(quality.fail_to_pass),
                            "p2p": len(quality.pass_to_pass),
                        },
                    )
                )
            else:
                if repair_workspace is None:
                    raise ContractError("Repair Agent workspace was not prepared")
                service = MaskRepairService(
                    workspace=MaskRepairWorkspace(repair_workspace, self.settings),
                    session=session,
                    index=index,
                    planner=self.planner,
                    traces=traces,
                    initial_plan=plan,
                    initial_quality=quality,
                    point_index=point_index,
                    validate_plan=validate_repair_plan,
                )
                await run_mask_repair_agent(
                    agents=self.repair_agents,
                    session=session,
                    service=service,
                    run_timeout_seconds=self.settings.agent.repair.run_timeout_seconds,
                    progress=lambda kind, data: self._agent_progress(
                        context,
                        "mask_repair",
                        kind,
                        {**data, "seed_test_file": plan.seed_test_file},
                    ),
                    metadata={
                        "environment_id": environment.environment_id,
                        "seed_test_file": plan.seed_test_file,
                    },
                )
                if service.accepted_attempt_id is None:
                    raise CandidateRejected(
                        f"repair Agent did not produce an accepted plan for {plan.seed_test_file}"
                    )
                plan = service.current_plan
                quality = service.current_quality

            # Search may return an earlier cached winner after probing later plans.
            # Re-materialize the accepted content before deriving the final patches.
            await MaskExecutor(index).apply(session, plan)

        if not quality.accepted:
            raise ContractError("candidate authoring requires one final accepted Mask")
        if self.diagnostics is None:
            raise ContractError("FeatureBench diagnostics disappeared before candidate authoring")
        self.diagnostics.seed(plan.seed_test_file).record_final_mask_selected(
            plan=plan,
            quality=quality,
            repair_strategy=repair_strategy_used,
        )
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="final_mask_selected",
                stage="featurebench",
                message="selected the accepted Mask for candidate authoring",
                data={
                    "seed_test_file": plan.seed_test_file,
                    "gold_patch_tier": plan.gold_patch_tier,
                    "repair_strategy": repair_strategy_used,
                    "gold_patch_lines": quality.gold_patch_lines,
                    "gold_patch_files": quality.gold_patch_files,
                    "f2p": len(quality.fail_to_pass),
                    "p2p": len(quality.pass_to_pass),
                },
            )
        )

        await context.progress.set_work_item_activity(
            plan.seed_test_file, "Authoring task", current=plan.seed_test_file
        )
        corruption_patch = await session.git_diff("--binary")
        gold_patch = await session.git_diff("-R", "--binary")
        if not corruption_patch.strip() or not gold_patch.strip():
            raise CandidateRejected("structural deletion produced an empty patch")
        hidden_sources, hidden_test_files = hide_test_points(
            index.sources,
            index=index,
            test_ids=quality.fail_to_pass,
            test_paths=test_paths,
            test_source_paths=test_source_paths,
        )
        for path in hidden_test_files:
            await session.write_file(path, hidden_sources[path].encode("utf-8"))
        hidden_tests_patch = await session.git_diff("-R", "--binary", "--", *hidden_test_files)
        if not hidden_tests_patch.strip():
            raise CandidateRejected("F2P test declarations could not be hidden reversibly")

        issue = await author_issue(
            agents=self.issue_author_agents,
            session=session,
            index=index,
            plan=plan,
            run_timeout_seconds=self.settings.agent.issue_author.run_timeout_seconds,
            progress=lambda kind, data: self._agent_progress(
                context,
                "issue_author",
                kind,
                {**data, "seed_test_file": plan.seed_test_file},
            ),
            workspace_root=self.diagnostics.seed(plan.seed_test_file).new_workspace("issue"),
            metadata={
                "environment_id": environment.environment_id,
                "seed_test_file": plan.seed_test_file,
            },
        )
        interfaces = build_interfaces(index, plan, issue)
        if not interfaces:
            raise CandidateRejected("Top objects could not be rendered as public interfaces")
        library_name = self.settings.library_name or environment.repository.rstrip("/").rsplit(
            "/", 1
        )[-1].removesuffix(".git")
        prompt = render_prompt(
            draft=issue,
            interfaces=interfaces,
            library_name=library_name,
            black_links=self.settings.black_links,
        )
        task = assemble_task_record(
            environment=environment,
            prompt=prompt,
            plan=plan,
            quality=quality,
            verified_points=verified_points,
            gold_patch=gold_patch,
            hidden_tests_patch=hidden_tests_patch,
            test_timeout_seconds=full_test_timeout_seconds,
        )
        self.diagnostics.seed(plan.seed_test_file).record_candidate(
            task=task,
            plan=plan,
            quality=quality,
            corruption_patch=corruption_patch,
            interfaces=interfaces,
            issue=issue,
        )
        await context.progress.set_work_item_activity(
            plan.seed_test_file, "Verifying Gold", current=plan.seed_test_file
        )
        return task

    async def _validate_mask_plan(
        self,
        *,
        environment: EnvironmentRecord,
        context: GeneratorContext,
        session: ContainerSessionProtocol,
        index: RepositoryIndex,
        plan: MaskPlan,
        test_paths: Mapping[str, str],
        test_source_paths: Mapping[str, str],
        baseline_test_ids: tuple[str, ...],
        attempt_index: int,
        validation_phase: str,
    ) -> QualityReport:
        """Apply one canonical plan and run the authoritative isolated real tests."""
        seed_diagnostics = (
            self.diagnostics.seed(plan.seed_test_file) if self.diagnostics is not None else None
        )
        if seed_diagnostics is not None:
            seed_diagnostics.record_validation_started(
                attempt=attempt_index,
                phase=validation_phase,
                plan=plan,
            )
        try:
            await MaskExecutor(index).apply(session, plan)
            candidate_patch = await session.git_diff("--binary")
        except BaseException as exc:
            if seed_diagnostics is not None:
                seed_diagnostics.record_validation_error(exc)
            raise
        if seed_diagnostics is not None:
            seed_diagnostics.record_candidate_patch(candidate_patch)

        async def run_batch(
            test_ids: tuple[str, ...],
            remaining_seconds: float,
            replay_patch: str = candidate_patch,
        ) -> RunnerResponse:
            return await self._run_candidate_batch_in_fresh_session(
                environment=environment,
                context=context,
                candidate_patch=replay_patch,
                test_ids=test_ids,
                timeout_seconds=min(
                    self.settings.full_test_timeout_seconds,
                    remaining_seconds,
                ),
            )

        try:
            selected_baseline, report = await self._isolated_candidate_report(
                plan,
                tuple(
                    test_id
                    for test_id in baseline_test_ids
                    if test_paths.get(test_id) == plan.seed_test_file
                ),
                test_paths,
                run_batch=run_batch,
                timeout_seconds=self.settings.validation_timeout_seconds,
            )
            quality = assess_quality(
                baseline=selected_baseline,
                report=report,
                plan=plan,
                test_paths=test_paths,
                settings=self.settings,
            )
            quality = apply_gold_patch_gates(
                quality,
                gold_patch=await session.git_diff("-R", "--binary"),
                settings=self.settings,
                tier=plan.gold_patch_tier,
            )
            quality = apply_f2p_hideability_gate(
                quality,
                index=index,
                test_paths=test_paths,
                test_source_paths=test_source_paths,
            )
        except BaseException as exc:
            if seed_diagnostics is not None:
                seed_diagnostics.record_validation_error(exc)
            raise
        if seed_diagnostics is not None:
            seed_diagnostics.record_validation_finished(quality)
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="mask_validation",
                stage="featurebench",
                message="mask accepted" if quality.accepted else "; ".join(quality.errors),
                data={
                    "seed_test_file": plan.seed_test_file,
                    "gold_patch_tier": plan.gold_patch_tier,
                    "repair_attempt": attempt_index,
                    "f2p": len(quality.fail_to_pass),
                    "p2p": len(quality.pass_to_pass),
                    "f2p_ratio": quality.f2p_ratio,
                    "top_objects": len(plan.top_objects),
                    "specific_objects": len(plan.specific_objects),
                    "removed_lines": plan.removed_lines,
                    "gold_patch_lines": quality.gold_patch_lines,
                    "gold_patch_files": quality.gold_patch_files,
                },
            )
        )
        return quality
