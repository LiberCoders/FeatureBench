from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol, cast

import pytest

from codeworld.agents.factory import AgentFactory
from codeworld.agents.feedback import AgentFeedback, AgentFeedbackError
from codeworld.config import AgentSettings, BaseImageSettings, EnvironmentSettings
from codeworld.environment.builder import (
    EnvironmentBuilder,
    _IsolatedBaselineRunner,  # pyright: ignore[reportPrivateUsage]
    assert_collection_can_reach_verified_floor,
    assert_quickcheck_health,
    baseline_timeout_response,
    derive_clone_plan,
    diagnose_environment_failure,
    group_test_ids_by_path,
    render_final_dockerfile,
    run_baseline_groups,
    select_quickcheck_test_ids,
    select_repair_runtime_spec,
    summarize_baseline_shortfall,
    summarize_runner_failure,
)
from codeworld.environment.diagnostics import FailureDiagnosticContext
from codeworld.environment.ecosystems import ecosystem_prompt_guidance
from codeworld.environment.feedback import FeedbackCode, issue
from codeworld.environment.planning import (
    SOURCE_MARKER,
    AnalysisAgentContext,
    ArtifactAgentContext,
    ArtifactRepairContract,
    ArtifactRepairHistoryEntry,
    ArtifactRepositoryFacts,
    BaseImageOption,
    ClonePlan,
    EnvironmentAnalysis,
    EnvironmentArtifactRepair,
    EnvironmentArtifacts,
    ExecutionContract,
    SelectedBaseContract,
    artifact_repair_agent_context,
    artifact_repair_contract,
    artifact_user_prompt,
    authorized_repair_output_type,
    materialize_artifact_repair,
    merge_artifact_repair,
    render_artifact_agent_context,
    runner_contract,
    validate_analysis,
    validate_artifact_repair_constraints,
    validate_artifacts,
)
from codeworld.environment.prompt_assets import load_prompt_asset
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
    DOCKER_HARNESS_STEP_MARKER,
    CommandTimeoutError,
    ContractError,
    DockerBuildError,
    DockerBuildOrigin,
    DockerBuildStep,
    EnvironmentValidationError,
    InfrastructureError,
    RunnerCollectionUnstableError,
)
from codeworld.events import Event
from codeworld.models import RunnerLimits, RuntimeResources, RuntimeSpec
from codeworld.runtime.container import CommandResult, DockerRuntime

URL = "https://github.com/example/project.git"
COMMIT = "a" * 40


def docker_build_error(
    *,
    stage: str,
    instruction: str,
    output: str,
    timeout_seconds: float | None = None,
    stderr: str | None = None,
    origin: DockerBuildOrigin = DockerBuildOrigin.ARTIFACT,
) -> DockerBuildError:
    return DockerBuildError(
        exit_code=None if timeout_seconds is not None else 1,
        timeout_seconds=timeout_seconds,
        stderr=stderr or output,
        failed_step=DockerBuildStep(
            step_id="8",
            stage=stage,
            instruction=instruction,
            output=output,
            origin=origin,
        ),
    )


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class ValidationMetrics(Protocol):
    observations: dict[str, object]

    def complete_phase(self) -> float: ...

    def begin_phase(self, phase: str) -> None: ...


def settings() -> EnvironmentSettings:
    return EnvironmentSettings(
        base_images=(
            BaseImageSettings(
                name="python-3.12",
                image="ubuntu:24.04",
                languages=("python",),
                runtime_family="python",
                runtime_version="3.12",
                recipe="python-3.12",
            ),
            BaseImageSettings(
                name="cuda",
                image="nvidia/cuda:12.4.1-devel-ubuntu22.04",
                languages=("generic",),
                runtime_family="cuda",
                runtime_version="12.4",
                gpu_support=True,
                recipe="cuda-12.4",
            ),
        )
    )


def analysis(*, gpu_requirement: str = "none") -> EnvironmentAnalysis:
    gpu = gpu_requirement in ("optional", "required")
    required = gpu_requirement == "required"
    return EnvironmentAnalysis(
        primary_language="python",
        languages=("python",),
        runtime_requirement=">=3.12",
        build_systems=("setuptools",),
        package_managers=("uv",),
        test_frameworks=("pytest",),
        manifest_paths=("pyproject.toml",),
        ci_paths=(),
        test_evidence_paths=("tests",),
        gpu_requirement=gpu_requirement,  # type: ignore[arg-type]
        gpu_evidence_paths=("tests/test_gpu.py",) if gpu else (),
        gpu_reason="GPU tests are optional" if gpu else None,
        selected_base_image_name="cuda" if required else "python-3.12",
        rejection_reason=None,
    )


def clone_plan() -> ClonePlan:
    return ClonePlan(
        repository_url=URL,
        commit=COMMIT,
        destination="/testbed",
        strategy="partial",
        recurse_submodules=True,
        fetch_lfs=True,
    )


def artifacts(*, image: str = "ubuntu:24.04") -> EnvironmentArtifacts:
    return EnvironmentArtifacts(
        dockerfile=(
            f"FROM {image}\n{SOURCE_MARKER}\nRUN python -m pip install --no-cache-dir -e .\n"
        ),
        activation=('#!/usr/bin/env bash\nexport PATH="/testbed/.venv/bin:$PATH"\ncd /testbed\n'),
        runner=(
            "#!/usr/bin/env bash\n"
            "# Runner contract fixture\n"
            "request=''\nresponse=''\n"
            "while (($#)); do case $1 in --request) request=$2; shift 2;; "
            "--response) response=$2; shift 2;; esac; done\n"
            'python /testbed/tools/runner.py "$request" "$response"\n'
        ),
    )


def test_repair_uses_real_runtime_after_activation_preflight() -> None:
    current_runtime = RuntimeSpec(
        activation="#!/usr/bin/env bash\nexport ACTIVE_ENV=1\ncd /testbed\n",
        runner=artifacts().runner,
    )
    failure = AgentFeedback(
        issues=(issue(FeedbackCode.ARTIFACT_RUNNER_CONTRACT, path=("runner",)),)
    )

    selected = select_repair_runtime_spec(
        built_image_available=True,
        current_runtime=current_runtime,
        failure=failure,
    )

    assert selected is current_runtime


def test_repair_falls_back_to_raw_runtime_when_activation_failed() -> None:
    current_runtime = RuntimeSpec(
        activation="#!/usr/bin/env bash\nexport ACTIVE_ENV=1\ncd /testbed\n",
        runner=artifacts().runner,
    )
    failure = AgentFeedback(
        issues=(issue(FeedbackCode.ARTIFACT_ACTIVATION_ENVIRONMENT, path=("activation",)),)
    )

    selected = select_repair_runtime_spec(
        built_image_available=True,
        current_runtime=current_runtime,
        failure=failure,
    )

    assert selected is not current_runtime
    assert "Planner image has no project environment yet" in selected.activation


def test_activation_failure_uses_raw_runtime_even_when_dockerfile_is_also_authorized() -> None:
    current_runtime = RuntimeSpec(
        activation="#!/usr/bin/env bash\nsource /testbed/.venv/bin/activate\ncd /testbed\n",
        runner=artifacts().runner,
    )
    failure = diagnose_environment_failure(
        EnvironmentValidationError(
            "activation failed: /testbed/.venv/bin/activate: No such file or directory",
            phase="activation",
            repair_fields=("activation", "dockerfile"),
        )
    )

    selected = select_repair_runtime_spec(
        built_image_available=True,
        current_runtime=current_runtime,
        failure=failure,
    )

    assert selected is not current_runtime
    assert "Planner image has no project environment yet" in selected.activation


def test_ambiguous_runtime_failure_keeps_the_real_activation_context() -> None:
    current_runtime = RuntimeSpec(
        activation="#!/usr/bin/env bash\nexport ACTIVE_ENV=1\ncd /testbed\n",
        runner=artifacts().runner,
    )
    failure = AgentFeedback(
        issues=tuple(
            issue(FeedbackCode.VALIDATION_RUNTIME, path=(field_name,))
            for field_name in ("dockerfile", "activation", "runner")
        )
    )

    selected = select_repair_runtime_spec(
        built_image_available=True,
        current_runtime=current_runtime,
        failure=failure,
    )

    assert selected is current_runtime


def test_repair_uses_planner_runtime_before_an_image_exists() -> None:
    current_runtime = RuntimeSpec(
        activation="#!/usr/bin/env bash\nexport ACTIVE_ENV=1\ncd /testbed\n",
        runner=artifacts().runner,
    )
    failure = AgentFeedback(issues=(issue(FeedbackCode.BUILD_COMMAND, path=("dockerfile",)),))

    selected = select_repair_runtime_spec(
        built_image_available=False,
        current_runtime=current_runtime,
        failure=failure,
    )

    assert selected is not current_runtime
    assert "Planner image has no project environment yet" in selected.activation


@pytest.mark.asyncio
async def test_every_validation_call_emits_one_terminal_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = RecordingEventSink()
    builder = EnvironmentBuilder(
        DockerRuntime(),
        settings(),
        agents=AgentFactory(AgentSettings(model="test")),
        events=events,
    )

    async def fake_validation(
        self: EnvironmentBuilder,
        *,
        image: str,
        runtime_spec: RuntimeSpec,
        commit: str,
        test_frameworks: tuple[str, ...],
        metrics: ValidationMetrics,
    ) -> tuple[CollectedTestPoint, ...]:
        del self, image, runtime_spec, test_frameworks
        metrics.complete_phase()
        metrics.begin_phase("quickcheck")
        metrics.observations["collected_test_point_count"] = 10
        if commit == COMMIT:
            raise ContractError("quickcheck failed")
        metrics.complete_phase()
        return (CollectedTestPoint(id="test_one", path="tests/test_one.py"),)

    monkeypatch.setattr(EnvironmentBuilder, "_validate_once", fake_validation)
    spec = RuntimeSpec(activation=artifacts().activation, runner=artifacts().runner)

    with pytest.raises(ContractError, match="quickcheck failed"):
        await builder.validate(image="sha256:failed", runtime_spec=spec, commit=COMMIT)
    await builder.validate(image="sha256:success", runtime_spec=spec, commit="b" * 40)

    measured = [event for event in events.events if event.kind == "validation_measured"]
    assert len(measured) == 2
    assert measured[0].data["status"] == "failed"
    assert measured[0].data["failed_phase"] == "quickcheck"
    assert measured[0].data["error_type"] == "ContractError"
    assert measured[0].data["collected_test_point_count"] == 10
    assert measured[1].data["status"] == "succeeded"
    assert measured[1].data["failed_phase"] is None
    assert measured[1].data["error"] is None
    phases = [event for event in events.events if event.kind == "validation_phase_started"]
    assert [event.data["phase"] for event in phases] == [
        "container_session",
        "container_session",
    ]


@pytest.mark.asyncio
async def test_long_docker_build_emits_liveness_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = RecordingEventSink()
    builder = EnvironmentBuilder(
        DockerRuntime(),
        settings(),
        agents=AgentFactory(AgentSettings(model="test")),
        events=events,
    )

    async def slow_build(**_: object) -> tuple[str, str]:
        await asyncio.sleep(0.025)
        return "sha256:image", "example:tag"

    monkeypatch.setattr(builder, "_build_final", slow_build)
    monkeypatch.setattr("codeworld.environment.builder._DOCKER_BUILD_HEARTBEAT_SECONDS", 0.01)

    result = await builder._build_with_progress(  # pyright: ignore[reportPrivateUsage]
        environment_id="environment",
        dockerfile="FROM scratch\n",
        attempt=2,
    )

    assert result == ("sha256:image", "example:tag")
    heartbeats = [
        event
        for event in events.events
        if event.kind == "build_progress" and event.data["heartbeat"] is True
    ]
    assert len(heartbeats) >= 2
    assert all(event.data["attempt"] == 2 for event in heartbeats)
    assert all(event.data["max_attempts"] == settings().max_build_attempts for event in heartbeats)


def test_analysis_contract_accepts_optional_gpu_on_cpu() -> None:
    output = analysis(gpu_requirement="optional")
    validate_analysis(output, settings())
    assert output.selected_base_image_name == "python-3.12"


def test_analysis_does_not_choose_artifacts_for_a_rejected_repository() -> None:
    output = EnvironmentAnalysis(
        primary_language="python",
        languages=("python",),
        runtime_requirement=None,
        build_systems=(),
        package_managers=(),
        test_frameworks=(),
        manifest_paths=("pyproject.toml",),
        ci_paths=(),
        test_evidence_paths=(),
        gpu_requirement="none",
        gpu_evidence_paths=(),
        gpu_reason=None,
        selected_base_image_name=None,
        rejection_reason="repository has no executable unit-test framework",
    )
    validate_analysis(output, settings())
    assert output.selected_base_image_name is None


def test_analysis_can_reject_a_repository_with_no_evidence_files() -> None:
    output = EnvironmentAnalysis(
        primary_language=None,
        languages=(),
        runtime_requirement=None,
        build_systems=(),
        package_managers=(),
        test_frameworks=(),
        manifest_paths=(),
        ci_paths=(),
        test_evidence_paths=(),
        gpu_requirement="none",
        gpu_evidence_paths=(),
        gpu_reason=None,
        selected_base_image_name=None,
        rejection_reason="repository contains no executable unit-test framework or test evidence",
    )

    validate_analysis(output, settings())


def test_analysis_requires_language_evidence_for_a_testable_repository() -> None:
    with pytest.raises(ValueError, match="unit-test framework requires language evidence"):
        EnvironmentAnalysis.model_validate(
            analysis().model_dump()
            | {
                "primary_language": None,
                "languages": (),
            }
        )


def test_analysis_rejects_when_no_configured_base_image_matches() -> None:
    output = analysis().model_copy(
        update={
            "selected_base_image_name": None,
            "rejection_reason": (
                "no configured base image satisfies the repository's required Python 3.13 runtime"
            ),
        }
    )

    validate_analysis(output, settings())
    assert output.has_unit_test_framework
    assert output.rejection_reason is not None


def test_analysis_rejection_reason_matches_acceptance_state() -> None:
    with pytest.raises(ValueError, match="rejected repository requires rejection_reason"):
        EnvironmentAnalysis.model_validate(
            analysis().model_dump() | {"selected_base_image_name": None, "rejection_reason": None}
        )
    with pytest.raises(ValueError, match="accepted repository must not include rejection_reason"):
        EnvironmentAnalysis.model_validate(
            analysis().model_dump() | {"rejection_reason": "base mismatch"}
        )


def test_harness_derives_clone_features_from_repository_files(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text("[submodule 'vendor']\n", encoding="utf-8")
    nested = tmp_path / "assets"
    nested.mkdir()
    (nested / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    derived = derive_clone_plan(tmp_path, repository_url=URL, commit=COMMIT)
    assert derived == clone_plan()


def test_quickcheck_samples_test_points_instead_of_files() -> None:
    points = tuple(
        CollectedTestPoint(id=f"tests/test_api.py::test_{index}", path="tests/test_api.py")
        for index in range(4)
    )
    selected = select_quickcheck_test_ids(points, commit=COMMIT, limit=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected).issubset({point.id for point in points})

    all_available = select_quickcheck_test_ids(points, commit=COMMIT, limit=10)
    assert len(all_available) == len(points)
    assert set(all_available) == {point.id for point in points}


def test_quickcheck_requires_half_of_the_collected_sample_to_really_pass() -> None:
    selected_ids = tuple(f"tests/test_api.py::test_{index}" for index in range(10))

    def response(passed: int) -> RunnerResponse:
        return RunnerResponse(
            action="run",
            exit_code=1,
            results=tuple(
                RunnerTestPointExecution(
                    id=test_id,
                    outcome="passed" if index < passed else "failed",
                )
                for index, test_id in enumerate(selected_ids)
            ),
            output="some sampled tests failed",
        )

    assert assert_quickcheck_health(
        response(5), selected_ids=selected_ids, minimum_pass_rate=0.5
    ) == (5, 5)
    with pytest.raises(ContractError, match="required_passed=5; observed_passed=4"):
        assert_quickcheck_health(response(4), selected_ids=selected_ids, minimum_pass_rate=0.5)


def test_collect_floor_returns_direct_runner_feedback_before_later_health_gates() -> None:
    collected = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(CollectedTestPoint(id="test_one", path="tests/test_one.py"),),
    )
    with pytest.raises(ContractError, match="required=10; received=1") as caught:
        assert_collection_can_reach_verified_floor(collected, minimum=10)
    feedback = diagnose_environment_failure(caught.value)
    assert {item.path for item in feedback.issues} == {("runner",)}
    assert feedback.issues[0].code == FeedbackCode.VALIDATION_COLLECTION_GRANULARITY


def test_complete_baseline_groups_test_points_deterministically_by_path() -> None:
    points = (
        CollectedTestPoint(id="tests/test_b.py::test_two", path="tests/test_b.py"),
        CollectedTestPoint(id="tests/test_a.py::test_one", path="tests/test_a.py"),
        CollectedTestPoint(id="tests/test_b.py::test_one", path="tests/test_b.py"),
    )

    assert group_test_ids_by_path(points, max_group_size=100) == (
        ("tests/test_a.py::test_one",),
        ("tests/test_b.py::test_one", "tests/test_b.py::test_two"),
    )

    assert group_test_ids_by_path(points, max_group_size=1) == (
        ("tests/test_a.py::test_one",),
        ("tests/test_b.py::test_one",),
        ("tests/test_b.py::test_two",),
    )
    with pytest.raises(ValueError, match="at least one"):
        group_test_ids_by_path(points, max_group_size=0)


@pytest.mark.asyncio
async def test_baseline_timeout_exhausts_the_shared_deadline() -> None:
    groups = (("tests/test_a.py::test_one",), ("tests/test_b.py::test_two",))

    class BaselineRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        async def run(
            self, test_ids: tuple[str, ...], *, timeout_seconds: float | None = None
        ) -> RunnerResponse:
            del timeout_seconds
            self.calls.append(test_ids)
            if len(self.calls) == 1:
                raise CommandTimeoutError("Runner run timed out after 120s")
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    fake = BaselineRunner()
    reports = await run_baseline_groups(
        cast(EnvironmentRunner, fake),
        groups,
        total_timeout_seconds=600,
    )

    assert fake.calls == [groups[0]]
    assert reports[0].results[0].outcome == "error"
    assert reports[0].exit_code == 124
    assert all("total timeout" in report.errors[0] for report in reports)


@pytest.mark.asyncio
async def test_isolated_baseline_runner_uses_and_closes_one_container_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = (("tests/test_a.py::test_one",), ("tests/test_b.py::test_two",))

    class FakeRuntime:
        def __init__(self) -> None:
            self.opened = 0
            self.closed = 0
            self.active = 0

        @asynccontextmanager
        async def session(self, image: str, *, runtime_spec: RuntimeSpec):
            assert image == "candidate:latest"
            assert self.active == 0
            self.opened += 1
            self.active += 1
            fake_session = type("FakeSession", (), {"runtime_spec": runtime_spec})()
            try:
                yield fake_session
            finally:
                self.active -= 1
                self.closed += 1

    class FakeEnvironmentRunner:
        def __init__(self, session: object) -> None:
            del session

        async def run(
            self, test_ids: tuple[str, ...], *, timeout_seconds: float | None = None
        ) -> RunnerResponse:
            assert timeout_seconds is not None
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    monkeypatch.setattr(
        "codeworld.environment.builder.EnvironmentRunner",
        FakeEnvironmentRunner,
    )
    fake_runtime = FakeRuntime()
    isolated = _IsolatedBaselineRunner(
        cast(DockerRuntime, fake_runtime),
        image="candidate:latest",
        runtime_spec=RuntimeSpec(
            activation=artifacts().activation,
            runner=artifacts().runner,
        ),
    )

    reports = await run_baseline_groups(
        isolated,
        groups,
        total_timeout_seconds=600,
    )

    assert [report.passed_ids for report in reports] == list(groups)
    assert fake_runtime.opened == len(groups)
    assert fake_runtime.closed == len(groups)
    assert fake_runtime.active == 0


@pytest.mark.asyncio
async def test_collected_path_validation_uses_a_fresh_image_and_checks_provenance() -> None:
    class FakePathSession:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        async def exec(self, command: tuple[str, ...], **kwargs: object) -> CommandResult:
            assert kwargs["workdir"] == "/"
            self.commands.append(command)
            return CommandResult(
                argv=command,
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=0,
            )

    class FakePathRuntime:
        def __init__(self) -> None:
            self.images: list[str] = []
            self.path_session = FakePathSession()

        @asynccontextmanager
        async def session(self, image: str, *, runtime_spec: RuntimeSpec):
            del runtime_spec
            self.images.append(image)
            yield self.path_session

    runtime = FakePathRuntime()
    builder = EnvironmentBuilder(
        cast(DockerRuntime, runtime),
        settings(),
        agents=AgentFactory(AgentSettings(model="test")),
    )
    spec = RuntimeSpec(activation=artifacts().activation, runner=artifacts().runner)

    await builder._assert_collected_test_paths(  # pyright: ignore[reportPrivateUsage]
        image="candidate:latest",
        runtime_spec=spec,
        paths=("tests/test_api.py",),
    )

    assert runtime.images == ["candidate:latest"]
    command = runtime.path_session.commands[0]
    assert "realpath -e" in command[4]
    assert "ls-files --error-unmatch" in command[4]
    assert 'case "$resolved"' in command[4]
    assert "test source traverses a symbolic link" not in command[4]
    assert command[-1] == "tests/test_api.py"


@pytest.mark.asyncio
async def test_baseline_total_timeout_stops_launching_groups_and_marks_rest_error() -> None:
    groups = (
        ("tests/test_a.py::test_one",),
        ("tests/test_b.py::test_two",),
        ("tests/test_c.py::test_three",),
    )

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    class BaselineRunner:
        def __init__(self, clock: FakeClock) -> None:
            self.clock = clock
            self.calls: list[tuple[tuple[str, ...], float | None]] = []

        async def run(
            self, test_ids: tuple[str, ...], *, timeout_seconds: float | None = None
        ) -> RunnerResponse:
            self.calls.append((test_ids, timeout_seconds))
            self.clock.now = 6.0
            return RunnerResponse(
                action="run",
                exit_code=0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="passed") for test_id in test_ids
                ),
            )

    clock = FakeClock()
    fake = BaselineRunner(clock)
    reports = await run_baseline_groups(
        cast(EnvironmentRunner, fake),
        groups,
        total_timeout_seconds=5,
        clock=clock,
    )

    assert fake.calls == [(groups[0], 5.0)]
    assert reports[0].passed_ids == groups[0]
    assert all(result.outcome == "error" for report in reports[1:] for result in report.results)
    assert all("total timeout" in report.errors[0] for report in reports[1:])


def test_baseline_shortfall_feedback_preserves_representative_root_causes() -> None:
    groups = (("tests/test_a.py::test_one",), ("tests/test_b.py::test_two",))
    timeout = baseline_timeout_response(
        groups[0], CommandTimeoutError("Runner run timed out after 120s")
    )
    dependency_error = RunnerResponse(
        action="run",
        exit_code=1,
        results=(
            RunnerTestPointExecution(
                id=groups[1][0],
                outcome="error",
                message="ModuleNotFoundError: missing_dependency",
            ),
        ),
        errors=("test setup failed",),
        output="ModuleNotFoundError: missing_dependency",
    )
    rendered = summarize_baseline_shortfall(
        required=10,
        verified=0,
        collected=2,
        group_ids=groups,
        reports=(timeout, dependency_error),
    )
    evidence = json.loads(rendered)
    assert len(rendered) <= 4500
    assert evidence["required_verified_test_points"] == 10
    assert evidence["observed_verified_test_points"] == 0
    assert evidence["nonpassing_group_count"] == 2
    assert "timed out" in json.dumps(evidence["representative_nonpassing_groups"])
    assert "ModuleNotFoundError" in json.dumps(evidence["representative_nonpassing_groups"])


def test_analysis_structured_output_requires_every_decision_field() -> None:
    schema = EnvironmentAnalysis.model_json_schema()
    assert set(schema["properties"]) == set(schema["required"])
    assert "clone" not in schema["properties"]
    assert "ecosystem_modules" not in schema["properties"]
    assert "selected_base_image" not in schema["properties"]
    assert "rejection_reason" in schema["properties"]
    assert "rationale" not in schema["properties"]


def test_analysis_prompt_scopes_multistack_repositories_to_one_coherent_suite() -> None:
    instructions = load_prompt_asset("analysis.instructions")
    user_prompt = load_prompt_asset("analysis.user")

    assert "one coherent" in instructions
    assert "independent optional subprojects" in instructions
    assert "prohibition on secondary build" in instructions
    assert "canonical top-level package suite" in user_prompt
    assert "native-extension toolchain" in user_prompt


def test_artifact_prompt_allows_only_repository_evidenced_suite_boundaries() -> None:
    instructions = load_prompt_asset("artifact.initial.instructions")
    repair = load_prompt_asset("artifact.repair.instructions")
    runner = load_prompt_asset("artifact.runner.user")

    assert "independent integration" in instructions
    assert "outcome-select tests" in instructions
    assert "fake credentials" in repair
    assert "suite-level path or canonical command" in runner
    assert "never select individual tests based on observed outcomes" in runner


def test_artifact_prompt_requires_live_repository_source() -> None:
    common = load_prompt_asset("artifact.common.user")
    dockerfile = load_prompt_asset("artifact.dockerfile.user")

    assert "sole source of project-under-test code" in common
    assert "Dependencies may be frozen; repository source must remain live" in common
    assert "sole source of project-under-test code" in artifact_user_prompt(("dockerfile",))
    assert "sole source of project-under-test code" in artifact_user_prompt(("runner",))
    assert "all third-party dependencies and required tools" in dockerfile
    assert "rebuild or regenerate only repository-owned artifacts offline" in dockerfile
    assert "never make a stale build-time project artifact" in dockerfile


def test_ecosystem_guidance_keeps_project_code_bound_to_testbed() -> None:
    python_guidance = " ".join(
        guidance
        for module in ecosystem_prompt_guidance(module_ids=("python",))
        for guidance in module.dockerfile_guidance
    )
    dotnet_guidance = " ".join(
        guidance
        for module in ecosystem_prompt_guidance(module_ids=("dotnet",))
        for guidance in module.dockerfile_guidance
    )

    assert "editable installation or another repository-supported mechanism" in python_guidance
    assert "current /testbed checkout" in python_guidance
    assert "dotnet test --no-restore" in dotnet_guidance
    assert "do not use --no-build or stale assemblies" in dotnet_guidance


def test_agent_contexts_are_strict_complete_contracts() -> None:
    selected = analysis()
    analysis_context = AnalysisAgentContext(
        base_image_options=tuple(
            BaseImageOption.from_settings(item) for item in settings().base_images
        ),
    )
    artifact_context = ArtifactAgentContext(
        repository=ArtifactRepositoryFacts.from_analysis(selected),
        selected_base=SelectedBaseContract.from_settings(settings().base_images[0]),
        execution_contract=ExecutionContract(
            workdir="/testbed",
            resources=RuntimeResources(),
            limits=RunnerLimits(),
        ),
        runner_contract=runner_contract(),
        ecosystem_guidance=ecosystem_prompt_guidance(
            languages=selected.languages,
            build_systems=selected.build_systems,
            package_managers=selected.package_managers,
            test_frameworks=selected.test_frameworks,
        ),
    )
    rendered_analysis = analysis_context.model_dump_json()
    assert "base_image_options" in rendered_analysis
    assert "repository_url" not in rendered_analysis
    assert "current_analysis" not in rendered_analysis
    assert {module.module for module in artifact_context.ecosystem_guidance} == {
        "python",
        "pytest",
    }
    rendered_artifact = render_artifact_agent_context(artifact_context)
    assert '"selected_base_image_name"' not in rendered_artifact
    assert "name" not in type(artifact_context.selected_base).model_fields
    assert '"minimum_test_points"' not in rendered_artifact
    assert '"quickcheck_test_points"' not in rendered_artifact
    assert '"collection_stability_runs"' not in rendered_artifact
    payload = json.loads(rendered_artifact)
    assert payload["execution_contract"] == {
        "workdir": "/testbed",
        "resources": {"network": "none"},
        "limits": {
            "collect_timeout_seconds": 120.0,
            "baseline_total_timeout_seconds": 3600.0,
        },
    }


def test_artifact_context_renders_nondefault_execution_contract_values() -> None:
    selected = analysis()
    context = ArtifactAgentContext(
        repository=ArtifactRepositoryFacts.from_analysis(selected),
        selected_base=SelectedBaseContract.from_settings(settings().base_images[0]),
        execution_contract=ExecutionContract(
            workdir="/testbed",
            resources=RuntimeResources(
                network="bridge",
                cpus=3,
                memory="6g",
                gpus="all",
            ),
            limits=RunnerLimits(
                collect_timeout_seconds=45,
                baseline_total_timeout_seconds=900,
            ),
        ),
        runner_contract=runner_contract(),
        ecosystem_guidance=(),
    )

    execution = json.loads(render_artifact_agent_context(context))["execution_contract"]
    assert execution == {
        "workdir": "/testbed",
        "resources": {
            "network": "bridge",
            "cpus": 3.0,
            "memory": "6g",
            "gpus": "all",
        },
        "limits": {
            "collect_timeout_seconds": 45.0,
            "baseline_total_timeout_seconds": 900.0,
        },
    }


def test_harness_derives_registered_modules_from_factual_labels() -> None:
    assert analysis().ecosystem_modules == ("pytest", "python")


def test_runner_contract_explains_external_file_path_transport() -> None:
    rendered = str(runner_contract().rules)
    assert "request file" in rendered
    assert "explicitly permitted by applicable ecosystem guidance" in rendered
    assert "incidental human-facing presentation output" in rendered
    assert "stdin-script argv sentinel" not in rendered
    node_guidance = " ".join(
        item
        for module in ecosystem_prompt_guidance(module_ids=("node",))
        for item in module.runner_guidance
    )
    assert "stdin-script argv sentinel" in node_guidance


def test_pytest_runner_guidance_requires_per_item_machine_reports() -> None:
    rendered = " ".join(
        item
        for module in ecosystem_prompt_guidance(module_ids=("pytest",))
        for item in module.runner_guidance
    )
    assert "pytest_runtest_logreport" in rendered
    assert "all requested test points in one pytest invocation" in rendered
    assert "<native_base>#<zero_based_occurrence>" in rendered
    assert "same pytest execution invocation" in rendered
    assert "Never resolve a nondeterministic native node ID in one pytest process" in rendered
    assert "aggregate return code" in rendered
    assert "zero aggregate exit status" in rendered
    assert "individual outcome" in rendered
    assert "never split a node ID on whitespace" in rendered
    assert "standard repository configuration" in rendered
    assert "presentation output" in rendered
    assert "pytest version is proven to be 8.2 or newer" in rendered
    assert "pytest_ignore_collect" in rendered
    assert "Do not upgrade or override the repository's pytest constraint" in rendered
    assert "CW_REQUEST_PATH" in rendered
    assert "environment-variable value" in rendered
    assert "plugin source" in rendered


def test_runner_contract_keeps_unbounded_ids_in_files() -> None:
    rendered = " ".join(runner_contract().rules)
    assert "request file or another selection file" in rendered
    assert "environment-variable value" in rendered
    assert "generated plugin source" in rendered


def test_runner_review_does_not_guess_repository_suite_boundaries() -> None:
    instructions = load_prompt_asset("runner-review.instructions")
    assert "fixed repository suite root" in instructions
    assert "reviewer has no repository evidence" in instructions
    assert "request or selection file" in instructions


def test_required_gpu_must_select_gpu_capable_base() -> None:
    output = analysis(gpu_requirement="required").model_copy(
        update={"selected_base_image_name": "python-3.12"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_analysis(output, settings())
    assert caught.value.feedback.issues[0].code == FeedbackCode.ANALYSIS_GPU_BASE_REQUIRED


def test_analysis_accepts_only_a_configured_base_name() -> None:
    output = analysis().model_copy(update={"selected_base_image_name": "python:3.12-slim-bookworm"})
    with pytest.raises(AgentFeedbackError) as caught:
        validate_analysis(output, settings())
    feedback = caught.value.feedback.issues[0]
    assert feedback.code == FeedbackCode.ANALYSIS_BASE_NAME
    assert feedback.expected == "python-3.12, cuda"


def test_environment_artifacts_render_a_pinned_online_clone() -> None:
    selected = analysis()
    authored = artifacts()
    validate_artifacts(authored, settings(), selected)
    dockerfile = render_final_dockerfile(
        authored,
        analysis=selected,
        clone=clone_plan(),
        settings=settings(),
    )
    assert "COPY repository/" not in dockerfile
    assert "AS featurebench_managed_base" in dockerfile
    assert "FROM featurebench_managed_base" in dockerfile
    assert "ARG REPOSITORY_COMMIT" not in dockerfile
    assert (
        f"git_transport -C /testbed fetch --force --no-tags --depth=1 "
        f'--filter=blob:none origin "{COMMIT}"' in dockerfile
    )
    partial_clone_config = (
        "git -C /testbed config core.repositoryFormatVersion 1",
        "git -C /testbed config extensions.partialClone origin",
        "git -C /testbed config remote.origin.promisor true",
        "git -C /testbed config remote.origin.partialCloneFilter blob:none",
    )
    for command in partial_clone_config:
        assert command in dockerfile
    filtered_fetch = dockerfile.index("git_transport -C /testbed fetch --force --no-tags --depth=1")
    assert all(dockerfile.index(command) < filtered_fetch for command in partial_clone_config)
    filtered_checkout = dockerfile.index(
        "git_transport -C /testbed checkout --detach FETCH_HEAD",
        filtered_fetch,
    )
    unfiltered_fetch = dockerfile.index(
        f'git_transport -C /testbed fetch --force --no-tags --depth=1 origin "{COMMIT}"',
        filtered_checkout,
    )
    unfiltered_checkout = dockerfile.index(
        "git_transport -C /testbed checkout --detach FETCH_HEAD",
        unfiltered_fetch,
    )
    fallback_clone = dockerfile.index(
        f'git_transport clone --depth=1 --no-checkout "{URL}" /testbed',
        unfiltered_checkout,
    )
    assert filtered_fetch < filtered_checkout < unfiltered_fetch < unfiltered_checkout
    assert "clone --filter=blob:none" not in dockerfile
    root_workdir = dockerfile.index("WORKDIR /\n")
    remove_old_checkout = dockerfile.index("rm -rf /testbed", root_workdir)
    assert root_workdir < remove_old_checkout < fallback_clone
    assert "ARG GITHUB_PROXY_PREFIX=" in dockerfile
    assert "ARG CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS=30" in dockerfile
    assert "ARG CW_DOWNLOAD_RETRIES=5" in dockerfile
    assert '--retry "$CW_DOWNLOAD_RETRIES"' in dockerfile
    assert '--connect-timeout "$CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS"' in dockerfile
    assert "ARG GIT_COMMAND_TIMEOUT_SECONDS=300" in dockerfile
    assert "ARG GIT_HTTP_LOW_SPEED_LIMIT=1024" in dockerfile
    assert "ARG GIT_HTTP_LOW_SPEED_TIME=30" in dockerfile
    assert "http.lowSpeedLimit=${GIT_HTTP_LOW_SPEED_LIMIT}" in dockerfile
    assert "http.lowSpeedTime=${GIT_HTTP_LOW_SPEED_TIME}" in dockerfile
    assert (
        'timeout --signal=TERM --kill-after=10s "${GIT_COMMAND_TIMEOUT_SECONDS}s" git' in dockerfile
    )
    assert "Git command timed out after ${GIT_COMMAND_TIMEOUT_SECONDS}s" in dockerfile
    assert f'git remote set-url origin "{URL}"' in dockerfile
    assert "git_transport -C /testbed checkout --detach FETCH_HEAD" in dockerfile
    assert f'git_transport checkout --detach "{COMMIT}"' in dockerfile
    assert "git_transport lfs pull; git reset --hard HEAD" in dockerfile
    assert "/opt/featurebench" not in dockerfile
    assert "featurebench-sources-" not in dockerfile
    assert "sources.yaml" not in dockerfile
    assert dockerfile.count(DOCKER_HARNESS_STEP_MARKER) == 2
    assert "/data2/" not in dockerfile
    assert "ARG MINIFORGE_DIST_URL=https://github.com/" in dockerfile
    assert "ARG CONDA_CHANNEL_URL=https://conda.anaconda.org/" in dockerfile
    assert authored.activation not in dockerfile
    assert authored.runner not in dockerfile


def test_environment_dockerfile_rejects_required_external_build_arguments() -> None:
    selected = analysis()
    required = artifacts().model_copy(
        update={"dockerfile": artifacts().dockerfile.replace("FROM", "ARG PRIVATE_INDEX\nFROM", 1)}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(required, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_BUILD_ARG_DEFAULT

    standalone = required.model_copy(
        update={
            "dockerfile": required.dockerfile.replace(
                "ARG PRIVATE_INDEX", "ARG PRIVATE_INDEX=https://example.com/simple"
            )
        }
    )
    validate_artifacts(standalone, settings(), selected)


def test_environment_artifacts_reject_wrong_images_and_embedded_agents() -> None:
    selected = analysis()
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(artifacts(image="python:3.99"), settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_BASE_REFERENCE
    poisoned = artifacts().model_copy(
        update={"dockerfile": artifacts().dockerfile + "RUN pip install pydantic-ai\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(poisoned, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_AGENT_CONTENT

    forged_origin = artifacts().model_copy(
        update={"dockerfile": artifacts().dockerfile + f"RUN {DOCKER_HARNESS_STEP_MARKER}; false\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(forged_origin, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_AGENT_CONTENT


def test_environment_artifacts_preserve_managed_path() -> None:
    selected = analysis()
    poisoned = artifacts().model_copy(
        update={"dockerfile": artifacts().dockerfile + "ENV PATH=/project/bin\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(poisoned, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_PATH_OVERRIDE

    inherited = artifacts().model_copy(
        update={"dockerfile": artifacts().dockerfile + 'ENV PATH="/project/bin:${PATH}"\n'}
    )
    validate_artifacts(inherited, settings(), selected)


def test_environment_artifacts_quote_shell_sensitive_python_requirements() -> None:
    selected = analysis()
    poisoned = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile
            + "RUN python -m pip install pytest>=8 coverage[toml]<9\n"
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(poisoned, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_SHELL_REQUIREMENT

    quoted = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile
            + 'RUN python -m pip install "pytest>=8" "coverage[toml]<9"\n'
        }
    )
    validate_artifacts(quoted, settings(), selected)


def test_artifact_repair_is_sparse_and_host_merged() -> None:
    current = artifacts()
    feedback = AgentFeedback(issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),))
    replacement = current.runner + "\n# repaired\n"
    merged = merge_artifact_repair(
        current,
        feedback,
        EnvironmentArtifactRepair(runner=replacement),
    )
    assert merged.runner == replacement
    assert merged.dockerfile == current.dockerfile
    assert merged.activation == current.activation

    with pytest.raises(AgentFeedbackError) as caught:
        merge_artifact_repair(
            current,
            feedback,
            EnvironmentArtifactRepair(runner=replacement, dockerfile=current.dockerfile + "\n"),
        )
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_UNRELATED_FIELD_CHANGED


def test_artifact_repair_ignores_repeated_current_values() -> None:
    current = artifacts()
    feedback = AgentFeedback(
        issues=(
            issue(FeedbackCode.VALIDATION_RUNTIME, path=("dockerfile",)),
            issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),
        )
    )
    replacement = current.dockerfile + "\n# repaired\n"
    merged = merge_artifact_repair(
        current,
        feedback,
        EnvironmentArtifactRepair(
            dockerfile=replacement,
            runner=current.runner,
            activation=current.activation,
        ),
    )
    assert merged.dockerfile == replacement
    assert merged.runner == current.runner
    assert merged.activation == current.activation

    with pytest.raises(AgentFeedbackError) as caught:
        merge_artifact_repair(
            current,
            feedback,
            EnvironmentArtifactRepair(
                dockerfile=current.dockerfile,
                runner=current.runner,
            ),
        )
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_FAILED_FIELD_UNCHANGED

    dockerfile_feedback = AgentFeedback(
        issues=(issue(FeedbackCode.BUILD_COMMAND, path=("dockerfile",)),)
    )
    activation_replacement = current.activation + "# activate repaired environment\n"
    consistent = merge_artifact_repair(
        current,
        dockerfile_feedback,
        EnvironmentArtifactRepair(
            dockerfile=replacement,
            activation=activation_replacement,
        ),
    )
    assert consistent.dockerfile == replacement
    assert consistent.activation == activation_replacement

    with pytest.raises(AgentFeedbackError) as caught:
        merge_artifact_repair(
            current,
            dockerfile_feedback,
            EnvironmentArtifactRepair(activation=activation_replacement),
        )
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_FAILED_FIELD_UNCHANGED


def test_repair_output_schema_exposes_only_feedback_authorized_fields() -> None:
    runner_only = authorized_repair_output_type(
        AgentFeedback(issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),))
    )
    assert set(runner_only.model_fields) == {"runner"}
    runner_schema = runner_only.model_json_schema()
    assert runner_schema["required"] == ["runner"]
    assert "MUST differ" in runner_schema["properties"]["runner"]["description"]
    corrected_runner = artifacts().runner + "\n# corrected\n"
    runner_only.model_validate({"runner": corrected_runner})
    with pytest.raises(ValueError):
        runner_only.model_validate(
            {"runner": corrected_runner, "dockerfile": artifacts().dockerfile}
        )

    contract = artifact_repair_contract(
        AgentFeedback(issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),)),
        runner_only,
    )
    assert isinstance(contract, ArtifactRepairContract)
    assert contract.output_model == "AuthorizedRunnerRepair"
    assert contract.authorized_fields == ("runner",)
    assert contract.directly_implicated_fields == ("runner",)
    assert contract.consistency_fields == ()
    assert "output_schema" not in type(contract).model_fields


def test_artifact_repair_output_contains_complete_authorized_values() -> None:
    current = artifacts()
    feedback = AgentFeedback(issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),))
    output_type = authorized_repair_output_type(feedback)
    corrected = current.runner.replace(
        "# Runner contract fixture",
        "# Runner V1 validated fixture",
    )
    output = output_type.model_validate({"runner": corrected})
    sparse = materialize_artifact_repair(output)
    assert sparse == EnvironmentArtifactRepair(runner=corrected)
    assert merge_artifact_repair(current, feedback, sparse).runner == corrected

    with pytest.raises(ValueError):
        output_type.model_validate({"runner": [{"old_text": "old", "new_text": "new"}]})


def test_repair_context_contains_only_field_relevant_contracts() -> None:
    selected = analysis()
    initial = ArtifactAgentContext(
        repository=ArtifactRepositoryFacts.from_analysis(selected),
        selected_base=SelectedBaseContract.from_settings(settings().base_images[0]),
        execution_contract=ExecutionContract(
            workdir="/testbed",
            resources=RuntimeResources(),
            limits=RunnerLimits(),
        ),
        runner_contract=runner_contract(),
        ecosystem_guidance=ecosystem_prompt_guidance(
            languages=selected.languages,
            build_systems=selected.build_systems,
            package_managers=selected.package_managers,
            test_frameworks=selected.test_frameworks,
        ),
    )
    current = artifacts()

    activation_failure = AgentFeedback(
        issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("activation",)),)
    )
    activation_context = artifact_repair_agent_context(
        initial=initial,
        current=current,
        failure=activation_failure,
        repair_history=(),
        accepted_constraints=None,
        repair_contract=artifact_repair_contract(activation_failure),
    )
    activation_payload = activation_context.model_dump(mode="json", exclude_none=True)
    assert activation_payload["current_artifacts"] == {"activation": current.activation}
    assert "runner_contract" not in activation_payload
    assert activation_payload["ecosystem_guidance"] == []

    runner_failure = AgentFeedback(
        issues=(issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),)
    )
    runner_context = artifact_repair_agent_context(
        initial=initial,
        current=current,
        failure=runner_failure,
        repair_history=(
            ArtifactRepairHistoryEntry(
                attempt=1,
                failed_phase="docker_build",
                feedback=activation_failure,
            ),
        ),
        accepted_constraints=None,
        repair_contract=artifact_repair_contract(runner_failure),
    )
    runner_payload = runner_context.model_dump(mode="json", exclude_none=True)
    assert runner_payload["current_artifacts"] == {"runner": current.runner}
    assert "runner_contract" in runner_payload
    assert runner_payload["repair_history"] == [
        {
            "attempt": 1,
            "failed_phase": "docker_build",
            "feedback": activation_failure.model_dump(mode="json", exclude_none=True),
        }
    ]
    assert {item["module"] for item in runner_payload["ecosystem_guidance"]} == {"pytest"}
    assert all(
        not item["dockerfile_guidance"] and not item["activation_guidance"]
        for item in runner_payload["ecosystem_guidance"]
    )

    activation_prompt = artifact_user_prompt(("activation",))
    assert "Activation:" in activation_prompt
    assert "Dockerfile:" not in activation_prompt
    assert "Runner:" not in activation_prompt
    assert "runner_contract" not in activation_prompt

    runner_prompt = artifact_user_prompt(("runner",))
    assert "Runner:" in runner_prompt
    assert "Dockerfile:" not in runner_prompt
    assert "Activation:" not in runner_prompt

    initial_prompt = artifact_user_prompt(("dockerfile", "activation", "runner"))
    assert all(section in initial_prompt for section in ("Dockerfile:", "Activation:", "Runner:"))
    assert "accepted local unit-test suite as the dependency boundary" in initial_prompt
    assert "a group's name alone is not evidence" in initial_prompt
    assert "smallest set that covers the accepted tests" in initial_prompt
    assert "each network-backed source service in its own RUN" in initial_prompt

    ambiguous = authorized_repair_output_type(
        AgentFeedback(
            issues=(
                issue(FeedbackCode.VALIDATION_RUNTIME, path=("dockerfile",)),
                issue(FeedbackCode.VALIDATION_RUNTIME, path=("runner",)),
            )
        )
    )
    assert set(ambiguous.model_fields) == {"dockerfile", "activation", "runner"}
    assert ambiguous.model_json_schema().get("required") is None

    dockerfile_only = authorized_repair_output_type(
        AgentFeedback(issues=(issue(FeedbackCode.BUILD_COMMAND, path=("dockerfile",)),))
    )
    assert set(dockerfile_only.model_fields) == {"dockerfile", "activation", "runner"}
    assert dockerfile_only.model_json_schema()["required"] == ["dockerfile"]


def test_runtime_failure_feedback_distinguishes_contract_and_ambiguous_test_failures() -> None:
    contract = diagnose_environment_failure(
        ContractError("runner did not produce a bounded response for collect")
    )
    assert {item.path for item in contract.issues} == {("runner",)}
    assert contract.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "--request <request-json-path>" in (contract.issues[0].expected or "")
    invalid_response = diagnose_environment_failure(
        ContractError("runner returned invalid JSON for collect: duplicate test IDs")
    )
    assert {item.path for item in invalid_response.issues} == {("runner",)}

    selected_mismatch = diagnose_environment_failure(
        ContractError("runner omitted or added selected test results: requested=10, returned=0")
    )
    assert {item.path for item in selected_mismatch.issues} == {("runner",)}
    assert selected_mismatch.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert invalid_response.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert invalid_response.issues[0].expected == (
        "one unique stable framework-addressable ID per individual test case"
    )
    assert invalid_response.issues[0].actual == "the response contains duplicate test IDs"

    agent_build = diagnose_environment_failure(
        docker_build_error(
            stage="stage-1 3/5",
            instruction="RUN ./configure --enable-project",
            output="#8 configure exited with status 2\n#8 ERROR: process exited with code 2",
        )
    )
    assert {item.path for item in agent_build.issues} == {("dockerfile",)}
    assert agent_build.issues[0].actual == "an Agent-controlled Dockerfile instruction failed"

    full_buildkit_log = diagnose_environment_failure(
        docker_build_error(
            stage="stage-1 4/8",
            instruction="RUN cd /testbed && uv sync --no-lock --group test",
            output=(
                "#17 0.2 error: unexpected argument '--no-lock' found\n"
                "#17 ERROR: process exited with code 2"
            ),
            stderr=(
                "#5 [featurebench_managed_base 2/4] RUN apt-get update\n"
                "#5 CACHED\n"
                "#12 [stage-1 2/8] RUN git_transport -C /testbed fetch origin deadbeef\n"
                "#12 DONE 0.4s\n"
                "#17 [stage-1 4/8] RUN cd /testbed && uv sync --no-lock --group test\n"
                "#17 0.2 error: unexpected argument '--no-lock' found\n"
                "#17 ERROR: process exited with code 2"
            ),
        )
    )
    assert {item.path for item in full_buildkit_log.issues} == {("dockerfile",)}
    assert full_buildkit_log.issues[0].actual == (
        "an Agent-controlled Dockerfile instruction failed"
    )

    agent_build_timeout = diagnose_environment_failure(
        docker_build_error(
            stage="stage-1 3/5",
            instruction="RUN make all",
            output="#8 still compiling",
            timeout_seconds=3600,
        )
    )
    assert {item.path for item in agent_build_timeout.issues} == {("dockerfile",)}
    assert "timed out" in (agent_build_timeout.issues[0].actual or "")

    unknown_build = diagnose_environment_failure(
        DockerBuildError(
            exit_code=1,
            stderr="Dockerfile parse failed before BuildKit exposed a numbered step",
        )
    )
    assert {item.path for item in unknown_build.issues} == {("dockerfile",)}
    assert unknown_build.issues[0].actual == (
        "the Docker build failed without enough provenance to exclude the Dockerfile"
    )

    runner_timeout = diagnose_environment_failure(
        CommandTimeoutError(
            "Runner run timed out: command timed out after 120s: docker exec example bash"
        )
    )
    assert {item.path for item in runner_timeout.issues} == {("runner",)}
    assert "configured timeout" in (runner_timeout.issues[0].actual or "")

    runner_transport = diagnose_environment_failure(
        EnvironmentValidationError(
            "large Runner request transport failed: Argument list too long",
            phase="runner_transport",
            repair_fields=("runner",),
        )
    )
    assert {item.path for item in runner_transport.issues} == {("runner",)}
    assert runner_transport.issues[0].context == {"failure_phase": "runner_transport"}

    for unrepairable in (
        InfrastructureError("cannot start Docker CLI: executable not found"),
        CommandTimeoutError("command timed out after 3600s: docker build example"),
        docker_build_error(
            stage="featurebench_managed_base 2/4",
            instruction="RUN apt-get update",
            output="#5 ModuleNotFoundError: managed-base-tool",
            origin=DockerBuildOrigin.HARNESS,
        ),
        docker_build_error(
            stage="stage-1 1/4",
            instruction="RUN git_transport -C /testbed fetch origin deadbeef",
            output="#7 connection reset",
            origin=DockerBuildOrigin.HARNESS,
        ),
        docker_build_error(
            stage="featurebench_managed_base 2/4",
            instruction="RUN apt-get update",
            output="#8 downloading packages",
            timeout_seconds=3600,
            origin=DockerBuildOrigin.HARNESS,
        ),
        docker_build_error(
            stage="stage-1 1/4",
            instruction="RUN git_transport -C /testbed fetch origin deadbeef",
            output="#8 fetching repository",
            timeout_seconds=3600,
            origin=DockerBuildOrigin.HARNESS,
        ),
    ):
        with pytest.raises((DockerBuildError, InfrastructureError)):
            diagnose_environment_failure(unrepairable)

    executed_during_collect = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: collected_count=710; output_tail=--- PASS: TestExample"
        )
    )
    assert {item.path for item in executed_during_collect.issues} == {("runner",)}
    assert executed_during_collect.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "without executing" in (executed_during_collect.issues[0].expected or "")

    unstable = diagnose_environment_failure(
        RunnerCollectionUnstableError(
            "Runner collect is not stable across repeated executions: missing_ids=['test one']"
        )
    )
    assert {item.path for item in unstable.issues} == {("runner",)}
    assert unstable.issues[0].code == FeedbackCode.ARTIFACT_COLLECTION_UNSTABLE
    assert "same unique framework-addressable IDs" in unstable.issues[0].correction

    granularity = diagnose_environment_failure(
        ContractError(
            "Runner collect returned too few framework-addressable individual test cases; "
            "required=10; received=5"
        )
    )
    assert granularity.issues[0].code == FeedbackCode.VALIDATION_COLLECTION_GRANULARITY
    assert granularity.issues[0].expected == ">= 10 individual test case IDs"
    assert granularity.issues[0].actual == "5 collected entries"

    aggregate_collection = diagnose_environment_failure(
        ContractError(
            "Runner collect returned too few framework-addressable individual test cases; "
            "required=10; received=0; "
            'response={"output_excerpt":"tests/test_api.py: 27\\n"}'
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert aggregate_collection.issues[0].code == FeedbackCode.VALIDATION_COLLECTION_GRANULARITY
    assert "pytest_collection_finish" in (aggregate_collection.issues[0].expected or "")
    assert "-qq" in (aggregate_collection.issues[0].actual or "")

    quoted_pytest_node = diagnose_environment_failure(
        ContractError(
            "quickcheck environment health gate failed: required_passed=5; "
            'observed_passed=0; sampled=10; output_excerpt="ERROR: file or directory '
            "not found: 'tests/test_api.py::test_read[item with spaces]'\""
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in quoted_pytest_node.issues} == {("runner",)}
    assert quoted_pytest_node.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "literal shell quotes" in (quoted_pytest_node.issues[0].actual or "")

    restricted_root_collision = diagnose_environment_failure(
        ContractError(
            "quickcheck environment health gate failed: required_passed=5; observed_passed=0; "
            "message=requested test ID was absent from restricted pytest collection; "
            "pytest selection collection failed; ImportPathMismatchError: "
            "('tests.conftest', '/testbed/nested/tests/conftest.py', "
            "PosixPath('/testbed/tests/conftest.py'))"
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in restricted_root_collision.issues} == {("runner",)}
    assert restricted_root_collision.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "fixed bounded invocation" in str(
        restricted_root_collision.issues[0].context.get("repair_strategy")
    )

    pytest_executed_during_collect = diagnose_environment_failure(
        ContractError(
            "Runner collect returned too few framework-addressable individual test cases; "
            'required=10; received=0; response={"output_tail":"27297 passed in 15.78s"}'
        )
    )
    assert (
        pytest_executed_during_collect.issues[0].code
        == FeedbackCode.VALIDATION_PYTEST_COLLECT_EXECUTED
    )
    assert "--collect-only" in pytest_executed_during_collect.issues[0].correction
    assert "pytest_collection_finish" in pytest_executed_during_collect.issues[0].correction
    assert "human-facing collection display" in (
        pytest_executed_during_collect.issues[0].correction
    )

    runtime_materialization = diagnose_environment_failure(
        ContractError(
            "quickcheck environment health gate failed: Creating virtual environment at: .venv"
        )
    )
    assert {item.path for item in runtime_materialization.issues} == {
        ("dockerfile",),
        ("runner",),
    }
    assert all(
        item.code == FeedbackCode.VALIDATION_RUNTIME_MATERIALIZATION
        for item in runtime_materialization.issues
    )

    offline_dependency = diagnose_environment_failure(
        ContractError(
            "Runner collect returned too few framework-addressable individual test cases; "
            "required=10; received=4; output=Unknown host repo.maven.apache.org: "
            "Temporary failure in name resolution"
        )
    )
    assert {item.path for item in offline_dependency.issues} == {("dockerfile",)}
    assert offline_dependency.issues[0].code == FeedbackCode.VALIDATION_RUNTIME_MATERIALIZATION

    tox_materialization = diagnose_environment_failure(
        ContractError("Runner collect failed: py3.12: uv-sync> /opt/conda/bin/uv sync --locked")
    )
    assert all(
        item.code == FeedbackCode.VALIDATION_RUNTIME_MATERIALIZATION
        for item in tox_materialization.issues
    )

    incompatible_pytest = diagnose_environment_failure(
        ContractError(
            "Runner collect failed or returned fewer than 10 unique test points: "
            "pytest.PytestRemovedIn10Warning"
        )
    )
    assert {item.path for item in incompatible_pytest.issues} == {("dockerfile",)}
    assert incompatible_pytest.issues[0].code == FeedbackCode.BUILD_DEPENDENCY_VERSION
    assert incompatible_pytest.issues[0].expected == "pytest<9"

    pytest_self_version = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: ERROR: /testbed/pyproject.toml: 'minversion' requires "
            "pytest-2.0, actual pytest-0.1.dev1+g68016f0ef"
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            package_managers=("pip",),
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in pytest_self_version.issues} == {("dockerfile",)}
    assert pytest_self_version.issues[0].code == FeedbackCode.BUILD_DEPENDENCY_VERSION
    assert "shallow source build" in (pytest_self_version.issues[0].actual or "")

    collected_but_unparsed = diagnose_environment_failure(
        ContractError(
            "Runner collect failed or returned fewer than 10 unique test points: "
            "received=0; tests/test_api.py: 42"
        )
    )
    assert {item.path for item in collected_but_unparsed.issues} == {("runner",)}

    dependency = diagnose_environment_failure(
        ContractError("Runner collect failed: ModuleNotFoundError in site-packages")
    )
    assert {item.path for item in dependency.issues} == {("dockerfile",)}

    local_reactor_dependency = diagnose_environment_failure(
        InfrastructureError(
            "Failed to execute goal on project test-jpms: Could not resolve dependencies "
            "for project com.example:test-jpms:jar:1.2.3-SNAPSHOT: Could not find artifact "
            "com.example:library:jar:1.2.3-SNAPSHOT"
        )
    )
    assert {item.path for item in local_reactor_dependency.issues} == {("dockerfile",)}
    assert local_reactor_dependency.issues[0].code == FeedbackCode.BUILD_LOCAL_REACTOR_DEPENDENCY
    assert "com.example:library:jar:1.2.3-SNAPSHOT" in (
        local_reactor_dependency.issues[0].actual or ""
    )
    assert "install" in local_reactor_dependency.issues[0].correction

    dynamic_maven_provider = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: Failed to execute goal "
            "org.apache.maven.plugins:maven-surefire-plugin:3.5.6:test on project gson: "
            "Failed to collect dependencies at "
            "org.apache.maven.surefire:surefire-junit4:jar:3.5.6"
        )
    )
    assert {item.path for item in dynamic_maven_provider.issues} == {("dockerfile",)}
    assert dynamic_maven_provider.issues[0].code == FeedbackCode.BUILD_MAVEN_TEST_PROVIDER
    assert "surefire-junit4" in (dynamic_maven_provider.issues[0].actual or "")
    assert "test goal" in dynamic_maven_provider.issues[0].correction

    missing_ruby_test_dependency = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: Failure/Error: require 'aruba/api'\n\n"
            "LoadError:\n  cannot load such file -- aruba/api"
        ),
        context=FailureDiagnosticContext(
            primary_language="ruby",
            package_managers=("bundler", "rubygems"),
            test_frameworks=("rspec",),
        ),
    )
    assert {item.path for item in missing_ruby_test_dependency.issues} == {("dockerfile",)}
    assert missing_ruby_test_dependency.issues[0].code == FeedbackCode.BUILD_RUBY_TEST_DEPENDENCY
    assert "aruba/api" in (missing_ruby_test_dependency.issues[0].actual or "")
    assert "Runner" in missing_ruby_test_dependency.issues[0].correction

    ruby_deployment_without_lock = diagnose_environment_failure(
        InfrastructureError(
            "bundle install failed: The deployment setting requires a lockfile. "
            "Please make sure you have checked your Gemfile.lock into version control."
        ),
        context=FailureDiagnosticContext(
            primary_language="ruby",
            build_systems=("bundler",),
            package_managers=("bundler",),
        ),
    )
    assert len(ruby_deployment_without_lock.issues) == 1
    assert ruby_deployment_without_lock.issues[0].path == ("dockerfile",)
    assert ruby_deployment_without_lock.issues[0].code == FeedbackCode.BUILD_RUBY_DEPLOYMENT_LOCK
    assert "deployment" in ruby_deployment_without_lock.issues[0].correction

    runner_plugin = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: ImportError: Error importing plugin "
            "'/tmp/task/featurebench_pytest_plugin.py': No module named '/tmp/task'"
        )
    )
    assert {item.path for item in runner_plugin.issues} == {("runner",)}
    assert runner_plugin.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "importable module name" in (runner_plugin.issues[0].expected or "")

    runner_helper = diagnose_environment_failure(
        ContractError(
            'Runner collect failed: Traceback: File "/tmp/pytest-runner-123/runner_plugin.py", '
            "line 39, in pytest_sessionfinish; NameError: name 'action' is not defined"
        )
    )
    assert {item.path for item in runner_helper.issues} == {("runner",)}
    assert runner_helper.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "temporary helper" in (runner_helper.issues[0].expected or "")

    compact_runner_helper = diagnose_environment_failure(
        ContractError(
            'Runner collect failed: INTERNALERROR> File "/tmp/task/p/fb.py", line 40; '
            "ValueError: plugin already registered"
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in compact_runner_helper.issues} == {("runner",)}

    truncated_runner_plugin_traceback = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: pytest assertion rewriting crashed while compiling "
            'the generated plugin: TypeError: required field "lineno" missing from alias'
        )
    )
    assert {item.path for item in truncated_runner_plugin_traceback.issues} == {("runner",)}
    assert truncated_runner_plugin_traceback.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT

    invalid_pytest_hook = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: INTERNALERROR> pluggy._manager.PluginValidationError: "
            "Plugin 'featurebench-artifact-plugin' for hook 'pytest_runtest_logreport'; "
            "hookimpl definition: pytest_runtest_logreport(r); "
            "Argument(s) {'r'} are declared in the hookimpl but can not be found in the hookspec"
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in invalid_pytest_hook.issues} == {("runner",)}
    assert invalid_pytest_hook.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "exact framework hook signatures" in (invalid_pytest_hook.issues[0].expected or "")

    current = artifacts()
    merged = merge_artifact_repair(
        current,
        dependency,
        EnvironmentArtifactRepair(dockerfile=current.dockerfile + "\n# dependency repair"),
    )
    assert merged.runner == current.runner

    shadowed_project = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: ImportError: cannot import name 'new_api' "
            "from 'project' (/opt/conda/lib/python3.12/site-packages/project.py)"
        )
    )
    assert {item.path for item in shadowed_project.issues} == {("dockerfile",)}
    assert shadowed_project.issues[0].code == FeedbackCode.BUILD_PROJECT_IMPORT_SHADOWED
    assert shadowed_project.issues[0].expected == (
        "the tested project import resolves to the pinned checkout under /testbed"
    )
    assert shadowed_project.issues[0].actual == (
        "the import resolved to /opt/conda/lib/python3.12/site-packages/project.py"
    )

    shadowed_inside_testbed_venv = diagnose_environment_failure(
        ContractError(
            "Runner collect failed: ImportError: cannot import name 'new_api' "
            "from 'project' (/testbed/.venv/lib/python3.12/site-packages/project.py)"
        )
    )
    assert shadowed_inside_testbed_venv.issues[0].code == (
        FeedbackCode.BUILD_PROJECT_IMPORT_SHADOWED
    )
    assert shadowed_inside_testbed_venv.issues[0].path == ("dockerfile",)

    ambiguous = diagnose_environment_failure(
        ContractError("Runner collect failed with exit code 1 and no structured response")
    )
    assert {item.path for item in ambiguous.issues} == {("dockerfile",), ("runner",)}

    unavailable_git = diagnose_environment_failure(
        ContractError("activation failed: git is unavailable after environment activation")
    )
    assert {item.path for item in unavailable_git.issues} == {
        ("activation",),
        ("dockerfile",),
    }

    activation_timeout = diagnose_environment_failure(
        ContractError("activation failed: activation timed out: command timed out after 120s")
    )
    assert {item.path for item in activation_timeout.issues} == {("activation",)}

    missing_activation_environment = diagnose_environment_failure(
        ContractError(
            "activation failed to enter /testbed: /testbed/.venv/bin/activate: "
            "No such file or directory"
        )
    )
    assert {item.path for item in missing_activation_environment.issues} == {
        ("activation",),
        ("dockerfile",),
    }

    quickcheck_timeout = diagnose_environment_failure(
        EnvironmentValidationError(
            "quickcheck environment health gate failed: Runner run timed out",
            phase="quickcheck",
            repair_fields=("dockerfile", "activation", "runner"),
        )
    )
    assert {item.path for item in quickcheck_timeout.issues} == {
        ("dockerfile",),
        ("activation",),
        ("runner",),
    }
    assert all(
        item.context == {"failure_phase": "quickcheck"} for item in quickcheck_timeout.issues
    )

    unhealthy_quickcheck = diagnose_environment_failure(
        ContractError(
            "quickcheck environment health gate failed: required_passed=5; "
            "observed_passed=4; sampled=10; assertion failures"
        )
    )
    assert {item.path for item in unhealthy_quickcheck.issues} == {
        ("dockerfile",),
        ("activation",),
        ("runner",),
    }

    unmapped_pytest_evidence = diagnose_environment_failure(
        ContractError(
            "quickcheck environment health gate failed: observed_passed=0; "
            "requested test produced no unambiguous pytest evidence; 10 passed in 1.29s"
        ),
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
        ),
    )
    assert {item.path for item in unmapped_pytest_evidence.issues} == {("runner",)}
    assert "same pytest process" in unmapped_pytest_evidence.issues[0].context["repair_strategy"]

    insufficient_baseline = diagnose_environment_failure(
        ContractError(
            "baseline sweep produced too few verified passing test points: "
            "required=10; received=7; collected=42; groups=6"
        )
    )
    assert {item.path for item in insufficient_baseline.issues} == {
        ("dockerfile",),
        ("activation",),
        ("runner",),
    }
    assert sum(item.evidence is not None for item in insufficient_baseline.issues) == 1

    unknown_run_id = diagnose_environment_failure(
        ContractError(
            "Runner execution reported test IDs absent from stable collection: "
            "action=run; unexpected_ids=['pytest::unknown']"
        )
    )
    assert {item.path for item in unknown_run_id.issues} == {("runner",)}
    assert unknown_run_id.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "stable collect inventory" in (unknown_run_id.issues[0].expected or "")


def test_dependency_version_repair_preserves_unrelated_pip_requirements() -> None:
    current = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile
            + "\nRUN python -m pip install pytest coverage pretend\n"
            + "RUN python -m pip install -e .\n"
        }
    )
    constraints = AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_DEPENDENCY_VERSION,
                path=("dockerfile",),
                expected="pytest<9",
            ),
        )
    )
    valid = current.model_copy(
        update={
            "dockerfile": current.dockerfile.replace(
                "pytest coverage pretend", "'pytest<9' coverage pretend"
            )
        }
    )
    validate_artifact_repair_constraints(current, valid, constraints, analysis())

    runtime_version_contract = AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_DEPENDENCY_VERSION,
                path=("dockerfile",),
                expected=(
                    "the installed pytest distribution reports a version satisfying "
                    "the repository's minversion 2.0"
                ),
            ),
        )
    )
    vcs_metadata_fix = current.model_copy(
        update={
            "dockerfile": current.dockerfile.replace(
                "RUN python -m pip install pytest coverage pretend",
                "RUN git fetch --tags --force\nRUN python -m pip install pytest coverage pretend",
            )
        }
    )
    validate_artifact_repair_constraints(
        current,
        vcs_metadata_fix,
        runtime_version_contract,
        analysis(),
    )

    regressed = current.model_copy(
        update={
            "dockerfile": current.dockerfile.replace(
                "pytest coverage pretend", "'pytest<9' coverage"
            )
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifact_repair_constraints(current, regressed, constraints, analysis())
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_REPAIR_REGRESSION
    assert "pretend" in (caught.value.feedback.issues[0].actual or "")

    lost_checkout = valid.model_copy(
        update={
            "dockerfile": valid.dockerfile.replace(
                "RUN python -m pip install --no-cache-dir -e .\n", ""
            ).replace("RUN python -m pip install -e .\n", "")
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifact_repair_constraints(current, lost_checkout, constraints, analysis())
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_REPAIR_REGRESSION
    assert "-e" not in (caught.value.feedback.issues[0].actual or "")
    assert "." in (caught.value.feedback.issues[0].actual or "")


def test_runner_failure_summary_keeps_failed_results_and_selection_mismatch() -> None:
    response = RunnerResponse(
        action="run",
        exit_code=1,
        output="ModuleNotFoundError: No module named 'optional_dependency'",
        results=(
            RunnerTestPointExecution(
                id="tests/test_api.py::test_import",
                outcome="error",
                message="ModuleNotFoundError: No module named 'optional_dependency'",
            ),
        ),
    )
    evidence = summarize_runner_failure(
        response,
        expected_ids=("tests/test_api.py::test_import", "tests/test_api.py::test_other"),
    )
    assert '"exit_code":1' in evidence
    assert "ModuleNotFoundError" in evidence
    assert "tests/test_api.py::test_other" in evidence
    feedback = diagnose_environment_failure(
        ContractError(f"quickcheck environment health gate failed: {evidence}")
    )
    assert {item.path for item in feedback.issues} == {("dockerfile",)}


def test_missing_pytest_plugin_is_attributed_from_runner_source() -> None:
    error = ContractError(
        "Runner collect failed: ImportError: Error importing plugin "
        "'pytest_json_plugin': No module named 'pytest_json_plugin'"
    )
    runner_plugin = diagnose_environment_failure(
        error,
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
            runner_source="exec pytest -p pytest_json_plugin --collect-only",
        ),
    )
    assert {item.path for item in runner_plugin.issues} == {("runner",)}
    assert runner_plugin.issues[0].code == FeedbackCode.ARTIFACT_RUNNER_CONTRACT
    assert "pytest_json_plugin" in (runner_plugin.issues[0].actual or "")

    repository_plugin = diagnose_environment_failure(
        error,
        context=FailureDiagnosticContext(
            primary_language="python",
            test_frameworks=("pytest",),
            runner_source="exec pytest --collect-only",
        ),
    )
    assert {item.path for item in repository_plugin.issues} == {("dockerfile",)}


def test_runner_failure_summary_keeps_both_root_cause_and_final_status() -> None:
    response = RunnerResponse(
        action="collect",
        exit_code=1,
        output=(
            "LoadError:\n  cannot load such file -- aruba/api\n"
            + "stack frame\n" * 500
            + "0 examples, 0 failures, 1 error occurred outside of examples\n"
        ),
    )
    evidence = summarize_runner_failure(response)
    assert "aruba/api" in evidence
    assert "output omitted" in evidence
    assert "1 error occurred outside of examples" in evidence
    feedback = diagnose_environment_failure(
        ContractError(f"Runner collect failed: {evidence}"),
        context=FailureDiagnosticContext(
            primary_language="ruby",
            package_managers=("bundler",),
            test_frameworks=("rspec",),
        ),
    )
    assert len(feedback.issues) == 1
    assert feedback.issues[0].path == ("dockerfile",)
    assert feedback.issues[0].code == FeedbackCode.BUILD_RUBY_TEST_DEPENDENCY


def test_activation_must_enter_testbed_without_exiting_the_caller() -> None:
    selected = analysis()
    missing_cd = artifacts().model_copy(
        update={"activation": "#!/usr/bin/env bash\n# activation deliberately does nothing\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(missing_cd, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_ACTIVATION_WORKDIR
    terminating = artifacts().model_copy(
        update={"activation": "#!/usr/bin/env bash\ncd /testbed\nexit 0\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(terminating, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_ACTIVATION_CONTROL


def test_dockerfile_must_not_create_project_state_before_source_marker() -> None:
    selected = analysis()
    premature_environment = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile.replace(
                SOURCE_MARKER,
                f"RUN python -m venv /testbed/.venv\n{SOURCE_MARKER}",
            )
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(premature_environment, settings(), selected)
    feedback = caught.value.feedback.issues[0]
    assert feedback.code == FeedbackCode.ARTIFACT_SOURCE_PREFIX_TESTBED
    assert feedback.path == ("dockerfile",)

    independent_environment = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile.replace(
                SOURCE_MARKER,
                f"RUN python -m venv /opt/build-env\n{SOURCE_MARKER}",
            )
        }
    )
    validate_artifacts(independent_environment, settings(), selected)


def test_uv_sync_requires_activation_of_the_built_project_environment() -> None:
    selected = analysis()
    authored = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile.replace(
                "python -m pip install --no-cache-dir -e .",
                "uv sync --frozen --group tests",
            ),
            "activation": "#!/usr/bin/env bash\ncd /testbed\n",
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(authored, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_ACTIVATION_ENVIRONMENT


def test_uv_sync_rejects_unsupported_no_lock_option_before_build() -> None:
    selected = analysis()
    authored = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile.replace(
                "python -m pip install --no-cache-dir -e .",
                "uv sync --no-lock --group tests",
            ),
            "activation": "#!/usr/bin/env bash\ncd /testbed\nexport PATH=/testbed/.venv/bin:$PATH\n",
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(authored, settings(), selected)
    feedback = caught.value.feedback.issues[0]
    assert feedback.code == FeedbackCode.BUILD_DEPENDENCY_LOCK
    assert feedback.path == ("dockerfile",)
    assert feedback.actual == "uv sync --no-lock is unsupported"


def test_uv_activation_rule_does_not_leak_outside_python_ecosystem() -> None:
    selected = analysis().model_copy(
        update={
            "primary_language": "javascript",
            "languages": ("javascript",),
            "build_systems": (),
            "package_managers": ("npm",),
            "test_frameworks": ("jest",),
        }
    )
    assert selected.ecosystem_modules == ("node",)
    authored = artifacts().model_copy(
        update={
            "dockerfile": artifacts().dockerfile.replace(
                "python -m pip install --no-cache-dir -e .",
                "uv sync --frozen --group tests",
            ),
            "activation": "#!/usr/bin/env bash\ncd /testbed\n",
        }
    )
    validate_artifacts(authored, settings(), selected)


def test_runner_must_not_synchronize_dependencies_at_runtime() -> None:
    selected = analysis()
    unsafe = artifacts().model_copy(update={"runner": artifacts().runner + "uv run pytest -q\n"})
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(unsafe, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_RUNTIME_MUTATION

    safe = artifacts().model_copy(
        update={"runner": artifacts().runner + "uv run --no-sync pytest -q\n"}
    )
    validate_artifacts(safe, settings(), selected)

    quoted_unsafe = artifacts().model_copy(
        update={
            "runner": artifacts().runner + "python - <<'PY'\ncmd = ['uv', 'run', 'pytest']\nPY\n"
        }
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(quoted_unsafe, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_RUNTIME_MUTATION

    tox_unsafe = artifacts().model_copy(
        update={"runner": artifacts().runner + "tox run -e py3.12\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(tox_unsafe, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_RUNTIME_MUTATION

    external_activation = artifacts().model_copy(
        update={"runner": artifacts().runner + "source /testbed/activate\n"}
    )
    with pytest.raises(AgentFeedbackError) as caught:
        validate_artifacts(external_activation, settings(), selected)
    assert caught.value.feedback.issues[0].code == FeedbackCode.ARTIFACT_EXTERNAL_RUNTIME_REFERENCE
