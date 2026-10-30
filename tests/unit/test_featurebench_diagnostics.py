from __future__ import annotations

import json
from pathlib import Path

from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.diagnostics import FeatureBenchDiagnostics
from codeworld.adapters.featurebench.models import (
    InterfaceContract,
    InterfaceDescription,
    IssueDraft,
    MaskPlan,
    QualityReport,
    SourceObject,
    TraceBundle,
)
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.errors import RunnerTimeoutError
from codeworld.models import RuntimeSpec, TaskRecord
from codeworld.selection import WorkState


def environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id="environment-1",
        repository="https://github.com/owner/repo.git",
        commit="a" * 40,
        language="python",
        gpu_requirement="none",
        image="owner/repo@sha256:" + "b" * 64,
        dockerfile="FROM python:3.12\nRUN python -m pip install pytest\n",
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Test-only Runner fixture with enough content for the public contract.\n"
                "echo '{\"exit_code\":0}'\n"
            ),
        ),
        verified_test_points=(
            CollectedTestPoint(id="tests/test_api.py::test_feature", path="tests/test_api.py"),
        ),
    )


def source() -> SourceObject:
    return SourceObject(
        object_id="pkg/api.py::feature::1",
        path="pkg/api.py",
        qualname="feature",
        name="feature",
        kind="function",
        line=1,
        start_line=1,
        end_line=20,
        signature="def feature(value: int) -> int:",
        classification="top",
    )


def plan() -> MaskPlan:
    return MaskPlan(
        seed_test_file="tests/test_api.py",
        top_objects=(source(),),
        p2p_test_ids=("tests/test_api.py::test_other",),
    )


def quality(*, accepted: bool) -> QualityReport:
    return QualityReport(
        accepted=accepted,
        fail_to_pass=("tests/test_api.py::test_feature",),
        pass_to_pass=("tests/test_api.py::test_other",),
        seed_pass_rate=0,
        f2p_ratio=0.5,
        gold_patch_lines=20,
        gold_patch_files=1,
        errors=() if accepted else ("P2P count is below minimum",),
        test_exit_code=1,
        test_output="real pytest output",
    )


def task() -> TaskRecord:
    return TaskRecord(
        task_id="featurebench-task-1",
        environment_id=environment().environment_id,
        image=environment().image,
        runtime=environment().runtime,
        adapter={"type": "featurebench", "metrics": {"gold_patch": {"lines": 20}}},
    )


def test_failed_seed_retains_complete_attempt_and_failure_ledger(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    seed.begin()
    seed.record_trace(
        {"successful_test_ids": ("tests/test_api.py::test_feature",)},
        TraceBundle(test_file="tests/test_api.py", records=()),
    )
    seed.record_top((source(),))
    seed.record_initial_plan(plan())
    seed.record_validation_started(
        attempt=0,
        phase="initial",
        plan=plan(),
    )
    seed.record_candidate_patch("diff --git a/pkg/api.py b/pkg/api.py\n")
    seed.record_validation_finished(quality(accepted=False))
    seed.finalize(
        "rejected",
        phase="deterministic_rollback",
        error_type="CandidateRejected",
        error_text="rollback exhausted",
    )
    store.finish(None)

    assert seed.work_state(frozenset()) is WorkState.REJECTED

    attempt = seed.root / "invocations/0001/attempts/0001"
    assert (attempt / "plan.json").is_file()
    assert (attempt / "quality.json").is_file()
    assert (attempt / "test-output.log").read_text() == "real pytest output"
    failure = json.loads(store.failures_path.read_text().splitlines()[0])
    assert "format" not in json.loads((store.root / "summary.json").read_text())
    assert "format" not in json.loads(seed.summary_path.read_text())
    assert failure["seed_test_file"] == "tests/test_api.py"
    assert failure["phase"] == "deterministic_rollback"
    assert failure["diagnostics_path"] == seed.relative_path


def test_clean_baseline_backoff_is_durable_before_a_failed_retry(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    seed.begin()
    seed.record_clean_baseline_backoff(
        phase="started",
        previous_p2p_test_points=100,
        maximum_p2p_test_points=50,
        error=RunnerTimeoutError("initial Runner run timed out"),
    )
    seed.record_clean_baseline_backoff(
        phase="replanned",
        previous_p2p_test_points=100,
        maximum_p2p_test_points=50,
        plan=plan(),
    )
    seed.record_clean_baseline_backoff(
        phase="retry_failed",
        previous_p2p_test_points=100,
        maximum_p2p_test_points=50,
        plan=plan(),
        error=RunnerTimeoutError("reduced Runner run timed out"),
    )

    invocation = seed.root / "invocations/0001"
    assert (invocation / "clean-baseline-replan.json").is_file()
    events = [json.loads(line) for line in seed.events_path.read_text().splitlines()]
    assert [event["kind"] for event in events[-3:]] == [
        "clean_baseline_p2p_backoff_started",
        "clean_baseline_p2p_backoff",
        "clean_baseline_p2p_backoff_retry_failed",
    ]
    assert events[-1]["data"]["error_type"] == "RunnerTimeoutError"


def test_runner_capacity_is_monotonic_and_requires_a_matching_signature(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    inventory = {
        "tests/test_api.py": (
            "tests/test_api.py::test_feature",
            "tests/test_api.py::test_other",
        )
    }
    signature = store.runner_capacity_signature(inventory)
    assert signature == store.runner_capacity_signature(
        {"tests/test_api.py": tuple(reversed(inventory["tests/test_api.py"]))}
    )

    assert store.load_runner_capacity(signature) is None
    assert (
        store.record_runner_capacity(
            signature=signature,
            test_capacity=200,
            source_seed_test_file="tests/test_api.py",
        )
        == 200
    )
    assert (
        store.record_runner_capacity(
            signature=signature,
            test_capacity=300,
            source_seed_test_file="tests/test_larger.py",
        )
        == 200
    )
    assert json.loads(store.runner_capacity_path.read_text())["source_seed_test_file"] == (
        "tests/test_api.py"
    )
    assert (
        store.record_runner_capacity(
            signature=signature,
            test_capacity=100,
            source_seed_test_file="tests/test_smaller.py",
        )
        == 100
    )
    assert json.loads(store.runner_capacity_path.read_text())["source_seed_test_file"] == (
        "tests/test_smaller.py"
    )

    resumed = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    assert resumed.load_runner_capacity(resumed.runner_capacity_signature(inventory)) == 100
    incompatible_concurrency = dict(signature)
    incompatible_concurrency["seed_concurrency"] = 1
    assert resumed.load_runner_capacity(incompatible_concurrency) is None
    assert (
        resumed.load_runner_capacity(
            resumed.runner_capacity_signature(
                {"tests/test_api.py": ("tests/test_api.py::test_different",)}
            )
        )
        is None
    )


def test_capacity_replan_records_the_exact_plan(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    seed.begin()
    seed.record_clean_baseline_capacity_replan(
        plan(),
        previous_p2p_test_points=100,
        maximum_p2p_test_points=50,
        test_capacity=51,
    )

    invocation = seed.root / "invocations/0001"
    assert (invocation / "clean-baseline-capacity-replan.json").is_file()
    event = json.loads(seed.events_path.read_text().splitlines()[-1])
    assert event["kind"] == "clean_baseline_capacity_replan"
    assert event["data"]["test_capacity"] == 51


def test_success_retention_keeps_summary_but_removes_heavy_evidence(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "trajectories"
    trajectory = trajectory_root / "featurebench_issue_author-run"
    trajectory.mkdir(parents=True)
    (trajectory / "step.json").write_text("{}\n")
    store = FeatureBenchDiagnostics(
        tmp_path / "run",
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    seed.begin()
    seed.record_agent_run(
        "issue_author",
        "agent_run_started",
        {"trajectory_path": str(trajectory)},
    )
    candidate = task()
    seed.record_candidate(
        task=candidate,
        plan=plan(),
        quality=quality(accepted=True),
        corruption_patch="diff --git a/pkg/api.py b/pkg/api.py\n",
        interfaces=(
            InterfaceDescription(path="pkg/api.py", code="def feature(): ...", description="x"),
        ),
        issue=IssueDraft(
            task_statement="Implement the feature.",
            interface_contracts=(
                InterfaceContract(object_id=source().object_id, docstring="Feature contract."),
            ),
        ),
    )
    seed.retain_success_summary(candidate, {"gold_patch_lines": 20})
    seed.finalize("succeeded", phase="completed")
    assert seed.work_state(frozenset()) is WorkState.UNSEEN
    assert seed.work_state(frozenset((candidate.task_id,))) is WorkState.SUCCEEDED
    seed.remove_success_trajectories(trajectory_root)
    seed.compact_success_audit()
    store.finish(None)

    assert not trajectory.exists()
    assert not (seed.root / "invocations").exists()
    assert (seed.root / "summary.json").is_file()
    assert (seed.root / "final-summary.json").is_file()
    assert json.loads(seed.summary_path.read_text())["status"] == "succeeded"


def test_featurebench_defaults_retain_only_failure_evidence() -> None:
    settings = FeatureBenchSettings()
    assert settings.audit.retention == "failures"
    assert settings.agent.trajectory.enabled is True
    assert settings.agent.trajectory.retention == "failures"


def test_featurebench_diagnostics_are_isolated_by_environment(tmp_path: Path) -> None:
    first = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    second = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment().model_copy(update={"environment_id": "environment-2"}),
        settings=FeatureBenchSettings(),
    )

    assert first.root != second.root
    assert first.root.parent == second.root.parent


def test_run_error_does_not_mark_an_unselected_seed_as_failed(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_unselected.py")

    store.finish(RuntimeError("another selected Seed failed"))

    assert seed.summary.status == "skipped"
    assert seed.work_state(frozenset()) is WorkState.UNSEEN


def test_running_seed_is_recovered_as_interrupted_work(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_interrupted.py")
    seed.begin()

    assert seed.work_state(frozenset()) is WorkState.INTERRUPTED

    store.finish(RuntimeError("simulated shutdown"))

    assert seed.summary.status == "interrupted"
    assert seed.work_state(frozenset()) is WorkState.INTERRUPTED
    failure = json.loads(store.failures_path.read_text().splitlines()[0])
    assert failure["disposition"] == "interrupted"
    assert failure["error_type"] == "RuntimeError"


def test_quality_gate_failure_retains_seed_owned_evidence(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    seed.begin()
    seed.record_candidate(
        task=task(),
        plan=plan(),
        quality=quality(accepted=True),
        corruption_patch="diff --git a/pkg/api.py b/pkg/api.py\n",
        interfaces=(
            InterfaceDescription(path="pkg/api.py", code="def feature(): ...", description="x"),
        ),
        issue=IssueDraft(
            task_statement="Implement the feature.",
            interface_contracts=(
                InterfaceContract(object_id=source().object_id, docstring="Feature contract."),
            ),
        ),
    )

    error = RuntimeError("Runner returned an invalid response")
    seed.record_quality_gate_error(error)
    seed.finalize("failed", phase="gold_closure", error=error)
    store.finish(None)

    evidence = json.loads((seed.root / "final-quality-gate-error.json").read_text())
    assert evidence == {
        "error": "Runner returned an invalid response",
        "error_type": "RuntimeError",
    }
    assert seed.summary.status == "failed"
    assert seed.work_state(frozenset()) is WorkState.FAILED


def test_seed_events_carry_the_explicit_seed_invocation_across_resume_attempts(
    tmp_path: Path,
) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_resume.py")
    seed.begin()
    seed.record_top((source(),))
    seed.finalize("interrupted", phase="top_selection", error=RuntimeError("restart"))
    seed.begin()
    seed.record_top((source(),))

    events = [json.loads(line) for line in seed.events_path.read_text().splitlines()]
    first_start = next(
        index for index, event in enumerate(events) if event["kind"] == "seed_started"
    )
    second_start = next(
        index
        for index, event in enumerate(events[first_start + 1 :], start=first_start + 1)
        if event["kind"] == "seed_started"
    )
    assert {event["seed_invocation"] for event in events[:second_start]} == {1}
    assert {event["seed_invocation"] for event in events[second_start:]} == {2}


def test_final_mask_selection_is_a_distinct_monotonic_stage(tmp_path: Path) -> None:
    store = FeatureBenchDiagnostics(
        tmp_path,
        run_id="generate-run",
        environment=environment(),
        settings=FeatureBenchSettings(),
    )
    seed = store.seed("tests/test_api.py")
    selected_plan = plan()
    accepted = quality(accepted=True)
    seed.begin()
    seed.record_validation_started(attempt=0, phase="initial", plan=selected_plan)
    seed.record_validation_finished(accepted)
    seed.record_final_mask_selected(
        plan=selected_plan,
        quality=accepted,
        repair_strategy="rollback",
    )

    selection = json.loads((seed.root / "invocations/0001/final-mask-selection.json").read_text())
    assert len(selection["plan_sha256"]) == 64
    assert selection["validation_count"] == 1
    assert selection["repair_strategy"] == "rollback"
    events = [json.loads(line) for line in seed.events_path.read_text().splitlines()]
    final = next(event for event in events if event["kind"] == "final_mask_selected")
    assert final["seed_invocation"] == 1
    assert final["data"]["plan_sha256"] == selection["plan_sha256"]
