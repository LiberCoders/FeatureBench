from __future__ import annotations

import json
import subprocess
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path

import pytest

from codeworld.agents.factory import AgentFactory
from codeworld.agents.feedback import AgentFeedback
from codeworld.config import AgentSettings, FactorySettings
from codeworld.environment.build_diagnostics import (
    BuildDiagnosticStore,
)
from codeworld.environment.builder import EnvironmentBuilder
from codeworld.environment.feedback import FeedbackCode, issue
from codeworld.environment.planning import (
    SOURCE_MARKER,
    EnvironmentAnalysis,
    EnvironmentArtifacts,
)
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.provenance import write_run_provenance
from codeworld.run_layout import RunLayout
from codeworld.runtime.container import DockerBuildProgress, DockerRuntime


def analysis() -> EnvironmentAnalysis:
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
        gpu_requirement="none",
        gpu_evidence_paths=(),
        gpu_reason=None,
        selected_base_image_name="python-3.12",
        rejection_reason=None,
    )


def artifacts() -> EnvironmentArtifacts:
    return EnvironmentArtifacts(
        dockerfile="FROM python:3.12\nRUN python -m pip install pytest\n",
        activation="#!/usr/bin/env bash\ncd /testbed\n",
        runner=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'request="$1"\n'
            'response="$2"\n'
            'python - "$request" "$response" <<\'PY\'\n'
            "import json, sys\n"
            "json.dump({'action': 'collect', 'exit_code': 0, 'collected': []}, open(sys.argv[2], 'w'))\n"
            "PY\n"
        ),
    )


def test_build_diagnostics_are_human_readable_and_content_addressed(tmp_path: Path) -> None:
    store = BuildDiagnosticStore(tmp_path / "run", run_id="build-run")
    repository = store.repository(
        repository="https://github.com/owner/repo.git",
        ref="main",
        task_id="owner/repo",
        github_id=123,
    )
    repository.begin()
    repository.set_commit("a" * 40, "environment-1")
    repository.record_analysis(analysis())
    candidate = artifacts()
    dockerfile = "FROM python:3.12\n" + candidate.dockerfile
    repository.record_attempt(attempt=1, dockerfile=dockerfile, artifacts=candidate)
    repository.record_feedback(
        attempt=1,
        feedback=AgentFeedback(issues=(issue(FeedbackCode.BUILD_COMMAND, path=("dockerfile",)),)),
        failed_phase="docker_build",
    )
    repository.record_attempt(attempt=2, dockerfile=dockerfile, artifacts=candidate)

    first = repository.root / "attempts" / "0001"
    second = repository.root / "attempts" / "0002"
    assert (first / "Dockerfile").read_text(encoding="utf-8") == dockerfile
    assert (first / "activate").read_text(encoding="utf-8") == candidate.activation
    assert (first / "runner").read_text(encoding="utf-8") == candidate.runner
    assert (first / "feedback.json").is_file()
    assert (repository.root / "analysis.json").is_file()
    assert (first / "Dockerfile").stat().st_ino == (second / "Dockerfile").stat().st_ino
    assert len(tuple(store.objects.iterdir())) == 3

    summary = json.loads(repository.summary_path.read_text(encoding="utf-8"))
    assert "format" not in summary
    assert summary["attempt_count"] == 2
    assert summary["failed_phase"] == "docker_build"
    assert repository.relative_path.startswith(".featurebench/diagnostics/repositories/")


def test_failure_retention_removes_only_successful_repository_trajectories(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectories"
    successful_trajectory = trajectory_root / "environment_analysis-success"
    failed_trajectory = trajectory_root / "environment_analysis-failure"
    successful_trajectory.mkdir(parents=True)
    failed_trajectory.mkdir()
    (successful_trajectory / "run.json").write_text("{}\n", encoding="utf-8")
    (failed_trajectory / "run.json").write_text("{}\n", encoding="utf-8")

    store = BuildDiagnosticStore(tmp_path / "run", run_id="build-run")
    successful = store.repository(
        repository="https://github.com/owner/success.git",
        ref="main",
        task_id="owner/success",
        github_id=1,
    )
    failed = store.repository(
        repository="https://github.com/owner/failure.git",
        ref="main",
        task_id="owner/failure",
        github_id=2,
    )
    successful.record_agent_run(
        role="analysis",
        kind="agent_run_started",
        attempt=None,
        data={"run_id": "success", "trajectory_path": str(successful_trajectory)},
    )
    failed.record_agent_run(
        role="analysis",
        kind="agent_run_started",
        attempt=None,
        data={"run_id": "failure", "trajectory_path": str(failed_trajectory)},
    )
    successful.finalize("succeeded")
    failed.finalize("failed", error_text="real failure", error_type="ContractError")

    resumed_successful = store.repository(
        repository="https://github.com/owner/success.git",
        ref="main",
        task_id="owner/success",
        github_id=1,
    )
    removed = resumed_successful.remove_successful_trajectories(trajectory_root)
    assert removed == (str(successful_trajectory.resolve()),)
    assert not successful_trajectory.exists()
    assert failed_trajectory.is_dir()


def test_build_agent_trajectory_retention_is_explicit() -> None:
    assert AgentSettings().trajectory.retention == "failures"
    assert AgentSettings.model_validate({"trajectory": {"enabled": True}}).trajectory.retention == (
        "failures"
    )
    configured = AgentSettings.model_validate(
        {"trajectory": {"enabled": True, "retention": "failures"}}
    )
    assert configured.trajectory.enabled is True
    assert configured.trajectory.retention == "failures"


def test_default_build_profile_keeps_only_failed_repository_trajectories() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = FactorySettings.from_files((project_root / "configs" / "default" / "build.yaml",))
    assert settings.agent.trajectory.enabled is True
    assert settings.agent.trajectory.retention == "failures"


def test_build_provenance_captures_resolved_settings_and_dirty_pipeline(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(("git", "init", "-q", str(project)), check=True)
    subprocess.run(
        ("git", "-C", str(project), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(project), "config", "user.name", "FeatureBench Test"),
        check=True,
    )
    source = project / "pipeline.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(project), "add", "pipeline.py"), check=True)
    subprocess.run(("git", "-C", str(project), "commit", "-qm", "initial"), check=True)
    source.write_text("VERSION = 2\n", encoding="utf-8")

    run_directory = tmp_path / "run"
    invocation_directory = RunLayout(run_directory).invocations / "0001"
    destination = write_run_provenance(
        invocation_directory,
        run_id="build-run",
        settings=FactorySettings(),
        project_root=project,
    )

    provenance = json.loads(destination.read_text(encoding="utf-8"))
    assert "format" not in provenance
    assert provenance["run_id"] == "build-run"
    assert provenance["featurebench_version"] == version("featurebench")
    assert "codeworld_version" not in provenance
    assert provenance["git_dirty"] is True
    assert len(provenance["git_commit"]) == 40
    assert len(provenance["pipeline_patch_sha256"]) == 64
    assert (invocation_directory / provenance["pipeline_patch_path"]).is_file()
    assert (invocation_directory / provenance["resolved_settings_path"]).is_file()
    assert not (run_directory / "provenance.json").exists()
    assert not (run_directory / "resolved-settings.json").exists()

    original = destination.read_bytes()
    write_run_provenance(
        RunLayout(run_directory).invocations / "0002",
        run_id="build-run",
        settings=FactorySettings.model_validate({"concurrency": 2}),
        project_root=project,
    )
    assert destination.read_bytes() == original


@pytest.mark.asyncio
async def test_builder_persists_candidate_before_the_real_build_attempt(tmp_path: Path) -> None:
    store = BuildDiagnosticStore(tmp_path / "run", run_id="build-run")
    diagnostics = store.repository(
        repository="https://github.com/owner/repo.git",
        ref="main",
        task_id="owner/repo",
        github_id=123,
    )
    diagnostics.begin()
    diagnostics.set_commit("a" * 40, "environment-1")
    candidate = artifacts().model_copy(
        update={"dockerfile": (f"FROM ubuntu:24.04\n{SOURCE_MARKER}\nRUN echo environment-ready\n")}
    )

    class FakeRuntime(DockerRuntime):
        async def prepare_image(self, image: str, *, pull: bool) -> str:
            del pull
            return image

    class RecordingBuilder(EnvironmentBuilder):
        async def _analyze(self, **_: object) -> EnvironmentAnalysis:
            return analysis().model_copy(update={"selected_base_image_name": "generic-ubuntu"})

        async def _author(self, **_: object) -> EnvironmentArtifacts:
            return candidate

        async def _build_final(
            self,
            *,
            environment_id: str,
            dockerfile: str,
            attempt: int,
            progress: Callable[[DockerBuildProgress], Awaitable[None]] | None = None,
        ) -> tuple[str, str]:
            del environment_id, progress
            attempt_directory = diagnostics.root / "attempts" / "0001"
            assert attempt == 1
            assert (attempt_directory / "Dockerfile").read_text(encoding="utf-8") == dockerfile
            assert (attempt_directory / "activate").is_file()
            assert (attempt_directory / "runner").is_file()
            return "sha256:" + "b" * 64, "codeworld/test:attempt-1"

        async def validate(self, **_: object) -> tuple[CollectedTestPoint, ...]:
            return (CollectedTestPoint(id="tests/test_api.py::test_one", path="tests/test_api.py"),)

    record = await RecordingBuilder(
        FakeRuntime(),
        FactorySettings().environment,
        agents=AgentFactory(AgentSettings(model="test")),
        diagnostics=diagnostics,
    ).build(
        repository=tmp_path,
        repository_url="https://github.com/owner/repo.git",
        commit="a" * 40,
    )

    assert record.dockerfile == (diagnostics.root / "attempts/0001/Dockerfile").read_text(
        encoding="utf-8"
    )
