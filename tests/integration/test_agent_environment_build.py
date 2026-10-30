from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from codeworld.agents.factory import AgentFactory
from codeworld.agents.telemetry import AgentRunMetric
from codeworld.config import (
    AgentMetricsSettings,
    AgentSettings,
    AgentTrajectorySettings,
    BaseImageSettings,
    DockerSettings,
    EnvironmentSettings,
    RepositorySettings,
)
from codeworld.environment.builder import EnvironmentBuilder
from codeworld.environment.jsonl import EnvironmentJsonlAppender, iter_environments
from codeworld.repositories.staging import RepositoryStager
from codeworld.runtime.container import DockerRuntime

pytestmark = pytest.mark.integration
_PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"


@pytest.mark.asyncio
async def test_real_agents_build_and_validate_public_python_repository(tmp_path: Path) -> None:
    if os.getenv("FEATUREBENCH_RUN_AGENT_TESTS") != "1":
        pytest.skip("set FEATUREBENCH_RUN_AGENT_TESTS=1 for the real Agent build smoke test")
    load_dotenv(_PROJECT_ENV, override=False)
    model = os.getenv("FEATUREBENCH_AGENT_TEST_MODEL", "openai:gpt-5.4-mini")
    base = os.getenv("FEATUREBENCH_AGENT_BASE_IMAGE")
    if not base:
        pytest.skip("FEATUREBENCH_AGENT_BASE_IMAGE must name a prepared local clean base image")

    repository_url = "https://github.com/pallets/itsdangerous.git"
    metrics_path = tmp_path / "agent-metrics.jsonl"
    trajectory_directory = tmp_path / "agent-trajectories"
    runtime = DockerRuntime(
        DockerSettings(
            pull=False,
            build_timeout_seconds=1800,
            command_timeout_seconds=300,
        )
    )
    environment_settings = EnvironmentSettings(
        planner_image=base,
        collect_timeout_seconds=180,
        max_build_attempts=3,
        base_images=(
            BaseImageSettings(
                name="local-python-agent-smoke",
                image=base,
                languages=("python",),
                runtime_family="python",
                runtime_version="3.10",
                capabilities=(
                    "bash",
                    "git",
                    "pip",
                    "pytest",
                    "uv",
                    "conda",
                    "build-toolchain",
                ),
            ),
        ),
    )
    await runtime.check_available()
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(update_submodules=False, fetch_lfs=False),
    )
    async with stager.stage(repository_url) as (repository, commit):
        record = await EnvironmentBuilder(
            runtime,
            environment_settings,
            agents=AgentFactory(
                AgentSettings(
                    model=model,
                    tool_retries=10,
                    output_retries=10,
                    analysis_run_timeout_seconds=600,
                    artifact_run_timeout_seconds=3600,
                    request_timeout_seconds=600,
                    tool_timeout_seconds=180,
                    metrics=AgentMetricsSettings(enabled=True, path=metrics_path),
                    trajectory=AgentTrajectorySettings(
                        enabled=True,
                        directory=trajectory_directory,
                    ),
                )
            ),
            run_id="agent-build-smoke",
            task_id="pallets/itsdangerous",
        ).build(
            repository=repository,
            repository_url=repository_url,
            commit=commit,
        )

    output = tmp_path / "environments.jsonl"
    with EnvironmentJsonlAppender(output) as writer:
        assert writer.append(record)
    assert tuple(iter_environments(output)) == (record,)
    assert record.repository == repository_url
    assert record.commit == commit
    assert record.image.startswith("sha256:")
    assert "/testbed" in record.dockerfile
    assert "/opt/featurebench" not in record.dockerfile
    assert record.runtime.workdir == "/testbed"
    metrics = [
        AgentRunMetric.model_validate_json(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {metric.agent_name for metric in metrics} >= {
        "environment_analysis",
        "environment_artifact",
    }
    assert all(metric.status == "succeeded" for metric in metrics)
    assert all(metric.requests > 0 and metric.total_tokens > 0 for metric in metrics)
    for metric in metrics:
        assert metric.trajectory_path is not None
        trajectory = Path(metric.trajectory_path)
        attempts = sorted(trajectory.glob("attempt-*"))  # noqa: ASYNC240 - post-run audit
        assert attempts
        assert all((attempt / "run.json").is_file() for attempt in attempts)
        assert all((attempt / "events.jsonl").is_file() for attempt in attempts)
        assert all(list((attempt / "snapshots").glob("*.json")) for attempt in attempts)
