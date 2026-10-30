from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from codeworld.adapters.featurebench.testing import PythonTracer
from codeworld.agents.factory import AgentFactory
from codeworld.config import (
    AgentSettings,
    DockerSettings,
    EnvironmentSettings,
    SourceEndpointSettings,
    SourceMetricsSettings,
    SourceServiceSettings,
    SourceSettings,
)
from codeworld.environment.builder import EnvironmentBuilder
from codeworld.environment.runner import EnvironmentRunner
from codeworld.models import RunnerLimits, RuntimeSpec
from codeworld.runtime.container import DockerRuntime
from codeworld.sources import SourceManager

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.asyncio
async def test_conda_source_materialization_passes_the_real_dockerfile_parser(
    tmp_path: Path,
) -> None:
    if os.getenv("FEATUREBENCH_RUN_DOCKER_TESTS") != "1":
        pytest.skip("set FEATUREBENCH_RUN_DOCKER_TESTS=1 for the real Docker smoke test")
    manager = SourceManager(
        SourceSettings(
            enabled=True,
            state_path=tmp_path / "source-state.json",
            metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
            services={
                "conda": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://mirror.example/anaconda/cloud/conda-forge",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://conda.anaconda.org/conda-forge",
                            canonical=True,
                        ),
                    )
                )
            },
        )
    )
    plan = await manager.build_plan()
    rendered = manager.materialize_build_sources(
        "FROM scratch\nRUN conda install -n base -y numpy scipy\n",
        plan,
    )
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(rendered, encoding="utf-8")

    process = await asyncio.create_subprocess_exec(
        "docker",
        "buildx",
        "build",
        "--check",
        "--file",
        str(dockerfile),
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    assert process.returncode == 0, (stdout + stderr).decode(errors="replace")


@pytest.mark.asyncio
async def test_real_image_activation_and_runner_contract() -> None:
    if os.getenv("FEATUREBENCH_RUN_DOCKER_TESTS") != "1":
        pytest.skip("set FEATUREBENCH_RUN_DOCKER_TESTS=1 for the real Docker smoke test")
    smoke = _FIXTURES / "smoke_environment"
    local_base = os.getenv("FEATUREBENCH_SMOKE_BASE_IMAGE")
    pull = local_base is None
    tag = f"featurebench/test-smoke:{os.getpid()}"
    runtime = DockerRuntime(
        DockerSettings(pull=pull, build_timeout_seconds=1800, command_timeout_seconds=300)
    )
    await runtime.check_available()
    image = await runtime.build(
        _FIXTURES,
        tag=tag,
        dockerfile=smoke / ("Dockerfile" if local_base is None else "Dockerfile.local"),
        build_args={"BASE_IMAGE": local_base} if local_base is not None else None,
        pull=pull,
    )
    spec = RuntimeSpec(
        activation=(smoke / "activate").read_text(encoding="utf-8"),
        runner=(smoke / "runner").read_text(encoding="utf-8"),
        limits=RunnerLimits(
            collect_timeout_seconds=120,
            baseline_total_timeout_seconds=600,
        ),
    )
    try:
        async with runtime.session(image, runtime_spec=spec) as session:
            commit_result = await session.exec(("git", "rev-parse", "HEAD"), check=True)
            runner = EnvironmentRunner(session)
            test_id = "tests/test_feature.py::test_feature_zero#0"
            assert (await runner.run((test_id,), timeout_seconds=120)).passed_ids == (test_id,)
            trace = await PythonTracer(session).trace(
                test_command=("pytest", "-q"),
                test_file="tests/test_feature.py",
                test_ids=(test_id,),
                timeout_seconds=120,
            )
            assert trace.passed_test_ids == (test_id,)
            feature_records = [
                record
                for record in trace.records
                if record.path == "src/featurebench_fixture/core.py"
                and record.qualname == "feature"
            ]
            assert len(feature_records) == 1
            assert feature_records[0].test_ids == (test_id,)
            assert feature_records[0].object_id.startswith(
                "src/featurebench_fixture/core.py::feature::"
            )
            await session.exec(
                (
                    "python",
                    "-c",
                    "from pathlib import Path; p=Path('src/featurebench_fixture/core.py'); "
                    "p.write_text(p.read_text().replace('bias = 3', 'bias = 30'))",
                ),
                activated=True,
                check=True,
            )
            changed = await runner.run((test_id,), timeout_seconds=120)
            assert changed.exit_code == 1
            assert changed.results[0].outcome == "failed"
        builder = EnvironmentBuilder(
            runtime,
            EnvironmentSettings(
                collect_timeout_seconds=120,
                min_verified_test_points=2,
            ),
            agents=AgentFactory(AgentSettings(model="test")),
        )
        verified_test_points = await builder.validate(
            image=image,
            runtime_spec=spec,
            commit=commit_result.stdout.strip(),
        )
        assert len(verified_test_points) >= 2
    finally:
        await runtime.remove_image(tag)
