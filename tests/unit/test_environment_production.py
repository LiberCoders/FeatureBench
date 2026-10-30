from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from codeworld.agents.factory import AgentFactory
from codeworld.config import AgentSettings, EnvironmentSettings, FactorySettings
from codeworld.environment.build_diagnostics import BuildDiagnosticStore
from codeworld.environment.job import (
    EnvironmentBuildJob,
    EnvironmentBuildOutcome,
    EnvironmentBuildTarget,
)
from codeworld.environment.jsonl import (
    EnvironmentFailureJsonlAppender,
    EnvironmentJsonlAppender,
)
from codeworld.environment.models import EnvironmentBuildFailure, EnvironmentRecord
from codeworld.environment.production import EnvironmentProducer, EnvironmentProductionSummary
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.errors import RepositoryMaterializationRejected
from codeworld.models import RuntimeSpec
from codeworld.repositories.jsonl import RepositoryJsonlAppender
from codeworld.repositories.models import RepositoryCandidate
from codeworld.repositories.staging import RepositoryStager
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import default_work_selection_registry


def candidate(index: int = 1) -> RepositoryCandidate:
    now = datetime.now(UTC)
    return RepositoryCandidate(
        discovery_run_id="test",
        github_id=index,
        full_name=f"example/project-{index}",
        name=f"project-{index}",
        owner="example",
        clone_url=f"https://github.com/example/project-{index}.git",
        html_url=f"https://github.com/example/project-{index}",
        api_url=f"https://api.github.com/repos/example/project-{index}",
        default_branch="main",
        stars=1,
        forks=0,
        open_issues=0,
        size_kb=1,
        language="Python",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_missing_git_object_is_a_materialization_rejection(tmp_path: Path) -> None:
    class FakeRuntime:
        sources = None

    class MissingStage:
        async def __aenter__(self) -> tuple[Path, str]:
            raise RepositoryMaterializationRejected("remote does not expose the requested commit")

        async def __aexit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    class MissingObjectStager:
        def stage(self, repository_url: str, ref: str) -> MissingStage:
            del repository_url, ref
            return MissingStage()

    agent_settings = AgentSettings(model="test")
    job = EnvironmentBuildJob(
        runtime=cast(DockerRuntime, FakeRuntime()),
        settings=EnvironmentSettings(),
        agents=AgentFactory(agent_settings),
        stager=cast(RepositoryStager, MissingObjectStager()),
        diagnostics=BuildDiagnosticStore(tmp_path, run_id="test"),
    )

    outcome = await job.run(
        EnvironmentBuildTarget.from_candidate(candidate()),
        existing_environment_ids=frozenset(),
    )

    assert outcome.failure is not None
    assert outcome.failure.disposition == "rejected"
    assert outcome.failure.category == "repository_object_unavailable"
    assert outcome.failure.failed_phase == "materialization"


@pytest.mark.asyncio
async def test_environment_preflight_checks_runtime_and_repository_host_capabilities(
    tmp_path: Path,
) -> None:
    checked: list[str] = []

    class FakeRuntime:
        sources = None

        async def check_available(self) -> None:
            checked.append("runtime")

    class FakeStager:
        async def check_available(self) -> None:
            checked.append("repository")

    settings = FactorySettings(workspace=tmp_path, agent=AgentSettings(model="test"))
    producer = EnvironmentProducer(
        settings=settings,
        runtime=cast(DockerRuntime, FakeRuntime()),
        agents=AgentFactory(settings.agent),
    )
    producer.stager = cast(RepositoryStager, FakeStager())

    await producer._preflight()  # pyright: ignore[reportPrivateUsage]

    assert checked == ["runtime", "repository"]


@pytest.mark.asyncio
async def test_repository_total_deadline_returns_a_categorized_failure(tmp_path: Path) -> None:
    class FakeRuntime:
        sources = None

        async def check_available(self) -> None:
            return None

    class HangingStager:
        async def check_available(self) -> None:
            return None

        @asynccontextmanager
        async def stage(self, repository_url: str, ref: str) -> AsyncGenerator[tuple[Path, str]]:
            del repository_url, ref
            await asyncio.Event().wait()
            yield tmp_path, "a" * 40

    agent_settings = AgentSettings(model="test")
    settings = FactorySettings(
        workspace=tmp_path,
        agent=agent_settings,
        environment=EnvironmentSettings(total_timeout_seconds=0.01),
    )
    producer = EnvironmentProducer(
        settings=settings,
        runtime=cast(DockerRuntime, FakeRuntime()),
        agents=AgentFactory(agent_settings),
    )
    producer.stager = cast(RepositoryStager, HangingStager())

    output = tmp_path / "environments.jsonl"
    failures = tmp_path / "failures.jsonl"
    with pytest.raises(TimeoutError):
        await producer.produce_one(
            repository=candidate().clone_url,
            ref=candidate().default_branch,
            output=output,
            failures=failures,
        )

    failure = EnvironmentBuildFailure.model_validate_json(failures.read_text(encoding="utf-8"))
    assert failure.disposition == "failed"
    assert failure.category == "environment_total_timeout"
    assert failure.error_type == "TimeoutError"
    assert "0.01s" in failure.error


@pytest.mark.asyncio
async def test_total_deadline_removes_a_built_image_not_yet_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40

    class FakeRuntime:
        sources = None

        def __init__(self) -> None:
            self.removed: list[str] = []

        async def remove_image(self, image: str) -> None:
            self.removed.append(image)

    class ExitHangingStager:
        @asynccontextmanager
        async def stage(self, repository_url: str, ref: str) -> AsyncGenerator[tuple[Path, str]]:
            del repository_url, ref
            yield tmp_path, commit
            await asyncio.Event().wait()

    record = EnvironmentRecord(
        environment_id="environment-id",
        repository="https://github.com/example/project.git",
        commit=commit,
        language="python",
        gpu_requirement="none",
        image="codeworld/environment-id:1",
        dockerfile="FROM ubuntu:24.04\n" + "# complete Dockerfile\n" * 2,
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=("#!/usr/bin/env bash\n" + "# complete Runner fixture\n" * 5 + "true\n"),
        ),
        verified_test_points=(
            CollectedTestPoint(id="tests/test_api.py::test_one", path="tests/test_api.py"),
        ),
    )

    class FakeBuilder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        @staticmethod
        def environment_id(repository_url: str, resolved_commit: str) -> str:
            del repository_url, resolved_commit
            return record.environment_id

        async def build(self, **kwargs: object) -> EnvironmentRecord:
            del kwargs
            return record

    monkeypatch.setattr("codeworld.environment.job.EnvironmentBuilder", FakeBuilder)
    agent_settings = AgentSettings(model="test")
    settings = FactorySettings(
        workspace=tmp_path,
        agent=agent_settings,
        environment=EnvironmentSettings(total_timeout_seconds=0.01),
    )
    runtime = FakeRuntime()
    job = EnvironmentBuildJob(
        runtime=cast(DockerRuntime, runtime),
        settings=settings.environment,
        agents=AgentFactory(agent_settings),
        stager=cast(RepositoryStager, ExitHangingStager()),
        diagnostics=BuildDiagnosticStore(tmp_path, run_id="test"),
    )

    outcome = await job.run(
        EnvironmentBuildTarget.from_candidate(candidate()),
        existing_environment_ids=frozenset(),
    )

    assert outcome.failure is not None
    assert outcome.failure.category == "environment_total_timeout"
    assert runtime.removed == [record.image]


def environment_for(item: RepositoryCandidate) -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id=f"environment-{item.github_id}",
        repository=item.clone_url,
        commit=f"{item.github_id:040x}",
        language="python",
        gpu_requirement="none",
        image=f"codeworld/environment-{item.github_id}:1",
        dockerfile="FROM ubuntu:24.04\n" + "# complete Dockerfile\n" * 2,
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=("#!/usr/bin/env bash\n" + "# complete Runner fixture\n" * 5 + "true\n"),
        ),
        verified_test_points=(
            CollectedTestPoint(id="tests/test_api.py::test_one", path="tests/test_api.py"),
        ),
    )


async def _selected_build_calls(
    root: Path,
    selection_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], EnvironmentProductionSummary]:
    repositories = root / "repositories.jsonl"
    environments = root / "environments.jsonl"
    failures = root / "failures.jsonl"
    items = tuple(candidate(index) for index in range(1, 6))
    with RepositoryJsonlAppender(repositories) as writer:
        for item in items:
            assert writer.append(item)
    with EnvironmentJsonlAppender(environments) as writer:
        assert writer.append(environment_for(items[0]))
    with EnvironmentFailureJsonlAppender(failures) as writer:
        writer.append(
            EnvironmentBuildFailure(
                run_id="test",
                task_id=items[1].full_name,
                github_id=items[1].github_id,
                repository=items[1].clone_url,
                ref=items[1].default_branch,
                disposition="failed",
                failed_phase="environment_build",
                attempts=1,
                diagnostics_path="diagnostics/repositories/fixture",
                error_type="RuntimeError",
                error="expected fixture failure",
            )
        )
        writer.append(
            EnvironmentBuildFailure(
                run_id="test",
                task_id=items[2].full_name,
                github_id=items[2].github_id,
                repository=items[2].clone_url,
                ref=items[2].default_branch,
                disposition="rejected",
                failed_phase="analysis_policy",
                attempts=0,
                diagnostics_path="diagnostics/repositories/rejected-fixture",
                error_type="RepositoryRejected",
                error="expected fixture rejection",
            )
        )
        writer.append(
            EnvironmentBuildFailure(
                run_id="test",
                task_id=items[3].full_name,
                github_id=items[3].github_id,
                repository=items[3].clone_url,
                ref=items[3].default_branch,
                disposition="interrupted",
                failed_phase="environment_build",
                attempts=1,
                diagnostics_path="diagnostics/repositories/interrupted-fixture",
                error_type="CancelledError",
                error="expected fixture interruption",
            )
        )

    class FakeRuntime:
        sources = None

    settings = FactorySettings(workspace=root, agent=AgentSettings(model="test"))
    producer = EnvironmentProducer(
        settings=settings,
        runtime=cast(DockerRuntime, FakeRuntime()),
        agents=AgentFactory(settings.agent),
        selection=default_work_selection_registry().resolve(selection_name),
    )
    calls: list[int] = []

    async def preflight() -> None:
        return None

    async def execute(
        job: EnvironmentBuildJob,
        target: EnvironmentBuildTarget,
        existing_ids: frozenset[str],
    ) -> EnvironmentBuildOutcome:
        del existing_ids
        assert target.github_id is not None
        calls.append(target.github_id)
        diagnostics = job.diagnostics.repository(
            repository=target.repository_url,
            ref=target.ref,
            task_id=target.task_id,
            github_id=target.github_id,
        )
        diagnostics.begin()
        return EnvironmentBuildOutcome(target=target, diagnostics=diagnostics, skipped=True)

    monkeypatch.setattr(producer, "_preflight", preflight)
    monkeypatch.setattr(producer, "_execute", execute)
    summary = await producer.produce(
        repositories=repositories,
        output=environments,
        failures=failures,
        limit=5,
    )
    return calls, summary


@pytest.mark.asyncio
async def test_default_resume_runs_only_objects_absent_from_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, summary = await _selected_build_calls(tmp_path, "unseen", monkeypatch)

    assert calls == [5]
    assert summary.submitted == 1


@pytest.mark.asyncio
async def test_non_success_resume_runs_unseen_and_retries_durable_non_success_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, summary = await _selected_build_calls(tmp_path, "non-success", monkeypatch)

    assert calls == [2, 3, 4, 5]
    assert summary.submitted == 4


@pytest.mark.asyncio
async def test_composed_resume_selects_only_requested_build_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, summary = await _selected_build_calls(tmp_path, "unseen+failed", monkeypatch)

    assert calls == [2, 5]
    assert summary.submitted == 2
