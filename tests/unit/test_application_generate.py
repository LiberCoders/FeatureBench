from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from pydantic import BaseModel, ConfigDict

from codeworld.adapters.base import (
    AdapterDefinition,
    AdapterSettings,
    GeneratorContext,
    QualityGateResult,
)
from codeworld.adapters.catalog import AdapterCatalog, AdapterInvocation, AdapterRegistration
from codeworld.application.generate import collect_validated_tasks, run_generate
from codeworld.config import FactorySettings
from codeworld.environment.jsonl import EnvironmentJsonlAppender, capture_environment_snapshot
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.errors import ContractError, NoWorkSelected
from codeworld.events import Event
from codeworld.models import RuntimeSpec, TaskRecord
from codeworld.run_layout import RunLayout
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import WorkState, default_work_selection_registry
from codeworld.state import JobStatus, StateStore
from codeworld.tasks.generation import iter_environment_generation_failures
from codeworld.tasks.jsonl import TaskJsonlAppender, read_tasks


def _append_environment_record(path: Path, record: EnvironmentRecord) -> None:
    with EnvironmentJsonlAppender(path) as writer:
        assert writer.append(record)


@pytest.fixture(autouse=True)
def stub_generate_resource_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str | None, str | None, Collection[str] | None]]:
    calls: list[tuple[str | None, str | None, Collection[str] | None]] = []

    async def cleanup(
        runtime: DockerRuntime,
        *,
        protected_images: Collection[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        calls.append((runtime.run_id, runtime.resource_scope, protected_images))
        return (), ()

    monkeypatch.setattr(DockerRuntime, "cleanup_run_resources", cleanup)
    return calls


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["example"]
    accepted: bool


def task(
    task_id: str,
    *,
    accepted: bool,
    environment_id: str = "environment-1",
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        environment_id=environment_id,
        image="example/image@sha256:" + "a" * 64,
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Minimal fixture implementing the external Runner artifact contract.\n"
                "# The application test never executes this script.\n"
                "echo '{}'\n"
            ),
        ),
        adapter={"type": "example", "accepted": accepted},
    )


class Generator:
    async def generate(
        self,
        environment: EnvironmentRecord,
        context: GeneratorContext,
    ) -> AsyncGenerator[TaskRecord, None]:
        del environment, context
        yield task("accepted", accepted=True)
        yield task("rejected", accepted=False)


class QualityGate:
    async def check(
        self,
        task: TaskRecord,
        payload: BaseModel,
        context: GeneratorContext,
    ) -> QualityGateResult:
        del task, context
        typed = cast(Payload, payload)
        return QualityGateResult(
            passed=typed.accepted,
            message="expected rejection" if not typed.accepted else "",
            metrics={"checked": True},
        )


class Events:
    def __init__(self) -> None:
        self.items: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.items.append(event)


class Progress:
    def __init__(self) -> None:
        self.created = 0
        self.rejected = 0
        self.failed = 0

    async def task_created(self) -> None:
        self.created += 1

    async def candidate_rejected(self, *, current: str | None = None) -> None:
        del current
        self.rejected += 1

    async def candidate_failed(self, *, current: str | None = None) -> None:
        del current
        self.failed += 1


@pytest.mark.asyncio
async def test_generate_application_writes_only_quality_gate_passes() -> None:
    events = Events()
    progress = Progress()
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=events,
            progress=progress,
            work_dir=None,
            run_id="run",
        ),
    )
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=Generator(),
        quality_gate=QualityGate(),
    )

    tasks = await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
    )

    assert [item.task_id for item in tasks] == ["accepted"]
    assert progress.created == 1
    assert progress.rejected == 1
    assert [item.kind for item in events.items] == [
        "quality_gate_started",
        "quality_gate_passed",
        "quality_gate_started",
        "quality_gate_rejected",
    ]


@pytest.mark.asyncio
async def test_generate_bounds_concurrent_quality_gates() -> None:
    class ManyCandidateGenerator:
        def __init__(self) -> None:
            self.generated: list[str] = []

        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            for index in range(7):
                task_id = f"candidate-{index}"
                self.generated.append(task_id)
                yield task(task_id, accepted=True)

    class BlockingQualityGate:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0
            self.window_filled = asyncio.Event()
            self.release = asyncio.Event()

        async def check(
            self,
            task: TaskRecord,
            payload: BaseModel,
            context: GeneratorContext,
        ) -> QualityGateResult:
            del task, payload, context
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if self.active == 3:
                self.window_filled.set()
            try:
                await self.release.wait()
                return QualityGateResult(passed=True)
            finally:
                self.active -= 1

    generator = ManyCandidateGenerator()
    quality_gate = BlockingQualityGate()
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=generator,
        quality_gate=quality_gate,
        quality_gate_concurrency=3,
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=Events(),
            progress=Progress(),
            work_dir=None,
            run_id="run",
        ),
    )

    production = asyncio.create_task(
        collect_validated_tasks(
            definition=definition,
            environment=batch_environment("environment-1"),
            context=context,
        )
    )
    await asyncio.wait_for(quality_gate.window_filled.wait(), timeout=1)
    assert generator.generated == ["candidate-0", "candidate-1", "candidate-2"]
    assert quality_gate.maximum_active == 3
    quality_gate.release.set()

    tasks = await asyncio.wait_for(production, timeout=1)
    assert {item.task_id for item in tasks} == {f"candidate-{index}" for index in range(7)}
    assert quality_gate.maximum_active == 3


@pytest.mark.asyncio
async def test_completed_quality_gates_are_reaped_while_the_next_candidate_is_slow() -> None:
    allow_last_candidate = asyncio.Event()

    class SlowFourthCandidateGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            for index in range(3):
                yield task(f"candidate-{index}", accepted=True)
            await allow_last_candidate.wait()
            yield task("candidate-3", accepted=True)

    progress = Progress()
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=SlowFourthCandidateGenerator(),
        quality_gate=QualityGate(),
        quality_gate_concurrency=4,
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=Events(),
            progress=progress,
            work_dir=None,
            run_id="run",
        ),
    )

    production = asyncio.create_task(
        collect_validated_tasks(
            definition=definition,
            environment=batch_environment("environment-1"),
            context=context,
        )
    )
    for _ in range(20):
        if progress.created == 3:
            break
        await asyncio.sleep(0)
    assert progress.created == 3
    assert not production.done()

    allow_last_candidate.set()
    tasks = await asyncio.wait_for(production, timeout=1)
    assert {item.task_id for item in tasks} == {
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    }


@pytest.mark.asyncio
async def test_task_limit_reserves_only_the_remaining_quality_gate_slots() -> None:
    class RecordingGenerator:
        def __init__(self) -> None:
            self.generated: list[str] = []

        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            for index in range(5):
                task_id = f"candidate-{index}"
                self.generated.append(task_id)
                yield task(task_id, accepted=True)

    generator = RecordingGenerator()
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=generator,
        quality_gate=QualityGate(),
        quality_gate_concurrency=4,
        maximum_tasks=2,
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=Events(),
            progress=Progress(),
            work_dir=None,
            run_id="run",
        ),
    )

    tasks = await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
    )

    assert {item.task_id for item in tasks} == {"candidate-0", "candidate-1"}
    assert generator.generated == ["candidate-0", "candidate-1"]


@pytest.mark.asyncio
async def test_generate_application_propagates_quality_gate_failures() -> None:
    class BrokenQualityGate:
        async def check(
            self,
            task: TaskRecord,
            payload: BaseModel,
            context: GeneratorContext,
        ) -> QualityGateResult:
            del task, payload, context
            raise RuntimeError("container unavailable")

    events = Events()
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=events,
            progress=Progress(),
            work_dir=None,
            run_id="run",
        ),
    )
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=Generator(),
        quality_gate=BrokenQualityGate(),
    )

    with pytest.raises(RuntimeError, match="container unavailable"):
        await collect_validated_tasks(
            definition=definition,
            environment=batch_environment("environment-1"),
            context=context,
        )

    assert [item.kind for item in events.items] == ["quality_gate_started"]


@pytest.mark.asyncio
async def test_adapter_lifecycle_isolates_quality_gate_failure_to_current_candidate() -> None:
    class TwoCandidateGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            yield task("broken", accepted=True)
            yield task("healthy", accepted=True)

    class FailFirstQualityGate:
        async def check(
            self,
            task: TaskRecord,
            payload: BaseModel,
            context: GeneratorContext,
        ) -> QualityGateResult:
            del payload, context
            if task.task_id == "broken":
                raise RuntimeError("runner response lost")
            return QualityGateResult(passed=True)

    class Lifecycle:
        def __init__(self) -> None:
            self.failed: list[str] = []
            self.checked: list[str] = []

        async def run_started(
            self, environment: EnvironmentRecord, context: GeneratorContext
        ) -> None:
            del environment, context

        async def candidate_checked(
            self,
            task: TaskRecord,
            result: QualityGateResult,
            context: GeneratorContext,
        ) -> None:
            del result, context
            self.checked.append(task.task_id)

        async def candidate_failed(
            self,
            task: TaskRecord,
            error: BaseException,
            context: GeneratorContext,
        ) -> None:
            del error, context
            self.failed.append(task.task_id)

        async def run_finished(
            self, context: GeneratorContext, error: BaseException | None
        ) -> None:
            del context, error

    lifecycle = Lifecycle()
    events = Events()
    progress = Progress()
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=TwoCandidateGenerator(),
        quality_gate=FailFirstQualityGate(),
        lifecycle=lifecycle,
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=events,
            progress=progress,
            work_dir=None,
            run_id="run",
        ),
    )

    tasks = await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
    )

    assert [candidate.task_id for candidate in tasks] == ["healthy"]
    assert lifecycle.failed == ["broken"]
    assert lifecycle.checked == ["healthy"]
    assert progress.failed == 1
    assert progress.created == 1
    assert [event.kind for event in events.items] == [
        "quality_gate_started",
        "quality_gate_failed",
        "quality_gate_started",
        "quality_gate_passed",
    ]


@pytest.mark.asyncio
async def test_generate_rejects_a_task_owned_by_another_environment() -> None:
    class WrongEnvironmentGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            yield task("wrong-owner", accepted=True, environment_id="environment-2")

    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=WrongEnvironmentGenerator(),
        quality_gate=QualityGate(),
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=Events(),
            progress=Progress(),
            work_dir=None,
            run_id="run",
        ),
    )

    with pytest.raises(ContractError, match="belongs to environment environment-2"):
        await collect_validated_tasks(
            definition=definition,
            environment=batch_environment("environment-1"),
            context=context,
        )


@pytest.mark.asyncio
async def test_task_limit_counts_quality_gate_passes_and_closes_generator() -> None:
    class SequencedGenerator:
        def __init__(self) -> None:
            self.generated: list[str] = []
            self.closed = False

        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del environment, context
            try:
                for task_id, accepted in (
                    ("rejected", False),
                    ("accepted", True),
                    ("unneeded", True),
                ):
                    self.generated.append(task_id)
                    yield task(task_id, accepted=accepted)
            finally:
                self.closed = True

    generator = SequencedGenerator()
    context = cast(
        GeneratorContext,
        SimpleNamespace(
            runtime=None,
            events=Events(),
            progress=Progress(),
            work_dir=None,
            run_id="run",
        ),
    )
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=generator,
        quality_gate=QualityGate(),
        maximum_tasks=1,
    )

    tasks = await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
    )

    assert [item.task_id for item in tasks] == ["accepted"]
    assert generator.generated == ["rejected", "accepted"]
    assert generator.closed


@pytest.mark.asyncio
async def test_generate_persists_each_quality_pass_before_observer_notification() -> None:
    order: list[str] = []

    class Lifecycle:
        async def run_started(
            self, environment: EnvironmentRecord, context: GeneratorContext
        ) -> None:
            del environment, context

        async def candidate_checked(
            self,
            task: TaskRecord,
            result: QualityGateResult,
            context: GeneratorContext,
        ) -> None:
            del task, result, context
            order.append("observed")

        async def candidate_failed(
            self,
            task: TaskRecord,
            error: BaseException,
            context: GeneratorContext,
        ) -> None:
            del task, error, context

        async def run_finished(
            self, context: GeneratorContext, error: BaseException | None
        ) -> None:
            del context, error

    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=Generator(),
        quality_gate=QualityGate(),
        lifecycle=Lifecycle(),
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(runtime=None, events=Events(), progress=Progress(), run_id="run"),
    )

    async def persist(candidate: TaskRecord) -> bool:
        order.append(f"persisted:{candidate.task_id}")
        return True

    await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
        persist=persist,
    )

    assert order == ["persisted:accepted", "observed", "observed"]


@pytest.mark.asyncio
async def test_generate_counts_only_newly_persisted_tasks_as_created() -> None:
    progress = Progress()
    definition = AdapterDefinition(
        type="example",
        payload_model=Payload,
        generator=Generator(),
        quality_gate=QualityGate(),
    )
    context = cast(
        GeneratorContext,
        SimpleNamespace(runtime=None, events=Events(), progress=progress, run_id="run"),
    )

    await collect_validated_tasks(
        definition=definition,
        environment=batch_environment("environment-1"),
        context=context,
        persist=lambda candidate: False,
    )

    assert progress.created == 0
    assert progress.rejected == 1


def test_task_jsonl_appender_is_idempotent_and_repairs_a_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    accepted = task("accepted", accepted=True)
    with TaskJsonlAppender(path) as writer:
        assert writer.append(accepted)
        assert not writer.append(accepted)

    with path.open("ab") as handle:
        handle.write(b'{"task_id":"torn"')
    with TaskJsonlAppender(path) as writer:
        assert writer.task_ids == frozenset({"accepted"})
        assert writer.task_ids_for_environment("environment-1") == frozenset({"accepted"})
        assert writer.task_ids_for_environment("environment-2") == frozenset()
        assert writer.append(task("second", accepted=True, environment_id="environment-2"))
        assert writer.task_ids_for_environment("environment-2") == frozenset({"second"})

    assert [item.task_id for item in read_tasks(path)] == ["accepted", "second"]


class BatchSettings(AdapterSettings):
    pass


def batch_environment(environment_id: str) -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id=environment_id,
        repository=f"https://example.com/{environment_id}.git",
        commit=(environment_id[-1] if environment_id[-1].isalnum() else "a") * 40,
        language="python",
        gpu_requirement="none",
        image=f"example/{environment_id}:latest",
        dockerfile=(
            "FROM python:3.13-slim\n"
            f"RUN git clone https://example.com/{environment_id}.git /testbed\n"
            "WORKDIR /testbed\n"
        ),
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\n# test activation\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Test-only Runner V1 fixture with enough source text for validation.\n"
                "echo '{}'\n"
            ),
        ),
        verified_test_points=(CollectedTestPoint(id="test_one", path="tests/test_one.py"),),
    )


def batch_task(environment: EnvironmentRecord) -> TaskRecord:
    return task(
        f"task-{environment.environment_id}",
        accepted=True,
        environment_id=environment.environment_id,
    ).model_copy(update={"image": environment.image, "runtime": environment.runtime})


async def run_batch_generate(
    tmp_path: Path,
    *,
    catalog: AdapterCatalog,
    environments: Path,
    limit: int | None = None,
) -> tuple[list[TaskRecord], list[Event], StateStore]:
    output_directory = tmp_path / "generate-output"
    settings = FactorySettings(
        workspace=tmp_path / "workspace",
        adapters={"example": {"environment_concurrency": 2}},
    )
    state = StateStore(RunLayout(output_directory).state)
    state.create_run("generate-run", "generate", settings.model_dump(mode="json"))
    events = Events()
    await run_generate(
        settings=settings,
        catalog=catalog,
        environment_path=environments,
        environment_id=None,
        limit=limit,
        adapter_type="example",
        output=output_directory / "tasks.jsonl",
        failures=output_directory / "failures.jsonl",
        events=events,
        state=state,
        run_id="generate-run",
        selection=default_work_selection_registry().resolve("unseen"),
    )
    return read_tasks(output_directory / "tasks.jsonl"), events.items, state


@pytest.mark.asyncio
async def test_generate_snapshots_and_processes_environments_in_adapter_owned_parallelism(
    tmp_path: Path,
    stub_generate_resource_cleanup: list[tuple[str | None, str | None, Collection[str] | None]],
) -> None:
    source = tmp_path / "environments.jsonl"
    for index in range(1, 4):
        _append_environment_record(source, batch_environment(f"environment-{index}"))
    active = 0
    maximum_active = 0
    processed: list[str] = []
    appended_after_start = False

    class BatchGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            nonlocal active, maximum_active, appended_after_start
            del context
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                if not appended_after_start:
                    appended_after_start = True
                    _append_environment_record(source, batch_environment("environment-4"))
                await asyncio.sleep(0.02)
                processed.append(environment.environment_id)
                yield batch_task(environment)
            finally:
                active -= 1

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=BatchGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )

    tasks, events, state = await run_batch_generate(
        tmp_path,
        catalog=catalog,
        environments=source,
    )
    try:
        assert maximum_active == 2
        assert sorted(processed) == [
            "environment-1",
            "environment-2",
            "environment-3",
        ]
        assert {item.task_id for item in tasks} == {
            "task-environment-1",
            "task-environment-2",
            "task-environment-3",
        }
        completed_event = next(event for event in events if event.kind == "completed")
        assert completed_event.data["tasks_created"] == 3
        assert completed_event.data["tasks_total"] == 3
        snapshot_event = next(
            event for event in events if event.kind == "generation_snapshot_captured"
        )
        assert snapshot_event.data["snapshot_environments"] == 3
        assert snapshot_event.data["environment_concurrency"] == 2
        snapshots = tuple(RunLayout(tmp_path / "generate-output").input_snapshots.glob("*.jsonl"))
        assert len(snapshots) == 1
        assert len(capture_environment_snapshot(snapshots[0]).records) == 3
        assert len(capture_environment_snapshot(source).records) == 4
        assert {job.status for job in state.jobs("generate-run")} == {"completed"}
        assert stub_generate_resource_cleanup == [
            ("generate-run", DockerRuntime.scope_for_workspace(tmp_path / "workspace"), None),
            ("generate-run", DockerRuntime.scope_for_workspace(tmp_path / "workspace"), None),
        ]
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_bounds_pending_jobs_to_environment_concurrency(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    for index in range(1, 6):
        _append_environment_record(source, batch_environment(f"environment-{index}"))
    window_full = asyncio.Event()
    release = asyncio.Event()
    started: list[str] = []

    class BlockingGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            started.append(environment.environment_id)
            if len(started) == 2:
                window_full.set()
            await release.wait()
            yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=BlockingGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )
    output_directory = tmp_path / "generate-output"
    settings = FactorySettings(
        workspace=tmp_path / "workspace",
        adapters={"example": {"environment_concurrency": 2}},
    )
    state = StateStore(RunLayout(output_directory).state)
    state.create_run("generate-run", "generate", settings.model_dump(mode="json"))
    running = asyncio.create_task(
        run_generate(
            settings=settings,
            catalog=catalog,
            environment_path=source,
            environment_id=None,
            limit=None,
            adapter_type="example",
            output=output_directory / "tasks.jsonl",
            failures=output_directory / "failures.jsonl",
            events=Events(),
            state=state,
            run_id="generate-run",
            selection=default_work_selection_registry().resolve("unseen"),
        )
    )
    try:
        await asyncio.wait_for(window_full.wait(), timeout=1)
        jobs = state.jobs("generate-run", "task-generation")
        assert len(jobs) == 2
        assert {job.status for job in jobs} == {"running"}

        release.set()
        await running
        assert len(state.jobs("generate-run", "task-generation")) == 5
        assert len(read_tasks(output_directory / "tasks.jsonl")) == 5
    finally:
        release.set()
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        state.close()


@pytest.mark.asyncio
async def test_generate_creates_one_adapter_invocation_for_all_environments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    for index in range(1, 4):
        _append_environment_record(source, batch_environment(f"environment-{index}"))
    invocation_count = 0
    definition_count = 0
    observed_services: list[object] = []

    class ScopedGenerator:
        def __init__(self, service: object) -> None:
            self.service = service

        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            observed_services.append(self.service)
            yield batch_task(environment)

    def factory(settings: AdapterSettings) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=ScopedGenerator(object()),
            quality_gate=QualityGate(),
        )

    def invocation_factory(settings: AdapterSettings) -> AdapterInvocation:
        nonlocal invocation_count
        del settings
        invocation_count += 1
        shared_service = object()

        def build_definition() -> AdapterDefinition:
            nonlocal definition_count
            definition_count += 1
            return AdapterDefinition(
                type="example",
                payload_model=Payload,
                generator=ScopedGenerator(shared_service),
                quality_gate=QualityGate(),
            )

        return AdapterInvocation(type="example", definition_factory=build_definition)

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
            invocation_factory=invocation_factory,
        )
    )

    tasks, _, state = await run_batch_generate(
        tmp_path,
        catalog=catalog,
        environments=source,
    )
    try:
        assert len(tasks) == 3
        assert invocation_count == 1
        assert definition_count == 3
        assert len({id(service) for service in observed_services}) == 1
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_treats_empty_adapter_selection_as_a_skip(tmp_path: Path) -> None:
    source = tmp_path / "environments.jsonl"
    _append_environment_record(source, batch_environment("environment-1"))

    class EmptySelectionGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            if environment.environment_id:
                raise NoWorkSelected("fixture selection matched no work")
            yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=EmptySelectionGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )

    tasks, events, state = await run_batch_generate(
        tmp_path,
        catalog=catalog,
        environments=source,
    )
    try:
        assert tasks == []
        assert (
            tuple(
                iter_environment_generation_failures(
                    tmp_path / "generate-output" / "failures.jsonl"
                )
            )
            == ()
        )
        job = state.get_job("generate-run", "environment-1")
        assert job is not None and job.status == "skipped"
        assert any(event.kind == "generation_environment_skipped" for event in events)
        assert not any(event.kind == "generation_environment_rejected" for event in events)
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_marks_environment_failed_when_all_candidates_fail_locally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    _append_environment_record(source, batch_environment("environment-1"))

    class OneCandidateGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            yield batch_task(environment)

    class BrokenQualityGate:
        async def check(
            self,
            task: TaskRecord,
            payload: BaseModel,
            context: GeneratorContext,
        ) -> QualityGateResult:
            del task, payload, context
            raise RuntimeError("runner response lost")

    class Lifecycle:
        async def run_started(
            self, environment: EnvironmentRecord, context: GeneratorContext
        ) -> None:
            del environment, context

        async def candidate_checked(
            self,
            task: TaskRecord,
            result: QualityGateResult,
            context: GeneratorContext,
        ) -> None:
            del task, result, context

        async def candidate_failed(
            self,
            task: TaskRecord,
            error: BaseException,
            context: GeneratorContext,
        ) -> None:
            del task, error, context

        async def run_finished(
            self, context: GeneratorContext, error: BaseException | None
        ) -> None:
            del context, error

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        generator = OneCandidateGenerator()
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=generator,
            quality_gate=BrokenQualityGate(),
            lifecycle=Lifecycle(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )

    tasks, events, state = await run_batch_generate(
        tmp_path,
        catalog=catalog,
        environments=source,
    )
    try:
        assert tasks == []
        job = state.get_job("generate-run", "environment-1")
        assert job is not None and job.status == "failed"
        assert job.error is not None and "1 candidates failed" in job.error
        assert any(event.kind == "generation_environment_failed" for event in events)
        assert not any(event.kind == "generation_environment_rejected" for event in events)
    finally:
        state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_status", "selection_name", "expected_states"),
    (
        ("queued", "unseen", frozenset({WorkState.UNSEEN})),
        (
            "queued",
            "non-success",
            frozenset(
                {
                    WorkState.UNSEEN,
                    WorkState.FAILED,
                    WorkState.REJECTED,
                    WorkState.INTERRUPTED,
                }
            ),
        ),
        (
            "running",
            "non-success",
            frozenset(
                {
                    WorkState.UNSEEN,
                    WorkState.FAILED,
                    WorkState.REJECTED,
                    WorkState.INTERRUPTED,
                }
            ),
        ),
        (
            "queued",
            "unseen+failed+interrupted",
            frozenset(
                {
                    WorkState.UNSEEN,
                    WorkState.FAILED,
                    WorkState.INTERRUPTED,
                }
            ),
        ),
    ),
)
async def test_generate_passes_the_shared_work_selection_to_adapter_children(
    tmp_path: Path,
    job_status: JobStatus,
    selection_name: str,
    expected_states: frozenset[WorkState],
) -> None:
    source = tmp_path / "environments.jsonl"
    environment = batch_environment("environment-1")
    _append_environment_record(source, environment)
    observed_states: list[frozenset[WorkState]] = []

    class InterruptedGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            observed_states.append(context.selection.states)
            if context.selection.accepts(WorkState.UNSEEN):
                yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=InterruptedGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )
    output_directory = tmp_path / "generate-output"
    settings = FactorySettings(
        workspace=tmp_path / "workspace",
        adapters={"example": {"environment_concurrency": 2}},
    )
    state = StateStore(RunLayout(output_directory).state)
    state.create_run("generate-run", "generate", settings.model_dump(mode="json"))
    state.upsert_job(
        "generate-run",
        environment.environment_id,
        "task-generation",
        job_status,
        {"environment_id": environment.environment_id},
        attempt=1,
        result={"task_ids": []},
    )
    try:
        await run_generate(
            settings=settings,
            catalog=catalog,
            environment_path=source,
            environment_id=None,
            limit=None,
            adapter_type="example",
            output=output_directory / "tasks.jsonl",
            failures=output_directory / "failures.jsonl",
            events=Events(),
            state=state,
            run_id="generate-run",
            selection=default_work_selection_registry().resolve(selection_name),
        )

        assert observed_states == [expected_states]
        assert [item.task_id for item in read_tasks(output_directory / "tasks.jsonl")] == [
            "task-environment-1"
        ]
        job = state.get_job("generate-run", environment.environment_id)
        assert job is not None and job.status == "completed" and job.attempt == 2
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_isolates_one_environment_failure_and_honors_snapshot_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    for index in range(1, 4):
        _append_environment_record(source, batch_environment(f"environment-{index}"))

    class FailingGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            if environment.environment_id == "environment-2":
                raise RuntimeError("isolated failure")
            yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=FailingGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )

    tasks, _, state = await run_batch_generate(
        tmp_path,
        catalog=catalog,
        environments=source,
        limit=2,
    )
    try:
        assert [item.task_id for item in tasks] == ["task-environment-1"]
        failures = tuple(
            iter_environment_generation_failures(tmp_path / "generate-output" / "failures.jsonl")
        )
        assert len(failures) == 1
        assert failures[0].environment_id == "environment-2"
        assert failures[0].error == "isolated failure"
        assert {job.job_id for job in state.jobs("generate-run")} == {
            "environment-1",
            "environment-2",
        }
        failed_job = state.get_job("generate-run", "environment-2")
        assert failed_job is not None
        assert failed_job.status == "failed"
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_resume_captures_a_new_snapshot_and_adds_new_environments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    _append_environment_record(source, batch_environment("environment-1"))
    processed: list[str] = []

    class ResumeGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            processed.append(environment.environment_id)
            yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=ResumeGenerator(),
            quality_gate=QualityGate(),
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )
    output_directory = tmp_path / "generate-output"
    settings = FactorySettings(
        workspace=tmp_path / "workspace",
        adapters={"example": {"environment_concurrency": 2}},
    )
    state = StateStore(RunLayout(output_directory).state)
    state.create_run("generate-run", "generate", settings.model_dump(mode="json"))
    task_summaries: list[tuple[int, int]] = []
    try:
        for resume in (False, True):
            events = Events()
            await run_generate(
                settings=settings,
                catalog=catalog,
                environment_path=source,
                environment_id=None,
                limit=None,
                adapter_type="example",
                output=output_directory / "tasks.jsonl",
                failures=output_directory / "failures.jsonl",
                events=events,
                state=state,
                run_id="generate-run",
                selection=default_work_selection_registry().resolve("unseen"),
            )
            completed_event = next(event for event in events.items if event.kind == "completed")
            task_summaries.append(
                (
                    cast(int, completed_event.data["tasks_created"]),
                    cast(int, completed_event.data["tasks_total"]),
                )
            )
            if not resume:
                _append_environment_record(source, batch_environment("environment-2"))

        assert {item.task_id for item in read_tasks(output_directory / "tasks.jsonl")} == {
            "task-environment-1",
            "task-environment-2",
        }
        assert processed == ["environment-1", "environment-1", "environment-2"]
        assert task_summaries == [(1, 1), (1, 2)]
        snapshots = sorted(RunLayout(output_directory).input_snapshots.glob("*.jsonl"))
        assert sorted(len(capture_environment_snapshot(path).records) for path in snapshots) == [
            1,
            2,
        ]
        first_job = state.get_job("generate-run", "environment-1")
        second_job = state.get_job("generate-run", "environment-2")
        assert first_job is not None and first_job.attempt == 2
        assert second_job is not None and second_job.attempt == 1
    finally:
        state.close()


@pytest.mark.asyncio
async def test_generate_recovers_environment_task_ownership_without_sqlite_jobs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    _append_environment_record(source, batch_environment("environment-1"))
    generated: list[str] = []

    class LimitedGenerator:
        async def generate(
            self,
            environment: EnvironmentRecord,
            context: GeneratorContext,
        ) -> AsyncGenerator[TaskRecord, None]:
            del context
            generated.append(environment.environment_id)
            yield batch_task(environment)

    def factory(settings: BaseModel) -> AdapterDefinition:
        del settings
        return AdapterDefinition(
            type="example",
            payload_model=Payload,
            generator=LimitedGenerator(),
            quality_gate=QualityGate(),
            maximum_tasks=1,
        )

    catalog = AdapterCatalog()
    catalog.register(
        AdapterRegistration(
            type="example",
            settings_model=BatchSettings,
            definition_factory=factory,
        )
    )
    output_directory = tmp_path / "generate-output"
    settings = FactorySettings(
        workspace=tmp_path / "workspace",
        adapters={"example": {"environment_concurrency": 2}},
    )
    state = StateStore(RunLayout(output_directory).state)
    state.create_run("generate-run", "generate", settings.model_dump(mode="json"))
    try:
        for _ in range(2):
            await run_generate(
                settings=settings,
                catalog=catalog,
                environment_path=source,
                environment_id=None,
                limit=None,
                adapter_type="example",
                output=output_directory / "tasks.jsonl",
                failures=output_directory / "failures.jsonl",
                events=Events(),
                state=state,
                run_id="generate-run",
                selection=default_work_selection_registry().resolve("unseen"),
            )
            state.delete_jobs("generate-run", ("task-generation",))

        assert generated == ["environment-1"]
        tasks = read_tasks(output_directory / "tasks.jsonl")
        assert [item.environment_id for item in tasks] == ["environment-1"]
    finally:
        state.close()
