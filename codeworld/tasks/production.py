"""Bounded scheduling and durable persistence for Adapter task generation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Iterable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from codeworld.adapters.base import AdapterDefinition, AdapterSettings, GeneratorContext
from codeworld.adapters.catalog import AdapterInvocation, AdapterRegistration
from codeworld.config import FactorySettings
from codeworld.environment.models import EnvironmentRecord
from codeworld.errors import CandidateRejected, ContractError, NoWorkSelected
from codeworld.events import Event, EventSink
from codeworld.models import StrictModel, TaskRecord
from codeworld.progress import AdapterProgressReporter
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import WorkSelection, WorkState
from codeworld.state import JobRecord, JobStatus, StateStore
from codeworld.tasks.generation import (
    EnvironmentGenerationFailure,
    EnvironmentGenerationFailureAppender,
)
from codeworld.tasks.jsonl import TaskJsonlAppender

TaskSink = Callable[[TaskRecord], bool | Awaitable[bool]]
GenerationDisposition = Literal["completed", "failed", "rejected", "skipped"]


class GenerationProductionSummary(StrictModel):
    """Terminal counts for one startup-snapshot Generate batch."""

    environments: int = Field(ge=0)
    completed: int = Field(ge=0)
    rejected: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    tasks_created: int = Field(ge=0)
    tasks_total: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _GenerationTarget:
    environment: EnvironmentRecord
    prior_job: JobRecord | None
    prior_task_ids: frozenset[str]
    payload: dict[str, object]
    attempt: int


def _environment_work_dir(root: Path, environment_id: str) -> Path:
    digest = hashlib.sha256(environment_id.encode()).hexdigest()[:12]
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in environment_id
    ).strip("._-")[-80:]
    return root / f"{safe or 'environment'}-{digest}"


def _job_work_state(job: JobRecord | None) -> WorkState:
    """Recover one Environment without treating never-started work as interrupted."""

    if job is None:
        return WorkState.UNSEEN
    match job.status:
        case "failed":
            return WorkState.FAILED
        case "rejected":
            return WorkState.REJECTED
        case "running" | "interrupted":
            return WorkState.INTERRUPTED
        case "completed":
            return WorkState.SUCCEEDED
        case "queued" | "skipped":
            return WorkState.UNSEEN


def _should_schedule_environment(
    job: JobRecord | None,
    *,
    task_ids: frozenset[str],
    selection: WorkSelection,
) -> bool:
    # Once an Environment has produced a task, the Adapter still owns finer-grained
    # Seed selection. Re-enter it on Resume so changed settings can expose new work.
    return bool(task_ids) or selection.accepts(_job_work_state(job))


def _job_attempt(job: JobRecord | None) -> int:
    if job is None:
        return 1
    return job.attempt + 1


def _job_payload(
    environment: EnvironmentRecord,
    *,
    snapshot_path: Path,
) -> dict[str, object]:
    return {
        "environment_id": environment.environment_id,
        "repository": environment.repository,
        "commit": environment.commit,
        "image": environment.image,
        "snapshot_path": str(snapshot_path),
    }


async def collect_validated_tasks(
    *,
    definition: AdapterDefinition,
    environment: EnvironmentRecord,
    context: GeneratorContext,
    persist: TaskSink | None = None,
    initial_accepted: int = 0,
) -> list[TaskRecord]:
    """Run a bounded candidate/quality-gate window and retain only real passes."""

    tasks: list[TaskRecord] = []
    accepted = initial_accepted
    if definition.maximum_tasks is not None and accepted >= definition.maximum_tasks:
        return tasks

    async def check_candidate(task: TaskRecord) -> tuple[TaskRecord | None, bool]:
        canonical = TaskRecord.model_validate_json(task.model_dump_json(exclude_none=True))
        if canonical.environment_id != environment.environment_id:
            raise ContractError(
                f"task {canonical.task_id} belongs to environment "
                f"{canonical.environment_id}, expected {environment.environment_id}"
            )
        payload = definition.validate_task(canonical)
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="quality_gate_started",
                stage=definition.type,
                task_id=canonical.task_id,
                message=f"checking {canonical.task_id}",
            )
        )
        try:
            result = await definition.quality_gate.check(canonical, payload, context)
        except Exception as error:
            if definition.lifecycle is None:
                raise
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="quality_gate_failed",
                    stage=definition.type,
                    task_id=canonical.task_id,
                    message=str(error) or repr(error),
                    data={"error_type": type(error).__name__},
                )
            )
            await definition.lifecycle.candidate_failed(canonical, error, context)
            await context.progress.candidate_failed(current=canonical.task_id)
            return None, False
        if not result.passed:
            await context.events.emit(
                Event(
                    run_id=context.run_id,
                    kind="quality_gate_rejected",
                    stage=definition.type,
                    task_id=canonical.task_id,
                    message=(result.message or "Adapter quality gate rejected the candidate"),
                    data=dict(result.metrics),
                )
            )
            if definition.lifecycle is not None:
                await definition.lifecycle.candidate_checked(canonical, result, context)
            await context.progress.candidate_rejected(current=canonical.task_id)
            return None, False

        appended = True
        if persist is not None:
            persisted = persist(canonical)
            appended = await persisted if inspect.isawaitable(persisted) else persisted
        await context.events.emit(
            Event(
                run_id=context.run_id,
                kind="quality_gate_passed",
                stage=definition.type,
                task_id=canonical.task_id,
                message=f"quality gate passed for {canonical.task_id}",
                data=dict(result.metrics),
            )
        )
        if definition.lifecycle is not None:
            await definition.lifecycle.candidate_checked(canonical, result, context)
        if persist is None or appended:
            await context.progress.task_created()
        return canonical if persist is None else None, appended

    async with aclosing(definition.generator.generate(environment, context)) as candidates:
        pending: set[asyncio.Task[tuple[TaskRecord | None, bool]]] = set()
        candidate_fetch: asyncio.Task[TaskRecord] | None = None
        exhausted = False
        try:
            while pending or candidate_fetch is not None or not exhausted:
                window = (
                    definition.quality_gate_concurrency
                    if definition.maximum_tasks is None
                    else min(
                        definition.quality_gate_concurrency,
                        definition.maximum_tasks - accepted,
                    )
                )
                if window <= 0:
                    break
                if not exhausted and candidate_fetch is None and len(pending) < window:
                    candidate_fetch = asyncio.create_task(anext(candidates))
                waiters: set[asyncio.Task[object]] = set(pending)
                if candidate_fetch is not None:
                    waiters.add(candidate_fetch)
                if not waiters:
                    break
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if candidate_fetch is not None and candidate_fetch in done:
                    fetched = candidate_fetch
                    candidate_fetch = None
                    try:
                        candidate = fetched.result()
                    except StopAsyncIteration:
                        exhausted = True
                    else:
                        pending.add(asyncio.create_task(check_candidate(candidate)))
                completed_checks = pending.intersection(done)
                pending.difference_update(completed_checks)
                for checked in completed_checks:
                    task, appended = await checked
                    if task is not None:
                        tasks.append(task)
                    if appended:
                        accepted += 1
        finally:
            if candidate_fetch is not None:
                candidate_fetch.cancel()
            for checked in pending:
                checked.cancel()
            if candidate_fetch is not None:
                await asyncio.gather(candidate_fetch, return_exceptions=True)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    return tasks


class GenerationProducer:
    """Run Adapter generation with a bounded pending window and durable outcomes."""

    def __init__(
        self,
        *,
        settings: FactorySettings,
        runtime: DockerRuntime,
        registration: AdapterRegistration,
        adapter_settings: AdapterSettings,
        events: EventSink,
        state: StateStore,
        run_id: str,
        selection: WorkSelection,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.adapter_invocation: AdapterInvocation = registration.begin_validated(adapter_settings)
        self.events = events
        self.state = state
        self.run_id = run_id
        self.selection = selection
        self.adapter_type = registration.type
        self.concurrency = adapter_settings.environment_concurrency
        self.work_root = settings.workspace / "runs" / run_id / "environments"
        self.work_root.mkdir(parents=True, exist_ok=True)

    async def _prepare_target(
        self,
        environment: EnvironmentRecord,
        *,
        snapshot_path: Path,
        writer: TaskJsonlAppender,
    ) -> _GenerationTarget | None:
        prior_job = self.state.get_job(self.run_id, environment.environment_id)
        prior_task_ids = writer.task_ids_for_environment(environment.environment_id)
        if not _should_schedule_environment(
            prior_job,
            task_ids=prior_task_ids,
            selection=self.selection,
        ):
            await self.events.emit(
                Event(
                    run_id=self.run_id,
                    kind="generation_environment_skipped",
                    stage=self.adapter_type,
                    message=f"selection {self.selection.name} skipped {environment.repository}",
                    data={
                        "environment_id": environment.environment_id,
                        "repository": environment.repository,
                        "selection": self.selection.name,
                    },
                )
            )
            return None

        target = _GenerationTarget(
            environment=environment,
            prior_job=prior_job,
            prior_task_ids=prior_task_ids,
            payload=_job_payload(environment, snapshot_path=snapshot_path),
            attempt=_job_attempt(prior_job),
        )
        self._state_job(target, "queued")
        return target

    def _state_job(
        self,
        target: _GenerationTarget,
        status: JobStatus,
        *,
        task_ids: set[str] | frozenset[str] | None = None,
        extra_result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        result: dict[str, object] = {
            "task_ids": sorted(target.prior_task_ids if task_ids is None else task_ids)
        }
        if extra_result is not None:
            result.update(extra_result)
        self.state.upsert_job(
            self.run_id,
            target.environment.environment_id,
            "task-generation",
            status,
            target.payload,
            attempt=target.attempt,
            result=result,
            error=error,
        )

    async def _run_adapter(
        self,
        *,
        definition: AdapterDefinition,
        target: _GenerationTarget,
        context: GeneratorContext,
        persist: TaskSink,
        task_ids: set[str],
    ) -> tuple[BaseException | None, NoWorkSelected | None]:
        error: BaseException | None = None
        no_work: NoWorkSelected | None = None
        lifecycle_started = False
        try:
            if definition.lifecycle is not None:
                lifecycle_started = True
                await definition.lifecycle.run_started(target.environment, context)
            await collect_validated_tasks(
                definition=definition,
                environment=target.environment,
                context=context,
                persist=persist,
                initial_accepted=len(target.prior_task_ids),
            )
            if not task_ids:
                if context.progress.candidate_failures:
                    error = ContractError(
                        "adapter produced no tasks because "
                        f"{context.progress.candidate_failures} candidates failed"
                    )
                else:
                    error = CandidateRejected(
                        "adapter produced no tasks that passed its quality gates"
                    )
        except NoWorkSelected as exc:
            no_work = exc
        except CandidateRejected as exc:
            if context.progress.candidate_failures:
                error = ContractError(
                    f"{context.progress.candidate_failures} candidates failed before "
                    f"the adapter exhausted its usable work: {exc}"
                )
            else:
                error = exc
        except asyncio.CancelledError as exc:
            error = exc
        except Exception as exc:
            error = exc
        finally:
            if definition.lifecycle is not None and lifecycle_started:
                if error is None:
                    try:
                        await definition.lifecycle.run_finished(context, None)
                    except asyncio.CancelledError as exc:
                        no_work = None
                        error = exc
                    except Exception as exc:
                        no_work = None
                        error = exc
                else:
                    with suppress(Exception):
                        await definition.lifecycle.run_finished(context, error)
        return error, no_work

    async def _finish_no_work(
        self,
        *,
        target: _GenerationTarget,
        context: GeneratorContext,
        progress: AdapterProgressReporter,
        task_ids: set[str],
        no_work: NoWorkSelected,
    ) -> GenerationDisposition:
        status = "completed" if task_ids else "skipped"
        detail = str(no_work) or "current selection matched no Adapter work"
        self._state_job(
            target,
            status,
            task_ids=task_ids,
            extra_result={"no_work": detail},
        )
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="generation_environment_skipped",
                stage=self.adapter_type,
                message=f"no selected work for {target.environment.repository}: {detail}",
                data={
                    "environment_id": target.environment.environment_id,
                    "repository": target.environment.repository,
                    "selection": context.selection.name,
                    "reason": detail,
                    "task_ids": sorted(task_ids),
                },
            )
        )
        await progress.complete()
        return "skipped"

    async def _finish_success(
        self,
        *,
        target: _GenerationTarget,
        progress: AdapterProgressReporter,
        task_ids: set[str],
    ) -> GenerationDisposition:
        self._state_job(target, "completed", task_ids=task_ids)
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="generation_environment_completed",
                stage=self.adapter_type,
                message=(
                    f"completed {target.environment.repository} with {len(task_ids)} durable tasks"
                ),
                data={
                    "environment_id": target.environment.environment_id,
                    "repository": target.environment.repository,
                    "task_ids": sorted(task_ids),
                },
            )
        )
        await progress.complete()
        return "completed"

    async def _finish_failure(
        self,
        *,
        target: _GenerationTarget,
        progress: AdapterProgressReporter,
        failure_writer: EnvironmentGenerationFailureAppender,
        task_ids: set[str],
        error: BaseException,
    ) -> GenerationDisposition:
        disposition: Literal["failed", "rejected", "interrupted"]
        if isinstance(error, asyncio.CancelledError):
            disposition = "interrupted"
        elif isinstance(error, CandidateRejected):
            disposition = "rejected"
        else:
            disposition = "failed"
        detail = str(error)[-50_000:] or repr(error)
        failure_writer.append(
            EnvironmentGenerationFailure(
                run_id=self.run_id,
                adapter=self.adapter_type,
                environment_id=target.environment.environment_id,
                repository=target.environment.repository,
                image=target.environment.image,
                disposition=disposition,
                task_ids=tuple(sorted(task_ids)),
                error_type=type(error).__name__,
                error=detail,
            )
        )
        self._state_job(target, disposition, task_ids=task_ids, error=detail)
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind=f"generation_environment_{disposition}",
                stage=self.adapter_type,
                message=f"{disposition}: {target.environment.repository}: {detail}",
                data={
                    "environment_id": target.environment.environment_id,
                    "repository": target.environment.repository,
                    "error_type": type(error).__name__,
                    "task_ids": sorted(task_ids),
                },
            )
        )
        with suppress(Exception):
            await progress.fail(stopped=disposition == "interrupted")
        if isinstance(error, asyncio.CancelledError):
            raise error
        return "rejected" if disposition == "rejected" else "failed"

    async def _execute(
        self,
        target: _GenerationTarget,
        *,
        output_directory: Path,
        writer: TaskJsonlAppender,
        failure_writer: EnvironmentGenerationFailureAppender,
        writer_lock: asyncio.Lock,
    ) -> GenerationDisposition:
        environment = target.environment
        definition = self.adapter_invocation.build_definition()
        progress = AdapterProgressReporter(
            events=self.events,
            run_id=self.run_id,
            adapter=self.adapter_type,
            adapter_name=definition.display_name or self.adapter_type.replace("-", " ").title(),
            phases=definition.progress_phases,
            environment_id=environment.environment_id,
        )
        context = GeneratorContext(
            runtime=self.runtime,
            events=self.events,
            progress=progress,
            work_dir=_environment_work_dir(self.work_root, environment.environment_id),
            run_id=self.run_id,
            output_dir=output_directory,
            completed_task_ids=writer.task_ids,
            selection=self.selection,
        )
        context.work_dir.mkdir(parents=True, exist_ok=True)
        task_ids = set(target.prior_task_ids)
        self._state_job(target, "running", task_ids=task_ids)
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="generation_environment_started",
                stage=self.adapter_type,
                message=f"generating tasks from {environment.repository}",
                data={
                    "environment_id": environment.environment_id,
                    "repository": environment.repository,
                    "attempt": target.attempt,
                },
            )
        )
        await progress.start()

        async def persist(candidate: TaskRecord) -> bool:
            async with writer_lock:
                appended = writer.append(candidate)
                writer.flush()
                task_ids.add(candidate.task_id)
                self._state_job(target, "running", task_ids=task_ids)
                return appended

        error, no_work = await self._run_adapter(
            definition=definition,
            target=target,
            context=context,
            persist=persist,
            task_ids=task_ids,
        )
        if error is None and no_work is not None:
            return await self._finish_no_work(
                target=target,
                context=context,
                progress=progress,
                task_ids=task_ids,
                no_work=no_work,
            )
        if error is None:
            return await self._finish_success(
                target=target,
                progress=progress,
                task_ids=task_ids,
            )
        return await self._finish_failure(
            target=target,
            progress=progress,
            failure_writer=failure_writer,
            task_ids=task_ids,
            error=error,
        )

    async def _emit_summary(self, summary: GenerationProductionSummary, output: Path) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="completed",
                stage=self.adapter_type,
                message=(
                    f"processed {summary.environments} environments: "
                    f"{summary.completed} completed, {summary.rejected} rejected, "
                    f"{summary.failed} failed, {summary.skipped} skipped; "
                    f"created {summary.tasks_created} tasks; "
                    f"{summary.tasks_total} total in {output}"
                ),
                data=summary.model_dump(mode="json"),
            )
        )

    async def produce(
        self,
        *,
        environments: Iterable[EnvironmentRecord],
        snapshot_path: Path,
        output: Path,
        failures: Path,
    ) -> GenerationProductionSummary:
        """Generate from a bounded pending window without materializing one Task per input."""

        counts: dict[GenerationDisposition, int] = {
            "completed": 0,
            "rejected": 0,
            "failed": 0,
            "skipped": 0,
        }
        source = iter(environments)
        exhausted = False
        pending: set[asyncio.Task[GenerationDisposition]] = set()
        writer_lock = asyncio.Lock()

        with (
            TaskJsonlAppender(
                output,
                fsync_interval=self.settings.jsonl_fsync_interval,
            ) as writer,
            EnvironmentGenerationFailureAppender(failures) as failure_writer,
        ):
            initial_task_count = len(writer.records)
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < self.concurrency:
                        try:
                            environment = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        target = await self._prepare_target(
                            environment,
                            snapshot_path=snapshot_path,
                            writer=writer,
                        )
                        if target is None:
                            counts["skipped"] += 1
                            continue
                        pending.add(
                            asyncio.create_task(
                                self._execute(
                                    target,
                                    output_directory=output.parent,
                                    writer=writer,
                                    failure_writer=failure_writer,
                                    writer_lock=writer_lock,
                                )
                            )
                        )
                    if not pending:
                        continue
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        counts[await task] += 1
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            writer.flush()
            task_count = len(writer.records)

        summary = GenerationProductionSummary(
            environments=sum(counts.values()),
            completed=counts["completed"],
            rejected=counts["rejected"],
            failed=counts["failed"],
            skipped=counts["skipped"],
            tasks_created=task_count - initial_task_count,
            tasks_total=task_count,
        )
        await self._emit_summary(summary, output)
        return summary
