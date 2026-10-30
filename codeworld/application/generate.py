"""Adapter task generation application workflow."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from codeworld.adapters.catalog import AdapterCatalog
from codeworld.application.resources import cleanup_owned_resources, finish_owned_cleanup
from codeworld.config import FactorySettings
from codeworld.environment.jsonl import (
    EnvironmentSnapshot,
    capture_environment_snapshot,
    write_environment_snapshot,
)
from codeworld.environment.models import EnvironmentRecord
from codeworld.events import Event, EventSink
from codeworld.run_layout import RunLayout
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import WorkSelection
from codeworld.sources import SourceManager
from codeworld.state import StateStore
from codeworld.tasks.production import GenerationProducer, collect_validated_tasks

__all__ = ["collect_validated_tasks", "run_generate"]


def _selected_environments(
    snapshot: EnvironmentSnapshot,
    *,
    environment_id: str | None,
    limit: int | None,
) -> tuple[EnvironmentRecord, ...]:
    if limit is not None and limit < 1:
        raise ValueError("Generate limit must be at least one")
    if environment_id is not None and limit is not None:
        raise ValueError("Generate limit cannot be combined with environment_id")
    if environment_id is not None:
        matches = tuple(
            record for record in snapshot.records if record.environment_id == environment_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"environment_id not found in startup snapshot of {snapshot.source}: "
                f"{environment_id}"
            )
        return matches
    if not snapshot.records:
        detail = (
            f"; ignored {snapshot.ignored_tail_bytes} bytes from an incomplete trailing record"
            if snapshot.ignored_tail_bytes
            else ""
        )
        raise ValueError(f"{snapshot.source} contains no complete environments{detail}")
    return snapshot.records[:limit]


async def run_generate(
    *,
    settings: FactorySettings,
    catalog: AdapterCatalog,
    environment_path: Path,
    environment_id: str | None,
    limit: int | None,
    adapter_type: str,
    output: Path,
    failures: Path,
    events: EventSink,
    state: StateStore,
    run_id: str,
    selection: WorkSelection,
) -> str:
    """Generate one startup snapshot while owning every transient Docker resource."""

    sources = SourceManager(settings.sources, run_id=run_id)
    runtime = DockerRuntime(
        settings.docker,
        sources=sources,
        run_id=run_id,
        resource_scope=DockerRuntime.scope_for_workspace(settings.workspace),
    )
    await cleanup_owned_resources(runtime, protected_images=None)
    try:
        snapshot = capture_environment_snapshot(environment_path)
        environments = _selected_environments(
            snapshot,
            environment_id=environment_id,
            limit=limit,
        )
        snapshot_path = (
            RunLayout(output.parent).input_snapshots / f"environments-{uuid4().hex}.jsonl"
        )
        write_environment_snapshot(snapshot_path, snapshot.records)

        adapter_configuration = settings.adapters.get(adapter_type)
        if adapter_configuration is None:
            raise ValueError(
                f"selected Adapter {adapter_type!r} has no YAML configuration; include its "
                "Adapter config explicitly"
            )
        registration = catalog.get(adapter_type)
        adapter_settings = registration.validate_settings(adapter_configuration)
        concurrency = adapter_settings.environment_concurrency

        await events.emit(
            Event(
                run_id=run_id,
                kind="generation_snapshot_captured",
                stage=adapter_type,
                message=(
                    f"captured {len(snapshot.records)} complete environments and selected "
                    f"{len(environments)}"
                ),
                data={
                    "source": str(snapshot.source),
                    "source_size_bytes": snapshot.source_size_bytes,
                    "snapshot_path": str(snapshot_path),
                    "snapshot_environments": len(snapshot.records),
                    "selected_environments": len(environments),
                    "ignored_tail_bytes": snapshot.ignored_tail_bytes,
                    "environment_concurrency": concurrency,
                    "limit": limit,
                    "environment_id": environment_id,
                },
            )
        )
        await events.emit(
            Event(
                run_id=run_id,
                kind="started",
                stage=adapter_type,
                message=(
                    f"generating tasks from {len(environments)} startup-snapshot "
                    f"environments with concurrency {concurrency}"
                ),
                data={
                    "environments": len(environments),
                    "environment_concurrency": concurrency,
                },
            )
        )

        producer = GenerationProducer(
            settings=settings,
            runtime=runtime,
            registration=registration,
            adapter_settings=adapter_settings,
            events=events,
            state=state,
            run_id=run_id,
            selection=selection,
        )
        summary = await producer.produce(
            environments=environments,
            snapshot_path=snapshot_path,
            output=output,
            failures=failures,
        )
        return (
            f"processed {summary.environments} environments "
            f"({summary.completed} completed, {summary.rejected} rejected, "
            f"{summary.failed} failed, {summary.skipped} skipped); "
            f"created {summary.tasks_created} tasks; "
            f"{summary.tasks_total} total in {output}"
        )
    finally:
        # Generate consumes Environment images; it never owns or deletes them.
        await finish_owned_cleanup(runtime, protected_images=None)
