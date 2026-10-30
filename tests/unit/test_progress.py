from __future__ import annotations

import asyncio

import pytest

from codeworld.events import Event
from codeworld.progress import AdapterPhase, AdapterProgressReporter, AdapterProgressSnapshot


class EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_adapter_progress_reporter_serializes_concurrent_updates() -> None:
    collector = EventCollector()
    reporter = AdapterProgressReporter(
        events=collector,
        run_id="run",
        adapter="example",
        adapter_name="Example",
        phases=(AdapterPhase(id="process", label="Processing", unit="items"),),
    )

    await reporter.start()
    await reporter.start_phase("process", total=10)
    await asyncio.gather(*(reporter.advance(current=f"item-{index}") for index in range(10)))
    await reporter.candidate_rejected()
    await reporter.candidate_failed()
    await reporter.task_created()
    await reporter.complete()

    snapshot = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert snapshot.status == "completed"
    assert snapshot.completed == 10
    assert snapshot.total == 10
    assert snapshot.tasks_created == 1
    assert snapshot.candidates_rejected == 1
    assert snapshot.candidates_failed == 1
    assert all(event.kind == "adapter_progress" for event in collector.events)


@pytest.mark.asyncio
async def test_adapter_progress_rejects_undeclared_phase() -> None:
    reporter = AdapterProgressReporter(
        events=EventCollector(),
        run_id="run",
        adapter="example",
        adapter_name="Example",
    )

    with pytest.raises(ValueError, match="not declared"):
        await reporter.start_phase("missing")


@pytest.mark.asyncio
async def test_adapter_progress_phase_current_is_scoped() -> None:
    collector = EventCollector()
    reporter = AdapterProgressReporter(
        events=collector,
        run_id="run",
        adapter="example",
        adapter_name="Example",
        phases=(
            AdapterPhase(id="prepare", label="Preparing"),
            AdapterPhase(id="process", label="Processing"),
        ),
    )

    await reporter.start_phase("prepare", total=1, current="input.jsonl")
    preparing = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert preparing.current == "input.jsonl"
    assert preparing.phase_label == "Preparing"

    await reporter.start_phase("process", total=2)
    next_phase = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert next_phase.current is None
    assert next_phase.phase_label == "Processing"


@pytest.mark.asyncio
async def test_adapter_progress_tracks_concurrent_work_items_independently() -> None:
    collector = EventCollector()
    reporter = AdapterProgressReporter(
        events=collector,
        run_id="run",
        adapter="example",
        adapter_name="Example",
        phases=(AdapterPhase(id="generate", label="Generating", unit="seeds"),),
    )

    await reporter.start_phase("generate", total=3)
    with pytest.raises(ValueError, match="work item ID"):
        await reporter.set_work_item_activity(" ", "Planning")
    with pytest.raises(ValueError, match="work item activity"):
        await reporter.set_work_item_activity("seed-a", " ")
    await asyncio.gather(
        reporter.set_work_item_activity("seed-b", "Planning"),
        reporter.set_work_item_activity("seed-a", "Targeting"),
    )
    active = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert tuple(item.item_id for item in active.work_items) == ("seed-a", "seed-b")
    assert tuple(item.activity for item in active.work_items) == ("Targeting", "Planning")

    await reporter.set_work_item_activity("seed-a", "Validating mask")
    await reporter.finish_work_item("seed-b")
    remaining = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert [(item.item_id, item.activity) for item in remaining.work_items] == [
        ("seed-a", "Validating mask")
    ]

    await reporter.finish_phase()
    finished = AdapterProgressSnapshot.model_validate(collector.events[-1].data)
    assert finished.work_items == ()
