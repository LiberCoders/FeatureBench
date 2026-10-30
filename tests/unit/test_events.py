from __future__ import annotations

import asyncio

import pytest

from codeworld.events import BestEffortEventSink, CompositeEventSink, Event


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class FailingSink:
    async def emit(self, event: Event) -> None:
        del event
        raise RuntimeError("display failed")


class GatedSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event: Event) -> None:
        self.started.set()
        await self.release.wait()
        await super().emit(event)


def event(index: int) -> Event:
    return Event(run_id="run", kind="progress", stage="test", message=str(index))


@pytest.mark.asyncio
async def test_best_effort_sink_isolates_projection_failures() -> None:
    durable = RecordingSink()
    projection = BestEffortEventSink(FailingSink())
    async with projection:
        events = CompositeEventSink((durable, projection))
        await events.emit(event(1))
        await asyncio.sleep(0)
        await events.emit(event(2))

    assert [item.message for item in durable.events] == ["1", "2"]
    assert isinstance(projection.error, RuntimeError)


@pytest.mark.asyncio
async def test_best_effort_sink_drops_old_progress_when_bounded() -> None:
    target = GatedSink()
    projection = BestEffortEventSink(target, capacity=2)
    async with projection:
        await projection.emit(event(0))
        await target.started.wait()
        for index in range(1, 6):
            await projection.emit(event(index))
        assert projection.dropped == 3
        target.release.set()

    assert [item.message for item in target.events] == ["0", "4", "5"]
