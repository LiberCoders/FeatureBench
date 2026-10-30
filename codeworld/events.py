"""Structured progress events; Rich is a view over these events, never the state store."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field

from codeworld.models import StrictModel
from codeworld.storage.jsonl import append_jsonl_record


class Event(StrictModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    kind: str
    stage: str
    task_id: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class EventSink(Protocol):
    async def emit(self, event: Event) -> None: ...


class NullEventSink:
    async def emit(self, event: Event) -> None:
        del event


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> None:
        async with self._lock:
            await asyncio.to_thread(
                append_jsonl_record,
                self.path,
                event,
                parse=Event.model_validate_json,
                serialize=lambda record: record.model_dump_json(exclude_none=True),
            )


class CompositeEventSink:
    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self.sinks = tuple(sinks)

    async def emit(self, event: Event) -> None:
        for sink in self.sinks:
            await sink.emit(event)


class BestEffortEventSink:
    """Project events through a bounded queue without delaying or failing producers."""

    def __init__(
        self,
        sink: EventSink,
        *,
        capacity: int = 64,
        close_timeout_seconds: float = 1.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("best-effort event queue capacity must be positive")
        if close_timeout_seconds <= 0:
            raise ValueError("best-effort event close timeout must be positive")
        self.sink = sink
        self.capacity = capacity
        self.close_timeout_seconds = close_timeout_seconds
        self.dropped = 0
        self.error: Exception | None = None
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=capacity)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def __aenter__(self) -> BestEffortEventSink:
        if self._worker is not None:
            raise RuntimeError("best-effort event sink is already running")
        self._worker = asyncio.create_task(self._consume())
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        await self.close()

    async def emit(self, event: Event) -> None:
        if self._worker is None or self._closed:
            return
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.dropped += 1
        self._queue.put_nowait(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=self.close_timeout_seconds,
            )
        except TimeoutError:
            worker.cancel()
        else:
            self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker, timeout=self.close_timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    try:
                        await self.sink.emit(item)
                    except Exception as exc:
                        self.error = exc
            finally:
                self._queue.task_done()
