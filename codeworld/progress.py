"""Adapter-neutral progress reporting for task generation."""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field, model_validator

from codeworld.events import Event, EventSink
from codeworld.models import StrictModel

AdapterProgressStatus = Literal[
    "preparing",
    "working",
    "waiting",
    "retrying",
    "completed",
    "failed",
    "stopped",
]


class AdapterPhase(StrictModel):
    """One user-facing phase declared by an Adapter."""

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    unit: str = Field(default="items", min_length=1)


class AdapterWorkItemProgress(StrictModel):
    """One independently progressing item inside the active Adapter phase."""

    item_id: str = Field(min_length=1)
    activity: str = Field(min_length=1)
    current: str | None = None


class AdapterProgressSnapshot(StrictModel):
    """Complete display state emitted after every meaningful progress change."""

    adapter: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    environment_id: str | None = None
    status: AdapterProgressStatus = "preparing"
    phase_id: str | None = None
    phase_label: str = "Preparing"
    completed: int = Field(default=0, ge=0)
    total: int | None = Field(default=None, ge=0)
    unit: str = Field(default="items", min_length=1)
    current: str | None = None
    work_items: tuple[AdapterWorkItemProgress, ...] = ()
    tasks_created: int = Field(default=0, ge=0)
    candidates_rejected: int = Field(default=0, ge=0)
    candidates_failed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_progress(self) -> AdapterProgressSnapshot:
        if self.total is not None and self.completed > self.total:
            raise ValueError("completed progress must not exceed total")
        return self


class AdapterProgressReporter:
    """Concurrency-safe Adapter API that emits normalized progress snapshots."""

    def __init__(
        self,
        *,
        events: EventSink,
        run_id: str,
        adapter: str,
        adapter_name: str,
        phases: tuple[AdapterPhase, ...] = (),
        environment_id: str | None = None,
    ) -> None:
        phase_ids = [phase.id for phase in phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("adapter progress phase IDs must be unique")
        self.events = events
        self.run_id = run_id
        self._phase_by_id = {phase.id: phase for phase in phases}
        self._lock = asyncio.Lock()
        self._work_items: dict[str, AdapterWorkItemProgress] = {}
        self._snapshot = AdapterProgressSnapshot(
            adapter=adapter,
            adapter_name=adapter_name,
            environment_id=environment_id,
        )

    @property
    def candidate_failures(self) -> int:
        """Return the current invocation's Seed/candidate failure count."""

        return self._snapshot.candidates_failed

    async def start(self) -> None:
        async with self._lock:
            await self._emit_locked()

    async def start_phase(
        self,
        phase_id: str,
        *,
        total: int | None = None,
        unit: str | None = None,
        current: str | None = None,
    ) -> None:
        try:
            phase = self._phase_by_id[phase_id]
        except KeyError as exc:
            raise ValueError(f"adapter progress phase is not declared: {phase_id}") from exc
        async with self._lock:
            self._work_items.clear()
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "working",
                    "phase_id": phase.id,
                    "phase_label": phase.label,
                    "completed": 0,
                    "total": total,
                    "unit": unit or phase.unit,
                    "current": current,
                    "work_items": (),
                }
            )
            await self._emit_locked()

    async def set_work_item_activity(
        self,
        item_id: str,
        activity: str,
        *,
        current: str | None = None,
    ) -> None:
        """Create or update one concurrent work item in the active phase."""

        if not item_id.strip():
            raise ValueError("adapter progress work item ID must not be empty")
        if not activity.strip():
            raise ValueError("adapter progress work item activity must not be empty")
        async with self._lock:
            self._work_items[item_id] = AdapterWorkItemProgress(
                item_id=item_id,
                activity=activity,
                current=current,
            )
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "working",
                    "work_items": self._ordered_work_items(),
                }
            )
            await self._emit_locked()

    async def finish_work_item(self, item_id: str) -> None:
        """Remove a completed concurrent work item, if it is still active."""

        async with self._lock:
            if self._work_items.pop(item_id, None) is None:
                return
            self._snapshot = self._snapshot.model_copy(
                update={"work_items": self._ordered_work_items()}
            )
            await self._emit_locked()

    async def advance(
        self,
        *,
        increment: int = 1,
        completed: int | None = None,
        current: str | None = None,
    ) -> None:
        if increment < 0 or (completed is not None and completed < 0):
            raise ValueError("adapter progress cannot move backwards")
        async with self._lock:
            next_completed = (
                completed if completed is not None else self._snapshot.completed + increment
            )
            if self._snapshot.total is not None:
                next_completed = min(next_completed, self._snapshot.total)
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "working",
                    "completed": next_completed,
                    "current": current if current is not None else self._snapshot.current,
                }
            )
            await self._emit_locked()

    async def finish_phase(self) -> None:
        async with self._lock:
            self._work_items.clear()
            completed = (
                self._snapshot.total
                if self._snapshot.total is not None
                else self._snapshot.completed
            )
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "working",
                    "completed": completed,
                    "current": None,
                    "work_items": (),
                }
            )
            await self._emit_locked()

    async def task_created(self) -> None:
        async with self._lock:
            self._snapshot = self._snapshot.model_copy(
                update={"tasks_created": self._snapshot.tasks_created + 1}
            )
            await self._emit_locked()

    async def candidate_rejected(self, *, current: str | None = None) -> None:
        async with self._lock:
            self._snapshot = self._snapshot.model_copy(
                update={
                    "candidates_rejected": self._snapshot.candidates_rejected + 1,
                    "current": current if current is not None else self._snapshot.current,
                }
            )
            await self._emit_locked()

    async def candidate_failed(self, *, current: str | None = None) -> None:
        async with self._lock:
            self._snapshot = self._snapshot.model_copy(
                update={
                    "candidates_failed": self._snapshot.candidates_failed + 1,
                    "current": current if current is not None else self._snapshot.current,
                }
            )
            await self._emit_locked()

    async def set_status(
        self,
        status: Literal["working", "waiting", "retrying"],
        *,
        current: str | None = None,
    ) -> None:
        async with self._lock:
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": status,
                    "current": current if current is not None else self._snapshot.current,
                }
            )
            await self._emit_locked()

    async def complete(self) -> None:
        async with self._lock:
            self._work_items.clear()
            completed = (
                self._snapshot.total
                if self._snapshot.total is not None
                else self._snapshot.completed
            )
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "completed",
                    "completed": completed,
                    "current": None,
                    "work_items": (),
                }
            )
            await self._emit_locked()

    async def fail(self, *, stopped: bool = False) -> None:
        async with self._lock:
            self._work_items.clear()
            self._snapshot = self._snapshot.model_copy(
                update={
                    "status": "stopped" if stopped else "failed",
                    "current": None,
                    "work_items": (),
                }
            )
            await self._emit_locked()

    def _ordered_work_items(self) -> tuple[AdapterWorkItemProgress, ...]:
        return tuple(self._work_items[key] for key in sorted(self._work_items))

    async def _emit_locked(self) -> None:
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind="adapter_progress",
                stage=self._snapshot.adapter,
                data=self._snapshot.model_dump(mode="json"),
            )
        )
