"""Strict streaming JSONL I/O for TaskRecord."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

from codeworld.models import TaskRecord
from codeworld.storage.files import atomic_binary_writer
from codeworld.storage.jsonl import DurableJsonl, iter_jsonl


def iter_tasks(path: str | Path) -> Iterator[TaskRecord]:
    yield from iter_jsonl(
        path,
        parse=TaskRecord.model_validate_json,
        record_name="task",
    )


def read_tasks(path: str | Path) -> list[TaskRecord]:
    return list(iter_tasks(path))


class TaskJsonlAppender:
    """Durable idempotent task writer used by long-running Adapter production."""

    def __init__(self, path: str | Path, *, fsync_interval: int = 1) -> None:
        self.path = Path(path)
        self._records: dict[str, TaskRecord] = {}
        self._task_ids_by_environment: dict[str, set[str]] = {}
        self._store = DurableJsonl(
            self.path,
            parse=TaskRecord.model_validate_json,
            serialize=lambda record: record.model_dump_json(exclude_none=True),
            record_name="task",
            fsync_interval=max(1, fsync_interval),
        )

    def __enter__(self) -> TaskJsonlAppender:
        self._store.__enter__()
        try:
            for record in self._store.records:
                previous = self._records.get(record.task_id)
                if previous is not None and previous != record:
                    raise ValueError(
                        f"conflicting duplicate task_id in {self.path}: {record.task_id}"
                    )
                self._records[record.task_id] = record
                self._task_ids_by_environment.setdefault(record.environment_id, set()).add(
                    record.task_id
                )
        except BaseException:
            self._store.close()
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self._store.close()

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        return tuple(self._records.values())

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def task_ids_for_environment(self, environment_id: str) -> frozenset[str]:
        """Return durable task IDs explicitly owned by one Environment."""

        return frozenset(self._task_ids_by_environment.get(environment_id, ()))

    def append(self, record: TaskRecord) -> bool:
        if not self._store.is_open:
            raise RuntimeError("TaskJsonlAppender must be used as a context manager")
        previous = self._records.get(record.task_id)
        if previous is not None:
            if previous == record:
                return False
            raise ValueError(f"task_id already exists with different content: {record.task_id}")
        self._store.append(record)
        self._records[record.task_id] = record
        self._task_ids_by_environment.setdefault(record.environment_id, set()).add(record.task_id)
        return True

    def flush(self) -> None:
        self._store.flush()


def write_tasks(path: str | Path, tasks: Iterable[TaskRecord]) -> int:
    """Atomically replace ``path`` with compact UTF-8 JSONL."""

    seen: set[str] = set()
    count = 0
    with atomic_binary_writer(path) as handle:
        for task in tasks:
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id: {task.task_id}")
            seen.add(task.task_id)
            line = task.model_dump_json(exclude_none=True).encode("utf-8")
            handle.write(line + b"\n")
            count += 1
    return count
