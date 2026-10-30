"""Concurrent-safe append-only JSONL storage for environment production."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from codeworld.environment.models import EnvironmentBuildFailure, EnvironmentRecord
from codeworld.repositories.identity import repository_aliases
from codeworld.selection import OutcomeIndex, WorkState
from codeworld.storage.files import atomic_binary_writer
from codeworld.storage.jsonl import DurableJsonl, iter_jsonl


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Complete Environment records visible at one captured source byte boundary."""

    source: Path
    source_size_bytes: int
    records: tuple[EnvironmentRecord, ...]
    ignored_tail_bytes: int = 0


def capture_environment_snapshot(path: str | Path) -> EnvironmentSnapshot:
    """Capture a non-blocking snapshot while an upstream Build may still append.

    The file size is fixed before reading. A final unterminated fragment is included
    only when it already validates as a complete Environment record; otherwise it is
    excluded from this invocation without waiting for the writer.
    """

    source = Path(path)
    with source.open("rb") as handle:
        boundary = os.fstat(handle.fileno()).st_size
        payload = handle.read(boundary)

    records: list[EnvironmentRecord] = []
    seen: dict[str, EnvironmentRecord] = {}
    ignored_tail_bytes = 0
    physical_lines = payload.splitlines(keepends=True)
    for line_number, raw_line in enumerate(physical_lines, start=1):
        terminated = raw_line.endswith((b"\n", b"\r"))
        encoded = raw_line.strip()
        if not encoded:
            continue
        try:
            line = encoded.decode("utf-8")
            record = EnvironmentRecord.model_validate_json(line)
        except Exception as exc:
            if line_number == len(physical_lines) and not terminated:
                ignored_tail_bytes = len(raw_line)
                break
            raise ValueError(f"invalid environment at {source}:{line_number}: {exc}") from exc
        previous = seen.get(record.environment_id)
        if previous is not None:
            detail = "different content" if previous != record else "identical content"
            raise ValueError(
                f"duplicate environment_id with {detail} at {source}:{line_number}: "
                f"{record.environment_id}"
            )
        seen[record.environment_id] = record
        records.append(record)
    return EnvironmentSnapshot(
        source=source,
        source_size_bytes=len(payload),
        records=tuple(records),
        ignored_tail_bytes=ignored_tail_bytes,
    )


def write_environment_snapshot(
    path: str | Path,
    records: Iterable[EnvironmentRecord],
) -> int:
    """Atomically persist the canonical records selected from one input snapshot."""

    count = 0
    with atomic_binary_writer(path) as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True).encode("utf-8") + b"\n")
            count += 1
    return count


def iter_environments(path: str | Path) -> Iterator[EnvironmentRecord]:
    yield from iter_jsonl(
        path,
        parse=EnvironmentRecord.model_validate_json,
        record_name="environment",
    )


def index_environment_build_outcomes(
    environments: Iterable[EnvironmentRecord],
    failures: Iterable[EnvironmentBuildFailure],
) -> OutcomeIndex:
    """Project already-loaded Build outputs without reopening their locked JSONL files."""

    index = OutcomeIndex()
    for record in environments:
        index.record(repository_aliases(record.repository), WorkState.SUCCEEDED)
    for failure in failures:
        aliases = repository_aliases(
            failure.repository,
            github_id=failure.github_id,
            full_name=failure.task_id if failure.github_id is not None else None,
        )
        state = {
            "failed": WorkState.FAILED,
            "rejected": WorkState.REJECTED,
            "interrupted": WorkState.INTERRUPTED,
        }[failure.disposition]
        index.record(aliases, state)
    return index


class EnvironmentJsonlAppender:
    """Long-lived idempotent writer that rejects identity/content conflicts."""

    def __init__(self, path: str | Path, *, fsync_interval: int = 1) -> None:
        self.path = Path(path)
        self._records: dict[str, EnvironmentRecord] = {}
        self._store = DurableJsonl(
            self.path,
            parse=EnvironmentRecord.model_validate_json,
            serialize=lambda record: record.model_dump_json(exclude_none=True),
            record_name="environment",
            fsync_interval=max(1, fsync_interval),
        )

    def __enter__(self) -> EnvironmentJsonlAppender:
        self._store.__enter__()
        try:
            for record in self._store.records:
                previous = self._records.get(record.environment_id)
                if previous is not None and previous != record:
                    raise ValueError(
                        f"conflicting duplicate environment_id in {self.path}: "
                        f"{record.environment_id}"
                    )
                self._records[record.environment_id] = record
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
    def records(self) -> tuple[EnvironmentRecord, ...]:
        return tuple(self._records.values())

    def append(self, record: EnvironmentRecord) -> bool:
        if not self._store.is_open:
            raise RuntimeError("EnvironmentJsonlAppender must be used as a context manager")
        previous = self._records.get(record.environment_id)
        if previous is not None:
            if previous == record:
                return False
            raise ValueError(
                f"environment_id already exists with different content: {record.environment_id}"
            )
        self._store.append(record)
        self._records[record.environment_id] = record
        return True

    def flush(self) -> None:
        self._store.flush()


class EnvironmentFailureJsonlAppender:
    """Append-only failure ledger; repeated attempts intentionally remain observable."""

    def __init__(self, path: str | Path) -> None:
        self._store = DurableJsonl(
            path,
            parse=EnvironmentBuildFailure.model_validate_json,
            serialize=lambda record: record.model_dump_json(exclude_none=True),
            record_name="environment failure",
            fsync_interval=1,
        )

    def __enter__(self) -> EnvironmentFailureJsonlAppender:
        self._store.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self._store.close()

    def append(self, failure: EnvironmentBuildFailure) -> None:
        if not self._store.is_open:
            raise RuntimeError("EnvironmentFailureJsonlAppender must be used as a context manager")
        self._store.append(failure)

    @property
    def records(self) -> tuple[EnvironmentBuildFailure, ...]:
        return self._store.records
