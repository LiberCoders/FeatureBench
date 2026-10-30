"""Locked, append-only JSONL durability without record-specific identity policy."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, cast

_TAIL_SCAN_BYTES = 64 * 1024


def _serialized_line[RecordT](record: RecordT, serialize: Callable[[RecordT], str]) -> bytes:
    payload = serialize(record)
    if "\n" in payload or "\r" in payload:
        raise ValueError("serialized JSONL record must occupy exactly one physical line")
    return payload.encode("utf-8")


def _unterminated_tail_start(handle: BinaryIO, end: int) -> int:
    position = end
    while position > 0:
        start = max(0, position - _TAIL_SCAN_BYTES)
        handle.seek(start)
        block = handle.read(position - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0


def _repair_unterminated_tail[RecordT](
    handle: BinaryIO,
    parse: Callable[[str], RecordT],
) -> bool:
    """Repair only an incomplete final physical line while the caller holds the lock."""
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end == 0:
        return False
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        handle.seek(0, os.SEEK_END)
        return False

    tail_start = _unterminated_tail_start(handle, end)
    handle.seek(tail_start)
    encoded = handle.read(end - tail_start).strip()
    try:
        line = encoded.decode("utf-8")
        if not line:
            raise ValueError("empty JSONL tail")
        parse(line)
    except Exception:
        handle.seek(tail_start)
        handle.truncate()
    else:
        handle.seek(0, os.SEEK_END)
        handle.write(b"\n")
    return True


def append_jsonl_record[RecordT](
    path: str | Path,
    record: RecordT,
    *,
    parse: Callable[[str], RecordT],
    serialize: Callable[[RecordT], str],
) -> None:
    """Lock, repair, append, and fsync one typed JSONL record without scanning history."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized_line(record, serialize)
    with destination.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            _repair_unterminated_tail(handle, parse)
            handle.seek(0, os.SEEK_END)
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_json_object(line: str) -> dict[str, object]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("JSONL diagnostic record must be an object")
    return cast(dict[str, object], value)


def append_jsonl_object(path: str | Path, value: Mapping[str, object]) -> None:
    """Durably append one untyped operational JSON object."""
    append_jsonl_record(
        path,
        dict(value),
        parse=_parse_json_object,
        serialize=lambda record: json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def iter_jsonl[RecordT](
    path: str | Path,
    *,
    parse: Callable[[str], RecordT],
    record_name: str,
) -> Iterator[RecordT]:
    """Read complete JSONL records without mutating or repairing the source."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield parse(line)
            except Exception as exc:
                raise ValueError(f"invalid {record_name} at {source}:{line_number}: {exc}") from exc


class DurableJsonl[RecordT]:
    """Own file locking, torn-tail recovery, typed loading, append, and fsync.

    Record identity and duplicate semantics deliberately remain in the stage-owned
    wrapper. This primitive only guarantees that every exposed existing record was
    parsed from a complete durable line and every appended record becomes one line.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        parse: Callable[[str], RecordT],
        serialize: Callable[[RecordT], str],
        record_name: str,
        fsync_interval: int = 1,
    ) -> None:
        self.path = Path(path)
        self.parse = parse
        self.serialize = serialize
        self.record_name = record_name
        self.fsync_interval = max(1, fsync_interval)
        self._handle: BinaryIO | None = None
        self._records: list[RecordT] = []
        self._unflushed = 0

    @property
    def records(self) -> tuple[RecordT, ...]:
        return tuple(self._records)

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def __enter__(self) -> DurableJsonl[RecordT]:
        if self._handle is not None:
            raise RuntimeError("DurableJsonl is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        try:
            if _repair_unterminated_tail(self._handle, self.parse):
                self._sync()
            self._load_existing()
        except BaseException:
            self.close(flush=False)
            raise
        return self

    def _load_existing(self) -> None:
        if self._handle is None:
            raise RuntimeError("DurableJsonl is not open")
        self._records.clear()
        self._handle.seek(0)
        line_number = 0
        while True:
            raw_line = self._handle.readline()
            if raw_line == b"":
                break
            line_number += 1
            encoded = raw_line.strip()
            try:
                line = encoded.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"invalid UTF-8 in {self.record_name} at {self.path}:{line_number}: {exc}"
                ) from exc
            if not line:
                continue
            try:
                record = self.parse(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid {self.record_name} at {self.path}:{line_number}: {exc}"
                ) from exc
            self._records.append(record)
        self._handle.seek(0, os.SEEK_END)

    def _sync(self) -> None:
        if self._handle is None:
            raise RuntimeError("DurableJsonl is not open")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def append(self, record: RecordT) -> None:
        if self._handle is None:
            raise RuntimeError("DurableJsonl must be used as a context manager")
        payload = _serialized_line(record, self.serialize)
        self._handle.write(payload + b"\n")
        self._records.append(record)
        self._unflushed += 1
        if self._unflushed >= self.fsync_interval:
            self.flush()

    def flush(self) -> None:
        if self._handle is None or self._unflushed == 0:
            return
        self._sync()
        self._unflushed = 0

    def close(self, *, flush: bool = True) -> None:
        if self._handle is None:
            return
        try:
            if flush:
                self.flush()
        finally:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()
