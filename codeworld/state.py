"""SQLite run ledger with an intentionally non-migrating current schema."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TextIO, cast

from pydantic import Field

from codeworld.errors import ContractError
from codeworld.events import Event
from codeworld.models import StrictModel

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, job_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    stage TEXT NOT NULL,
    task_id TEXT,
    message TEXT NOT NULL,
    data_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS events_run_sequence ON events(run_id, sequence);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


JobStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "rejected",
    "interrupted",
    "skipped",
]


class JobRecord(StrictModel):
    """One validated durable orchestration checkpoint."""

    run_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: JobStatus
    attempt: int = Field(ge=0)
    payload: dict[str, object]
    result: dict[str, object] | None = None
    error: str | None = None
    updated_at: datetime


class RunLock:
    """Kernel-backed single-writer lease for one orchestration run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError(f"run lock is already acquired: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ContractError(f"run is already active in another process: {self.path}") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        self._handle = handle

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class StateStore:
    """Persistent source of truth for resumable orchestration and progress views."""

    def __init__(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(destination, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if current not in (0, _SCHEMA_VERSION):
                raise ContractError(
                    f"state schema is {current}, FeatureBench requires {_SCHEMA_VERSION}; "
                    "use a fresh workspace"
                )
            self._connection.executescript(_SCHEMA)
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_run(self, run_id: str, command: str, config: dict[str, Any]) -> None:
        now = _now()
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    "INSERT INTO runs VALUES (?, ?, 'running', ?, ?, ?, NULL)",
                    (run_id, command, now, now, json.dumps(config, sort_keys=True)),
                )
        except sqlite3.IntegrityError as exc:
            raise ContractError(
                f"run_id already exists: {run_id}; use --resume with its run directory"
            ) from exc

    def resume_run(self, run_id: str, command: str, config: dict[str, Any]) -> None:
        """Run the same output directory again under its currently resolved YAML config."""
        resolved_config = json.dumps(config, sort_keys=True)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT command FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ContractError(f"cannot resume unknown run_id: {run_id}")
            if row["command"] != command:
                raise ContractError(f"run {run_id} belongs to {row['command']!r}, not {command!r}")
            self._connection.execute(
                "UPDATE runs SET status = 'running', updated_at = ?, config_json = ?, "
                "error = NULL WHERE run_id = ?",
                (_now(), resolved_config, run_id),
            )

    def load_run_config(self, run_id: str, command: str) -> tuple[str, dict[str, Any]]:
        """Return the last resolved config cached for an inactive editable run."""
        with self._lock:
            row = self._connection.execute(
                "SELECT command, status, config_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise ContractError(f"cannot configure unknown run_id: {run_id}")
        if row["command"] != command:
            raise ContractError(f"run {run_id} belongs to {row['command']!r}, not {command!r}")
        status = str(row["status"])
        value: Any = json.loads(str(row["config_json"]))
        if not isinstance(value, dict):
            raise ContractError(f"run {run_id} has an invalid stored configuration")
        return status, cast(dict[str, Any], value)

    def replace_run_config(
        self,
        run_id: str,
        command: str,
        *,
        expected: dict[str, Any],
        replacement: dict[str, Any],
    ) -> None:
        """Atomically replace the cached config after validating edited run inputs."""
        expected_json = json.dumps(expected, sort_keys=True)
        replacement_json = json.dumps(replacement, sort_keys=True)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT command, status, config_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ContractError(f"cannot configure unknown run_id: {run_id}")
            if row["command"] != command:
                raise ContractError(f"run {run_id} belongs to {row['command']!r}, not {command!r}")
            if row["config_json"] != expected_json:
                raise ContractError(
                    "run configuration changed after the editor opened; reopen the editor"
                )
            self._connection.execute(
                "UPDATE runs SET config_json = ?, updated_at = ? WHERE run_id = ?",
                (replacement_json, _now(), run_id),
            )

    def finish_run(self, run_id: str, *, error: str | None = None) -> None:
        status = "failed" if error else "completed"
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, error = ? WHERE run_id = ?",
                (status, _now(), error, run_id),
            )

    def interrupt_run(self, run_id: str, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE runs SET status = 'interrupted', updated_at = ?, error = ? WHERE run_id = ?",
                (_now(), error, run_id),
            )

    def upsert_job(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        status: JobStatus,
        payload: dict[str, object],
        *,
        attempt: int = 0,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        try:
            record = JobRecord(
                run_id=run_id,
                job_id=job_id,
                stage=stage,
                status=status,
                attempt=attempt,
                payload=payload,
                result=result,
                error=error,
                updated_at=datetime.now(UTC),
            )
        except ValueError as exc:
            raise ContractError(f"invalid job checkpoint {run_id}/{job_id}: {exc}") from exc
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO jobs
                   (run_id, job_id, stage, status, attempt, payload_json, result_json, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, job_id) DO UPDATE SET
                     stage=excluded.stage, status=excluded.status, attempt=excluded.attempt,
                     payload_json=excluded.payload_json, result_json=excluded.result_json,
                     error=excluded.error, updated_at=excluded.updated_at""",
                (
                    record.run_id,
                    record.job_id,
                    record.stage,
                    record.status,
                    record.attempt,
                    json.dumps(record.payload, sort_keys=True),
                    json.dumps(record.result, sort_keys=True)
                    if record.result is not None
                    else None,
                    record.error,
                    record.updated_at.isoformat(),
                ),
            )

    def get_job(self, run_id: str, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND job_id = ?", (run_id, job_id)
            ).fetchone()
        return self._decode_job(row) if row is not None else None

    def jobs(self, run_id: str, stage: str | None = None) -> list[JobRecord]:
        query = "SELECT * FROM jobs WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if stage is not None:
            query += " AND stage = ?"
            parameters.append(stage)
        query += " ORDER BY job_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._decode_job(row) for row in rows]

    def delete_jobs(self, run_id: str, stages: tuple[str, ...]) -> None:
        """Discard derived checkpoints while preserving durable stage outputs."""

        if not stages:
            return
        placeholders = ",".join("?" for _ in stages)
        with self._lock, self._connection:
            self._connection.execute(
                f"DELETE FROM jobs WHERE run_id = ? AND stage IN ({placeholders})",
                (run_id, *stages),
            )

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> JobRecord:
        values = dict(row)
        run_id = str(values.get("run_id", "<unknown>"))
        job_id = str(values.get("job_id", "<unknown>"))
        try:
            values["payload"] = json.loads(values.pop("payload_json"))
            raw_result = values.pop("result_json")
            values["result"] = json.loads(raw_result) if raw_result is not None else None
            return JobRecord.model_validate(values)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid job checkpoint {run_id}/{job_id}: {exc}") from exc

    async def emit(self, event: Event) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO events
                   (event_id, run_id, timestamp, kind, stage, task_id, message, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.run_id,
                    event.timestamp.isoformat(),
                    event.kind,
                    event.stage,
                    event.task_id,
                    event.message,
                    json.dumps(event.data, sort_keys=True),
                ),
            )
