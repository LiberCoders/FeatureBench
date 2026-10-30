"""Durable Environment-level outcomes for batched task generation."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import Field

from codeworld.models import StrictModel
from codeworld.storage.jsonl import DurableJsonl, iter_jsonl


class EnvironmentGenerationFailure(StrictModel):
    """One append-only terminal non-success for an Adapter Environment attempt."""

    run_id: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    image: str = Field(min_length=1)
    disposition: Literal["failed", "rejected", "interrupted"]
    task_ids: tuple[str, ...] = ()
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def iter_environment_generation_failures(
    path: str | Path,
) -> Iterator[EnvironmentGenerationFailure]:
    yield from iter_jsonl(
        path,
        parse=EnvironmentGenerationFailure.model_validate_json,
        record_name="environment generation failure",
    )


class EnvironmentGenerationFailureAppender:
    """Long-lived append-only failure writer shared by Generate workers."""

    def __init__(self, path: str | Path) -> None:
        self._store = DurableJsonl(
            path,
            parse=EnvironmentGenerationFailure.model_validate_json,
            serialize=lambda record: record.model_dump_json(exclude_none=True),
            record_name="environment generation failure",
            fsync_interval=1,
        )

    def __enter__(self) -> EnvironmentGenerationFailureAppender:
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

    def append(self, failure: EnvironmentGenerationFailure) -> None:
        if not self._store.is_open:
            raise RuntimeError(
                "EnvironmentGenerationFailureAppender must be used as a context manager"
            )
        self._store.append(failure)
