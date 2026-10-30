"""Append-only JSONL interface for repository records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

from codeworld.repositories.identity import repository_aliases
from codeworld.repositories.models import RepositoryCandidate
from codeworld.selection import OutcomeIndex, WorkState
from codeworld.storage.jsonl import DurableJsonl, iter_jsonl


def iter_repositories(path: str | Path) -> Iterator[RepositoryCandidate]:
    yield from iter_jsonl(
        path,
        parse=RepositoryCandidate.model_validate_json,
        record_name="repository",
    )


def index_repository_outcomes(repositories: Iterable[RepositoryCandidate]) -> OutcomeIndex:
    """Project already-loaded Discover outputs into the shared outcome vocabulary."""

    index = OutcomeIndex()
    for repository in repositories:
        index.record(
            repository_aliases(
                repository.clone_url,
                github_id=repository.github_id,
                full_name=repository.full_name,
            ),
            WorkState.SUCCEEDED,
        )
    return index


class RepositoryJsonlAppender:
    """Long-lived deduplicating writer that checkpoints discovery while it runs."""

    def __init__(self, path: str | Path, *, fsync_interval: int = 50) -> None:
        self.path = Path(path)
        self._ids: set[int] = set()
        self._names: set[str] = set()
        self._id_names: dict[int, str] = {}
        self._name_ids: dict[str, int] = {}
        self._run_counts: dict[str, int] = {}
        self._store = DurableJsonl(
            self.path,
            parse=RepositoryCandidate.model_validate_json,
            serialize=lambda record: record.model_dump_json(exclude_none=True),
            record_name="repository",
            fsync_interval=max(1, fsync_interval),
        )
        self.appended = 0

    def __enter__(self) -> RepositoryJsonlAppender:
        self._store.__enter__()
        try:
            for line_number, repository in enumerate(self._store.records, start=1):
                name = repository.full_name.casefold()
                previous_name = self._id_names.get(repository.github_id)
                previous_id = self._name_ids.get(name)
                if previous_name is not None or previous_id is not None:
                    raise ValueError(
                        f"duplicate repository identity at {self.path}:{line_number}: "
                        f"github_id={repository.github_id}, full_name={repository.full_name}"
                    )
                self._ids.add(repository.github_id)
                self._names.add(name)
                self._id_names[repository.github_id] = name
                self._name_ids[name] = repository.github_id
                self._run_counts[repository.discovery_run_id] = (
                    self._run_counts.get(repository.discovery_run_id, 0) + 1
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

    def append(self, repository: RepositoryCandidate) -> bool:
        if not self._store.is_open:
            raise RuntimeError("RepositoryJsonlAppender must be used as a context manager")
        name = repository.full_name.casefold()
        if repository.github_id in self._ids or name in self._names:
            return False
        self._store.append(repository)
        self._ids.add(repository.github_id)
        self._names.add(name)
        self._id_names[repository.github_id] = name
        self._name_ids[name] = repository.github_id
        self._run_counts[repository.discovery_run_id] = (
            self._run_counts.get(repository.discovery_run_id, 0) + 1
        )
        self.appended += 1
        return True

    @property
    def records(self) -> tuple[RepositoryCandidate, ...]:
        return self._store.records

    def count_for_run(self, run_id: str) -> int:
        return self._run_counts.get(run_id, 0)

    def flush(self) -> None:
        self._store.flush()
