"""Canonical public and internal paths for one data-factory run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Keep stage outputs visible and orchestration state behind one private boundary."""

    root: Path

    @property
    def internal(self) -> Path:
        return self.root / ".featurebench"

    @property
    def manifest(self) -> Path:
        return self.root / "run.yaml"

    @property
    def repositories(self) -> Path:
        return self.root / "repositories.jsonl"

    @property
    def environments(self) -> Path:
        return self.root / "environments.jsonl"

    @property
    def failures(self) -> Path:
        return self.root / "failures.jsonl"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks.jsonl"

    @property
    def state(self) -> Path:
        return self.internal / "state.sqlite3"

    @property
    def events(self) -> Path:
        return self.internal / "events.jsonl"

    @property
    def lock(self) -> Path:
        return self.internal / "run.lock"

    @property
    def config_changes(self) -> Path:
        return self.internal / "config-changes.jsonl"

    @property
    def invocations(self) -> Path:
        return self.internal / "invocations"

    @property
    def diagnostics(self) -> Path:
        return self.internal / "diagnostics"

    @property
    def removals(self) -> Path:
        return self.internal / "removals"

    @property
    def input_snapshots(self) -> Path:
        return self.internal / "input-snapshots"

    def existing_run_entries(self) -> tuple[Path, ...]:
        """Return pre-existing content other than the lock acquired for this launch."""

        if not self.root.is_dir():
            return ()
        entries = [entry for entry in self.root.iterdir() if entry != self.internal]
        if self.internal.is_dir():
            entries.extend(entry for entry in self.internal.iterdir() if entry != self.lock)
        elif self.internal.exists():
            entries.append(self.internal)
        return tuple(entries)
