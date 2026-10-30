"""Validated run-input editing independent of the terminal UI."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import yaml

from codeworld.adapters.catalog import AdapterCatalog, default_adapter_catalog
from codeworld.config import FactorySettings
from codeworld.console.models import (
    RunManifest,
    load_run_manifest,
    replace_run_manifest,
    validate_config_paths,
    validate_manifest_inputs,
)
from codeworld.errors import ConfigurationError
from codeworld.run_layout import RunLayout
from codeworld.state import RunLock, StateStore
from codeworld.storage.files import atomic_write_bytes
from codeworld.storage.jsonl import append_jsonl_object

type ConfigPath = tuple[str | int, ...]

_EXECUTION_KEY = "@execution"


@dataclass(frozen=True, slots=True)
class ConfigurationGroup:
    """One first-level configuration owner shown by the terminal editor."""

    id: str
    label: str


CONFIGURATION_GROUPS = (ConfigurationGroup("execution", "Execution"),)


@dataclass(frozen=True, slots=True)
class ConfigurationEntry:
    """One navigable container or editable value in the resolved config tree."""

    name: str
    path: ConfigPath
    value: object
    is_container: bool

    @property
    def rendered_value(self) -> str:
        if self.is_container:
            return f"{_child_count(self.value)} items"
        return json.dumps(self.value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class SearchResult:
    group: ConfigurationGroup
    entry: ConfigurationEntry
    score: int

    @property
    def label(self) -> str:
        return f"{self.group.label}.{format_config_path(self.entry.path)}"


@dataclass(frozen=True, slots=True)
class ConfigurationEditResult:
    path: ConfigPath
    old_value: object
    new_value: object


@dataclass(frozen=True, slots=True)
class ConfigurationSaveResult:
    """One confirmed persistence of every net change in the current draft."""

    changes: tuple[ConfigurationEditResult, ...]
    warning: str | None = None

    @property
    def count(self) -> int:
        return len(self.changes)


def _child_count(value: object) -> int:
    if isinstance(value, dict):
        return len(cast(dict[object, object], value))
    if isinstance(value, list):
        return len(cast(list[object], value))
    return 0


def _is_container(value: object) -> bool:
    if isinstance(value, dict):
        return bool(cast(dict[object, object], value))
    if isinstance(value, list):
        return any(isinstance(item, (dict, list)) for item in cast(list[object], value))
    return False


def _path_value(document: object, path: ConfigPath) -> object:
    current = document
    for component in path:
        if isinstance(component, str) and isinstance(current, dict):
            current = cast(dict[str, object], current)[component]
        elif isinstance(component, int) and isinstance(current, list):
            current = cast(list[object], current)[component]
        else:
            raise KeyError(format_config_path(path))
    return current


def _set_path_value(document: object, path: ConfigPath, value: object) -> None:
    if not path:
        raise ValueError("the configuration root cannot be replaced")
    parent = _path_value(document, path[:-1])
    final = path[-1]
    if isinstance(final, str) and isinstance(parent, dict):
        cast(dict[str, object], parent)[final] = value
        return
    if isinstance(final, int) and isinstance(parent, list):
        cast(list[object], parent)[final] = value
        return
    raise KeyError(format_config_path(path))


def _entries(
    document: dict[str, object], paths: tuple[ConfigPath, ...]
) -> tuple[ConfigurationEntry, ...]:
    entries: list[ConfigurationEntry] = []
    for path in paths:
        value = _path_value(document, path)
        component = path[-1]
        entries.append(
            ConfigurationEntry(
                name=str(component) if isinstance(component, str) else f"[{component}]",
                path=path,
                value=value,
                is_container=_is_container(value),
            )
        )
    return tuple(entries)


def format_config_path(path: ConfigPath) -> str:
    rendered = ""
    for component in path:
        if isinstance(component, int):
            rendered += f"[{component}]"
        else:
            if component == _EXECUTION_KEY:
                continue
            rendered += ("." if rendered else "") + component
    return rendered


def _execution_document(manifest: RunManifest) -> dict[str, object]:
    common: dict[str, object] = {"config_files": list(manifest.config_files)}
    if manifest.command == "discover":
        return {**common, "source": manifest.repository_source}
    if manifest.command == "build":
        return {
            **common,
            "source": manifest.repository_source,
            "ref": manifest.ref,
            "limit": manifest.limit,
        }
    return {
        **common,
        "environment": manifest.environment_source,
        "environment_id": manifest.environment_id,
        "limit": manifest.limit,
    }


def _manifest_updates(document: dict[str, object], manifest: RunManifest) -> dict[str, object]:
    raw = document.get(_EXECUTION_KEY)
    if not isinstance(raw, dict):
        raise ConfigurationError("run execution settings must be an object")
    execution = cast(dict[str, object], raw)
    common = {"config_files": execution.get("config_files")}
    if manifest.command == "discover":
        return {**common, "repository_source": execution.get("source")}
    if manifest.command == "build":
        return {
            **common,
            "repository_source": execution.get("source"),
            "ref": execution.get("ref"),
            "limit": execution.get("limit"),
        }
    return {
        **common,
        "environment_source": execution.get("environment"),
        "environment_id": execution.get("environment_id"),
        "limit": execution.get("limit"),
    }


def _children(document: dict[str, object], path: ConfigPath) -> tuple[ConfigurationEntry, ...]:
    value = _path_value(document, path)
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return _entries(document, tuple((*path, key) for key in mapping))
    if isinstance(value, list):
        items = cast(list[object], value)
        return _entries(document, tuple((*path, index) for index in range(len(items))))
    return ()


def _subsequence_score(query: str, candidate: str) -> int | None:
    query = query.casefold().strip()
    candidate = candidate.casefold()
    if not query:
        return 0
    direct = candidate.find(query)
    if direct >= 0:
        return 10_000 - direct * 10 - len(candidate)
    position = -1
    gap = 0
    for character in query:
        found = candidate.find(character, position + 1)
        if found < 0:
            return None
        gap += found - position - 1
        position = found
    return 1_000 - gap * 10 - len(candidate)


class RunConfigurationSession:
    """Single-writer editor for one run's YAML paths and execution inputs."""

    def __init__(
        self,
        project_root: str | Path,
        run_directory: str | Path,
        *,
        adapter_catalog: AdapterCatalog | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        candidate = Path(run_directory)
        self.run_directory = (
            candidate if candidate.is_absolute() else self.project_root / candidate
        ).resolve()
        self.catalog = adapter_catalog
        self.layout = RunLayout(self.run_directory)
        self.history_path = self.layout.config_changes
        self._lock = RunLock(self.layout.lock)
        self._state: StateStore | None = None
        self._manifest_path: Path | None = None
        self._manifest: RunManifest | None = None
        self._stored_config: dict[str, Any] | None = None
        self._document: dict[str, object] | None = None
        self._baseline_document: dict[str, object] | None = None
        self._pending_changes: dict[ConfigPath, ConfigurationEditResult] = {}
        self.status = ""

    def __enter__(self) -> RunConfigurationSession:
        self._lock.acquire()
        state: StateStore | None = None
        try:
            manifest_path, manifest = load_run_manifest(
                self.project_root,
                self.run_directory,
            )
            state = StateStore(self.layout.state)
            status, stored = state.load_run_config(manifest.run_id, manifest.command)
            if self.catalog is None and manifest.adapter is not None:
                self.catalog = default_adapter_catalog()
            FactorySettings.from_files(
                manifest.invocation(self.project_root, resume=True).config_files,
                adapter_catalog=self.catalog,
            )
        except BaseException:
            if state is not None:
                state.close()
            self._lock.close()
            raise
        self._manifest_path = manifest_path
        self._manifest = manifest
        self._state = state
        # Preserve the exact SQLite baseline for compare-and-swap. Adapter defaults may
        # normalize equivalent numeric representations (for example 600 -> 600.0) when
        # reconstructed, which must not look like a concurrent edit.
        self._stored_config = stored
        self._document = {_EXECUTION_KEY: _execution_document(manifest)}
        self._baseline_document = copy.deepcopy(self._document)
        self._pending_changes.clear()
        self.status = status
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._state is not None:
            self._state.close()
            self._state = None
        self._lock.close()

    @property
    def manifest(self) -> RunManifest:
        if self._manifest is None:
            raise RuntimeError("RunConfigurationSession is not open")
        return self._manifest

    @property
    def document(self) -> dict[str, object]:
        if self._document is None:
            raise RuntimeError("RunConfigurationSession is not open")
        return self._document

    @property
    def pending_changes(self) -> tuple[ConfigurationEditResult, ...]:
        """Return the draft's distinct net changes in first-edit order."""
        return tuple(self._pending_changes.values())

    @property
    def pending_change_count(self) -> int:
        return len(self._pending_changes)

    def group_entries(self, group_id: str) -> tuple[ConfigurationEntry, ...]:
        if group_id == "execution":
            return _children(self.document, (_EXECUTION_KEY,))
        raise KeyError(group_id)

    def children(self, path: ConfigPath) -> tuple[ConfigurationEntry, ...]:
        return _children(self.document, path)

    def entry(self, path: ConfigPath) -> ConfigurationEntry:
        value = _path_value(self.document, path)
        component = path[-1]
        return ConfigurationEntry(
            name=str(component) if isinstance(component, str) else f"[{component}]",
            path=path,
            value=value,
            is_container=_is_container(value),
        )

    def _leaf_entries(
        self,
        group: ConfigurationGroup,
        entries: tuple[ConfigurationEntry, ...],
    ) -> list[tuple[ConfigurationGroup, ConfigurationEntry]]:
        leaves: list[tuple[ConfigurationGroup, ConfigurationEntry]] = []
        for entry in entries:
            if entry.is_container:
                leaves.extend(self._leaf_entries(group, self.children(entry.path)))
            else:
                leaves.append((group, entry))
        return leaves

    def search(self, query: str) -> tuple[SearchResult, ...]:
        results: list[SearchResult] = []
        for group in CONFIGURATION_GROUPS:
            for owner, entry in self._leaf_entries(group, self.group_entries(group.id)):
                label = f"{owner.label} {format_config_path(entry.path)} {entry.rendered_value}"
                token_scores = [
                    _subsequence_score(token, label) for token in query.casefold().split()
                ]
                if any(score is None for score in token_scores):
                    continue
                results.append(
                    SearchResult(
                        group=owner,
                        entry=entry,
                        score=sum(cast(int, score) for score in token_scores),
                    )
                )
        return tuple(sorted(results, key=lambda item: (-item.score, item.label)))

    def _validate_document(
        self,
        candidate_document: dict[str, object],
    ) -> tuple[dict[str, object], FactorySettings, RunManifest, dict[str, object]]:
        execution_updates = _manifest_updates(candidate_document, self.manifest)
        updated_manifest_document = self.manifest.model_dump(mode="python")
        updated_manifest_document.update(execution_updates)
        validated_manifest = RunManifest.model_validate(updated_manifest_document)
        validate_config_paths(validated_manifest.config_files, self.project_root)
        validate_manifest_inputs(validated_manifest, self.project_root)
        settings = FactorySettings.from_files(
            validated_manifest.invocation(self.project_root, resume=True).config_files,
            adapter_catalog=self.catalog,
        )
        normalized_document: dict[str, object] = {
            _EXECUTION_KEY: _execution_document(validated_manifest)
        }
        return normalized_document, settings, validated_manifest, execution_updates

    def edit(self, path: ConfigPath, raw_value: str) -> ConfigurationEditResult:
        """Validate one value and apply it only to the in-memory draft."""
        if self._state is None or self._manifest_path is None:
            raise RuntimeError("RunConfigurationSession is not open")
        try:
            parsed: object = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML value: {exc}") from exc
        old_value = copy.deepcopy(_path_value(self.document, path))
        candidate_document = copy.deepcopy(self.document)
        _set_path_value(candidate_document, path, parsed)
        normalized_document, _, _, _ = self._validate_document(candidate_document)
        new_value = copy.deepcopy(_path_value(normalized_document, path))
        self._document = normalized_document

        baseline = self._baseline_document
        if baseline is None:
            raise RuntimeError("RunConfigurationSession has no draft baseline")
        baseline_value = copy.deepcopy(_path_value(baseline, path))
        if new_value == baseline_value:
            self._pending_changes.pop(path, None)
        else:
            self._pending_changes[path] = ConfigurationEditResult(
                path=path,
                old_value=baseline_value,
                new_value=new_value,
            )
        return ConfigurationEditResult(
            path=path,
            old_value=old_value,
            new_value=new_value,
        )

    def save(self) -> ConfigurationSaveResult:
        """Atomically persist the validated draft after explicit user confirmation."""
        if self._state is None or self._manifest_path is None:
            raise RuntimeError("RunConfigurationSession is not open")
        changes = self.pending_changes
        if not changes:
            return ConfigurationSaveResult(changes=())

        normalized_document, settings, _, execution_updates = self._validate_document(
            copy.deepcopy(self.document)
        )
        replacement = settings.model_dump(mode="json")
        expected = self._stored_config
        if expected is None:
            raise RuntimeError("RunConfigurationSession has no stored baseline")

        old_manifest = self._manifest_path.read_bytes()
        manifest = self.manifest
        try:
            updated_manifest = replace_run_manifest(
                self._manifest_path,
                manifest,
                updates=execution_updates,
            )
            self._state.replace_run_config(
                manifest.run_id,
                manifest.command,
                expected=expected,
                replacement=replacement,
            )
        except BaseException:
            atomic_write_bytes(self._manifest_path, old_manifest)
            raise

        self._manifest = updated_manifest
        self._stored_config = replacement
        self._document = normalized_document
        self._baseline_document = copy.deepcopy(normalized_document)
        self._pending_changes.clear()
        warning: str | None = None
        try:
            append_jsonl_object(
                self.history_path,
                {
                    "run_id": manifest.run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "changes": [
                        {
                            "path": format_config_path(change.path),
                            "old_value": change.old_value,
                            "new_value": change.new_value,
                        }
                        for change in changes
                    ],
                },
            )
        except OSError as exc:
            warning = f"run inputs were saved, but their audit event failed: {exc}"
        return ConfigurationSaveResult(
            changes=changes,
            warning=warning,
        )
