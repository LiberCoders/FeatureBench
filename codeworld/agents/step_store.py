"""Human-readable prefix-compacting Harness trajectory persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
)

from codeworld.storage.files import atomic_write_bytes

_HASH_PREFIX = "sha256:"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_ROOT_NAMES = frozenset({"media", "objects"})
_REQUEST_ENRICHMENT_FIELDS = (
    "conversation_id",
    "instructions",
    "run_id",
    "timestamp",
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "sequence",
        "run_id",
        "step_index",
        "conversation_id",
        "parent_run_id",
        "agent_name",
        "timestamp",
        "state",
        "message_count",
        "messages_sha256",
        "messages",
    }
)
_OBJECT_DICT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_MESSAGE_OBJECTS_ADAPTER: TypeAdapter[list[dict[str, object]]] = TypeAdapter(
    list[dict[str, object]]
)


class PrefixCompactingJsonStepStore:
    """Harness ``StepStore`` with standalone, directly auditable JSON snapshots.

    Every retained snapshot embeds its complete Pydantic AI message history. After a
    new snapshot is durably written, older snapshots whose histories are exact prefixes
    are removed because the new file fully supersedes them. Non-prefix histories remain
    as audit checkpoints across context compaction or rewriting. A latest interrupted
    snapshot never removes the newest complete prefix required by Harness's safe default
    resume path.
    """

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)
        self._metadata = FileStepStore(self._root)

    async def register_run(self, record: RunRecord) -> None:
        _validate_run_id(record.run_id)
        await self._metadata.register_run(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        _validate_run_id(run_id)
        return await self._metadata.get_run(run_id=run_id)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await self._metadata.list_runs(
            parent_run_id=parent_run_id,
            conversation_id=conversation_id,
        )

    async def append_event(self, event: StepEvent) -> None:
        _validate_run_id(event.run_id)
        await self._metadata.append_event(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        _validate_run_id(run_id)
        return await self._metadata.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await asyncio.to_thread(self._save_snapshot, snapshot)

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        return await asyncio.to_thread(self._load_latest_snapshot, run_id, include_interrupted)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        _validate_run_id(record.run_id)
        await self._metadata.record_tool_effect(record)

    async def get_tool_effect(
        self,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        _validate_run_id(run_id)
        return await self._metadata.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        _validate_run_id(run_id)
        return await self._metadata.list_unresolved_tool_effects(run_id=run_id)

    def _save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        _validate_run_id(snapshot.run_id)
        snapshot_directory = self._root / snapshot.run_id / "snapshots"
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        existing = _load_snapshots(snapshot_directory)
        sequence = max(existing, default=-1) + 1
        messages = _message_objects(snapshot.messages)
        manifest = {
            "sequence": sequence,
            "run_id": snapshot.run_id,
            "step_index": snapshot.step_index,
            "conversation_id": snapshot.conversation_id,
            "parent_run_id": snapshot.parent_run_id,
            "agent_name": snapshot.agent_name,
            "timestamp": snapshot.timestamp.isoformat(),
            "state": snapshot.state,
            "message_count": len(messages),
            "messages_sha256": _HASH_PREFIX + _history_digest(messages),
            "messages": messages,
        }
        encoded = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        atomic_write_bytes(snapshot_directory / f"{sequence}.json", encoded)
        _remove_superseded_snapshots(
            snapshot_directory=snapshot_directory,
            existing=existing,
            new_messages=messages,
            new_state=snapshot.state,
        )

    def _load_latest_snapshot(
        self,
        run_id: str,
        include_interrupted: bool,
    ) -> ContinuableSnapshot | None:
        _validate_run_id(run_id)
        snapshots = _load_snapshots(self._root / run_id / "snapshots")
        for sequence in sorted(snapshots, reverse=True):
            manifest = snapshots[sequence]
            state = _snapshot_state(manifest.get("state"))
            if not include_interrupted and state != "complete":
                continue
            return _restore_snapshot(run_id, manifest, state)
        return None


def _remove_superseded_snapshots(
    *,
    snapshot_directory: Path,
    existing: dict[int, dict[str, object]],
    new_messages: list[dict[str, object]],
    new_state: SnapshotState,
) -> None:
    dominated_interrupted: list[int] = []
    dominated_complete: list[int] = []
    for sequence, manifest in existing.items():
        old_messages = _validated_messages(manifest)
        if not _is_history_prefix(old_messages, new_messages):
            continue
        if _snapshot_state(manifest.get("state")) == "complete":
            dominated_complete.append(sequence)
        else:
            dominated_interrupted.append(sequence)

    removable = dominated_interrupted
    if new_state == "complete":
        removable = [*removable, *dominated_complete]
    elif dominated_complete:
        newest_complete = max(dominated_complete)
        removable = [
            *removable,
            *(sequence for sequence in dominated_complete if sequence != newest_complete),
        ]
    for sequence in removable:
        (snapshot_directory / f"{sequence}.json").unlink(missing_ok=True)


def _restore_snapshot(
    expected_run_id: str,
    manifest: dict[str, object],
    state: SnapshotState,
) -> ContinuableSnapshot:
    if manifest.get("run_id") != expected_run_id:
        raise ValueError("trajectory snapshot run_id does not match its directory")
    step_index = manifest.get("step_index")
    timestamp = manifest.get("timestamp")
    if type(step_index) is not int or not isinstance(timestamp, str):
        raise ValueError("trajectory snapshot metadata has invalid types")
    messages = ModelMessagesTypeAdapter.validate_python(_validated_messages(manifest))
    return ContinuableSnapshot(
        run_id=expected_run_id,
        step_index=step_index,
        messages=messages,
        conversation_id=_optional_string(manifest.get("conversation_id")),
        parent_run_id=_optional_string(manifest.get("parent_run_id")),
        agent_name=_optional_string(manifest.get("agent_name")),
        timestamp=datetime.fromisoformat(timestamp),
        state=state,
    )


def _load_snapshots(snapshot_directory: Path) -> dict[int, dict[str, object]]:
    if not snapshot_directory.is_dir():
        return {}
    snapshots: dict[int, dict[str, object]] = {}
    for path in snapshot_directory.glob("*.json"):
        try:
            sequence = int(path.stem)
        except ValueError:
            continue
        manifest = _load_json_object(path)
        if manifest.keys() != _SNAPSHOT_FIELDS:
            raise ValueError(f"trajectory snapshot fields do not match the current schema: {path}")
        stored_sequence = manifest.get("sequence")
        if type(stored_sequence) is not int or stored_sequence != sequence:
            raise ValueError(f"trajectory snapshot sequence does not match filename: {path}")
        snapshots[sequence] = manifest
    return snapshots


def _validated_messages(manifest: dict[str, object]) -> list[dict[str, object]]:
    try:
        messages = _MESSAGE_OBJECTS_ADAPTER.validate_python(
            manifest.get("messages"),
            strict=True,
        )
    except ValidationError as exc:
        raise ValueError("trajectory snapshot messages must contain JSON objects") from exc
    message_count = manifest.get("message_count")
    if type(message_count) is not int or message_count < 0 or message_count != len(messages):
        raise ValueError("trajectory snapshot message_count does not match messages")
    expected_digest = _parse_history_digest(manifest.get("messages_sha256"))
    if _history_digest(messages) != expected_digest:
        raise ValueError("trajectory snapshot history hash mismatch")
    return messages


def _message_objects(messages: list[ModelMessage]) -> list[dict[str, object]]:
    return _MESSAGE_OBJECTS_ADAPTER.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def _is_history_prefix(
    possible_prefix: list[dict[str, object]],
    messages: list[dict[str, object]],
) -> bool:
    return len(possible_prefix) <= len(messages) and all(
        _same_or_enriched_request(old, new)
        for old, new in zip(possible_prefix, messages, strict=False)
    )


def _same_or_enriched_request(
    old: dict[str, object],
    new: dict[str, object],
) -> bool:
    if old == new:
        return True
    if old.get("kind") != "request" or new.get("kind") != "request":
        return False
    enriched = dict(old)
    for field in _REQUEST_ENRICHMENT_FIELDS:
        if enriched.get(field) is None and new.get(field) is not None:
            enriched[field] = new[field]
    return enriched == new


def _history_digest(messages: list[dict[str, object]]) -> str:
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        return _OBJECT_DICT_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"cannot read trajectory snapshot: {path}") from exc


def _parse_history_digest(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(_HASH_PREFIX):
        raise ValueError("trajectory snapshot messages_sha256 is invalid")
    digest = value.removeprefix(_HASH_PREFIX)
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("trajectory snapshot messages_sha256 is invalid")
    return digest


def _snapshot_state(value: object) -> SnapshotState:
    if value == "complete":
        return "complete"
    if value == "interrupted":
        return "interrupted"
    raise ValueError(f"unknown trajectory snapshot state: {value!r}")


def _optional_string(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"expected trajectory string or null, got {type(value).__name__}")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id or run_id in _RESERVED_ROOT_NAMES:
        raise ValueError(f"invalid trajectory run_id: {run_id!r}")
