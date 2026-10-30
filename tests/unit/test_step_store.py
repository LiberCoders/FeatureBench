from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    SnapshotState,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
)

from codeworld.agents.step_store import PrefixCompactingJsonStepStore


def _request(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content)])


def _response(content: str, *, interrupted: bool = False) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content)],
        state="interrupted" if interrupted else "complete",
    )


def _snapshot(
    run_id: str,
    step_index: int,
    messages: list[ModelRequest | ModelResponse],
    *,
    state: SnapshotState = "complete",
) -> ContinuableSnapshot:
    return ContinuableSnapshot(
        run_id=run_id,
        conversation_id="conversation-1",
        parent_run_id="parent-1",
        agent_name="artifact",
        step_index=step_index,
        timestamp=datetime(2026, 7, 29, 8, step_index, tzinfo=UTC),
        messages=messages,
        state=state,
    )


def _snapshot_files(root: Path, run_id: str = "artifact-1") -> list[Path]:
    return sorted((root / run_id / "snapshots").glob("*.json"))


@pytest.mark.asyncio
async def test_new_complete_history_replaces_every_older_prefix_with_readable_json(
    tmp_path: Path,
) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    assert isinstance(store, StepStore)
    await store.register_run(RunRecord(run_id="artifact-1", agent_name="artifact"))
    messages: list[ModelRequest | ModelResponse] = [_request("build the environment")]
    for step in range(1, 6):
        await store.save_snapshot(_snapshot("artifact-1", step, list(messages)))
        messages.append(_response(f"model response {step}"))

    files = _snapshot_files(tmp_path)
    assert len(files) == 1
    manifest_text = files[0].read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "format" not in manifest
    assert manifest["message_count"] == 5
    assert len(manifest["messages"]) == 5
    assert "build the environment" in manifest_text
    assert "model response 4" in manifest_text
    assert "message_refs" not in manifest

    restored = await store.latest_snapshot(run_id="artifact-1")
    assert restored is not None
    assert ModelMessagesTypeAdapter.dump_json(restored.messages) == (
        ModelMessagesTypeAdapter.dump_json(messages[:5])
    )


@pytest.mark.asyncio
async def test_interrupted_tail_preserves_latest_complete_snapshot_for_safe_resume(
    tmp_path: Path,
) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(
        RunRecord(
            run_id="artifact-1",
            conversation_id="conversation-1",
            agent_name="artifact",
        )
    )
    complete = _snapshot("artifact-1", 1, [_request("build")])
    interrupted = _snapshot(
        "artifact-1",
        2,
        [_request("build"), _response("partial", interrupted=True)],
        state="interrupted",
    )
    await store.save_snapshot(complete)
    await store.save_snapshot(interrupted)

    assert len(_snapshot_files(tmp_path)) == 2
    latest_complete = await store.latest_snapshot(run_id="artifact-1")
    latest_any = await store.latest_snapshot(run_id="artifact-1", include_interrupted=True)
    assert latest_complete is not None
    assert latest_any is not None
    assert latest_complete.step_index == 1
    assert latest_complete.state == "complete"
    assert latest_any.step_index == 2
    assert latest_any.state == "interrupted"
    assert latest_any.conversation_id == "conversation-1"
    assert latest_any.parent_run_id == "parent-1"
    assert latest_any.agent_name == "artifact"


@pytest.mark.asyncio
async def test_non_prefix_history_remains_as_human_audit_checkpoint(tmp_path: Path) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(RunRecord(run_id="artifact-1"))
    before_compaction = _snapshot(
        "artifact-1",
        1,
        [_request("original task"), _response("long investigation")],
    )
    after_compaction = _snapshot(
        "artifact-1",
        2,
        [_request("summary of previous conversation"), _response("recent work")],
    )
    await store.save_snapshot(before_compaction)
    await store.save_snapshot(after_compaction)

    files = _snapshot_files(tmp_path)
    assert len(files) == 2
    contents = [path.read_text(encoding="utf-8") for path in files]
    assert any("long investigation" in content for content in contents)
    assert any("summary of previous conversation" in content for content in contents)


@pytest.mark.asyncio
async def test_harness_pending_request_metadata_enrichment_still_replaces_old_prefix(
    tmp_path: Path,
) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(RunRecord(run_id="artifact-1"))
    parts = [UserPromptPart("tool result")]
    pending = ModelRequest(parts=parts)
    committed = ModelRequest(
        parts=parts,
        timestamp=datetime(2026, 7, 29, 9, tzinfo=UTC),
        instructions="artifact instructions",
        run_id="run-1",
        conversation_id="conversation-1",
    )
    await store.save_snapshot(_snapshot("artifact-1", 1, [pending]))
    await store.save_snapshot(_snapshot("artifact-1", 2, [committed, _response("done")]))

    files = _snapshot_files(tmp_path)
    assert len(files) == 1
    assert "artifact instructions" in files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_harness_step_persistence_uses_prefix_compacting_store_transparently(
    tmp_path: Path,
) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    agent = Agent(
        TestModel(),
        capabilities=[
            StepPersistence(
                store=store,
                run_id="artifact-1",
                agent_name="artifact",
            )
        ],
    )
    await agent.run("build the environment")

    restored = await store.latest_snapshot(run_id="artifact-1")
    assert restored is not None
    assert len(restored.messages) == 2
    files = _snapshot_files(tmp_path)
    assert len(files) == 1
    manifest = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(manifest["messages"]) == 2


@pytest.mark.asyncio
async def test_logical_agent_run_keeps_transport_attempts_in_distinct_subdirectories(
    tmp_path: Path,
) -> None:
    logical_run = tmp_path / "mask-repair-logical-run"
    store = PrefixCompactingJsonStepStore(logical_run)
    agent = Agent(TestModel())

    for attempt in range(2):
        await agent.run(
            "repair the mask",
            conversation_id="mask-repair-logical-run",
            capabilities=(
                StepPersistence(
                    store=store,
                    run_id=f"attempt-{attempt:04d}",
                    agent_name="featurebench_mask_repair",
                    metadata={"logical_run_id": "mask-repair-logical-run"},
                ),
            ),
        )

    for attempt in range(2):
        attempt_root = logical_run / f"attempt-{attempt:04d}"
        run = json.loads((attempt_root / "run.json").read_text(encoding="utf-8"))
        assert run["conversation_id"] == "mask-repair-logical-run"
        assert run["metadata"]["logical_run_id"] == "mask-repair-logical-run"
        assert (attempt_root / "events.jsonl").is_file()
        assert list((attempt_root / "snapshots").glob("*.json"))


@pytest.mark.asyncio
async def test_prefix_compacting_store_retains_events_and_tool_effect_ledger(
    tmp_path: Path,
) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(RunRecord(run_id="artifact-1", agent_name="artifact"))
    event = StepEvent(run_id="artifact-1", kind="run_started", step_index=0)
    started = ToolEffectRecord(
        run_id="artifact-1",
        tool_call_id="tool-1",
        tool_name="run_command",
        status="started",
    )
    await store.append_event(event)
    await store.record_tool_effect(started)

    assert await store.list_events(run_id="artifact-1") == [event]
    assert await store.get_tool_effect(run_id="artifact-1", tool_call_id="tool-1") == started
    assert await store.list_unresolved_tool_effects(run_id="artifact-1") == [started]

    completed = ToolEffectRecord(
        run_id="artifact-1",
        tool_call_id="tool-1",
        tool_name="run_command",
        status="completed",
        started_at=started.started_at,
        ended_at=datetime.now(UTC),
    )
    await store.record_tool_effect(completed)
    assert await store.list_unresolved_tool_effects(run_id="artifact-1") == []


@pytest.mark.asyncio
async def test_modified_snapshot_is_rejected_by_full_history_hash(tmp_path: Path) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(RunRecord(run_id="artifact-1"))
    await store.save_snapshot(_snapshot("artifact-1", 1, [_request("build")]))
    snapshot_file = _snapshot_files(tmp_path)[0]
    manifest = json.loads(snapshot_file.read_text(encoding="utf-8"))
    manifest["messages"][0]["parts"][0]["content"] = "tampered"
    snapshot_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="history hash mismatch"):
        await store.latest_snapshot(run_id="artifact-1")


@pytest.mark.asyncio
async def test_snapshot_rejects_fields_outside_the_current_schema(tmp_path: Path) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    await store.register_run(RunRecord(run_id="artifact-1"))
    await store.save_snapshot(_snapshot("artifact-1", 1, [_request("build")]))
    snapshot_file = _snapshot_files(tmp_path)[0]
    manifest = json.loads(snapshot_file.read_text(encoding="utf-8"))
    manifest["format"] = "codeworld.prefix-compacted-json-snapshot.v1"
    snapshot_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="current schema"):
        await store.latest_snapshot(run_id="artifact-1")


@pytest.mark.asyncio
async def test_reserved_internal_directory_cannot_be_used_as_a_run_id(tmp_path: Path) -> None:
    store = PrefixCompactingJsonStepStore(tmp_path)
    with pytest.raises(ValueError, match="invalid trajectory run_id"):
        await store.register_run(RunRecord(run_id="objects"))
