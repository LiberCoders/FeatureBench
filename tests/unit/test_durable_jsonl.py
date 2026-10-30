from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from codeworld.models import StrictModel
from codeworld.storage.jsonl import DurableJsonl, append_jsonl_record, iter_jsonl


class Record(StrictModel):
    key: str


def store(path: Path, *, fsync_interval: int = 1) -> DurableJsonl[Record]:
    return DurableJsonl(
        path,
        parse=Record.model_validate_json,
        serialize=lambda record: record.model_dump_json(),
        record_name="test record",
        fsync_interval=fsync_interval,
    )


def append(path: Path, record: Record) -> None:
    append_jsonl_record(
        path,
        record,
        parse=Record.model_validate_json,
        serialize=lambda value: value.model_dump_json(),
    )


def test_valid_unterminated_record_is_preserved_and_terminated(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(Record(key="你好").model_dump_json().encode("utf-8"))

    with store(path) as writer:
        assert writer.records == (Record(key="你好"),)

    assert path.read_bytes().endswith(b"\n")


def test_invalid_unterminated_tail_is_repaired_at_exact_byte_offset(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    first = Record(key="非 ASCII 前缀").model_dump_json().encode("utf-8")
    path.write_bytes(first + b'\n{"key":"torn"')

    with store(path) as writer:
        assert writer.records == (Record(key="非 ASCII 前缀"),)
        writer.append(Record(key="second"))

    assert [
        record.key
        for record in iter_jsonl(
            path,
            parse=Record.model_validate_json,
            record_name="test record",
        )
    ] == ["非 ASCII 前缀", "second"]


def test_invalid_complete_line_is_rejected_without_rewriting_history(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    content = b'{"key":"first"}\nnot-json\n'
    path.write_bytes(content)

    with pytest.raises(ValueError, match=r"records\.jsonl:2"), store(path):
        pass

    assert path.read_bytes() == content


def test_append_flushes_on_configured_interval_and_context_exit(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    with store(path, fsync_interval=2) as writer:
        writer.append(Record(key="one"))
        writer.append(Record(key="two"))
        writer.append(Record(key="three"))

    assert [record.key for record in writer.records] == ["one", "two", "three"]
    assert path.read_bytes().count(b"\n") == 3


def test_one_shot_append_repairs_a_torn_tail_without_loading_history(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    first = Record(key="中文历史").model_dump_json().encode("utf-8")
    path.write_bytes(first + b'\n{"key":"torn"')

    append(path, Record(key="second"))

    assert [
        record.key
        for record in iter_jsonl(
            path,
            parse=Record.model_validate_json,
            record_name="test record",
        )
    ] == ["中文历史", "second"]


def test_one_shot_append_is_process_lock_safe_for_parallel_writers(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [Record(key=str(index)) for index in range(32)]

    def append_record(record: Record) -> None:
        append(path, record)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_record, records))

    loaded = list(
        iter_jsonl(
            path,
            parse=Record.model_validate_json,
            record_name="test record",
        )
    )
    assert {record.key for record in loaded} == {record.key for record in records}
    assert len(loaded) == len(records)
