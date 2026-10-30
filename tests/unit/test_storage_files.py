from __future__ import annotations

import stat
from pathlib import Path

import pytest

from codeworld.storage.files import atomic_binary_writer, atomic_write_bytes


def test_atomic_write_bytes_creates_and_replaces_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "artifact.txt"

    atomic_write_bytes(destination, b"first")
    atomic_write_bytes(destination, b"second")

    assert destination.read_bytes() == b"second"
    assert not tuple(destination.parent.glob(f".{destination.name}.*"))


def test_atomic_write_bytes_can_make_script_executable(tmp_path: Path) -> None:
    destination = tmp_path / "activate"

    atomic_write_bytes(destination, b"#!/bin/sh\n", executable=True)

    assert destination.stat().st_mode & stat.S_IXUSR


def test_atomic_binary_writer_preserves_target_when_streaming_fails(tmp_path: Path) -> None:
    destination = tmp_path / "tasks.jsonl"
    destination.write_bytes(b"existing\n")

    with pytest.raises(RuntimeError, match="stop"), atomic_binary_writer(destination) as handle:
        handle.write(b"replacement\n")
        raise RuntimeError("stop")

    assert destination.read_bytes() == b"existing\n"
    assert not tuple(destination.parent.glob(f".{destination.name}.*"))
