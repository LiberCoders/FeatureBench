"""Durable atomic replacement for small host-side artifact files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def atomic_binary_writer(
    path: str | Path,
    *,
    executable: bool = False,
) -> Generator[BinaryIO]:
    """Yield a temporary binary file and durably replace the destination on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary.chmod(0o755)
        temporary.replace(destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    executable: bool = False,
) -> None:
    """Fsync a same-directory temporary file, replace the target, then fsync its directory."""
    with atomic_binary_writer(path, executable=executable) as handle:
        handle.write(content)
