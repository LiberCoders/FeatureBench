"""Canonical public test IDs shared by pytest integrations."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PUBLIC_ID = re.compile(r"^(?P<native_base>.+)#(?P<occurrence>0|[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class PytestPublicId:
    """One stable pytest identity independent of parameter display values."""

    native_base: str
    occurrence: int


def pytest_native_base(node_id: str, original_name: str | None = None) -> str:
    """Return pytest's structural item identity without its parameter display."""
    if original_name:
        parent, separator, _ = node_id.rpartition("::")
        if separator:
            return f"{parent}::{original_name}"
    return node_id


def format_pytest_public_id(native_base: str, occurrence: int) -> str:
    """Format the mandatory ``<native_base>#<zero-based occurrence>`` ID."""
    if not native_base:
        raise ValueError("pytest public-ID base must not be empty")
    if occurrence < 0:
        raise ValueError("pytest public-ID occurrence must not be negative")
    return f"{native_base}#{occurrence}"


def parse_pytest_public_id(value: str) -> PytestPublicId:
    """Parse one canonical pytest public ID or reject a different grammar."""
    match = _PUBLIC_ID.fullmatch(value)
    if match is None:
        raise ValueError("pytest public IDs must use <native_base>#<zero-based occurrence>")
    return PytestPublicId(
        native_base=match.group("native_base"),
        occurrence=int(match.group("occurrence")),
    )
