"""Validation for identifiers that become filesystem and Docker resource names."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import cast

_INSTANCE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,199}\Z")


def normalize_instance_id(value: object, *, field: str = "instance_id") -> str:
    """Return one path-safe, Docker-safe instance ID."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if _INSTANCE_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{field} must contain 1-200 ASCII letters, digits, '.', '_', or '-', "
            "and must not start with '.'"
        )
    return normalized


def normalize_test_id(value: object, *, field: str = "test ID") -> str:
    """Return one repository-relative test ID without traversal or control characters."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if "\\" in normalized or any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise ValueError(f"{field} contains an invalid character")

    file_part, separator, selector = normalized.partition("::")
    if separator and not selector:
        raise ValueError(f"{field} has an empty test selector")
    if file_part.startswith("/testbed/"):
        file_part = file_part[len("/testbed/") :]
    elif file_part.startswith("/"):
        raise ValueError(f"{field} must resolve inside /testbed")
    while file_part.startswith("./"):
        file_part = file_part[2:]

    path = PurePosixPath(file_part)
    if not file_part or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must not traverse outside /testbed")
    canonical_path = path.as_posix()
    if canonical_path in {"", "."}:
        raise ValueError(f"{field} must name a test file")
    return canonical_path + (f"::{selector}" if separator else "")


def normalize_test_ids(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> list[str]:
    """Validate and normalize a dataset test-ID list."""
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise ValueError(f"{field} must be a {qualifier}list of test IDs")
    items = cast(list[object], value)
    normalized = [normalize_test_id(item, field=f"{field} item") for item in items]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


def testbed_test_id(value: object, *, field: str = "test ID") -> str:
    """Return an absolute /testbed test ID suitable for a quoted CLI argument."""
    return f"/testbed/{normalize_test_id(value, field=field)}"


def testbed_test_file(value: object, *, field: str = "test ID") -> str:
    """Return the absolute test file portion of a test ID."""
    return testbed_test_id(value, field=field).split("::", 1)[0]
