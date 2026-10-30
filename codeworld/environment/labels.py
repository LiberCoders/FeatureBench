"""Canonical matching for Agent-detected ecosystem labels."""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[-_./]+")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_WHITESPACE = re.compile(r"\s+")


def normalize_ecosystem_label(value: str) -> str:
    """Normalize presentation differences without maintaining alias branches in code."""
    separated = _SEPARATORS.sub(" ", value.strip().casefold())
    separated = _LETTER_DIGIT_BOUNDARY.sub(" ", separated)
    return _WHITESPACE.sub(" ", separated)
