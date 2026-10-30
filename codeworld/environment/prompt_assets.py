"""Load immutable environment Agent prompt text from packaged assets."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

PromptAssetName = Literal[
    "analysis.user",
    "analysis.instructions",
    "artifact.common.user",
    "artifact.dockerfile.user",
    "artifact.activation.user",
    "artifact.runner.user",
    "artifact.initial.instructions",
    "artifact.repair.instructions",
    "artifact.runner-repair.instructions",
    "runner-review.user",
    "runner-review.instructions",
]

_DIRECTORY = Path(__file__).resolve().parent / "assets" / "prompts" / "environment"


@cache
def load_prompt_asset(name: PromptAssetName) -> str:
    """Return normalized non-empty text for one known packaged prompt asset."""
    path = _DIRECTORY / f"{name}.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read environment prompt asset {path}: {exc}") from exc
    if not content:
        raise ValueError(f"environment prompt asset is empty: {path}")
    return content
