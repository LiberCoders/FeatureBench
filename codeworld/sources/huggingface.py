"""Hugging Face Hub source semantics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import SourceAdapter, SourceBuildContext

_PUBLIC_HUB_URL = re.compile(
    r"(?<!\$\{HF_ENDPOINT\})(?<!/)https://huggingface\.co(?:/[^\s'\"\\]*)?",
    re.IGNORECASE,
)
_CANONICAL_HUB = "https://huggingface.co"
_ENDPOINT_REFERENCE = "${HF_ENDPOINT}"


def _redirect_public_url(match: re.Match[str]) -> str:
    """Redirect public literal Hub URLs while leaving token-like URLs canonical."""

    url = match.group(0)
    if "?" in url or "#" in url:
        return url
    return f"{_ENDPOINT_REFERENCE}{url[len(_CANONICAL_HUB) :]}"


@dataclass(frozen=True, slots=True)
class HuggingFaceSourceAdapter(SourceAdapter):
    """Apply a Hub endpoint to clients and public literal artifact downloads."""

    name: str = "huggingface"
    priority: int = 150
    services: frozenset[str] = frozenset({"huggingface"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        return {"HF_ENDPOINT": url} if service == "huggingface" else {}

    def transform(self, source: str, context: SourceBuildContext) -> str:
        if "huggingface" not in context.services:
            return source
        return _PUBLIC_HUB_URL.sub(_redirect_public_url, source)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "huggingface":
            return ()
        return (
            "huggingface.co",
            "hf-mirror.com",
            "HF_ENDPOINT",
            "huggingface-cli",
            "hf download",
        )
