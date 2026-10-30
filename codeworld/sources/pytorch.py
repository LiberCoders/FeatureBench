"""CPU PyTorch wheel source semantics owned by one ecosystem adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import SourceAdapter, SourceBuildContext

_OFFICIAL_CPU_OPTION = re.compile(
    r"(?<!\S)(?:--index-url|--extra-index-url|--index|-i|--find-links|-f)(?:\s+|=)"
    r"(?P<quote>['\"]?)https://download\.pytorch\.org/whl/"
    r"(?:cpu(?:/torch_stable\.html)?/?|torch_stable\.html)(?P=quote)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PytorchCpuSourceAdapter(SourceAdapter):
    """Keep the normal Python index while supplying CPU wheels as find-links."""

    name: str = "pytorch_cpu"
    priority: int = 150
    services: frozenset[str] = frozenset({"pytorch_cpu"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        if service != "pytorch_cpu":
            return {}
        return {"TORCH_CPU_FIND_LINKS": url}

    def transform(self, source: str, context: SourceBuildContext) -> str:
        if "pytorch_cpu" not in context.services:
            return source
        # Replacing only the official CPU option leaves PIP_INDEX_URL authoritative
        # for ordinary dependencies and deliberately does not redirect CUDA/ROCm
        # backends. A service-specific argument also avoids changing unrelated pip
        # commands through a process-wide PIP_FIND_LINKS value.
        return _OFFICIAL_CPU_OPTION.sub("--find-links=${TORCH_CPU_FIND_LINKS}", source)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "pytorch_cpu":
            return ()
        return (
            "download.pytorch.org/whl/cpu",
            "pytorch-wheels/cpu",
            "TORCH_CPU_FIND_LINKS",
        )
