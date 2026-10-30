"""Unified source management and independently registered ecosystem adapters."""

from codeworld.sources.apt import AptSourceAdapter
from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceBuildFragments,
    SourceFailureKind,
    SourceImageRequest,
    SourceMatchConfidence,
    harness_run,
)
from codeworld.sources.conda import CondaSourceAdapter
from codeworld.sources.defaults import default_source_adapters
from codeworld.sources.docker import DockerSourceAdapter
from codeworld.sources.git import GitSourceAdapter
from codeworld.sources.gradle import GradleSourceAdapter
from codeworld.sources.huggingface import HuggingFaceSourceAdapter
from codeworld.sources.manager import (
    SourceAttemptError,
    SourceAttemptMetric,
    SourceBuildPlan,
    SourceChoice,
    SourceManager,
)
from codeworld.sources.maven import MavenSourceAdapter
from codeworld.sources.playwright import PlaywrightSourceAdapter
from codeworld.sources.pypi import PypiSourceAdapter
from codeworld.sources.pytorch import PytorchCpuSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry
from codeworld.sources.rustup import RustupSourceAdapter

__all__ = [
    "AptSourceAdapter",
    "ArgumentSourceAdapter",
    "CondaSourceAdapter",
    "DockerSourceAdapter",
    "GitSourceAdapter",
    "GradleSourceAdapter",
    "HuggingFaceSourceAdapter",
    "MavenSourceAdapter",
    "PlaywrightSourceAdapter",
    "PypiSourceAdapter",
    "PytorchCpuSourceAdapter",
    "RustupSourceAdapter",
    "SourceAdapter",
    "SourceAdapterRegistry",
    "SourceAttemptError",
    "SourceAttemptMetric",
    "SourceBuildContext",
    "SourceBuildFragments",
    "SourceBuildPlan",
    "SourceChoice",
    "SourceFailureKind",
    "SourceImageRequest",
    "SourceManager",
    "SourceMatchConfidence",
    "default_source_adapters",
    "harness_run",
]
