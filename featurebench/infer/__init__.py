"""
FeatureBench Inference Module

This module provides inference functionality for running various agents
on FeatureBench instances and generating patches.

Main components:
- run_infer: Main entry point for running inference
- agents: Agent implementations (claude_code, gemini_cli, mini_swe_agent, openhands, codex)
- container: Docker container management
- runtime: Runtime initialization and completion
- output: Thread-safe output management

Usage:
    python -m featurebench.infer.run_infer --agent claude_code --model claude-sonnet-4-20250514
    python -m featurebench.infer.run_infer --agent openhands --model gpt-4o --n-concurrent 4
"""

# Lazy imports to avoid circular import issues when running as module
# Import run_infer only when explicitly accessed to prevent RuntimeWarning
# when using: python -m featurebench.infer.run_infer

from featurebench.infer.models import (
    AgentName,
    TaskInstance,
    InferConfig,
    InferResult,
    RunMetadata,
    TaskPaths
)
from featurebench.infer.config import InferConfigLoader, DatasetLoader
from featurebench.infer.container import ContainerManager
from featurebench.infer.runtime import RuntimeHandler
from featurebench.infer.output import OutputManager
from featurebench.infer.agents import (
    get_agent,
    BaseAgent,
    ClaudeCodeAgent,
    GeminiCliAgent,
    MiniSweAgent,
    OpenHandsAgent,
    CodexAgent,
)


def __getattr__(name):
    """Lazy import for run_infer to avoid RuntimeWarning when running as module."""
    if name in ("main", "InferenceRunner"):
        from featurebench.infer import run_infer
        return getattr(run_infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main entry points (lazy loaded)
    "main",
    "InferenceRunner",
    # Models
    "AgentName",
    "TaskInstance",
    "InferConfig",
    "InferResult",
    "RunMetadata",
    "TaskPaths",
    # Config
    "InferConfigLoader",
    "DatasetLoader",
    # Container
    "ContainerManager",
    # Runtime
    "RuntimeHandler",
    # Output
    "OutputManager",
    # Agents
    "get_agent",
    "BaseAgent",
    "ClaudeCodeAgent",
    "GeminiCliAgent",
    "MiniSweAgent",
    "OpenHandsAgent",
    "CodexAgent",
]
