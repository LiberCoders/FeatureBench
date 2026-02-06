"""
ACE-Bench Inference Module

This module provides inference functionality for running various agents
on ACE-Bench instances and generating patches.

Main components:
- run_infer: Main entry point for running inference
- agents: Agent implementations (claude_code, gemini_cli, mini_swe_agent, openhands, codex)
- container: Docker container management
- runtime: Runtime initialization and completion
- output: Thread-safe output management

Usage:
    python -m acebench.infer.run_infer --agent claude_code --model claude-sonnet-4-20250514
    python -m acebench.infer.run_infer --agent openhands --model gpt-4o --n-concurrent 4
"""

# Lazy imports to avoid circular import issues when running as module
# Import run_infer only when explicitly accessed to prevent RuntimeWarning
# when using: python -m acebench.infer.run_infer

from acebench.infer.models import (
    AgentName,
    TaskInstance,
    InferConfig,
    InferResult,
    RunMetadata,
    TaskPaths
)
from acebench.infer.config import InferConfigLoader, DatasetLoader
from acebench.infer.container import ContainerManager
from acebench.infer.runtime import RuntimeHandler
from acebench.infer.output import OutputManager
from acebench.infer.agents import (
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
        from acebench.infer import run_infer
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
