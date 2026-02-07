"""
Agent implementations for FeatureBench inference.
"""

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.agents.codex import CodexAgent
from featurebench.infer.agents.claude_code import ClaudeCodeAgent
from featurebench.infer.agents.gemini_cli import GeminiCliAgent
from featurebench.infer.agents.mini_swe_agent import MiniSweAgent
from featurebench.infer.agents.openhands import OpenHandsAgent

__all__ = [
    "BaseAgent",
    "CodexAgent",
    "ClaudeCodeAgent",
    "GeminiCliAgent",
    "MiniSweAgent",
    "OpenHandsAgent"
]


def get_agent(agent_name: str, **kwargs) -> BaseAgent:
    """
    Get an agent by name.
    
    Args:
        agent_name: Name of the agent (claude_code, openhands)
        **kwargs: Additional arguments for the agent
        
    Returns:
        Agent instance
    """
    agents = {
        "codex": CodexAgent,
        "claude_code": ClaudeCodeAgent,
        "gemini_cli": GeminiCliAgent,
        "mini_swe_agent": MiniSweAgent,
        "openhands": OpenHandsAgent
    }
    
    agent_class = agents.get(agent_name.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    
    return agent_class(**kwargs)
