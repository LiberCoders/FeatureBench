"""
Agent implementations for ACE-Bench inference.
"""

from acebench.infer.agents.base import BaseAgent
from acebench.infer.agents.codex import CodexAgent
from acebench.infer.agents.claude_code import ClaudeCodeAgent
from acebench.infer.agents.gemini_cli import GeminiCliAgent
from acebench.infer.agents.openhands import OpenHandsAgent

__all__ = [
    "BaseAgent",
    "CodexAgent",
    "ClaudeCodeAgent",
    "GeminiCliAgent",
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
        "openhands": OpenHandsAgent
    }
    
    agent_class = agents.get(agent_name.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    
    return agent_class(**kwargs)

