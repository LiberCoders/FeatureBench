from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


CommandType = Union[str, List[str]]


def _should_use_uv(specs: Optional[Dict[str, Any]]) -> bool:
    return bool(specs and specs.get("use_uv", False))


def apply_uv_run_prefix(command: CommandType, specs: Optional[Dict[str, Any]] = None) -> CommandType:
    """Optionally prefix a command with "uv run" when use_uv is enabled.

    Args:
        command: Command string or list of arguments.
        specs: Repo specs dict; reads `use_uv` boolean.

    Returns:
        The command with "uv run" prefixed if enabled and not already present.
    """
    if not command or not _should_use_uv(specs):
        return command

    if isinstance(command, list):
        if len(command) >= 2 and command[0] == "uv" and command[1] == "run":
            return command
        return ["uv", "run", *command]

    if isinstance(command, str):
        stripped = command.lstrip()
        if stripped == "uv run" or stripped.startswith("uv run "):
            return command
        return f"uv run {command}"

    return command
