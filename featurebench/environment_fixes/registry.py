"""Registry and execution helpers for task-environment fixes."""

import logging
from typing import TypeVar

from featurebench.environment_fixes.base import CommandRunner, EnvironmentFix


_FIXES: list[EnvironmentFix] = []
_EnvironmentFixType = TypeVar("_EnvironmentFixType", bound=type[EnvironmentFix])


def register_environment_fix(fix_type: _EnvironmentFixType) -> _EnvironmentFixType:
    """Register an environment-fix class at import time."""
    fix = fix_type()
    if any(existing.name == fix.name for existing in _FIXES):
        raise ValueError(f"Duplicate environment fix name: {fix.name}")
    _FIXES.append(fix)
    return fix_type


def apply_environment_fixes(
    repository: str,
    run_command: CommandRunner,
    logger: logging.Logger,
) -> bool:
    """Apply every registered fix matching ``repository`` in registration order."""
    matching_fixes = [fix for fix in _FIXES if fix.applies_to(repository)]
    if not matching_fixes:
        return True

    for fix in matching_fixes:
        logger.info("Applying environment fix %s for %s", fix.name, repository)
        try:
            if not fix.apply(run_command, logger):
                logger.error("Environment fix %s failed for %s", fix.name, repository)
                return False
        except Exception:
            logger.exception("Environment fix %s raised for %s", fix.name, repository)
            return False
        logger.info("Environment fix %s completed for %s", fix.name, repository)

    return True
