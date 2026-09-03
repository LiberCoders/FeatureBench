"""Interfaces for repository-specific task-environment fixes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
import logging


CommandRunner = Callable[[str], tuple[int, str | bytes]]


class EnvironmentFix(ABC):
    """A targeted cleanup applied before the task repository becomes the Git baseline."""

    name: str
    repositories: frozenset[str]

    def applies_to(self, repository: str) -> bool:
        """Return whether this fix applies to a canonical ``owner/repo`` name."""
        normalized = repository.strip().casefold()
        return normalized in {repo.casefold() for repo in self.repositories}

    @abstractmethod
    def apply(self, run_command: CommandRunner, logger: logging.Logger) -> bool:
        """Apply and verify the fix, returning ``False`` on failure."""
