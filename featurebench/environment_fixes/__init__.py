"""Repository-specific fixes for leaked or duplicated task-environment artifacts.

Add a fix by implementing :class:`EnvironmentFix`, decorating it with
``register_environment_fix``, and importing its module below. Each plugin is
matched by repository and is shared by inference and Oracle evaluation.
"""

from featurebench.environment_fixes.registry import apply_environment_fixes

# Import built-in plugins so they register themselves.
from featurebench.environment_fixes import mlflow as _mlflow  # noqa: F401, E402

__all__ = ["apply_environment_fixes"]
