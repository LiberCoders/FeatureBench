"""MLflow-specific task-environment cleanup."""

import logging

from featurebench.environment_fixes.base import CommandRunner, EnvironmentFix
from featurebench.environment_fixes.registry import register_environment_fix


@register_environment_fix
class MlflowParallelSourcesFix(EnvironmentFix):
    """Remove unmasked MLflow package copies shipped in the monorepo's ``libs`` tree."""

    name = "mlflow_parallel_sources"
    repositories = frozenset({"mlflow/mlflow"})
    paths = (
        "/testbed/libs/skinny/mlflow",
        "/testbed/libs/tracing/mlflow",
    )

    def apply(self, run_command: CommandRunner, logger: logging.Logger) -> bool:
        paths = " ".join(self.paths)
        command = f"""
        paths=({paths})
        for path in "${{paths[@]}}"; do
            if [ -e "$path" ] || [ -L "$path" ]; then
                rm -rf -- "$path" || exit 1
                echo "Removed MLflow parallel source: $path"
            fi
            if [ -e "$path" ] || [ -L "$path" ]; then
                echo "MLflow parallel source still exists: $path" >&2
                exit 1
            fi
        done
        """
        exit_code, output = run_command(command)
        if exit_code != 0:
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            logger.error("Failed to remove MLflow parallel sources: %s", output)
            return False
        return True
