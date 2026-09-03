import logging

from featurebench.environment_fixes import apply_environment_fixes


class RecordingRunner:
    def __init__(self, exit_code=0, output="ok"):
        self.exit_code = exit_code
        self.output = output
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        return self.exit_code, self.output


def test_mlflow_fix_removes_and_verifies_both_parallel_source_trees():
    runner = RecordingRunner()

    assert apply_environment_fixes(
        "MLflow/MLflow",
        runner,
        logging.getLogger("test"),
    )

    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert "/testbed/libs/skinny/mlflow" in command
    assert "/testbed/libs/tracing/mlflow" in command
    assert 'rm -rf -- "$path"' in command
    assert '[ -e "$path" ] || [ -L "$path" ]' in command


def test_mlflow_fix_failure_is_fatal():
    runner = RecordingRunner(exit_code=1, output=b"permission denied")

    assert not apply_environment_fixes(
        "mlflow/mlflow",
        runner,
        logging.getLogger("test"),
    )


def test_unrelated_repository_does_not_run_mlflow_fix():
    runner = RecordingRunner()

    assert apply_environment_fixes(
        "pandas-dev/pandas",
        runner,
        logging.getLogger("test"),
    )
    assert runner.commands == []
