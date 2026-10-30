from __future__ import annotations

import logging
import shlex

import pandas as pd
import pytest

from featurebench.harness import runtime


@pytest.mark.parametrize(
    "runner",
    [runtime.run_instance_level1, runtime.run_instance_level2],
)
def test_runtime_runs_every_f2p_test_with_quoted_arguments(monkeypatch, tmp_path, runner) -> None:
    commands: list[str] = []

    def fake_exec(container, command, timeout=None):
        commands.append(command)
        return 0, b""

    monkeypatch.setattr(runtime, "exec_run_with_timeout", fake_exec)
    monkeypatch.setattr(runtime, "copy_to_container", lambda *args: None)
    monkeypatch.setattr(runtime, "apply_environment_fixes", lambda *args: True)

    f2p_tests = [
        "tests/test one.py::test_feature",
        "tests/test_two.py::test_feature[param's]",
    ]
    instance = pd.Series(
        {
            "instance_id": "owner__repo.task.lv1",
            "repo": "owner/repo",
            "patch": "",
            "test_patch": "",
            "FAIL_TO_PASS": f2p_tests,
            "PASS_TO_PASS": [],
            "repo_settings": {"test_cmd": "pytest -rA", "timeout_one": 0},
        }
    )

    result = runner(
        instance,
        {"model_patch": "diff --git a/x b/x\n"},
        object(),
        logging.getLogger("test-harness-runtime"),
        tmp_path,
    )

    assert result["f2p_success"] is True
    test_command = next(
        command for command in commands if "pytest -rA" in command and "/testbed/tests/" in command
    )
    arguments = shlex.split(test_command)
    assert arguments[-2:] == [f"/testbed/{test_id}" for test_id in f2p_tests]

    if runner is runtime.run_instance_level1:
        delete_commands = [command for command in commands if command.startswith("rm -f --")]
        assert delete_commands
        assert all("::" not in command for command in delete_commands)
