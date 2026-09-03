import logging
from pathlib import Path

from featurebench.infer.models import TaskInstance
from featurebench.infer.runtime import RuntimeHandler


class FakeContainerManager:
    def __init__(self, patch_exit_code=0, bytecode_exit_code=0):
        self.patch_exit_code = patch_exit_code
        self.bytecode_exit_code = bytecode_exit_code
        self.commands = []
        self.copied_patch = None

    def copy_to_container(self, container, source, destination):
        self.copied_patch = (Path(source), destination)

    def exec_command(self, container, command, log_file=None):
        self.commands.append(command)
        if "git apply --whitespace=fix /tmp/mask.patch" in command:
            return self.patch_exit_code, "patch output"
        if "__pycache__" in command:
            return self.bytecode_exit_code, "bytecode cleanup output"
        return 0, "ok"


def _level1_instance():
    return TaskInstance(
        instance_id="owner__repo.commit.task.lv1",
        problem_statement="task",
        image_name="image:latest",
        level=1,
        patch="diff --git a/module.py b/module.py\n",
        fail_to_pass=[],
    )


def test_level1_removes_container_mask_patch_after_success(tmp_path):
    manager = FakeContainerManager()
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert handler._initialize_level1(
        object(),
        _level1_instance(),
        tmp_path / "infer.log",
    )

    source_path, destination = manager.copied_patch
    assert destination == "/tmp/mask.patch"
    assert not source_path.exists()

    apply_command = next(command for command in manager.commands if "git apply" in command)
    assert "status=$?" in apply_command
    assert "rm -f -- /tmp/mask.patch" in apply_command
    assert 'exit "$status"' in apply_command


def test_level1_applies_mlflow_fix_before_git_baseline(tmp_path):
    manager = FakeContainerManager()
    handler = RuntimeHandler(manager, logging.getLogger("test"))
    instance = _level1_instance()
    instance.metadata["repo_settings"] = {"repository": "mlflow/mlflow"}

    assert handler._initialize_level1(
        object(),
        instance,
        tmp_path / "infer.log",
    )

    cleanup_index = next(
        index
        for index, command in enumerate(manager.commands)
        if "/testbed/libs/skinny/mlflow" in command
    )
    git_init_index = next(
        index
        for index, command in enumerate(manager.commands)
        if command == "cd /testbed && git init"
    )
    assert cleanup_index < git_init_index


def test_level1_removes_python_bytecode_before_git_baseline(tmp_path):
    manager = FakeContainerManager()
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert handler._initialize_level1(
        object(),
        _level1_instance(),
        tmp_path / "infer.log",
    )

    bytecode_index = next(
        index
        for index, command in enumerate(manager.commands)
        if "__pycache__" in command
    )
    bytecode_command = manager.commands[bytecode_index]
    git_init_index = next(
        index
        for index, command in enumerate(manager.commands)
        if command == "cd /testbed && git init"
    )

    assert "-name '*.pyc'" in bytecode_command
    assert "-name '*.pyo'" in bytecode_command
    assert bytecode_index < git_init_index


def test_level1_bytecode_cleanup_failure_is_fatal(tmp_path):
    manager = FakeContainerManager(bytecode_exit_code=1)
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert not handler._initialize_level1(
        object(),
        _level1_instance(),
        tmp_path / "infer.log",
    )

    assert not any(command == "cd /testbed && git init" for command in manager.commands)


def test_level1_mask_failure_is_fatal_and_still_removes_patch(tmp_path):
    manager = FakeContainerManager(patch_exit_code=1)
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert not handler._initialize_level1(
        object(),
        _level1_instance(),
        tmp_path / "infer.log",
    )

    apply_command = next(command for command in manager.commands if "git apply" in command)
    assert "rm -f -- /tmp/mask.patch" in apply_command
    assert not any("git init" in command for command in manager.commands)


def test_package_cache_cleanup_clears_internal_caches(tmp_path):
    manager = FakeContainerManager()
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert handler.clear_package_caches(object(), tmp_path / "infer.log")

    command = manager.commands[-1]
    assert "/download" in command
    assert "/opt/miniconda3/pkgs" in command
    assert "/opt/conda/pkgs" in command
    assert "/root/.cache/pip" in command
    assert "/root/.cache/uv" in command
    assert "/proc/self/mountinfo" in command
    assert "Skipping mounted package cache" in command


def test_package_cache_cleanup_failure_is_fatal(tmp_path):
    class FailingContainerManager(FakeContainerManager):
        def exec_command(self, container, command, log_file=None):
            self.commands.append(command)
            return 1, "mounted cache"

    manager = FailingContainerManager()
    handler = RuntimeHandler(manager, logging.getLogger("test"))

    assert not handler.clear_package_caches(object(), tmp_path / "infer.log")
