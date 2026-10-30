import io
import json
import logging
import threading

import pandas as pd
from rich.console import Console

from featurebench.harness import run_evaluation
from featurebench.harness.constants import KEY_INSTANCE_ID, LOG_REPORT
from featurebench.harness.container import EvalContainerManager


class FakeContainer:
    def __init__(self, container_id="container-1", status="running"):
        self.id = container_id
        self.short_id = container_id[:12]
        self.status = status
        self.killed = False
        self.removed = False

    def reload(self):
        return None

    def kill(self):
        self.killed = True
        self.status = "exited"

    def remove(self, force=False):
        self.removed = True


def _eval_cleanup():
    cleanup = run_evaluation.EvalContainerCleanup.__new__(run_evaluation.EvalContainerCleanup)
    cleanup.run_id = "eval-run-1"
    cleanup.console = Console(file=io.StringIO(), force_terminal=False)
    cleanup._active_containers_lock = threading.RLock()
    cleanup._active_containers = {}
    cleanup._shutdown_requested = threading.Event()
    cleanup._cleanup_lock = threading.RLock()
    cleanup._cleanup_in_progress = False
    cleanup._cleanup_interrupt_notice_printed = False
    return cleanup


def test_eval_container_manager_passes_labels_to_docker(monkeypatch):
    captured = {}

    class FakeContainers:
        def get(self, container_name):
            raise run_evaluation.docker.errors.NotFound("missing")

        def run(self, *args, **kwargs):
            captured.update(kwargs)
            return FakeContainer()

    class FakeImages:
        def get(self, image_name):
            return object()

    class FakeClient:
        containers = FakeContainers()
        images = FakeImages()

    monkeypatch.setattr("featurebench.harness.container.docker.from_env", lambda: FakeClient())

    manager = EvalContainerManager(logging.getLogger("test"))
    labels = {"featurebench.run": "eval-run-1", "featurebench.kind": "eval"}
    manager.create_container("image:latest", "task-1", labels=labels)

    assert captured["labels"] == labels


def test_eval_cleanup_labeled_containers_filters_current_run(monkeypatch):
    cleanup = _eval_cleanup()
    container = FakeContainer()
    captured = {}

    class FakeContainers:
        def list(self, **kwargs):
            captured.update(kwargs)
            return [container]

    class FakeClient:
        containers = FakeContainers()

    monkeypatch.setattr(run_evaluation.docker, "from_env", lambda: FakeClient())

    removed = cleanup._cleanup_labeled_containers()

    assert removed == 1
    assert container.killed
    assert container.removed
    assert captured["all"] is True
    assert captured["filters"] == {
        "label": [
            "featurebench.run=eval-run-1",
            "featurebench.kind=eval",
        ]
    }


def test_eval_interrupted_task_does_not_save_failure_report(monkeypatch, tmp_path):
    cleanup = _eval_cleanup()

    class FakeEvalContainerManager:
        def __init__(self, logger):
            self.logger = logger

        def pull_image_if_needed(self, image_name):
            return None

        def create_container(self, *args, **kwargs):
            return FakeContainer()

        def cleanup_container(self, container):
            return None

    def fake_run_level1(*args, **kwargs):
        cleanup.request_shutdown()
        raise RuntimeError("container removed during shutdown")

    monkeypatch.setattr(run_evaluation, "EvalContainerManager", FakeEvalContainerManager)
    monkeypatch.setattr(run_evaluation, "get_docker_image_name", lambda instance: "image:latest")
    monkeypatch.setattr(run_evaluation, "run_instance_level1", fake_run_level1)

    instance = pd.Series(
        {
            KEY_INSTANCE_ID: "repo.task.lv1",
            "level": 1,
            "repo": "repo",
        }
    )
    result = run_evaluation.run_instance(
        instance,
        {"prediction": "patch"},
        tmp_path,
        docker_runtime_config={"need_gpu": False},
        container_cleanup=cleanup,
    )

    report_path = tmp_path / "eval_outputs" / "repo.task.lv1" / "attempt-1" / LOG_REPORT
    assert result["completed"] is False
    assert "Interrupted during shutdown" in result["error"]
    assert not report_path.exists()


def test_eval_interrupted_runtime_result_does_not_save_report(monkeypatch, tmp_path):
    cleanup = _eval_cleanup()

    class FakeEvalContainerManager:
        def __init__(self, logger):
            self.logger = logger

        def pull_image_if_needed(self, image_name):
            return None

        def create_container(self, *args, **kwargs):
            return FakeContainer()

        def cleanup_container(self, container):
            return None

    def fake_run_level1(*args, **kwargs):
        cleanup.request_shutdown()
        return {
            "patch_applied": True,
            "f2p_success": False,
            "p2p_success": False,
            "error": None,
        }

    monkeypatch.setattr(run_evaluation, "EvalContainerManager", FakeEvalContainerManager)
    monkeypatch.setattr(run_evaluation, "get_docker_image_name", lambda instance: "image:latest")
    monkeypatch.setattr(run_evaluation, "run_instance_level1", fake_run_level1)

    instance = pd.Series(
        {
            KEY_INSTANCE_ID: "repo.task.lv1",
            "level": 1,
            "repo": "repo",
        }
    )
    result = run_evaluation.run_instance(
        instance,
        {"prediction": "patch"},
        tmp_path,
        docker_runtime_config={"need_gpu": False},
        container_cleanup=cleanup,
    )

    report_path = tmp_path / "eval_outputs" / "repo.task.lv1" / "attempt-1" / LOG_REPORT
    assert result["completed"] is False
    assert "Interrupted during shutdown" in result["error"]
    assert not report_path.exists()


def test_eval_report_without_completion_marker_is_rerun(monkeypatch, tmp_path):
    instance_id = "repo.task.lv1"
    log_dir = tmp_path / "eval_outputs" / instance_id / "attempt-1"
    log_dir.mkdir(parents=True)
    report_path = log_dir / LOG_REPORT
    report_path.write_text(
        json.dumps(
            {
                instance_id: {
                    "resolved": False,
                    "patch_successfully_applied": False,
                }
            }
        ),
        encoding="utf-8",
    )
    called = False

    class FakeEvalContainerManager:
        def __init__(self, logger):
            self.logger = logger

        def pull_image_if_needed(self, image_name):
            return None

        def create_container(self, *args, **kwargs):
            return FakeContainer()

        def cleanup_container(self, container):
            return None

    def fake_run_level1(*args, **kwargs):
        nonlocal called
        called = True
        return {
            "patch_applied": True,
            "f2p_success": True,
            "p2p_success": True,
            "error": None,
        }

    monkeypatch.setattr(run_evaluation, "EvalContainerManager", FakeEvalContainerManager)
    monkeypatch.setattr(run_evaluation, "get_docker_image_name", lambda instance: "image:latest")
    monkeypatch.setattr(run_evaluation, "run_instance_level1", fake_run_level1)

    instance = pd.Series(
        {
            KEY_INSTANCE_ID: instance_id,
            "level": 1,
            "repo": "repo",
        }
    )
    result = run_evaluation.run_instance(
        instance,
        {"prediction": "patch"},
        tmp_path,
        docker_runtime_config={"need_gpu": False},
    )

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert called
    assert result["completed"] is True
    assert saved_report[instance_id][run_evaluation.EVAL_REPORT_COMPLETED_KEY] is True
