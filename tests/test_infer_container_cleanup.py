import io
import logging
import threading

from rich.console import Console

from featurebench import cli
from featurebench.infer import run_infer
from featurebench.infer.container import ContainerManager


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


def _runner_for_cleanup():
    runner = run_infer.InferenceRunner.__new__(run_infer.InferenceRunner)
    runner.run_timestamp = "2026-06-06__21-38-09"
    runner.console = Console(file=io.StringIO(), force_terminal=False)
    runner._active_containers_lock = threading.RLock()
    runner._active_containers = {}
    runner._shutdown_requested = threading.Event()
    runner._cleanup_lock = threading.RLock()
    runner._cleanup_in_progress = False
    runner._cleanup_interrupt_notice_printed = False
    return runner


def test_container_manager_passes_labels_to_docker(monkeypatch):
    captured = {}

    class FakeImages:
        def get(self, image_name):
            return object()

    class FakeContainers:
        def run(self, **kwargs):
            captured.update(kwargs)
            return FakeContainer()

    class FakeClient:
        images = FakeImages()
        containers = FakeContainers()

    monkeypatch.setattr("featurebench.infer.container.docker.from_env", lambda: FakeClient())

    manager = ContainerManager(logger=logging.getLogger("test"), env_vars={})
    labels = {"featurebench.run": "run-1", "featurebench.kind": "infer"}
    manager.create_container("image:latest", labels=labels)

    assert captured["labels"] == labels


def test_cleanup_active_containers_kills_and_removes_registered_container(monkeypatch):
    runner = _runner_for_cleanup()
    container = FakeContainer()
    runner._register_container(container)
    monkeypatch.setattr(runner, "_cleanup_labeled_containers", lambda: 0)

    removed = runner._cleanup_active_containers("test")

    assert removed == 1
    assert container.killed
    assert container.removed
    assert runner._active_containers == {}


def test_cleanup_labeled_containers_filters_current_run(monkeypatch):
    runner = _runner_for_cleanup()
    container = FakeContainer()
    captured = {}

    class FakeContainers:
        def list(self, **kwargs):
            captured.update(kwargs)
            return [container]

    class FakeClient:
        containers = FakeContainers()

    monkeypatch.setattr(run_infer.docker, "from_env", lambda: FakeClient())

    removed = runner._cleanup_labeled_containers()

    assert removed == 1
    assert container.killed
    assert container.removed
    assert captured["all"] is True
    assert captured["filters"] == {
        "label": [
            "featurebench.run=2026-06-06__21-38-09",
            "featurebench.kind=infer",
        ]
    }


def test_cli_converts_keyboard_interrupt_to_130():
    def interrupted():
        raise KeyboardInterrupt

    assert cli._run_with_patched_argv("fb infer", [], interrupted) == 130
