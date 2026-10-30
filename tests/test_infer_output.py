from __future__ import annotations

import json

import pytest

from featurebench.infer import run_infer
from featurebench.infer.models import InferConfig, InferResult, TaskInstance, TaskPaths
from featurebench.infer.output import OutputManager


def _result(*, patch: str = "patch", success: bool = True) -> InferResult:
    return InferResult(
        instance_id="owner__repo.task.lv1",
        model_patch=patch,
        agent="mini_swe_agent",
        model="provider/model",
        n_attempt=1,
        metadata={},
        success=success,
    )


def test_output_manager_writes_and_replaces_results_atomically(tmp_path) -> None:
    manager = OutputManager(tmp_path)

    manager.write_result(_result(patch="first"))
    manager.write_result(_result(patch="second", success=False))

    records = [
        json.loads(line)
        for line in manager.output_jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["model_patch"] == "second"
    assert records[0]["success"] is False
    assert not manager.output_jsonl_path.with_suffix(".jsonl.tmp").exists()


def test_output_manager_does_not_overwrite_corrupt_results(tmp_path) -> None:
    manager = OutputManager(tmp_path)
    valid_record = _result(patch="preserve").to_jsonl()
    original = f"{valid_record}\n{{broken\n"
    manager.output_jsonl_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        manager.write_result(_result())

    assert manager.output_jsonl_path.read_text(encoding="utf-8") == original


def test_output_manager_propagates_summary_write_failures(monkeypatch, tmp_path) -> None:
    manager = OutputManager(tmp_path)

    def fail_summary() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_save_run_summary", fail_summary)

    with pytest.raises(OSError, match="disk full"):
        manager.write_result(_result())


def test_resume_rejects_corrupt_output(tmp_path) -> None:
    manager = OutputManager(tmp_path)
    manager.output_jsonl_path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        manager.load_completed_tasks()


def test_resume_rejects_unsafe_instance_id(tmp_path) -> None:
    manager = OutputManager(tmp_path)
    manager.output_jsonl_path.write_text(
        json.dumps(
            {
                "instance_id": "../../escape.lv1",
                "n_attempt": 1,
                "model_patch": "patch",
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instance_id"):
        manager.load_completed_tasks()


def test_task_paths_reject_unsafe_instance_id_before_creating_directories(tmp_path) -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskPaths(tmp_path, "../../escape.lv1", 1)

    assert list(tmp_path.iterdir()) == []


def test_parallel_runner_does_not_reclassify_output_failures(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[env_vars]\nHF_TOKEN = ""\n', encoding="utf-8")
    config = InferConfig(
        agent="mini_swe_agent",
        model="provider/model",
        dataset_revision="v1.1",
        n_concurrent=2,
        output_dir=tmp_path / "runs",
        split="lite",
    )
    runner = run_infer.InferenceRunner(config, config_path=config_path)
    instance = TaskInstance(
        instance_id="owner__repo.task.lv1",
        problem_statement="Implement the feature",
        image_name="example/image",
        level=1,
        fail_to_pass=["tests/test_feature.py::test_feature"],
    )
    writes: list[InferResult] = []

    monkeypatch.setattr(runner, "_load_dataset", lambda: [instance])
    monkeypatch.setattr(runner, "_process_single_task", lambda *_: _result())
    monkeypatch.setattr(runner, "_start_agent_network", lambda: None)
    monkeypatch.setattr(runner, "_close_agent_network", lambda: None)

    def fail_write(result: InferResult) -> None:
        writes.append(result)
        raise OSError("disk full")

    monkeypatch.setattr(runner.output_manager, "write_result", fail_write)

    with pytest.raises(OSError, match="disk full"):
        runner.run()

    assert len(writes) == 1
    assert writes[0].success is True
    run_infer.atexit.unregister(runner._atexit_cleanup)
