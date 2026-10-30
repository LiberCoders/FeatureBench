from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import datasets
import pandas as pd
import pytest
from rich.console import Console

from featurebench.harness import run_evaluation
from featurebench.infer import config as infer_config_module
from featurebench.infer import run_infer
from featurebench.infer.config import (
    DEFAULT_DATASET_REVISION,
    DatasetLoader,
    InferConfigLoader,
)
from featurebench.infer.models import InferConfig, RunMetadata, TaskInstance


def _write_config(path, dataset_revision: str | None = None) -> None:
    content = '[env_vars]\nHF_TOKEN = ""\n'
    if dataset_revision is not None:
        content += f'\n[dataset]\nrevision = "{dataset_revision}"\n'
    path.write_text(content, encoding="utf-8")


def test_dataset_revision_defaults_to_v11(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    loader = InferConfigLoader(config_path)

    assert DEFAULT_DATASET_REVISION == "v1.1"
    assert loader.get_dataset_revision() == "v1.1"


def test_task_instance_validates_and_separates_dataset_metadata() -> None:
    instance = TaskInstance.from_dict(
        {
            "instance_id": "owner__repo.commit.test.hash.lv1",
            "problem_statement": "Implement the feature",
            "image_name": "docker.io/example/image:latest",
            "level": 1,
            "patch": "diff --git a/x b/x\n",
            "FAIL_TO_PASS": ["tests/test_feature.py::test_feature"],
            "repo_settings": "{}",
        }
    )

    assert instance.metadata == {"repo_settings": "{}"}
    instance.instance_id = "updated.lv1"
    assert instance.to_dict()["instance_id"] == "updated.lv1"


def test_task_instance_preserves_prompt_and_protects_canonical_fields() -> None:
    problem_statement = "  Keep leading context\n\nand the final newline.\n"
    instance = TaskInstance.from_dict(
        {
            "instance_id": " owner__repo.commit.test.hash.lv1 ",
            "problem_statement": problem_statement,
            "image_name": " docker.io/example/image:latest ",
            "level": 1,
            "FAIL_TO_PASS": [" tests/test_feature.py::test_feature "],
        }
    )

    assert instance.instance_id == "owner__repo.commit.test.hash.lv1"
    assert instance.problem_statement == problem_statement
    assert instance.image_name == "docker.io/example/image:latest"
    assert instance.fail_to_pass == ["tests/test_feature.py::test_feature"]

    directly_constructed = TaskInstance(
        instance_id="canonical.lv1",
        problem_statement="canonical prompt",
        image_name="canonical/image",
        level=1,
        fail_to_pass=["tests/test_feature.py::test_feature"],
        metadata={"instance_id": "shadowed.lv2", "level": 2},
    )
    serialized = directly_constructed.to_dict()
    assert serialized["instance_id"] == "canonical.lv1"
    assert serialized["level"] == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"problem_statement": ""}, "problem_statement"),
        ({"image_name": None}, "image_name"),
        ({"level": 3}, "level"),
        ({"FAIL_TO_PASS": "tests/test_feature.py"}, "FAIL_TO_PASS"),
        ({"FAIL_TO_PASS": []}, "FAIL_TO_PASS"),
        ({"patch": 1}, "patch"),
    ],
)
def test_task_instance_rejects_malformed_dataset_rows(updates, message) -> None:
    row = {
        "instance_id": "owner__repo.commit.test.hash.lv1",
        "problem_statement": "Implement the feature",
        "image_name": "docker.io/example/image:latest",
        "level": 1,
        "patch": None,
        "FAIL_TO_PASS": ["tests/test_feature.py::test_feature"],
    }
    row.update(updates)

    with pytest.raises(ValueError, match=message):
        TaskInstance.from_dict(row)


@pytest.mark.parametrize(
    "instance_id",
    ["../../escape.lv1", "/tmp/escape.lv1", ".hidden.lv1", "bad/id.lv1", "bad id.lv1"],
)
def test_task_instance_rejects_unsafe_instance_ids(instance_id) -> None:
    with pytest.raises(ValueError, match="instance_id"):
        TaskInstance.from_dict(
            {
                "instance_id": instance_id,
                "problem_statement": "Implement the feature",
                "image_name": "docker.io/example/image:latest",
                "level": 1,
                "FAIL_TO_PASS": ["tests/test_feature.py::test_feature"],
            }
        )


@pytest.mark.parametrize(
    "test_ids",
    [
        ["../../outside.py::test_feature"],
        ["/etc/passwd"],
        ["tests/test_feature.py\n::test_feature"],
        ["tests/test_feature.py::"],
        ["tests/test_feature.py", "/testbed/tests/test_feature.py"],
    ],
)
def test_task_instance_rejects_unsafe_or_duplicate_test_ids(test_ids) -> None:
    with pytest.raises(ValueError, match="FAIL_TO_PASS"):
        TaskInstance.from_dict(
            {
                "instance_id": "owner__repo.commit.test.hash.lv1",
                "problem_statement": "Implement the feature",
                "image_name": "docker.io/example/image:latest",
                "level": 1,
                "FAIL_TO_PASS": test_ids,
            }
        )


def test_task_instance_normalizes_testbed_test_ids() -> None:
    instance = TaskInstance.from_dict(
        {
            "instance_id": "owner__repo.commit.test.hash.lv1",
            "problem_statement": "Implement the feature",
            "image_name": "docker.io/example/image:latest",
            "level": 1,
            "FAIL_TO_PASS": ["/testbed/tests/test_feature.py::test_feature"],
        }
    )

    assert instance.fail_to_pass == ["tests/test_feature.py::test_feature"]


def test_infer_config_defaults_to_current_working_directory(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.1")
    monkeypatch.chdir(tmp_path)

    loader = InferConfigLoader()

    assert loader.config_path == config_path
    assert loader.get_dataset_revision() == "v1.1"


def test_infer_config_missing_from_current_working_directory_is_explicit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="current working directory"):
        InferConfigLoader()


def test_dataset_revision_reads_config(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")

    loader = InferConfigLoader(config_path)

    assert loader.get_dataset_revision() == "v1.0"


def test_infer_dataset_loader_passes_revision(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")
    captured = {}

    def fake_load_dataset(name, *, split, token, revision):
        captured.update(name=name, split=split, token=token, revision=revision)
        return [{"instance_id": "owner__repo.commit.test_case.hash.lv1"}]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        infer_config_module,
        "resolve_hf_dataset_revision",
        lambda dataset, revision, token, endpoint: f"resolved-{revision}",
    )
    loader = DatasetLoader(InferConfigLoader(config_path))

    rows = loader.load_dataset("LiberCoders/FeatureBench", split="lite")

    assert captured == {
        "name": "LiberCoders/FeatureBench",
        "split": "lite",
        "token": False,
        "revision": "resolved-v1.0",
    }
    assert rows[0]["level"] == 1

    loader.load_dataset(
        "LiberCoders/FeatureBench",
        split="lite",
        revision="main",
    )
    assert captured["revision"] == "resolved-main"


def test_infer_dataset_loader_does_not_turn_load_failure_into_empty_success(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    def fail_load_dataset(*args, **kwargs):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(datasets, "load_dataset", fail_load_dataset)
    monkeypatch.setattr(
        infer_config_module,
        "resolve_hf_dataset_revision",
        lambda *args: "resolved-v1.1",
    )

    with pytest.raises(RuntimeError, match="dataset unavailable"):
        DatasetLoader(InferConfigLoader(config_path)).load_dataset(
            "LiberCoders/FeatureBench",
            split="lite",
        )


def test_infer_dataset_loader_rejects_unsafe_instance_id(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: [{"instance_id": "../../escape.lv1"}],
    )
    monkeypatch.setattr(
        infer_config_module,
        "resolve_hf_dataset_revision",
        lambda *args: "resolved-v1.1",
    )

    with pytest.raises(ValueError, match="instance_id"):
        DatasetLoader(InferConfigLoader(config_path)).load_dataset(
            "LiberCoders/FeatureBench",
            split="lite",
        )


@pytest.mark.parametrize(
    ("rows", "task_ids", "message"),
    [
        (
            [
                {"instance_id": "owner__repo.duplicate.lv1"},
                {"instance_id": "owner__repo.duplicate.lv1"},
            ],
            None,
            "duplicate instance_id",
        ),
        (
            [{"instance_id": "owner__repo.available.lv1"}],
            ["owner__repo.missing.lv1"],
            "requested task IDs",
        ),
        (
            [{"instance_id": "owner__repo.available.lv1"}],
            ["owner__repo.available.lv1"],
            "selected dataset and levels",
        ),
    ],
)
def test_infer_dataset_loader_rejects_ambiguous_or_missing_tasks(
    monkeypatch, tmp_path, rows, task_ids, message
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        infer_config_module,
        "resolve_hf_dataset_revision",
        lambda *args: "resolved-v1.1",
    )

    levels = [2] if task_ids == ["owner__repo.available.lv1"] else None
    with pytest.raises(ValueError, match=message):
        DatasetLoader(InferConfigLoader(config_path)).load_dataset(
            "LiberCoders/FeatureBench",
            split="lite",
            levels=levels,
            task_ids=task_ids,
        )


def test_eval_dataset_loader_passes_cli_revision(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")
    captured = {}

    def fake_load_dataset(name, *, split, token, revision):
        captured.update(name=name, split=split, token=token, revision=revision)
        return [
            {
                "instance_id": "owner__repo.commit.test_case.hash.lv2",
                "FAIL_TO_PASS": ["tests/test_feature.py::test_feature"],
            }
        ]

    monkeypatch.setattr(run_evaluation, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        run_evaluation,
        "resolve_hf_dataset_revision",
        lambda *args: "resolved-v1.1",
    )

    frame = run_evaluation.load_dataset_from_hf(
        Console(),
        "full",
        "LiberCoders/FeatureBench",
        str(config_path),
        "v1.1",
    )

    assert captured == {
        "name": "LiberCoders/FeatureBench",
        "split": "full",
        "token": False,
        "revision": "resolved-v1.1",
    }
    assert frame["level"].tolist() == [2]


def test_eval_dataset_loader_rejects_missing_explicit_config(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        run_evaluation.load_dataset_from_hf(
            Console(),
            "lite",
            "LiberCoders/FeatureBench",
            str(tmp_path / "missing.toml"),
        )


def test_dataset_revision_verification_rejects_stale_cache(tmp_path) -> None:
    stale_dataset = SimpleNamespace(
        cache_files=[
            {"filename": str(tmp_path / "default" / "0.0.0" / "old-commit" / "data.arrow")}
        ]
    )

    try:
        infer_config_module.verify_loaded_dataset_revision(stale_dataset, "new-commit")
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("stale dataset cache was accepted")


def test_inference_runner_resolves_revision_from_config(capsys, monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")
    config = InferConfig(
        agent="mini_swe_agent",
        model="provider/model",
        output_dir=tmp_path / "runs",
        split="lite",
        cost_limit=0,
        api_key="must-not-be-persisted",
    )

    runner = run_infer.InferenceRunner(config, config_path=config_path)

    assert runner.config.dataset_revision == "v1.0"
    runner._save_run_metadata(["task-1"])
    saved_metadata = json.loads(
        (runner.output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["dataset_revision"] == "v1.0"
    assert saved_metadata["cost_limit"] == 0
    assert "api_key" not in saved_metadata
    assert "api_key" not in config.to_dict()
    runner.output_manager.update_metadata_end_time()
    updated_metadata = RunMetadata.load(runner.output_dir / "run_metadata.json")
    assert updated_metadata.end_time is not None
    assert not (runner.output_dir / "run_metadata.json.tmp").exists()
    monkeypatch.setattr(runner, "_load_dataset", list)
    assert runner.run() == 1
    assert "must-not-be-persisted" not in capsys.readouterr().out
    run_infer.atexit.unregister(runner._atexit_cleanup)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"n_concurrent": 0}, "n_concurrent"),
        ({"n_attempts": 0}, "n_attempts"),
        ({"timeout": 0}, "timeout"),
        ({"proxy_port": 70000}, "proxy_port"),
        ({"max_iters": 0}, "max_iters"),
        ({"cost_limit": -1}, "cost_limit"),
        ({"task_ids": ["task.lv1", "task.lv1"]}, "duplicates"),
        ({"level": []}, "level"),
        ({"split": " "}, "split"),
    ],
)
def test_infer_config_rejects_invalid_execution_boundaries(updates, message) -> None:
    values = {
        "agent": "mini_swe_agent",
        "model": "provider/model",
        "split": "lite",
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        InferConfig(**values)


def test_cli_accepts_data_version(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fb infer", "--data-version", "v1.0"])
    infer_args = run_infer.parse_args()
    assert infer_args.dataset_version == "v1.0"
    assert infer_args._dataset_version_provided is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["fb eval", "--predictions-path", "predictions.jsonl", "--data-version", "main"],
    )
    eval_args = run_evaluation.parse_args()
    assert eval_args.dataset_version == "main"


@pytest.mark.parametrize(
    "argv",
    [
        ["fb infer", "--n-concurrent", "0"],
        ["fb infer", "--n-attempts", "0"],
        ["fb infer", "--timeout", "0"],
        ["fb infer", "--proxy-port", "65536"],
    ],
)
def test_infer_cli_rejects_invalid_positive_integer_options(monkeypatch, argv) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        run_infer.parse_args()

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["fb eval", "--predictions-path", "output.jsonl", "--n-concurrent", "0"],
        ["fb eval", "--predictions-path", "output.jsonl", "--timeout", "-1"],
        ["fb eval", "--predictions-path", "output.jsonl", "--proxy-port", "0"],
    ],
)
def test_eval_cli_rejects_invalid_positive_integer_options(monkeypatch, argv) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        run_evaluation.parse_args()

    assert exc_info.value.code == 2


def test_force_rerun_rejects_a_missing_list_file(tmp_path) -> None:
    missing = str(tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError, match=r"missing\.txt"):
        run_infer._load_force_rerun_ids([missing])
    with pytest.raises(FileNotFoundError, match=r"missing\.txt"):
        run_evaluation._load_force_rerun_ids([missing])


def test_eval_dataset_revision_follows_run_metadata() -> None:
    assert (
        run_evaluation.resolve_eval_dataset_revision(
            None,
            {"dataset_revision": "v1.1"},
        )
        == "v1.1"
    )
    assert (
        run_evaluation.resolve_eval_dataset_revision(
            "v1.0",
            {"dataset_revision": "v1.1"},
        )
        == "v1.0"
    )


def test_eval_dataset_revision_rejects_metadata_without_revision() -> None:
    with pytest.raises(ValueError, match="dataset_revision"):
        run_evaluation.resolve_eval_dataset_revision(None, {"agent": "current"})
    assert run_evaluation.resolve_eval_dataset_revision(None, None) is None


def test_run_metadata_requires_dataset_revision(tmp_path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "agent": "mini_swe_agent",
                "model": "provider/model",
                "dataset": "LiberCoders/FeatureBench",
                "n_concurrent": 1,
                "n_attempts": 1,
                "task_ids": [],
                "output_path": str(tmp_path),
                "start_time": "2026-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="dataset_revision"):
        RunMetadata.load(metadata_path)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["dataset_revision"] = "  "
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        RunMetadata.load(metadata_path)


def test_run_metadata_rejects_persisted_credentials(tmp_path) -> None:
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "agent": "mini_swe_agent",
                "model": "provider/model",
                "dataset": "LiberCoders/FeatureBench",
                "dataset_revision": "v1.1",
                "n_concurrent": 1,
                "n_attempts": 1,
                "task_ids": [],
                "output_path": str(tmp_path),
                "start_time": "2026-01-01T00:00:00",
                "api_key": "secret",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="api_key"):
        RunMetadata.load(metadata_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"n_concurrent": 0}, "n_concurrent"),
        ({"task_ids": []}, "task_ids"),
        ({"level": [1, 1]}, "duplicates"),
        ({"white_box": "false"}, "white_box"),
        ({"native_tool_calling": "yes"}, "native_tool_calling"),
        ({"max_iters": 0}, "max_iters"),
    ],
)
def test_run_metadata_rejects_invalid_resume_boundaries(tmp_path, updates, message) -> None:
    payload = {
        "agent": "mini_swe_agent",
        "model": "provider/model",
        "dataset": "LiberCoders/FeatureBench",
        "dataset_revision": "v1.1",
        "n_concurrent": 1,
        "n_attempts": 1,
        "task_ids": ["task.lv1"],
        "output_path": str(tmp_path),
        "start_time": "2026-01-01T00:00:00",
        "timeout": 3600,
        "split": "lite",
    }
    payload.update(updates)
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        RunMetadata.load(metadata_path)


def test_eval_rejects_malformed_inference_metadata(monkeypatch, tmp_path) -> None:
    predictions_path = tmp_path / "output.jsonl"
    (tmp_path / "run_metadata.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(
        run_evaluation,
        "parse_args",
        lambda: SimpleNamespace(predictions_path=str(predictions_path)),
    )

    with pytest.raises(RuntimeError, match="Invalid inference metadata"):
        run_evaluation.main()


def test_eval_returns_failure_when_no_tasks_match(monkeypatch, tmp_path) -> None:
    predictions_path = tmp_path / "output.jsonl"
    args = SimpleNamespace(
        predictions_path=str(predictions_path),
        dataset_version=None,
        dataset="LiberCoders/FeatureBench",
        split="lite",
        n_concurrent=1,
        timeout=None,
        gpu_ids=None,
        proxy_port=None,
        review_codes=False,
        config_path=None,
        include_failed=False,
        task_ids=None,
        force_rerun=None,
    )
    monkeypatch.setattr(run_evaluation, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_evaluation,
        "load_dataset_from_hf",
        lambda *args, **kwargs: pd.DataFrame({"instance_id": []}),
    )
    monkeypatch.setattr(run_evaluation, "get_predictions_from_file", lambda path: [])

    assert run_evaluation.main() == 1
