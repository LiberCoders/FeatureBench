from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import datasets
from rich.console import Console

from featurebench.harness import run_evaluation
from featurebench.infer import run_infer
from featurebench.infer import config as infer_config_module
from featurebench.infer.config import (
    DEFAULT_DATASET_REVISION,
    DatasetLoader,
    InferConfigLoader,
)
from featurebench.infer.models import InferConfig, RunMetadata


def _write_config(path, dataset_revision: str | None = None) -> None:
    content = "[env_vars]\nHF_TOKEN = \"\"\n"
    if dataset_revision is not None:
        content += f'\n[dataset]\nrevision = "{dataset_revision}"\n'
    path.write_text(content, encoding="utf-8")


def test_dataset_revision_defaults_to_v11(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    loader = InferConfigLoader(config_path)

    assert DEFAULT_DATASET_REVISION == "v1.1"
    assert loader.get_dataset_revision() == "v1.1"


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


def test_eval_dataset_loader_passes_cli_revision(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")
    captured = {}

    def fake_load_dataset(name, *, split, token, revision):
        captured.update(name=name, split=split, token=token, revision=revision)
        return [{"instance_id": "owner__repo.commit.test_case.hash.lv2"}]

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


def test_dataset_revision_verification_rejects_stale_cache(tmp_path) -> None:
    stale_dataset = SimpleNamespace(
        cache_files=[
            {
                "filename": str(
                    tmp_path / "default" / "0.0.0" / "old-commit" / "data.arrow"
                )
            }
        ]
    )

    try:
        infer_config_module.verify_loaded_dataset_revision(stale_dataset, "new-commit")
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("stale dataset cache was accepted")


def test_inference_runner_resolves_revision_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "v1.0")
    config = InferConfig(
        agent="mini_swe_agent",
        model="provider/model",
        output_dir=tmp_path / "runs",
        split="lite",
    )

    runner = run_infer.InferenceRunner(config, config_path=config_path)

    assert runner.config.dataset_revision == "v1.0"
    runner._save_run_metadata(["task-1"])
    saved_metadata = json.loads(
        (runner.output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["dataset_revision"] == "v1.0"
    run_infer.atexit.unregister(runner._atexit_cleanup)


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


def test_eval_dataset_revision_handles_legacy_and_standalone_runs() -> None:
    assert run_evaluation.resolve_eval_dataset_revision(None, {"agent": "legacy"}) == "v1.0"
    assert run_evaluation.resolve_eval_dataset_revision(None, None) is None


def test_legacy_run_metadata_resumes_against_v10(tmp_path) -> None:
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

    metadata = RunMetadata.load(metadata_path)

    assert metadata.dataset_revision == "v1.0"
