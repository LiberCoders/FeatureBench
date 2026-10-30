from __future__ import annotations

import json

import pandas as pd
import pytest

from featurebench.harness.constants import KEY_INSTANCE_ID
from featurebench.harness.utils import (
    get_docker_image_name,
    get_docker_runtime_config,
    get_predictions_from_file,
    parse_repo_settings,
)
from featurebench.infer.models import TaskInstance


def test_docker_image_uses_the_current_dataset_field() -> None:
    instance = pd.Series(
        {
            KEY_INSTANCE_ID: "owner__repo.task.lv1",
            "image_name": "LiberCoders/FeatureBench-Specs_Repo-Instance_1234",
        }
    )

    assert (
        get_docker_image_name(instance)
        == "docker.io/libercoders/featurebench-specs_repo-instance_1234"
    )


def test_docker_image_rejects_removed_instance_image_field() -> None:
    instance = pd.Series(
        {
            KEY_INSTANCE_ID: "owner__repo.task.lv1",
            "instance_image": "featurebench-specs_repo-instance_1234",
        }
    )

    with pytest.raises(ValueError, match="image_name not found"):
        get_docker_image_name(instance)


def test_predictions_loader_validates_and_normalizes_current_schema(tmp_path) -> None:
    predictions_path = tmp_path / "output.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": " task.lv1 ",
                "model_patch": "diff --git a/x b/x\n",
                "model_name_or_path": " model ",
                "n_attempt": 2,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = get_predictions_from_file(predictions_path)

    assert predictions[0]["instance_id"] == "task.lv1"
    assert predictions[0]["model_name_or_path"] == "model"
    assert predictions[0]["n_attempt"] == 2


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"instance_id": "task.lv1"}], "model_patch"),
        ([{"instance_id": "task.lv1", "model_patch": None}], "model_patch"),
        ([{"instance_id": "task.lv1", "model_patch": "", "n_attempt": 0}], "n_attempt"),
        ([{"instance_id": "task.lv1", "model_patch": "", "success": "yes"}], "success"),
        (
            [
                {"instance_id": "task.lv1", "model_patch": "one"},
                {"instance_id": "task.lv1", "model_patch": "two"},
            ],
            "Duplicate prediction",
        ),
    ],
)
def test_predictions_loader_rejects_invalid_or_duplicate_records(
    tmp_path, records, message
) -> None:
    predictions_path = tmp_path / "output.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        get_predictions_from_file(predictions_path)


def test_predictions_loader_reports_invalid_json_line(tmp_path) -> None:
    predictions_path = tmp_path / "output.jsonl"
    predictions_path.write_text("\n{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"output\.jsonl:2"):
        get_predictions_from_file(predictions_path)


def test_predictions_loader_rejects_unsafe_instance_id(tmp_path) -> None:
    predictions_path = tmp_path / "output.jsonl"
    predictions_path.write_text(
        json.dumps({"instance_id": "../../escape.lv1", "model_patch": "patch"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instance_id"):
        get_predictions_from_file(predictions_path)


def test_runtime_config_uses_the_published_dataset_schema_for_infer_and_eval() -> None:
    repo_settings = {
        "docker_specs": {
            "run_args": {
                "cuda_visible_devices": "4,5,6,7",
                "cuda_visible_num": None,
                "shm_size": "8g",
                "number_once": 2,
            },
            "custom_docker_args": [
                "-e KEEP=value",
                "-ee EXPORT=value=with=equals",
                "-e Http_Proxy=http://secret-proxy",
                "-v /host:/container",
                3,
            ],
        }
    }
    expected = {
        "need_gpu": True,
        "shm_size": "8g",
        "number_once": 2,
        "env_vars": {"KEEP": "value"},
        "env_exports": ["export EXPORT=value=with=equals"],
    }
    instance = TaskInstance(
        instance_id="owner__repo.task.lv1",
        problem_statement="Implement the feature.",
        image_name="libercoders/example",
        level=1,
        fail_to_pass=["tests/test_feature.py::test_feature"],
        metadata={"repo_settings": repo_settings},
    )

    assert get_docker_runtime_config(repo_settings) == expected
    assert instance.get_docker_runtime_config() == expected


@pytest.mark.parametrize("value", ["{broken", "[]", [], 3])
def test_repo_settings_rejects_malformed_dataset_values(value) -> None:
    row = pd.Series({"repo_settings": value})
    instance = TaskInstance(
        instance_id="owner__repo.task.lv1",
        problem_statement="Implement the feature.",
        image_name="libercoders/example",
        level=1,
        fail_to_pass=["tests/test_feature.py::test_feature"],
        metadata={"repo_settings": value},
    )

    with pytest.raises(ValueError, match="repo_settings"):
        parse_repo_settings(row)
    with pytest.raises(ValueError, match="repo_settings"):
        instance.get_repo_settings()


@pytest.mark.parametrize(
    "repo_settings", [{}, {"docker_specs": None}, {"docker_specs": {"run_args": []}}]
)
def test_runtime_config_handles_missing_or_malformed_optional_sections(repo_settings) -> None:
    assert get_docker_runtime_config(repo_settings) == {
        "need_gpu": False,
        "shm_size": None,
        "number_once": 1,
        "env_vars": {},
        "env_exports": [],
    }


@pytest.mark.parametrize(
    ("raw_shm_size", "expected"),
    [(" 8g ", "8g"), ("", None), (64, None)],
)
def test_runtime_config_normalizes_shared_memory_size(raw_shm_size, expected) -> None:
    config = get_docker_runtime_config({"docker_specs": {"run_args": {"shm_size": raw_shm_size}}})

    assert config["shm_size"] == expected
