"""Utility functions for FeatureBench evaluation harness."""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from featurebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_N_ATTEMPT,
    KEY_PREDICTION,
)
from featurebench.utils.docker_images import normalize_image_name
from featurebench.utils.identifiers import normalize_instance_id
from featurebench.utils.runtime_config import DockerRuntimeConfig, parse_docker_runtime_config


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""


def get_predictions_from_file(file_path: str | Path) -> list[dict]:
    """
    Load predictions from a JSONL file.

    Each line should be a JSON object with at least:
    - instance_id: str
    - model_patch: str (or test_result.git_patch)
    - model_name_or_path: str (optional)

    Args:
        file_path: Path to predictions JSONL file

    Returns:
        List of prediction dictionaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {file_path}")

    predictions = []
    seen_keys: set[tuple[str, int]] = set()
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                pred = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {file_path}:{line_number}") from exc
            if not isinstance(pred, dict):
                raise ValueError(f"Prediction at {file_path}:{line_number} must be a JSON object")

            try:
                pred[KEY_INSTANCE_ID] = normalize_instance_id(pred.get(KEY_INSTANCE_ID))
            except ValueError as exc:
                raise ValueError(
                    f"Prediction at {file_path}:{line_number} has an invalid "
                    f"'{KEY_INSTANCE_ID}': {exc}"
                ) from exc

            model_patch = pred.get(KEY_PREDICTION)
            if not isinstance(model_patch, str):
                raise ValueError(
                    f"Prediction at {file_path}:{line_number} has an invalid '{KEY_PREDICTION}'"
                )

            model = pred.get(KEY_MODEL, "unknown")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"Prediction at {file_path}:{line_number} has an invalid '{KEY_MODEL}'"
                )
            pred[KEY_MODEL] = model.strip()

            n_attempt = pred.get(KEY_N_ATTEMPT, 1)
            if isinstance(n_attempt, bool) or not isinstance(n_attempt, int) or n_attempt < 1:
                raise ValueError(
                    f"Prediction at {file_path}:{line_number} has an invalid '{KEY_N_ATTEMPT}'"
                )
            pred[KEY_N_ATTEMPT] = n_attempt

            if "success" in pred and not isinstance(pred["success"], bool):
                raise ValueError(
                    f"Prediction at {file_path}:{line_number} has an invalid 'success'"
                )

            prediction_key = (pred[KEY_INSTANCE_ID], n_attempt)
            if prediction_key in seen_keys:
                raise ValueError(
                    "Duplicate prediction for "
                    f"instance_id={prediction_key[0]!r}, n_attempt={prediction_key[1]}"
                )
            seen_keys.add(prediction_key)
            predictions.append(pred)

    return predictions


def filter_predictions_by_ids(
    predictions: list[dict],
    instance_ids: list[str] | None = None
) -> list[dict]:
    """
    Filter predictions by instance IDs.

    Args:
        predictions: List of predictions
        instance_ids: List of instance IDs to keep (None = keep all)

    Returns:
        Filtered list of predictions
    """
    if instance_ids is None:
        return predictions

    instance_ids_set = set(instance_ids)
    return [p for p in predictions if p[KEY_INSTANCE_ID] in instance_ids_set]


def get_instance_from_dataset(
    dataset: pd.DataFrame,
    instance_id: str
) -> pd.Series | None:
    """
    Get a single instance from the dataset by instance_id.

    Args:
        dataset: FeatureBench dataset DataFrame
        instance_id: Instance ID to retrieve

    Returns:
        Instance as pandas Series, or None if not found
    """
    matches = dataset[dataset[KEY_INSTANCE_ID] == instance_id]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def get_docker_image_name(instance: pd.Series) -> str:
    """
    Get Docker image name for an instance.

    Args:
        instance: Instance data as pandas Series (from HuggingFace)

    Returns:
        Docker image name (e.g., docker.io/libercoders/image_name)
    """
    if "image_name" not in instance or pd.isna(instance["image_name"]):
        raise ValueError(f"image_name not found for {instance[KEY_INSTANCE_ID]}")
    return normalize_image_name(instance["image_name"])


def str2bool(v: str) -> bool:
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Boolean value expected, got: {v}")


def parse_repo_settings(instance: pd.Series) -> dict:
    """
    Parse repo_settings from instance data.

    Args:
        instance: Instance data as pandas Series

    Returns:
        Dictionary containing repo settings
    """
    repo_settings_str = instance.get("repo_settings", "{}")
    if repo_settings_str is None or (
        pd.api.types.is_scalar(repo_settings_str) and pd.isna(repo_settings_str)
    ):
        return {}
    if isinstance(repo_settings_str, str) and not repo_settings_str.strip():
        return {}

    if isinstance(repo_settings_str, str):
        try:
            parsed = json.loads(repo_settings_str)
        except json.JSONDecodeError as exc:
            raise ValueError("repo_settings must contain valid JSON") from exc
    elif isinstance(repo_settings_str, dict):
        parsed = repo_settings_str
    else:
        raise ValueError("repo_settings must be a JSON object")
    if not isinstance(parsed, dict):
        raise ValueError("repo_settings must be a JSON object")
    return parsed


def get_test_config_from_repo_settings(repo_settings: dict) -> dict:
    """
    Extract test configuration from repo_settings.

    Args:
        repo_settings: Repository settings dictionary

    Returns:
        Dictionary with test_cmd, timeout_run, timeout_one
    """
    return {
        "test_cmd": repo_settings.get("test_cmd", "pytest -rA -p no:cacheprovider --color=no"),
        "timeout_run": repo_settings.get("timeout_run", 600),
        "timeout_one": repo_settings.get("timeout_one", 10),
        "use_uv": repo_settings.get("use_uv", False),
    }


def get_docker_runtime_config(repo_settings: dict) -> DockerRuntimeConfig:
    """Extract Docker runtime configuration from repository settings."""
    return parse_docker_runtime_config(repo_settings)


def build_test_command(test_cmd: str, timeout_one: Optional[int] = None) -> str:
    """
    Build test command with timeout configuration.

    Args:
        test_cmd: Base test command
        timeout_one: Timeout per test case (seconds)

    Returns:
        Complete test command string
    """
    if timeout_one is not None and timeout_one > 0:
        return f"{test_cmd} --timeout={timeout_one}"
    return test_cmd


def preprocess_hf_patch(patch_text: str, raw_fail_to_pass: Any) -> str:
    """
    Preprocess HF patch for harness evaluation.

    Steps:
    1) Reverse patch direction (dataset stores corruption patch).
    2) Parse FAIL_TO_PASS paths.
    3) Remove patch operations touching FAIL_TO_PASS files.
    """
    reversed_patch = reverse_unified_patch(patch_text)
    fail_to_pass_paths = parse_fail_to_pass_paths(raw_fail_to_pass)
    return remove_patch_operations_for_paths(reversed_patch, fail_to_pass_paths)


def reverse_unified_patch(patch_text: str) -> str:
    """Reverse a unified diff patch so it can be applied in the opposite direction."""
    if not patch_text:
        return patch_text

    hunk_header_pattern = re.compile(r"^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)$")
    reversed_lines: list[str] = []

    for line in patch_text.splitlines():
        if line.startswith("--- "):
            reversed_lines.append("+++ " + line[4:])
            continue
        if line.startswith("+++ "):
            reversed_lines.append("--- " + line[4:])
            continue

        match = hunk_header_pattern.match(line)
        if match:
            old_range, new_range, tail = match.groups()
            reversed_lines.append(f"@@ -{new_range} +{old_range} @@{tail}")
            continue

        if line.startswith("new file mode "):
            reversed_lines.append(line.replace("new file mode ", "deleted file mode ", 1))
            continue
        if line.startswith("deleted file mode "):
            reversed_lines.append(line.replace("deleted file mode ", "new file mode ", 1))
            continue
        if line.startswith("old mode "):
            reversed_lines.append(line.replace("old mode ", "new mode ", 1))
            continue
        if line.startswith("new mode "):
            reversed_lines.append(line.replace("new mode ", "old mode ", 1))
            continue
        if line.startswith("rename from "):
            reversed_lines.append(line.replace("rename from ", "rename to ", 1))
            continue
        if line.startswith("rename to "):
            reversed_lines.append(line.replace("rename to ", "rename from ", 1))
            continue
        if line.startswith("copy from "):
            reversed_lines.append(line.replace("copy from ", "copy to ", 1))
            continue
        if line.startswith("copy to "):
            reversed_lines.append(line.replace("copy to ", "copy from ", 1))
            continue

        if line.startswith("+") and not line.startswith("+++ "):
            reversed_lines.append("-" + line[1:])
            continue
        if line.startswith("-") and not line.startswith("--- "):
            reversed_lines.append("+" + line[1:])
            continue

        reversed_lines.append(line)

    return "\n".join(reversed_lines) + ("\n" if patch_text.endswith("\n") else "")


def _normalize_patch_path(path: str) -> str:
    path = str(path).strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if path.startswith("./"):
        path = path[2:]
    return path


def parse_fail_to_pass_paths(raw_fail_to_pass: Any) -> set[str]:
    """Parse FAIL_TO_PASS field into normalized file-path set."""
    if raw_fail_to_pass is None:
        return set()

    values: Any = raw_fail_to_pass
    if isinstance(raw_fail_to_pass, str):
        try:
            values = json.loads(raw_fail_to_pass)
        except json.JSONDecodeError:
            return set()

    if not isinstance(values, list):
        return set()

    result: set[str] = set()
    for item in values:
        if isinstance(item, str) and item.strip():
            result.add(_normalize_patch_path(item))
    return result


def remove_patch_operations_for_paths(patch_text: str, excluded_paths: set[str]) -> str:
    """
    Remove file-level operations in unified diff when file path is excluded.

    This works on `diff --git ...` blocks and drops whole blocks whose target
    file path matches any path in `excluded_paths`.
    """
    if not patch_text or not excluded_paths:
        return patch_text

    normalized_excluded = {_normalize_patch_path(p) for p in excluded_paths if p}
    lines = patch_text.splitlines(keepends=True)
    output: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.startswith("diff --git "):
            output.append(line)
            i += 1
            continue

        block_start = i
        i += 1
        while i < len(lines) and not lines[i].startswith("diff --git "):
            i += 1
        block = lines[block_start:i]
        header = block[0].strip()
        parts = header.split()
        if len(parts) >= 4:
            path_a = _normalize_patch_path(parts[2])
            path_b = _normalize_patch_path(parts[3])
            should_drop = path_a in normalized_excluded or path_b in normalized_excluded
        else:
            should_drop = False

        if not should_drop:
            output.extend(block)

    return "".join(output)
