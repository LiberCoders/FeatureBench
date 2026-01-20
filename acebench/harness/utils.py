"""Utility functions for ACE-Bench evaluation harness."""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from acebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_N_ATTEMPT,
    KEY_PREDICTION,
)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


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
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                # Normalize prediction format
                if KEY_PREDICTION not in pred:
                    # Try to extract from test_result.git_patch
                    if "test_result" in pred and "git_patch" in pred["test_result"]:
                        pred[KEY_PREDICTION] = pred["test_result"]["git_patch"]
                    else:
                        pred[KEY_PREDICTION] = ""

                # Ensure instance_id exists
                if KEY_INSTANCE_ID not in pred:
                    raise ValueError(f"Prediction missing '{KEY_INSTANCE_ID}': {pred}")

                # Set default model name if not provided
                if KEY_MODEL not in pred:
                    pred[KEY_MODEL] = "unknown"

                # Set default n_attempt if not provided
                if KEY_N_ATTEMPT not in pred:
                    pred[KEY_N_ATTEMPT] = 1

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
        dataset: ACE-Bench dataset DataFrame
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
    # HuggingFace dataset uses 'image_name' field
    if "image_name" in instance and pd.notna(instance["image_name"]):
        image_name = instance["image_name"]
        # If doesn't have registry prefix, add docker.io
        if '/' not in image_name or not image_name.startswith(('docker.io/', 'gcr.io/', 'ghcr.io/')):
            return f"docker.io/{image_name}".lower()
        return image_name.lower()

    # Fallback to old instance_image field for backward compatibility
    if "instance_image" not in instance or pd.isna(instance["instance_image"]):
        raise ValueError(f"image_name not found for {instance[KEY_INSTANCE_ID]}")

    image_name = instance["instance_image"]
    return f"docker.io/libercoders/{image_name}".lower()


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
    if not repo_settings_str or pd.isna(repo_settings_str):
        return {}

    try:
        if isinstance(repo_settings_str, str):
            return json.loads(repo_settings_str)
        elif isinstance(repo_settings_str, dict):
            return repo_settings_str
        else:
            return {}
    except json.JSONDecodeError:
        return {}


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
    }


def get_docker_runtime_config(repo_settings: dict) -> dict:
    """
    Extract Docker runtime configuration from repo_settings.

    Args:
        repo_settings: Repository settings dictionary

    Returns:
        Dictionary with docker runtime config (need_gpu, shm_size, number_once, env_vars, env_exports)
    """
    docker_specs = repo_settings.get("docker_specs", {})
    run_args = docker_specs.get("run_args", {})
    custom_docker_args = docker_specs.get("custom_docker_args", [])

    # check if need GPU
    need_gpu = True if run_args.get("cuda_visible_devices", None) else False

    # Parse shm_size
    shm_size = run_args.get("shm_size")

    # Parse number_once (GPU count)
    number_once = run_args.get("number_once", 1)
    if not isinstance(number_once, int) or number_once <= 0:
        number_once = 1

    # Parse environment variables from custom_docker_args (-e and -ee)
    # Ignore -v (volume) and proxy-related -e
    env_vars = {}
    env_exports = []  # For -ee (to be written to .bashrc)

    proxy_keywords = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']

    if isinstance(custom_docker_args, list):
        for arg in custom_docker_args:
            if not isinstance(arg, str):
                continue

            if arg.startswith('-e '):
                # Regular environment variable
                env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                if '=' in env_part:
                    key = env_part.split('=')[0].strip()
                    # Skip proxy-related variables
                    if key not in proxy_keywords:
                        env_vars[key] = env_part.split('=', 1)[1]

            elif arg.startswith('-ee '):
                # Environment variable to be written to .bashrc
                env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                if '=' in env_part:
                    key = env_part.split('=')[0].strip()
                    # Skip proxy-related variables
                    if key not in proxy_keywords:
                        env_exports.append(f'export {env_part}')

            # Ignore -v (volume) arguments

    return {
        "need_gpu": need_gpu,
        "shm_size": shm_size,
        "number_once": number_once,
        "env_vars": env_vars,
        "env_exports": env_exports,
    }


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

