"""Data models for the inference module."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from featurebench.utils.identifiers import normalize_instance_id, normalize_test_ids
from featurebench.utils.runtime_config import DockerRuntimeConfig, parse_docker_runtime_config


class AgentName(str, Enum):
    """Supported agent names."""

    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli"
    MINI_SWE_AGENT = "mini_swe_agent"
    OPENHANDS = "openhands"
    CODEX = "codex"


@dataclass
class TaskInstance:
    """Represents a single task instance from the dataset."""

    instance_id: str
    problem_statement: str
    image_name: str
    level: int
    patch: str | None = None
    fail_to_pass: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instance_id = normalize_instance_id(
            self.instance_id,
            field="task field 'instance_id'",
        )
        for name in ("problem_statement", "image_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task field {name!r} must be a non-empty string")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level not in {1, 2}:
            raise ValueError("task field 'level' must be 1 or 2")
        if self.patch is not None and not isinstance(self.patch, str):
            raise ValueError("task field 'patch' must be a string or null")
        self.fail_to_pass = normalize_test_ids(
            self.fail_to_pass,
            field="task field 'FAIL_TO_PASS'",
            allow_empty=False,
        )
        if not isinstance(self.metadata, dict):
            raise ValueError("task metadata must be an object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInstance":
        """Create a validated task instance from one dataset row."""
        required: dict[str, str] = {}
        for key in ("instance_id", "problem_statement", "image_name"):
            value = data.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task field {key!r} must be a non-empty string")
            if key == "instance_id":
                required[key] = normalize_instance_id(
                    value,
                    field="task field 'instance_id'",
                )
            else:
                required[key] = value if key == "problem_statement" else value.strip()

        raw_level = data.get("level")
        if isinstance(raw_level, bool) or not isinstance(raw_level, int) or raw_level not in {1, 2}:
            raise ValueError("task field 'level' must be 1 or 2")

        fail_to_pass = normalize_test_ids(
            data.get("FAIL_TO_PASS"),
            field="task field 'FAIL_TO_PASS'",
            allow_empty=False,
        )

        patch = data.get("patch")
        if patch is not None and not isinstance(patch, str):
            raise ValueError("task field 'patch' must be a string or null")

        canonical_keys = {
            "instance_id",
            "problem_statement",
            "image_name",
            "level",
            "patch",
            "FAIL_TO_PASS",
        }
        return cls(
            instance_id=required["instance_id"],
            problem_statement=required["problem_statement"],
            image_name=required["image_name"],
            level=raw_level,
            patch=patch,
            fail_to_pass=fail_to_pass,
            metadata={key: value for key, value in data.items() if key not in canonical_keys},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **self.metadata,
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "image_name": self.image_name,
            "level": self.level,
            "patch": self.patch,
            "FAIL_TO_PASS": self.fail_to_pass,
        }

    def get_repo_settings(self) -> dict[str, Any]:
        """
        Parse and return repo_settings from metadata.

        Returns:
            Dictionary containing repo settings
        """
        repo_settings_str = self.metadata.get("repo_settings", "{}")
        if repo_settings_str is None:
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

    def get_docker_runtime_config(self) -> DockerRuntimeConfig:
        """Extract Docker runtime configuration from repository settings."""
        return parse_docker_runtime_config(self.get_repo_settings())


@dataclass
class InferConfig:
    """Configuration for inference run."""

    agent: str
    model: str
    # HuggingFace dataset repo name (e.g., "LiberCoders/FeatureBench")
    dataset: str = "LiberCoders/FeatureBench"
    # Hugging Face dataset tag, branch, or commit SHA. None defers to config.toml/default.
    dataset_revision: str | None = None
    n_concurrent: int = 1
    n_attempts: int = 1
    task_ids: list[str] | None = None
    level: list[int] | None = None
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    timeout: int = 7200  # 2 hours default timeout
    proxy_port: int | None = None
    runtime_proxy: bool | None = None
    gpu_ids: str | None = None  # Comma-separated GPU IDs (e.g., "0,1,2,3"), None means all
    # OpenHands only: max step/iteration limit. None means "do not override" (OpenHands default applies).
    max_iters: int | None = None
    # mini-swe-agent only: per-task cost limit in USD. 0 disables the limit.
    cost_limit: float | None = None
    split: str = "full"  # HuggingFace dataset split name (e.g., "lite", "full")
    # If True, remove the "## Interface Descriptions" section from the task prompt.
    without_interface_descriptions: bool = False
    # If True, enable white-box mode: agent can see FAIL_TO_PASS test file(s).
    white_box: bool = False
    # OpenHands only: True/False forces LLM_NATIVE_TOOL_CALLING, None leaves OpenHands default.
    native_tool_calling: bool | None = None
    # If True, send prior assistant reasoning content back to OpenHands LLM requests.
    send_reasoning_content: bool = False
    # OpenHands only: JSON object passed through to LLM.litellm_extra_body.
    litellm_extra_body: str | None = None
    # OpenHands only: attach an X-Session-Id header per task attempt and release it after the run.
    session_cache: bool = False
    # Optional task IDs to force rerun even if completed.
    force_rerun_ids: list[str] | None = None
    # If True, treat prior TIMEOUT attempts as completed when resuming (skip reruns).
    force_timeout: bool = False
    # Validated non-secret execution settings restored from run metadata on resume.
    openhands_reasoning_effort: str | None = None
    codex_reasoning_effort: str | None = None
    # Optional: CLI overrides for agent auth/endpoint/version.
    api_key: str | None = None
    base_url: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        for name in ("agent", "model", "dataset"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            setattr(self, name, value.strip())

        if self.dataset_revision is not None:
            if not isinstance(self.dataset_revision, str) or not self.dataset_revision.strip():
                raise ValueError("dataset_revision must be a non-empty string when provided")
            self.dataset_revision = self.dataset_revision.strip()

        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        self.split = self.split.strip()

        for name in ("n_concurrent", "n_attempts", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.proxy_port is not None and (
            isinstance(self.proxy_port, bool)
            or not isinstance(self.proxy_port, int)
            or not 1 <= self.proxy_port <= 65535
        ):
            raise ValueError("proxy_port must be between 1 and 65535")
        if self.max_iters is not None and (
            isinstance(self.max_iters, bool)
            or not isinstance(self.max_iters, int)
            or self.max_iters < 1
        ):
            raise ValueError("max_iters must be a positive integer")
        if self.cost_limit is not None and (
            isinstance(self.cost_limit, bool)
            or not isinstance(self.cost_limit, (int, float))
            or self.cost_limit < 0
        ):
            raise ValueError("cost_limit must be a non-negative number")

        self.task_ids = self._validate_ids("task_ids", self.task_ids)
        self.force_rerun_ids = self._validate_ids("force_rerun_ids", self.force_rerun_ids)
        if self.level is not None:
            if (
                not isinstance(self.level, list)
                or not self.level
                or any(isinstance(value, bool) or value not in {1, 2} for value in self.level)
            ):
                raise ValueError("level must be a non-empty list containing only 1 or 2")
            if len(self.level) != len(set(self.level)):
                raise ValueError("level must not contain duplicates")

    @staticmethod
    def _validate_ids(name: str, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list")
        normalized = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must contain only non-empty strings")
            normalized.append(normalize_instance_id(value, field=f"{name} item"))
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} must not contain duplicates")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent,
            "model": self.model,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "n_concurrent": self.n_concurrent,
            "n_attempts": self.n_attempts,
            "task_ids": self.task_ids,
            "level": self.level,
            "output_dir": str(self.output_dir),
            "timeout": self.timeout,
            "proxy_port": self.proxy_port,
            "runtime_proxy": self.runtime_proxy,
            "gpu_ids": self.gpu_ids,
            "max_iters": self.max_iters,
            "cost_limit": self.cost_limit,
            "split": self.split,
            "without_interface_descriptions": self.without_interface_descriptions,
            "white_box": self.white_box,
            "native_tool_calling": self.native_tool_calling,
            "send_reasoning_content": self.send_reasoning_content,
            "litellm_extra_body": self.litellm_extra_body,
            "session_cache": self.session_cache,
            "force_rerun_ids": self.force_rerun_ids,
            "force_timeout": self.force_timeout,
            "openhands_reasoning_effort": self.openhands_reasoning_effort,
            "codex_reasoning_effort": self.codex_reasoning_effort,
            "base_url": self.base_url,
            "version": self.version,
        }


@dataclass
class InferResult:
    """Result of a single inference run."""

    instance_id: str
    model_patch: str
    agent: str
    model: str
    n_attempt: int
    metadata: dict[str, Any]
    success: bool = True
    error: str | None = None
    agent_exit_status: str | None = None

    def __post_init__(self) -> None:
        self.instance_id = normalize_instance_id(self.instance_id)
        for name in ("agent", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            setattr(self, name, value.strip())
        if not isinstance(self.model_patch, str):
            raise ValueError("model_patch must be a string")
        if isinstance(self.n_attempt, bool) or not isinstance(self.n_attempt, int) or self.n_attempt < 1:
            raise ValueError("n_attempt must be a positive integer")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for output.jsonl."""
        return {
            "instance_id": self.instance_id,
            "n_attempt": self.n_attempt,
            "model_patch": self.model_patch,
            "agent": self.agent,
            "model": self.model,
            "task_metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "agent_exit_status": self.agent_exit_status,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL format string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RunMetadata:
    """Metadata for a complete inference run."""

    agent: str
    model: str
    dataset: str
    n_concurrent: int
    n_attempts: int
    task_ids: list[str]
    output_path: str
    start_time: str
    dataset_revision: str
    timeout: int = 7200
    proxy_port: int | None = None
    runtime_proxy: bool | None = None
    gpu_ids: str | None = None
    max_iters: int | None = None
    cost_limit: float | None = None
    openhands_reasoning_effort: str | None = None
    codex_reasoning_effort: str | None = None
    split: str | None = None  # HuggingFace dataset split name
    level: list[int] | None = None  # Level filter (1, 2)
    without_interface_descriptions: bool = False
    white_box: bool = False
    native_tool_calling: bool | None = None
    send_reasoning_content: bool = False
    litellm_extra_body: str | None = None
    session_cache: bool = False
    force_timeout: bool = False
    base_url: str | None = None
    version: str | None = None
    end_time: str | None = None

    def __post_init__(self) -> None:
        for name in ("agent", "model", "dataset", "output_path", "start_time"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            setattr(self, name, value.strip())
        if not isinstance(self.dataset_revision, str) or not self.dataset_revision.strip():
            raise ValueError("dataset_revision must be a non-empty string")
        self.dataset_revision = self.dataset_revision.strip()
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        self.split = self.split.strip()
        for name in ("n_concurrent", "n_attempts", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.task_ids, list) or not self.task_ids or any(
            not isinstance(task_id, str) or not task_id.strip() for task_id in self.task_ids
        ):
            raise ValueError("task_ids must be a non-empty list of strings")
        self.task_ids = [
            normalize_instance_id(task_id, field="task_ids item") for task_id in self.task_ids
        ]
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must not contain duplicates")
        if self.level is not None and (
            not isinstance(self.level, list)
            or not self.level
            or any(isinstance(value, bool) or value not in {1, 2} for value in self.level)
        ):
            raise ValueError("level must be a non-empty list containing only 1 or 2")
        if self.level is not None and len(self.level) != len(set(self.level)):
            raise ValueError("level must not contain duplicates")
        if self.proxy_port is not None and (
            isinstance(self.proxy_port, bool)
            or not isinstance(self.proxy_port, int)
            or not 1 <= self.proxy_port <= 65535
        ):
            raise ValueError("proxy_port must be between 1 and 65535")
        if self.max_iters is not None and (
            isinstance(self.max_iters, bool)
            or not isinstance(self.max_iters, int)
            or self.max_iters < 1
        ):
            raise ValueError("max_iters must be a positive integer")
        if self.cost_limit is not None and (
            isinstance(self.cost_limit, bool)
            or not isinstance(self.cost_limit, (int, float))
            or self.cost_limit < 0
        ):
            raise ValueError("cost_limit must be a non-negative number")
        for name in (
            "without_interface_descriptions",
            "white_box",
            "send_reasoning_content",
            "session_cache",
            "force_timeout",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.runtime_proxy is not None and not isinstance(self.runtime_proxy, bool):
            raise ValueError("runtime_proxy must be a boolean or null")
        if self.native_tool_calling is not None and not isinstance(
            self.native_tool_calling, bool
        ):
            raise ValueError("native_tool_calling must be a boolean or null")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent,
            "model": self.model,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "n_concurrent": self.n_concurrent,
            "n_attempts": self.n_attempts,
            "task_ids": self.task_ids,
            "output_path": self.output_path,
            "start_time": self.start_time,
            "timeout": self.timeout,
            "proxy_port": self.proxy_port,
            "runtime_proxy": self.runtime_proxy,
            "gpu_ids": self.gpu_ids,
            "max_iters": self.max_iters,
            "cost_limit": self.cost_limit,
            "openhands_reasoning_effort": self.openhands_reasoning_effort,
            "codex_reasoning_effort": self.codex_reasoning_effort,
            "split": self.split,
            "level": self.level,
            "without_interface_descriptions": self.without_interface_descriptions,
            "white_box": self.white_box,
            "native_tool_calling": self.native_tool_calling,
            "send_reasoning_content": self.send_reasoning_content,
            "litellm_extra_body": self.litellm_extra_body,
            "session_cache": self.session_cache,
            "force_timeout": self.force_timeout,
            "base_url": self.base_url,
            "version": self.version,
            "end_time": self.end_time,
        }

    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
        """Load metadata from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TaskPaths:
    """Paths for a single task's files."""

    run_dir: Path
    task_id: str
    attempt: int

    def __post_init__(self) -> None:
        self.task_id = normalize_instance_id(self.task_id, field="task_id")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")

    @property
    def task_dir(self) -> Path:
        """Task directory: runs/{timestamp}/run_outputs/{task_id}"""
        return self.run_dir / "run_outputs" / self.task_id

    @property
    def attempt_dir(self) -> Path:
        """Attempt directory: runs/{timestamp}/run_outputs/{task_id}/attempt-{attempt}"""
        return self.task_dir / f"attempt-{self.attempt}"

    @property
    def infer_log_path(self) -> Path:
        """Path to infer.log file."""
        return self.attempt_dir / "infer.log"

    @property
    def run_log_path(self) -> Path:
        """Path to run.log file (detailed execution log)."""
        return self.attempt_dir / "run.log"

    @property
    def patch_path(self) -> Path:
        """Path to patch.diff file."""
        return self.attempt_dir / "patch.diff"

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
