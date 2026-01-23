"""
Data models for the inference module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class AgentName(str, Enum):
    """Supported agent names."""
    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli"
    OPENHANDS = "openhands"
    CODEX = "codex"


@dataclass
class TaskInstance:
    """Represents a single task instance from the dataset."""
    instance_id: str
    problem_statement: str
    image_name: str
    level: int
    patch: Optional[str] = None
    fail_to_pass: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInstance":
        """Create a TaskInstance from a dictionary."""
        return cls(
            instance_id=data.get("instance_id", ""),
            problem_statement=data.get("problem_statement", ""),
            image_name=data.get("image_name", ""),
            level=int(data.get("level", 1)),
            patch=data.get("patch"),
            fail_to_pass=data.get("FAIL_TO_PASS"),
            metadata=data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "image_name": self.image_name,
            "level": self.level,
            "patch": self.patch,
            "FAIL_TO_PASS": self.fail_to_pass,
            **self.metadata
        }
    
    def get_repo_settings(self) -> Dict[str, Any]:
        """
        Parse and return repo_settings from metadata.
        
        Returns:
            Dictionary containing repo settings
        """
        repo_settings_str = self.metadata.get("repo_settings", "{}")
        if not repo_settings_str:
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
    
    def get_docker_runtime_config(self) -> Dict[str, Any]:
        """
        Extract Docker runtime configuration from repo_settings.
        
        Returns:
            Dictionary with docker runtime config (need_gpu, shm_size, env_vars, env_exports, number_once)
        """
        repo_settings = self.get_repo_settings()
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


@dataclass
class InferConfig:
    """Configuration for inference run."""
    agent: str
    model: str
    # HuggingFace dataset repo name (e.g., "LiberCoders/ACE-Bench")
    dataset: str = "LiberCoders/ACE-Bench"
    n_concurrent: int = 1
    n_attempts: int = 1
    task_ids: Optional[List[str]] = None
    level: Optional[List[int]] = None
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    timeout: int = 7200  # 2 hours default timeout
    proxy_port: Optional[int] = None
    runtime_proxy: Optional[bool] = None
    gpu_ids: Optional[str] = None  # Comma-separated GPU IDs (e.g., "0,1,2,3"), None means all
    # OpenHands only: max step/iteration limit. None means "do not override" (OpenHands default applies).
    max_iters: Optional[int] = None
    split: Optional[str] = None  # HuggingFace dataset split name (e.g., "lite", "full")
    # If True, remove the "## Interface Descriptions" section from the task prompt.
    without_interface_descriptions: bool = False
    # If True, enable white-box mode: agent can see FAIL_TO_PASS test file(s).
    white_box: bool = False
    # If True, force OpenHands to use native tool calling (LLM_NATIVE_TOOL_CALLING=true).
    force_native_tool_calling: bool = False
    # Optional task IDs to force rerun even if completed.
    force_rerun_ids: Optional[List[str]] = None
    # If True, treat prior TIMEOUT attempts as completed when resuming (skip reruns).
    force_timeout: bool = False
    # Optional: CLI overrides for agent auth/endpoint/version.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent,
            "model": self.model,
            "dataset": self.dataset,
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
            "split": self.split,
            "without_interface_descriptions": self.without_interface_descriptions,
            "white_box": self.white_box,
            "force_native_tool_calling": self.force_native_tool_calling,
            "force_rerun_ids": self.force_rerun_ids,
            "force_timeout": self.force_timeout,
            "api_key": self.api_key,
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
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output.jsonl."""
        return {
            "instance_id": self.instance_id,
            "n_attempt": self.n_attempt,
            "model_patch": self.model_patch,
            "agent": self.agent,
            "model": self.model,
            "task_metadata": self.metadata,
            "success": self.success,
            "error": self.error
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
    task_ids: List[str]
    output_path: str
    start_time: str
    timeout: int = 7200
    proxy_port: Optional[int] = None
    runtime_proxy: Optional[bool] = None
    gpu_ids: Optional[str] = None
    max_iters: Optional[int] = None
    openhands_reasoning_effort: Optional[str] = None
    codex_reasoning_effort: Optional[str] = None
    split: Optional[str] = None  # HuggingFace dataset split name
    level: Optional[List[int]] = None  # Level filter (1, 2)
    without_interface_descriptions: bool = False
    white_box: bool = False
    force_native_tool_calling: bool = False
    force_timeout: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    version: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent": self.agent,
            "model": self.model,
            "dataset": self.dataset,
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
            "openhands_reasoning_effort": self.openhands_reasoning_effort,
            "codex_reasoning_effort": self.codex_reasoning_effort,
            "split": self.split,
            "level": self.level,
            "without_interface_descriptions": self.without_interface_descriptions,
            "white_box": self.white_box,
            "force_native_tool_calling": self.force_native_tool_calling,
            "force_timeout": self.force_timeout,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "version": self.version,
            "end_time": self.end_time
        }
    
    def save(self, path: Path) -> None:
        """Save metadata to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
        """Load metadata from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TaskPaths:
    """Paths for a single task's files."""
    run_dir: Path
    task_id: str
    attempt: int
    
    @property
    def task_dir(self) -> Path:
        """Task directory: runs/{timestamp}/run_outputs/{task_id}"""
        return self.run_dir / 'run_outputs' / self.task_id
    
    @property
    def attempt_dir(self) -> Path:
        """Attempt directory: runs/{timestamp}/run_outputs/{task_id}/attempt-{attempt}"""
        return self.task_dir / f'attempt-{self.attempt}'
    
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

