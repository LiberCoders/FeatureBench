"""
ACE-Bench Inference Runner

Main entry point for running agents on ACE-Bench instances.
Supports parallel execution, multiple attempts, and generates output.jsonl.

Usage:
    python -m acebench.infer.run_infer --agent claude_code --model claude-sonnet-4-20250514
    python -m acebench.infer.run_infer --agent openhands --model gpt-4o --n-concurrent 4
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from acebench.infer.agents import get_agent
from acebench.infer.config import InferConfigLoader, DatasetLoader
from acebench.infer.container import ContainerManager
from acebench.infer.models import InferConfig, InferResult, RunMetadata, TaskInstance, TaskPaths
from acebench.infer.output import OutputManager
from acebench.infer.runtime import RuntimeHandler

from acebench.infer.gpu_scheduler import GpuLease, GpuScheduler, detect_host_gpu_ids, parse_gpu_id_list


# Configure console logging - minimal output to terminal
console = Console()


def setup_console_logging():
    """Setup minimal console logging."""
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up a minimal console handler that only shows WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)


def create_task_file_logger(log_file: Path, task_id: str) -> logging.Logger:
    """
    Create a dedicated file logger for a specific task.
    
    Args:
        log_file: Path to the log file (run.log)
        task_id: Task ID for logger naming
        
    Returns:
        Logger instance that writes to the file
    """
    # Create a unique logger for this task
    logger_name = f"task_{task_id.replace('/', '_').replace('.', '_')}"
    task_logger = logging.getLogger(logger_name)
    task_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    task_logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    task_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (avoid terminal output)
    task_logger.propagate = False
    
    return task_logger


def cleanup_task_logger(task_logger: logging.Logger) -> None:
    """Clean up a task logger by closing and removing its handlers."""
    for handler in task_logger.handlers[:]:
        handler.close()
        task_logger.removeHandler(handler)


def _strip_interface_descriptions(problem_statement: str) -> str:
    """Remove the '## Interface Descriptions' section (header to EOF) from a prompt."""
    if not problem_statement:
        return problem_statement

    match = re.search(r"(?m)^##\s+Interface\s+Descriptions\b", problem_statement)
    if not match:
        return problem_statement

    # Keep everything before the header.
    cut_idx = match.start()
    kept = problem_statement[:cut_idx]
    return kept.rstrip() + "\n"


def _normalize_f2p_test_path(test_path: str) -> str:
    """Normalize a FAIL_TO_PASS test path to an absolute /testbed path for display.

    The dataset typically stores paths relative to the repository root (e.g. "tests/...").
    In the container, the repository lives at /testbed, so we display /testbed/<path>.
    """
    raw = (test_path or "").strip()
    if not raw:
        return raw

    # Already absolute to the repo root inside the container.
    if raw == "/testbed" or raw.startswith("/testbed/"):
        return raw

    # Be forgiving about common relative prefixes.
    if raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("testbed/"):
        raw = raw[len("testbed/") :]

    return f"/testbed/{raw.lstrip('/')}"


def _load_force_rerun_ids(raw_values: Optional[List[str]]) -> List[str]:
    if not raw_values:
        return []

    ids: List[str] = []
    for value in raw_values:
        if not value:
            continue
        candidate = str(value).strip()
        if not candidate:
            continue
        if candidate.endswith(".txt"):
            path = Path(candidate)
            if not path.exists():
                logger.warning(f"--force-rerun file not found: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    task_id = line.strip()
                    if task_id:
                        ids.append(task_id)
            continue
        ids.append(candidate)

    # Preserve order while de-duplicating.
    deduped = list(dict.fromkeys(ids))
    return deduped


def _get_primary_f2p_test_path(instance: TaskInstance) -> Optional[str]:
    """Pick a single representative FAIL_TO_PASS test file path for white-box messaging."""
    f2p = instance.fail_to_pass
    if not f2p:
        return None
    f2p_list = f2p if isinstance(f2p, list) else [f2p]
    # Keep behavior deterministic.
    for item in f2p_list:
        if isinstance(item, str) and item.strip():
            return _normalize_f2p_test_path(item)
    return None


def _inject_white_box_note(problem_statement: str, test_file_path: str) -> str:
    """Append a white-box bullet under the **NOTE** bullet list (best-effort)."""
    if not problem_statement:
        return problem_statement
    if not test_file_path:
        return problem_statement

    bullet = (
        f"- You are given access to the test file `{test_file_path}` for this task "
        f"(white-box testing). Use this information to help you solve the problem.\n"
    )

    # Avoid duplicating if resuming or re-running.
    if bullet.strip() in problem_statement:
        return problem_statement

    note_idx = problem_statement.find("**NOTE**")
    if note_idx == -1:
        # If NOTE block is absent, append near the top.
        return bullet + "\n" + problem_statement

    # Insert after the NOTE bullet list: contiguous lines starting with "- ".
    after_note = problem_statement[note_idx:]
    lines = after_note.splitlines(keepends=True)

    insert_at = None
    in_bullets = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("- "):
            in_bullets = True
            continue
        if in_bullets:
            insert_at = i
            break
    if in_bullets and insert_at is None:
        insert_at = len(lines)

    if insert_at is None:
        # No bullet list found after NOTE; place right after NOTE header line.
        for i, line in enumerate(lines):
            if "**NOTE**" in line:
                insert_at = i + 1
                break
        if insert_at is None:
            insert_at = 0

    lines.insert(insert_at, bullet)
    injected = "".join(lines)
    return problem_statement[:note_idx] + injected


# Initialize console logging
setup_console_logging()
logger = logging.getLogger(__name__)


class InferenceRunner:
    """Main inference runner class."""
    
    def __init__(self, config: InferConfig, resume_dir: Optional[Path] = None):
        """
        Initialize the inference runner.
        
        Args:
            config: Inference configuration
            resume_dir: Optional path to resume from a previous run
        """
        self.config = config
        self.console = Console()
        self.resume_mode = resume_dir is not None
        
        # Load configuration
        self.config_loader = InferConfigLoader()
        self.cache_dir = self.config_loader.get_cache_dir()
        
        # Set output directory
        if resume_dir:
            # Resume mode: use existing directory
            self.output_dir = resume_dir
            self.run_timestamp = resume_dir.name
        else:
            # Normal mode: create new directory with timestamp
            self.run_timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            self.output_dir = config.output_dir / self.run_timestamp
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize output manager (uses its own logger)
        self.output_manager = OutputManager(self.output_dir)
        
        # Get agent environment variables
        self.agent_env_vars = self.config_loader.get_agent_env_vars(config.agent)
        if self.cache_dir:
            self.agent_env_vars.setdefault("AGENT_DOWNLOAD_CACHE", "/download")
        if getattr(config, "runtime_proxy", None) is not None:
            self.agent_env_vars["ACE_RUNTIME_PROXY"] = "true" if config.runtime_proxy else "false"

        # Apply CLI overrides (api_key/base_url/version) with warnings on conflict.
        api_key_map = {
            "openhands": "LLM_API_KEY",
            "claude_code": "ANTHROPIC_API_KEY",
            "gemini_cli": "GEMINI_API_KEY",
            "codex": "OPENAI_API_KEY",
        }
        base_url_map = {
            "openhands": "LLM_BASE_URL",
            "claude_code": "ANTHROPIC_BASE_URL",
            "gemini_cli": "GOOGLE_GEMINI_BASE_URL",
            "codex": "OPENAI_BASE_URL",
        }
        version_map = {
            "openhands": "OPENHANDS_VERSION",
            "claude_code": "CLAUDE_CODE_VERSION",
            "gemini_cli": "GEMINI_CLI_VERSION",
            "codex": "CODEX_VERSION",
        }

        def _apply_override(flag_name: str, value: Optional[str], key_map: Dict[str, str]) -> None:
            if value is None:
                return
            raw = str(value).strip()
            if not raw:
                return
            env_key = key_map.get(config.agent)
            if not env_key:
                return
            existing = self.agent_env_vars.get(env_key)
            if existing is not None and str(existing).strip() and str(existing).strip() != raw:
                self.console.print(
                    f"[yellow]Warning: {flag_name} overrides {env_key} from config.toml[/]"
                )
            self.agent_env_vars[env_key] = raw

        _apply_override("--api-key", config.api_key, api_key_map)
        _apply_override("--base-url", config.base_url, base_url_map)
        _apply_override("--version", config.version, version_map)

        # Force native tool calling for OpenHands when requested.
        if config.agent == "openhands":
            if getattr(config, "force_native_tool_calling", False):
                self.agent_env_vars["LLM_NATIVE_TOOL_CALLING"] = "true"
            else:
                # Avoid inheriting config/env values when not forcing.
                self.agent_env_vars.pop("LLM_NATIVE_TOOL_CALLING", None)

            # Surface force-timeout behavior to the agent via env.
            if getattr(config, "force_timeout", False):
                self.agent_env_vars["ACE_FORCE_TIMEOUT"] = "true"
            else:
                self.agent_env_vars.pop("ACE_FORCE_TIMEOUT", None)

        # Resume mode: metadata is the source of truth for reproducibility.
        # Non-resume: config/env wins; CLI only fills when config has no setting.
        if self.resume_mode:
            metadata: Dict[str, Any] = {}
            metadata_path = self.output_dir / "run_metadata.json"
            try:
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
            except Exception:
                # Best-effort only; resume should still work.
                metadata = {}

            if config.agent == "openhands":
                # Force native tool calling in resume mode strictly follows metadata.
                force_native = bool(metadata.get("force_native_tool_calling"))
                if force_native:
                    self.agent_env_vars["LLM_NATIVE_TOOL_CALLING"] = "true"
                else:
                    self.agent_env_vars.pop("LLM_NATIVE_TOOL_CALLING", None)

                recorded = metadata.get("openhands_reasoning_effort")
                if recorded is not None and str(recorded).strip():
                    self.agent_env_vars["LLM_REASONING_EFFORT"] = str(recorded).strip()
                else:
                    self.agent_env_vars.pop("LLM_REASONING_EFFORT", None)

                # OpenHands max iterations in resume mode comes from resume config
                # (which itself is built from run_metadata.json).
                if config.max_iters is not None:
                    self.agent_env_vars["OPENHANDS_MAX_ITERATIONS"] = str(config.max_iters)
                else:
                    self.agent_env_vars.pop("OPENHANDS_MAX_ITERATIONS", None)

            if config.agent == "codex":
                recorded = metadata.get("codex_reasoning_effort")
                if recorded is not None and str(recorded).strip():
                    self.agent_env_vars["CODEX_REASONING_EFFORT"] = str(recorded).strip()
                else:
                    self.agent_env_vars.pop("CODEX_REASONING_EFFORT", None)
        else:
            # OpenHands max iterations in non-resume mode.
            if config.agent == "openhands" and config.max_iters is not None:
                has_env_setting = bool(self.agent_env_vars.get("OPENHANDS_MAX_ITERATIONS"))
                if not has_env_setting:
                    self.agent_env_vars["OPENHANDS_MAX_ITERATIONS"] = str(config.max_iters)
        
        # Load completed tasks for resume functionality
        self._completed_tasks: Set[Tuple[str, int]] = set()
        if self.resume_mode:
            self._completed_tasks = self.output_manager.load_completed_tasks()
        self._force_rerun_ids: Set[str] = set(self.config.force_rerun_ids or [])

        # Optional GPU scheduler (initialized in run() after dataset is loaded)
        self._gpu_scheduler: Optional[GpuScheduler] = None
    
    def _load_dataset(self) -> List[TaskInstance]:
        """Load dataset from HuggingFace."""
        self.console.print("[bold blue]Loading dataset from HuggingFace...[/]")
        
        dataset_loader = DatasetLoader(self.config_loader)
        
        # Parse levels (None means all levels)
        levels = self.config.level if self.config.level else None

        task_ids = self.config.task_ids
        if task_ids is not None and self.config.force_rerun_ids:
            task_ids = list(dict.fromkeys(list(task_ids) + list(self.config.force_rerun_ids)))
        
        # Load data with split parameter
        raw_data = dataset_loader.load_dataset(
            dataset=self.config.dataset,
            split=self.config.split,
            levels=levels,
            task_ids=task_ids
        )
        
        # Convert to TaskInstance objects
        instances = [TaskInstance.from_dict(item) for item in raw_data]
        
        self.console.print(f"[green]Loaded {len(instances)} task instances[/]")
        return instances

    def _get_image_name(self, instance: TaskInstance) -> str:
        """Get Docker image name for an instance."""
        image_name = instance.image_name
        
        # Add docker.io prefix if needed
        if '/' not in image_name or not image_name.startswith(('docker.io/', 'gcr.io/', 'ghcr.io/')):
            image_name = f'docker.io/{image_name}'
        
        return image_name.lower()
    
    def _process_single_task(
        self,
        instance: TaskInstance,
        attempt: int
    ) -> InferResult:
        """
        Process a single task instance.
        
        Args:
            instance: Task instance
            attempt: Attempt number (1-based)
            
        Returns:
            InferResult with the outcome
        """
        task_id = instance.instance_id
        
        # Create task paths
        task_paths = TaskPaths(self.output_dir, task_id, attempt)
        task_paths.ensure_dirs()
        
        # Get log file paths
        log_file = task_paths.infer_log_path
        run_log_file = task_paths.run_log_path
        
        # Create dedicated file logger for this task
        task_logger = create_task_file_logger(run_log_file, f"{task_id}-{attempt}")
        task_logger.info(f"Processing {task_id} (attempt {attempt}/{self.config.n_attempts})")
        
        # Initialize infer log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"ACE-Bench Inference Log\n")
            f.write(f"Instance: {task_id}\n")
            f.write(f"Attempt: {attempt}/{self.config.n_attempts}\n")
            f.write(f"Agent: {self.config.agent}\n")
            f.write(f"Model: {self.config.model}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
        
        container = None
        gpu_lease: Optional[GpuLease] = None
        result = InferResult(
            instance_id=task_id,
            model_patch="",
            agent=self.config.agent,
            model=self.config.model,
            n_attempt=attempt,
            metadata=instance.metadata,
            success=False
        )
        
        try:
            # Get Docker image
            image_name = self._get_image_name(instance)
            
            # Get docker runtime config from repo_settings
            docker_runtime_config = instance.get_docker_runtime_config()
            task_logger.info(f"Docker runtime config: need_gpu={docker_runtime_config.get('need_gpu')}, "
                           f"shm_size={docker_runtime_config.get('shm_size')}, "
                           f"number_once={docker_runtime_config.get('number_once')}, "
                           f"env_vars={list(docker_runtime_config.get('env_vars', {}).keys())}, "
                           f"env_exports={len(docker_runtime_config.get('env_exports', []))} items")

            # Allocate GPUs for this task if needed.
            task_gpu_ids = self.config.gpu_ids
            if docker_runtime_config.get("need_gpu") and self._gpu_scheduler is not None:
                requested = docker_runtime_config.get("number_once", 1)
                if not isinstance(requested, int) or requested <= 0:
                    requested = 1
                gpu_lease = self._gpu_scheduler.allocate(requested)
                task_gpu_ids = gpu_lease.gpu_ids_str
                task_logger.info(
                    f"GPU scheduling: allocated {requested} GPU(s): {task_gpu_ids} "
                    f"(pool={','.join(self._gpu_scheduler.gpu_pool)})"
                )
            
            # Create container manager with task-specific logger
            cm = ContainerManager(task_logger, self.agent_env_vars)
            
            # Pull image if needed
            task_logger.info(f"Ensuring image {image_name} is available...")
            cm.pull_image(image_name)
            
            # Create container with timestamp to avoid name conflicts
            container_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            container_name = f"ace-infer-{task_id.replace('/', '-').replace('.', '-')}-attempt-{attempt}-{container_timestamp}"
            task_logger.info(f"Creating container {container_name}...")

            # Mount cache volume
            volumes = None
            if self.cache_dir:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                # Bind host cache directory to /download inside container
                volumes = {str(self.cache_dir): {"bind": "/download", "mode": "rw"}}
            
            container = cm.create_container(
                image_name=image_name,
                container_name=container_name,
                working_dir="/testbed",
                proxy_port=self.config.proxy_port,
                gpu_ids=task_gpu_ids,
                docker_runtime_config=docker_runtime_config,
                volumes=volumes
            )
            
            # Initialize runtime with task-specific logger
            runtime_handler = RuntimeHandler(cm, task_logger)
            
            if not runtime_handler.initialize_runtime(
                container,
                instance,
                log_file,
                white_box=getattr(self.config, "white_box", False),
            ):
                result.error = "Runtime initialization failed"
                return result
            
            # Create and install agent with task-specific logger
            agent = get_agent(
                self.config.agent,
                container_manager=cm,
                env_vars=self.agent_env_vars,
                logger=task_logger,
                model=self.config.model,
                version=self.config.version,
            )
            
            if not agent.install(container, log_file):
                result.error = "Agent installation failed"
                return result
            
            # Pre-run setup hook (for custom container processing)
            task_logger.info("Running pre-run setup...")
            if not agent.pre_run_setup(container, instance, log_file):
                task_logger.warning("Pre-run setup returned False (non-fatal)")
            
            # Run agent
            instruction = instance.problem_statement
            if getattr(self.config, "white_box", False):
                f2p_path = _get_primary_f2p_test_path(instance)
                if f2p_path:
                    instruction = _inject_white_box_note(instruction, f2p_path)
            if getattr(self.config, "without_interface_descriptions", False):
                instruction = _strip_interface_descriptions(instruction)
            agent_success = agent.run(
                container,
                instruction,
                log_file,
                timeout=self.config.timeout
            )
            
            # if not agent_success, raise an exception and will not save patch
            if not agent_success:
                raise RuntimeError(f"Agent did not complete successfully")
            
            # Complete runtime and get patch
            patch = runtime_handler.complete_runtime(container, instance, log_file)
            
            if patch is None:
                result.error = "Failed to generate patch"
            else:
                result.model_patch = patch
                result.success = True

                # Save patch file
                self.output_manager.save_patch(task_paths, patch)

                task_logger.info(f"Successfully processed {task_id} (patch: {len(patch)} chars)")
            
        except Exception as e:
            task_logger.error(f"Error processing {task_id}: {e}")
            result.error = str(e)
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\nFATAL ERROR: {e}\n")
        
        finally:
            # Clean up container
            if container is not None:
                try:
                    cm.stop_container(container, force=True)
                except Exception as e:
                    task_logger.warning(f"Error cleaning up container: {e}")

            # Release GPU lease even if container creation failed.
            if gpu_lease is not None and self._gpu_scheduler is not None:
                try:
                    self._gpu_scheduler.release(gpu_lease)
                except Exception as e:
                    task_logger.warning(f"Error releasing GPU lease: {e}")
            
            # Write end time to log
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\nEnd Time: {datetime.now().isoformat()}\n")
            
            # Clean up task logger
            cleanup_task_logger(task_logger)
        
        return result

    def _warmup_cache(self, instance: TaskInstance) -> None:
        """Run a single install to populate shared cache before parallel runs."""
        image_name = self._get_image_name(instance)
        warmup_log = self.output_dir / "warmup.log"

        # Minimal file logger for warmup
        task_logger = create_task_file_logger(warmup_log, "warmup")
        task_logger.info(f"Warmup cache using image {image_name}")

        container = None
        gpu_lease: Optional[GpuLease] = None
        try:
            cm = ContainerManager(task_logger, self.agent_env_vars)
            cm.pull_image(image_name)

            volumes = None
            if self.cache_dir:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                volumes = {str(self.cache_dir): {"bind": "/download", "mode": "rw"}}

            docker_runtime_config = instance.get_docker_runtime_config()
            task_gpu_ids = self.config.gpu_ids
            if docker_runtime_config.get("need_gpu") and self._gpu_scheduler is not None:
                requested = docker_runtime_config.get("number_once", 1)
                if not isinstance(requested, int) or requested <= 0:
                    requested = 1
                gpu_lease = self._gpu_scheduler.allocate(requested)
                task_gpu_ids = gpu_lease.gpu_ids_str
                task_logger.info(
                    f"GPU scheduling (warmup): allocated {requested} GPU(s): {task_gpu_ids} "
                    f"(pool={','.join(self._gpu_scheduler.gpu_pool)})"
                )

            container = cm.create_container(
                image_name=image_name,
                working_dir="/testbed",
                proxy_port=self.config.proxy_port,
                gpu_ids=task_gpu_ids,
                docker_runtime_config=docker_runtime_config,
                volumes=volumes
            )

            agent = get_agent(
                self.config.agent,
                container_manager=cm,
                env_vars=self.agent_env_vars,
                logger=task_logger,
                model=self.config.model,
                version=self.config.version,
            )

            agent.install(container, warmup_log)

        finally:
            if container is not None:
                try:
                    cm.stop_container(container, force=True)
                except Exception as e:
                    task_logger.warning(f"Error cleaning up warmup container: {e}")

            if gpu_lease is not None and self._gpu_scheduler is not None:
                try:
                    self._gpu_scheduler.release(gpu_lease)
                except Exception as e:
                    task_logger.warning(f"Error releasing warmup GPU lease: {e}")

    
    def _save_run_metadata(self, task_ids: List[str]) -> None:
        """Save run metadata."""
        # Persist the *effective* max iters used for OpenHands.
        effective_max_iters: Optional[int] = None
        if self.config.agent == "openhands":
            raw = self.agent_env_vars.get("OPENHANDS_MAX_ITERATIONS")
            if raw is not None and str(raw).strip():
                try:
                    effective_max_iters = int(str(raw).strip())
                except Exception:
                    effective_max_iters = None

        # Persist the *effective* reasoning effort used by the agent (if any).
        openhands_reasoning_effort: Optional[str] = None
        codex_reasoning_effort: Optional[str] = None
        if self.config.agent == "openhands":
            raw = self.agent_env_vars.get("LLM_REASONING_EFFORT")
            if raw is not None and str(raw).strip():
                openhands_reasoning_effort = str(raw).strip()
        elif self.config.agent == "codex":
            raw = self.agent_env_vars.get("CODEX_REASONING_EFFORT")
            if raw is not None and str(raw).strip():
                codex_reasoning_effort = str(raw).strip()

        metadata = RunMetadata(
            agent=self.config.agent,
            model=self.config.model,
            dataset=self.config.dataset,
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            task_ids=task_ids,
            output_path=str(self.output_dir),
            start_time=datetime.now().isoformat(),
            timeout=self.config.timeout,
            proxy_port=self.config.proxy_port,
            runtime_proxy=self.config.runtime_proxy,
            gpu_ids=self.config.gpu_ids,
            max_iters=effective_max_iters,
            openhands_reasoning_effort=openhands_reasoning_effort,
            codex_reasoning_effort=codex_reasoning_effort,
            split=self.config.split,
            level=self.config.level,
            without_interface_descriptions=self.config.without_interface_descriptions,
            white_box=getattr(self.config, "white_box", False),
            force_native_tool_calling=getattr(self.config, "force_native_tool_calling", False),
            force_timeout=getattr(self.config, "force_timeout", False),
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            version=self.config.version,
        )
        self.output_manager.save_metadata(metadata)
    
    def run(self) -> None:
        """Run the inference pipeline."""
        # Print startup banner to console
        self.console.print()
        self.console.print("[bold cyan]" + "=" * 60 + "[/]")
        if self.resume_mode:
            self.console.print("[bold cyan]Resuming ACE-Bench Inference[/]")
        else:
            self.console.print("[bold cyan]Starting ACE-Bench Inference[/]")
        self.console.print(f"[white]Agent:[/] [green]{self.config.agent}[/]")
        self.console.print(f"[white]Model:[/] [green]{self.config.model}[/]")
        if self.config.api_key is not None and str(self.config.api_key).strip():
            self.console.print(f"[white]API key:[/] [green]{self.config.api_key}[/]")
        if self.config.base_url is not None and str(self.config.base_url).strip():
            self.console.print(f"[white]Base URL:[/] [green]{self.config.base_url}[/]")
        if self.config.version is not None and str(self.config.version).strip():
            self.console.print(f"[white]Agent version:[/] [green]{self.config.version}[/]")
        self.console.print(f"[white]Dataset:[/] [green]{self.config.dataset}[/]")
        self.console.print(f"[white]Split:[/] [green]{self.config.split}[/]")
        if self.config.level:
            self.console.print(f"[white]Levels:[/] [green]{self.config.level}[/]")
        else:
            self.console.print(f"[white]Levels:[/] [green]all (lv1, lv2)[/]")
        self.console.print(f"[white]Concurrent:[/] [green]{self.config.n_concurrent}[/]")
        self.console.print(f"[white]Attempts:[/] [green]{self.config.n_attempts}[/]")
        if getattr(self.config, "force_timeout", False):
            self.console.print("[white]Force timeout skip:[/] [yellow]enabled[/]")
        if getattr(self.config, "without_interface_descriptions", False):
            self.console.print("[white]Prompt:[/] [yellow]without interface descriptions[/]")
        if getattr(self.config, "white_box", False):
            self.console.print("[white]Prompt:[/] [yellow]white-box (tests visible)[/]")
        if self.config.agent == "openhands":
            if getattr(self.config, "force_native_tool_calling", False):
                self.console.print("[white]Tool calling:[/] [yellow]forced native[/]")
            effective = self.agent_env_vars.get("OPENHANDS_MAX_ITERATIONS")
            if effective is not None and str(effective).strip():
                self.console.print(f"[white]Max iters:[/] [green]{effective}[/]")
            # Only show reasoning_effort when the effective model looks like an OpenAI gpt/o-series model.
            model_lower = str(self.config.model).strip().lower() if self.config.model is not None else ""
            model_tail = model_lower.split("/", 1)[-1]
            # Treat all gpt-series models as eligible for displaying reasoning_effort.
            is_gpt_series = ("gpt" in model_tail) or model_tail.startswith("gpt")
            if is_gpt_series:
                reasoning_effort = self.agent_env_vars.get("LLM_REASONING_EFFORT")
                if reasoning_effort is not None and str(reasoning_effort).strip():
                    self.console.print(
                        f"[white]Reasoning effort:[/] [green]{str(reasoning_effort).strip()}[/]"
                    )
        if self.config.agent == "codex":
            reasoning_effort = self.agent_env_vars.get("CODEX_REASONING_EFFORT")
            if reasoning_effort is not None and str(reasoning_effort).strip():
                self.console.print(f"[white]Reasoning effort:[/] [green]{str(reasoning_effort).strip()}[/]")
        if self.config.gpu_ids:
            self.console.print(f"[white]GPUs:[/] [green]{self.config.gpu_ids}[/]")
        else:
            self.console.print(f"[white]GPUs:[/] [green]all available[/]")
        self.console.print(f"[white]Output:[/] [green]{self.output_dir}[/]")
        if self.resume_mode:
            self.console.print(f"[white]Completed:[/] [yellow]{len(self._completed_tasks)} tasks already done[/]")
        self.console.print("[bold cyan]" + "=" * 60 + "[/]")
        self.console.print()
        
        # Load dataset
        instances = self._load_dataset()
        
        if not instances:
            self.console.print("[bold red]No task instances to process[/]")
            return

        # Initialize GPU scheduler (best-effort) if there are any GPU tasks.
        try:
            need_gpu_any = any(inst.get_docker_runtime_config().get("need_gpu") for inst in instances)
        except Exception:
            need_gpu_any = False

        if need_gpu_any:
            gpu_pool: Optional[List[str]] = None
            if self.config.gpu_ids:
                gpu_pool = parse_gpu_id_list(self.config.gpu_ids)
            else:
                gpu_pool = detect_host_gpu_ids()

            if gpu_pool:
                # Validate max number_once against pool size.
                max_required = 1
                for inst in instances:
                    cfg = inst.get_docker_runtime_config()
                    if not cfg.get("need_gpu"):
                        continue
                    n = cfg.get("number_once", 1)
                    if isinstance(n, int) and n > max_required:
                        max_required = n

                if max_required > len(gpu_pool):
                    raise RuntimeError(
                        f"GPU scheduling pool too small: max number_once={max_required} but pool={','.join(gpu_pool)}"
                    )

                self._gpu_scheduler = GpuScheduler(gpu_pool)
                self.console.print(
                    f"[bold cyan]GPU scheduling enabled[/]: pool=[green]{','.join(gpu_pool)}[/] "
                )
            else:
                self.console.print(
                    "[bold yellow]GPU scheduling disabled[/]: failed to detect GPU pool (nvidia-smi unavailable). "
                    "Falling back to previous behavior (GPU tasks may all default to GPU0)."
                )

        # Warm up cache once before parallel runs (if enabled)
        if self.cache_dir and self.config.n_concurrent > 1:
            self.console.print("[bold blue]Warming cache before parallel execution...[/]")
            try:
                self._warmup_cache(instances[0])
            except Exception as e:
                self.console.print(f"[bold yellow]Cache warmup failed (continuing): {e}[/]")
        
        # Save run metadata (only in non-resume mode, or update in resume mode)
        task_ids = [inst.instance_id for inst in instances]
        if not self.resume_mode:
            self._save_run_metadata(task_ids)
        
        # Start output manager
        self.output_manager.start()
        
        # Track success/failure counts
        success_count = 0
        failure_count = 0
        skipped_count = 0
        
        try:
            # Build task list: (instance, attempt) pairs, filtering completed tasks
            all_tasks = []
            tasks = []
            for instance in instances:
                for attempt in range(1, self.config.n_attempts + 1):
                    all_tasks.append((instance, attempt))
                    task_key = (instance.instance_id, attempt)
                    if task_key not in self._completed_tasks or instance.instance_id in self._force_rerun_ids:
                        tasks.append((instance, attempt))
                    else:
                        skipped_count += 1
            
            total_all_tasks = len(all_tasks)
            total_tasks = len(tasks)
            
            if self.resume_mode and skipped_count > 0:
                self.console.print(f"[bold blue]Total tasks: {total_all_tasks} (skipping {skipped_count} completed)[/]")
            self.console.print(f"[bold blue]Tasks to process: {total_tasks}[/]")
            self.console.print()
            
            if total_tasks == 0:
                self.console.print("[bold green]All tasks already completed![/]")
                return
            
            # Process tasks with rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[green]{task.fields[success]}✓[/] [red]{task.fields[failure]}✗[/]"),
                TimeElapsedColumn(),
                console=self.console,
                refresh_per_second=10,
            ) as progress:
                
                task_progress = progress.add_task(
                    "[cyan]Processing...",
                    total=total_tasks,
                    success=0,
                    failure=0
                )
                
                if self.config.n_concurrent == 1:
                    # Sequential processing
                    for instance, attempt in tasks:
                        result = self._process_single_task(instance, attempt)
                        self.output_manager.write_result(result)
                        
                        if result.success:
                            success_count += 1
                        else:
                            failure_count += 1
                        
                        progress.update(
                            task_progress, 
                            advance=1,
                            success=success_count,
                            failure=failure_count
                        )
                else:
                    # Parallel processing
                    with ThreadPoolExecutor(max_workers=self.config.n_concurrent) as executor:
                        # Submit all tasks
                        future_to_task = {
                            executor.submit(self._process_single_task, inst, att): (inst.instance_id, att)
                            for inst, att in tasks
                        }
                        
                        # Process results as they complete
                        for future in as_completed(future_to_task):
                            task_id, attempt = future_to_task[future]
                            try:
                                result = future.result()
                                self.output_manager.write_result(result)
                                
                                if result.success:
                                    success_count += 1
                                else:
                                    failure_count += 1
                                    
                            except Exception as e:
                                failure_count += 1
                                # Write error result
                                error_result = InferResult(
                                    instance_id=task_id,
                                    model_patch="",
                                    agent=self.config.agent,
                                    model=self.config.model,
                                    n_attempt=attempt,
                                    metadata={},
                                    success=False,
                                    error=str(e)
                                )
                                self.output_manager.write_result(error_result)
                            
                            progress.update(
                                task_progress,
                                advance=1,
                                success=success_count,
                                failure=failure_count
                            )
        
        finally:
            # Stop output manager
            self.output_manager.stop()
            
            # Update end time
            self.output_manager.update_metadata_end_time()
        
        # Print completion banner
        self.console.print()
        self.console.print("[bold cyan]" + "=" * 60 + "[/]")
        self.console.print("[bold cyan]Inference Complete[/]")
        self.console.print(f"[green]Success: {success_count}[/] | [red]Failure: {failure_count}[/]")
        self.console.print(f"[white]Results saved to:[/] [green]{self.output_dir / 'output.jsonl'}[/]")
        self.console.print(f"[white]Run summary:[/] [green]{self.output_manager.run_summary_path}[/]")
        self.console.print("[bold cyan]" + "=" * 60 + "[/]")
        self.console.print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run agents on ACE-Bench instances",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Track whether certain args were explicitly provided on CLI.
    # This matters because some args have defaults; in --resume mode we only want to warn
    # when the user explicitly attempted to override metadata.
    argv = sys.argv[1:]
    split_provided = "--split" in argv
    dataset_provided = "--dataset" in argv
    
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=None,
        choices=["claude_code", "gemini_cli", "openhands", "codex"],
        help="Agent to use for inference (required unless --resume is used)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (e.g., claude-sonnet-4-20250514, gpt-4o) (required unless --resume is used)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override agent API key (takes precedence over config)"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override agent base URL (takes precedence over config)"
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Override agent version (takes precedence over config)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset repo name (e.g., 'LiberCoders/ACE-Bench'). "
            "Default: 'LiberCoders/ACE-Bench' (only applies when not using --resume)"
        ),
    )
    
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=None,
        help=(
            "Number of concurrent tasks (default: 1). "
            "In --resume mode, this overrides the value stored in run_metadata.json when explicitly provided."
        )
    )
    
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=1,
        help="Number of attempts per task (default: 1)"
    )
    
    parser.add_argument(
        "--task-id", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to process (default: all)"
    )

    parser.add_argument(
        "--force-rerun",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Force rerun task IDs (space-separated), or provide a .txt file with one task_id per line. "
            "These tasks are processed even if already completed."
        ),
    )
    
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2],
        help="Task levels to filter (1=lv1, 2=lv2). Level is determined by instance_id suffix. Default: all"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset split name (e.g., 'lite', 'full'). "
            "Default: 'full' (only applies when not using --resume)"
        )
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="runs",
        help="Output directory (default: runs)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per task in seconds (default: 3600 = 1 hour)"
    )

    parser.add_argument(
        "--force-timeout",
        action="store_true",
        help=(
            "When resuming OpenHands runs, treat attempts with existing infer.log TIMEOUT markers as completed "
            "so they are not rerun."
        ),
    )

    parser.add_argument(
        "--without",
        action="store_true",
        help=(
            "Ablation: remove the '## Interface Descriptions' section from the task prompt "
            "before sending it to the agent (from that header to EOF). "
            "In --resume mode, this flag is ignored and the value from run_metadata.json is used."
        ),
    )

    parser.add_argument(
        "--white",
        action="store_true",
        help=(
            "Enable white-box mode: agent can see the FAIL_TO_PASS test file for the task. "
            "This keeps the test file in the container and appends a NOTE bullet with its path to the prompt. "
            "In --resume mode, this flag is ignored and the value from run_metadata.json is used."
        ),
    )

    parser.add_argument(
        "--native-tool-calling",
        action="store_true",
        help=(
            "OpenHands only: force native tool calling (sets LLM_NATIVE_TOOL_CALLING=true inside the container). "
            "In --resume mode, this flag is ignored and the value from run_metadata.json is used."
        ),
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help=(
            "OpenHands only: maximum iterations/steps (sets OPENHANDS_MAX_ITERATIONS). "
            "Precedence: config/env wins; CLI only applies when no config value is set. "
            "Default: do not override (OpenHands upstream default applies, typically 500). "
            "In --resume mode, this argument is ignored (uses run_metadata.json)."
        )
    )
    
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=None,
        help="Proxy port to use for inference (default: None)"
    )

    parser.add_argument(
        "--runtime-proxy",
        type=str,
        choices=["on", "off"],
        default=None,
        help=(
            "Enable/disable HTTP_PROXY/HTTPS_PROXY at agent runtime (after installation). "
            "Default: on when --proxy-port is provided, otherwise off. "
            "This is ignored when full-body capture is enabled."
        ),
    )
    
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). Default: all available GPUs"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a previous run directory (e.g., runs/2025-12-02__16-06-04). "
            "Most args are loaded from run_metadata.json; --n-concurrent, --timeout, --proxy-port, --gpu-ids, --force-timeout can be overridden"
        ),
    )
    
    args = parser.parse_args()

    # Stash explicit-provided flags for resume-mode warning logic.
    args._split_provided = split_provided
    args._dataset_provided = dataset_provided

    return args


def load_resume_config(resume_dir: Path, args: argparse.Namespace) -> Tuple[InferConfig, Path]:
    """
    Load configuration from a previous run for resume.
    
    Args:
        resume_dir: Path to the previous run directory
        args: Parsed command line arguments
        
    Returns:
        Tuple of (InferConfig, output_dir)
    """
    console = Console()
    
    # Load run_metadata.json
    metadata_path = resume_dir / "run_metadata.json"
    if not metadata_path.exists():
        console.print(f"[bold red]Error: run_metadata.json not found in {resume_dir}[/]")
        sys.exit(1)
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Warn about ignored arguments (only if user explicitly specified them)
    warnings = []
    if args.agent is not None:
        warnings.append(f"--agent (using '{metadata['agent']}' from metadata)")
    if args.model is not None:
        warnings.append(f"--model (using '{metadata['model']}' from metadata)")
    if getattr(args, "_dataset_provided", False):
        warnings.append(f"--dataset (using '{metadata.get('dataset')}' from metadata)")
    if args.n_attempts != 1:  # Default is 1
        warnings.append(f"--n-attempts (using '{metadata['n_attempts']}' from metadata)")
    if args.task_id is not None:
        warnings.append(f"--task-id (using task_ids from metadata)")
    if args.level is not None:
        warnings.append(f"--level (using level from metadata)")
    if getattr(args, "_split_provided", False):
        warnings.append(f"--split (using '{metadata.get('split')}' from metadata)")
    if args.output_dir != "runs":  # Default is "runs"
        warnings.append(f"--output-dir (using '{resume_dir}')")
    if args.max_iters is not None:
        warnings.append(f"--max-iters (using '{metadata.get('max_iters')}' from metadata)")
    if getattr(args, "without", False):
        warnings.append(
            "--without (using 'without_interface_descriptions' from metadata)"
        )
    if getattr(args, "white", False):
        warnings.append(
            "--white (using 'white_box' from metadata)"
        )
    if getattr(args, "native_tool_calling", False):
        warnings.append(
            "--native-tool-calling (using 'force_native_tool_calling' from metadata)"
        )
    
    if warnings:
        console.print("[bold yellow]Warning: The following arguments are ignored in resume mode:[/]")
        for w in warnings:
            console.print(f"  [yellow]• {w}[/]")
        console.print()
    
    # Determine n_concurrent: use command line if explicitly provided, otherwise use metadata
    metadata_n_concurrent = metadata.get('n_concurrent', 1)
    n_concurrent = args.n_concurrent if args.n_concurrent is not None else metadata_n_concurrent
    if args.n_concurrent is not None and args.n_concurrent != metadata_n_concurrent:
        console.print(
            f"[cyan]Using --n-concurrent={args.n_concurrent} (overriding metadata value {metadata_n_concurrent})[/]"
        )
    
    # Determine timeout: use command line if specified, otherwise use metadata
    default_timeout = 7200
    metadata_timeout = metadata.get('timeout', default_timeout)
    timeout = args.timeout if args.timeout != default_timeout else metadata_timeout
    if args.timeout != default_timeout and args.timeout != metadata_timeout:
        console.print(f"[cyan]Using --timeout={args.timeout} (overriding metadata value {metadata_timeout})[/]")
    
    # Determine proxy_port: use command line if specified, otherwise use metadata
    metadata_proxy_port = metadata.get('proxy_port')
    proxy_port = args.proxy_port if args.proxy_port is not None else metadata_proxy_port
    if args.proxy_port is not None and args.proxy_port != metadata_proxy_port:
        console.print(f"[cyan]Using --proxy-port={args.proxy_port} (overriding metadata value {metadata_proxy_port})[/]")

    # Determine runtime_proxy: CLI overrides; otherwise use metadata; fallback to proxy_port.
    runtime_proxy_arg = None
    if args.runtime_proxy is not None:
        runtime_proxy_arg = str(args.runtime_proxy).strip().lower() == "on"
    metadata_runtime_proxy = metadata.get("runtime_proxy")
    runtime_proxy = (
        runtime_proxy_arg
        if runtime_proxy_arg is not None
        else metadata_runtime_proxy
    )
    if runtime_proxy is None:
        runtime_proxy = bool(proxy_port)
    if runtime_proxy_arg is not None and runtime_proxy_arg != metadata_runtime_proxy:
        console.print(f"[cyan]Using --runtime-proxy={args.runtime_proxy} (overriding metadata value {metadata_runtime_proxy})[/]")
    
    # Determine gpu_ids: use command line if specified, otherwise use metadata
    metadata_gpu_ids = metadata.get('gpu_ids')
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else metadata_gpu_ids
    if args.gpu_ids is not None and args.gpu_ids != metadata_gpu_ids:
        console.print(f"[cyan]Using --gpu-ids={args.gpu_ids} (overriding metadata value {metadata_gpu_ids})[/]")

    metadata_force_timeout = bool(metadata.get("force_timeout", False))
    force_timeout = metadata_force_timeout or bool(getattr(args, "force_timeout", False))
    if getattr(args, "force_timeout", False) and not metadata_force_timeout:
        console.print("[cyan]Using --force-timeout (overriding metadata value False)[/]")

    # Determine max_iters: always use metadata in resume mode.
    max_iters = metadata.get('max_iters')

    # Determine without_interface_descriptions: always use metadata in resume mode.
    without_interface_descriptions = bool(metadata.get("without_interface_descriptions"))

    # Determine white_box: always use metadata in resume mode.
    white_box = bool(metadata.get("white_box"))

    # Determine force_native_tool_calling: always use metadata in resume mode.
    force_native_tool_calling = bool(metadata.get("force_native_tool_calling"))

    # Determine api_key/base_url/version: CLI overrides; otherwise use metadata.
    metadata_api_key = metadata.get("api_key")
    api_key = args.api_key if args.api_key is not None else metadata_api_key
    metadata_base_url = metadata.get("base_url")
    base_url = args.base_url if args.base_url is not None else metadata_base_url
    metadata_version = metadata.get("version")
    version = args.version if args.version is not None else metadata_version
    
    # Create config from metadata (n_attempts, split, level always from metadata)
    config = InferConfig(
        agent=metadata['agent'],
        model=metadata['model'],
        dataset=str(metadata.get('dataset') or "LiberCoders/ACE-Bench"),
        n_concurrent=n_concurrent,
        n_attempts=metadata.get('n_attempts', 1),
        task_ids=metadata.get('task_ids'),
        level=metadata.get('level'),  # Use level from metadata
        output_dir=resume_dir.parent,  # Parent of timestamp folder
        timeout=timeout,
        proxy_port=proxy_port,
        runtime_proxy=runtime_proxy,
        gpu_ids=gpu_ids,
        max_iters=max_iters,
        split=metadata.get('split'),  # Use split from metadata
        without_interface_descriptions=without_interface_descriptions,
        white_box=white_box,
        force_native_tool_calling=force_native_tool_calling,
        force_timeout=force_timeout,
        force_rerun_ids=_load_force_rerun_ids(getattr(args, "force_rerun", None)),
        api_key=api_key,
        base_url=base_url,
        version=version,
    )
    
    return config, resume_dir


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle resume mode
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            console.print(f"[bold red]Error: Resume directory not found: {resume_dir}[/]")
            sys.exit(1)
        
        config, output_dir = load_resume_config(resume_dir, args)
        
        # Run inference in resume mode
        runner = InferenceRunner(config, resume_dir=output_dir)
        runner.run()
    else:
        # Normal mode - validate required arguments
        if args.agent is None:
            console.print("[bold red]Error: --agent is required (unless using --resume)[/]")
            sys.exit(1)
        if args.model is None:
            console.print("[bold red]Error: --model is required (unless using --resume)[/]")
            sys.exit(1)
        
        # Create new config
        config = InferConfig(
            agent=args.agent,
            model=args.model,
            dataset=(
                args.dataset.strip()
                if args.dataset is not None and str(args.dataset).strip()
                else "LiberCoders/ACE-Bench"
            ),
            n_concurrent=args.n_concurrent if args.n_concurrent is not None else 1,
            n_attempts=args.n_attempts,
            task_ids=args.task_id,
            level=args.level,
            output_dir=Path(args.output_dir),
            timeout=args.timeout,
            proxy_port=args.proxy_port,
            runtime_proxy=(
                True
                if str(args.runtime_proxy or "").strip().lower() == "on"
                else False
                if str(args.runtime_proxy or "").strip().lower() == "off"
                else bool(args.proxy_port)
            ),
            gpu_ids=args.gpu_ids,
            max_iters=args.max_iters,
            split=args.split if args.split is not None else "full",
            without_interface_descriptions=bool(getattr(args, "without", False)),
            white_box=bool(getattr(args, "white", False)),
            force_native_tool_calling=bool(getattr(args, "native_tool_calling", False)),
            force_timeout=bool(getattr(args, "force_timeout", False)),
            force_rerun_ids=_load_force_rerun_ids(getattr(args, "force_rerun", None)),
            api_key=args.api_key,
            base_url=args.base_url,
            version=args.version,
        )
        
        # Run inference
        runner = InferenceRunner(config)
        runner.run()


if __name__ == "__main__":
    main()

