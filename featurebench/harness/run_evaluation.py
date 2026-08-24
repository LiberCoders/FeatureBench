"""
FeatureBench Evaluation Runner

This script runs evaluation for FeatureBench predictions by:
1. Loading predictions from JSONL file
2. Loading dataset from HuggingFace (using config.toml settings)
3. Creating Docker containers for each instance
4. Applying patches and running tests
5. Collecting and reporting results

Usage:
    python -m featurebench.harness.run_evaluation --predictions-path runs/xxx/output.jsonl
    python -m featurebench.harness.run_evaluation --predictions-path runs/xxx/output.jsonl --split lite
    python -m featurebench.harness.run_evaluation --predictions-path runs/xxx/output.jsonl --split full --n-concurrent 8
    python -m featurebench.harness.run_evaluation --predictions-path runs/xxx/output.jsonl --dataset LiberCoders/FeatureBench
    python -m featurebench.harness.run_evaluation --predictions-path gold --split full
"""

import argparse
import atexit
import json
import logging
import os
import shutil
import signal
import threading
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import docker
import pandas as pd
from datasets import load_dataset
from docker.models.containers import Container
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from featurebench.harness.constants import (
    EvalType,
    FAIL_ONLY_REPOS,
    KEY_INSTANCE_ID,
    KEY_N_ATTEMPT,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_PATCH,
    LOG_REPORT,
    UTF8,
)
from featurebench.harness.container import DOCKER_HOST_GATEWAY, EvalContainerManager
from featurebench.harness.report import (
    build_test_status,
    generate_error_report,
    generate_instance_report,
    generate_summary_report,
    parse_test_outputs,
    save_report,
)
from featurebench.harness.review_codes import (
    save_review_codes_level1,
    save_review_codes_level2,
)
from featurebench.harness.runtime import run_instance_level1, run_instance_level2
from featurebench.harness.utils import (
    filter_predictions_by_ids,
    get_docker_image_name,
    get_docker_runtime_config,
    get_instance_from_dataset,
    get_predictions_from_file,
    preprocess_hf_patch,
    parse_repo_settings,
)

from featurebench.infer.config import (
    DEFAULT_DATASET_REVISION,
    InferConfigLoader,
    configure_hf_access,
    resolve_hf_dataset_revision,
    verify_loaded_dataset_revision,
)
from featurebench.infer.gpu_scheduler import GpuLease, GpuScheduler, detect_host_gpu_ids, parse_gpu_id_list

# Global console for rich output
console = Console()


DEFAULT_DATASET = "LiberCoders/FeatureBench"
FEATUREBENCH_RUN_LABEL = "featurebench.run"
FEATUREBENCH_KIND_LABEL = "featurebench.kind"
FEATUREBENCH_TASK_LABEL = "featurebench.task"
FEATUREBENCH_ATTEMPT_LABEL = "featurebench.attempt"
EVAL_REPORT_COMPLETED_KEY = "featurebench_eval_completed"
EVAL_REPORT_COMPLETED_AT_KEY = "featurebench_eval_completed_at"


class RunningTasksTracker:
    """Track currently running evaluation tasks for rich live display."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running_tasks: dict[tuple[str, int], datetime] = {}

    @staticmethod
    def _format_elapsed(total_seconds: float) -> str:
        seconds = max(0, int(total_seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def mark_started(self, instance_id: str, n_attempt: int) -> None:
        with self._lock:
            self._running_tasks[(instance_id, n_attempt)] = datetime.now()

    def mark_finished(self, instance_id: str, n_attempt: int) -> None:
        with self._lock:
            self._running_tasks.pop((instance_id, n_attempt), None)

    def _snapshot(self) -> list[tuple[str, int, datetime]]:
        with self._lock:
            items = list(self._running_tasks.items())
        items.sort(key=lambda item: item[1])
        return [(instance_id, n_attempt, started_at) for (instance_id, n_attempt), started_at in items]

    def build_table(self) -> Table:
        table = Table(
            show_header=False,
            box=None,
            pad_edge=False,
            expand=True,
        )
        table.add_column("Running Task", style="bright_black")
        table.add_column("Elapsed", justify="right", width=10, style="bright_black")

        running = self._snapshot()
        if not running:
            table.add_row("[dim]No active task[/]", "")
            return table

        now = datetime.now()
        for idx, (instance_id, n_attempt, started_at) in enumerate(running, start=1):
            label = instance_id if n_attempt == 1 else f"{instance_id} (attempt {n_attempt})"
            indexed_label = f"[{idx}] {label}"
            elapsed = self._format_elapsed((now - started_at).total_seconds())
            table.add_row(indexed_label, elapsed)
        return table


class RunningTasksView:
    """Live renderable for currently running harness tasks."""

    def __init__(self, tracker: RunningTasksTracker):
        self._tracker = tracker

    def __rich__(self) -> Table:
        return self._tracker.build_table()


def _mark_report_completed(report: dict[str, Any], instance_id: str) -> None:
    instance_report = report.get(instance_id)
    if not isinstance(instance_report, dict):
        return
    instance_report[EVAL_REPORT_COMPLETED_KEY] = True
    instance_report[EVAL_REPORT_COMPLETED_AT_KEY] = datetime.now().isoformat()


def _report_has_completion_marker(report: dict[str, Any], instance_id: str) -> bool:
    instance_report = report.get(instance_id)
    return (
        isinstance(instance_report, dict)
        and instance_report.get(EVAL_REPORT_COMPLETED_KEY) is True
    )


def _legacy_report_looks_interrupted(log_dir: Path) -> bool:
    """Best-effort detection for reports written by old interrupted runs."""
    log_file = log_dir / LOG_INSTANCE
    if not log_file.exists():
        return False

    try:
        log_text = log_file.read_text(encoding=UTF8, errors="replace").lower()
    except Exception:
        return False

    if "evaluation interrupted during shutdown" in log_text:
        return True
    if "interrupted before container creation" in log_text:
        return True

    cleanup_failed = "failed to cleanup container" in log_text
    cleanup_raced = (
        "already in progress" in log_text
        or "no such container" in log_text
        or "not found" in log_text
    )
    return cleanup_failed and cleanup_raced


def _existing_report_is_reusable(
    report: dict[str, Any],
    log_dir: Path,
    instance_id: str,
) -> bool:
    if _report_has_completion_marker(report, instance_id):
        return True

    # Reports produced before this marker existed are still reusable unless
    # their logs show they were written while Ctrl+C cleanup was racing the
    # worker thread.
    return not _legacy_report_looks_interrupted(log_dir)


class EvalContainerCleanup:
    """Track eval containers so interrupts can remove them reliably."""

    def __init__(self, run_id: str, console: Console):
        self.run_id = run_id
        self.console = console
        self._active_containers_lock = threading.RLock()
        self._active_containers: dict[str, Container] = {}
        self._shutdown_requested = threading.Event()
        self._cleanup_lock = threading.RLock()
        self._cleanup_in_progress = False
        self._cleanup_interrupt_notice_printed = False
        self._atexit_cleanup = self._cleanup_active_containers_at_exit
        atexit.register(self._atexit_cleanup)

    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def request_shutdown(self) -> None:
        self._shutdown_requested.set()

    def container_labels(
        self,
        instance_id: str,
        n_attempt: int,
        purpose: str = "eval",
    ) -> dict[str, str]:
        return {
            FEATUREBENCH_RUN_LABEL: self.run_id,
            FEATUREBENCH_KIND_LABEL: "eval",
            FEATUREBENCH_TASK_LABEL: str(instance_id),
            FEATUREBENCH_ATTEMPT_LABEL: str(n_attempt),
            "featurebench.purpose": purpose,
        }

    def register(self, container: Container) -> None:
        container_id = getattr(container, "id", None)
        if not container_id:
            return
        with self._active_containers_lock:
            self._active_containers[container_id] = container

    def unregister(self, container: Container) -> None:
        container_id = getattr(container, "id", None)
        if not container_id:
            return
        with self._active_containers_lock:
            self._active_containers.pop(container_id, None)

    def _ignore_interrupt_during_cleanup(self, signum, frame) -> None:
        self.request_shutdown()
        if not self._cleanup_interrupt_notice_printed:
            self._cleanup_interrupt_notice_printed = True
            try:
                self.console.print(
                    "[yellow]Cleanup already in progress; ignoring additional Ctrl+C.[/]"
                )
            except Exception:
                pass

    def _remove_container_best_effort(self, container: Container) -> bool:
        try:
            container_id = getattr(container, "short_id", None) or getattr(container, "id", "")
            try:
                container.reload()
            except docker.errors.NotFound:
                return False
            except Exception:
                pass

            try:
                if getattr(container, "status", None) == "running":
                    container.kill()
            except docker.errors.NotFound:
                return False
            except Exception:
                pass

            try:
                container.remove(force=True)
                return True
            except docker.errors.NotFound:
                return False
            except Exception as exc:
                try:
                    self.console.print(
                        f"[yellow]Warning: failed to remove eval container {container_id}: {exc}[/]"
                    )
                except Exception:
                    pass
                return False
        finally:
            self.unregister(container)

    def _cleanup_labeled_containers(self) -> int:
        removed = 0
        try:
            client = docker.from_env()
            label_filter = [
                f"{FEATUREBENCH_RUN_LABEL}={self.run_id}",
                f"{FEATUREBENCH_KIND_LABEL}=eval",
            ]
            containers = client.containers.list(
                all=True,
                filters={"label": label_filter},
            )
        except Exception as exc:
            try:
                self.console.print(
                    f"[yellow]Warning: failed to scan eval containers by label: {exc}[/]"
                )
            except Exception:
                pass
            return removed

        for container in containers:
            if self._remove_container_best_effort(container):
                removed += 1
        return removed

    def cleanup_active_containers(self, reason: str) -> int:
        with self._cleanup_lock:
            if self._cleanup_in_progress:
                return 0
            self._cleanup_in_progress = True

        previous_sigint = None
        signal_replaced = False
        try:
            if threading.current_thread() is threading.main_thread():
                previous_sigint = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._ignore_interrupt_during_cleanup)
                signal_replaced = True

            with self._active_containers_lock:
                containers = list(self._active_containers.values())

            removed = 0
            if containers:
                try:
                    self.console.print(
                        f"[yellow]Cleaning {len(containers)} active eval container(s) after {reason}...[/]"
                    )
                except Exception:
                    pass

            for container in containers:
                if self._remove_container_best_effort(container):
                    removed += 1

            removed += self._cleanup_labeled_containers()
            if removed:
                try:
                    self.console.print(
                        f"[yellow]Removed {removed} eval container(s).[/]"
                    )
                except Exception:
                    pass
            return removed
        finally:
            if signal_replaced:
                try:
                    signal.signal(signal.SIGINT, previous_sigint)
                except Exception:
                    pass
            with self._cleanup_lock:
                self._cleanup_in_progress = False

    def _cleanup_active_containers_at_exit(self) -> None:
        self.cleanup_active_containers("process exit")


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Set up logger for instance evaluation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []

    # File handler only - no console output
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _load_force_rerun_ids(raw_values: list[str] | None) -> set[str]:
    if not raw_values:
        return set()

    ids: list[str] = []
    for value in raw_values:
        if not value:
            continue
        candidate = str(value).strip()
        if not candidate:
            continue
        if candidate.endswith(".txt"):
            path = Path(candidate)
            if not path.exists():
                logging.getLogger(__name__).warning(
                    "--force-rerun file not found: %s",
                    path,
                )
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    task_id = line.strip()
                    if task_id:
                        ids.append(task_id)
            continue
        ids.append(candidate)

    return set(dict.fromkeys(ids))


def run_instance(
    instance: pd.Series,
    pred: dict,
    output_dir: Path,
    timeout: int | None = None,
    gpu_ids: str | None = None,
    review_codes: bool = False,
    proxy_port: int | None = None,
    white: bool = False,
    docker_runtime_config: dict | None = None,
    gpu_scheduler: GpuScheduler | None = None,
    force_rerun_ids: set[str] | None = None,
    running_tasks_tracker: RunningTasksTracker | None = None,
    container_cleanup: EvalContainerCleanup | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for a single instance.

    Args:
        instance: Instance data from FeatureBench dataset
        pred: Prediction dictionary with patch
        output_dir: Output directory for evaluation results
        timeout: Test timeout in seconds
        gpu_ids: Comma-separated GPU IDs to use
        review_codes: Whether to save agent-generated code for review
        proxy_port: Proxy port for network access

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]
    n_attempt = pred.get(KEY_N_ATTEMPT, 1)

    def _interrupted_result(message: str) -> dict[str, Any]:
        return {
            "instance_id": instance_id,
            "n_attempt": n_attempt,
            "completed": False,
            "resolved": False,
            "patch_applied": False,
            "error": message,
        }

    # Setup logging directory
    log_dir = output_dir / "eval_outputs" / instance_id / f"attempt-{n_attempt}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check if report already exists
    report_path = log_dir / LOG_REPORT
    if report_path.exists() and (force_rerun_ids is None or instance_id not in force_rerun_ids):
        existing_report = None
        try:
            with open(report_path, "r", encoding=UTF8) as f:
                existing_report = json.load(f)
        except Exception:
            existing_report = None

        if (
            isinstance(existing_report, dict)
            and _existing_report_is_reusable(existing_report, log_dir, instance_id)
        ):
            return {
                "instance_id": instance_id,
                "n_attempt": n_attempt,
                "completed": True,
                "resolved": existing_report.get(instance_id, {}).get("resolved", False),
                "patch_applied": existing_report.get(instance_id, {}).get("patch_successfully_applied", False),
                "report": existing_report,
            }

        try:
            report_path.unlink()
        except FileNotFoundError:
            pass

    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(f"{instance_id}-{n_attempt}", log_file)

    logger.info(f"{'=' * 60}")
    logger.info(f"Starting evaluation for instance: {instance_id}")
    logger.info(f"Level: {instance['level']}")
    logger.info(f"Attempt: {n_attempt}")
    logger.info(f"{'=' * 60}")

    container = None
    gpu_lease: GpuLease | None = None
    container_manager = EvalContainerManager(logger)

    try:
        if running_tasks_tracker is not None:
            running_tasks_tracker.mark_started(instance_id, n_attempt)

        if container_cleanup is not None and container_cleanup.shutdown_requested():
            return _interrupted_result("Interrupted before container creation")

        # Parse repo_settings for docker runtime config (or reuse precomputed)
        docker_runtime_config = docker_runtime_config or {}
        if not docker_runtime_config:
            repo_settings = parse_repo_settings(instance)
            docker_runtime_config = get_docker_runtime_config(repo_settings)
        logger.info(f"Docker runtime config: shm_size={docker_runtime_config.get('shm_size')}, "
                    f"number_once={docker_runtime_config.get('number_once')}, "
                    f"env_vars={list(docker_runtime_config.get('env_vars', {}).keys())}, "
                    f"env_exports={len(docker_runtime_config.get('env_exports', []))} items")

        # Allocate GPUs for this instance if needed.
        task_gpu_ids = gpu_ids
        if docker_runtime_config.get("need_gpu") and gpu_scheduler is not None:
            requested = docker_runtime_config.get("number_once", 1)
            if not isinstance(requested, int) or requested <= 0:
                requested = 1
            gpu_lease = gpu_scheduler.allocate(requested)
            task_gpu_ids = gpu_lease.gpu_ids_str
            logger.info(
                f"GPU scheduling: allocated {requested} GPU(s): {task_gpu_ids} "
                f"(pool={','.join(gpu_scheduler.gpu_pool)})"
            )

        # Get Docker image and create container
        docker_image = get_docker_image_name(instance)
        logger.info(f"Using Docker image: {docker_image}")

        container_manager.pull_image_if_needed(docker_image)
        labels = None
        if container_cleanup is not None:
            labels = container_cleanup.container_labels(instance_id, n_attempt)

        if container_cleanup is not None and container_cleanup.shutdown_requested():
            return _interrupted_result("Interrupted before container creation")

        container = container_manager.create_container(
            docker_image,
            instance_id,
            n_attempt,
            task_gpu_ids,
            proxy_port,
            docker_runtime_config,
            labels=labels,
        )
        if container_cleanup is not None:
            container_cleanup.register(container)

        # Run evaluation based on level
        level = int(instance["level"])
        if level == 1:
            results = run_instance_level1(instance, pred, container, logger, log_dir, timeout, white=white)
        elif level == 2:
            results = run_instance_level2(instance, pred, container, logger, log_dir, timeout)
        else:
            raise ValueError(f"Unsupported level: {level}")

        if container_cleanup is not None and container_cleanup.shutdown_requested():
            logger.warning(
                "Evaluation interrupted during shutdown; not saving report."
            )
            return _interrupted_result("Interrupted during shutdown")

        # Save patch
        patch_content = pred.get(KEY_PREDICTION, "")
        patch_path = log_dir / LOG_PATCH
        with open(patch_path, "w", encoding=UTF8) as f:
            f.write(patch_content)

        # Parse test outputs and build report
        repo_name = instance.get("repo", "")
        f2p_parsed, p2p_parsed_list = parse_test_outputs(log_dir, repo_name, level)

        # Determine eval_type based on repo
        eval_type = EvalType.FAIL_ONLY if repo_name in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

        f2p_success, f2p_failure, p2p_success, p2p_failure = build_test_status(
            f2p_parsed, p2p_parsed_list, eval_type
        )

        # Generate and save report
        report = generate_instance_report(
            instance_id=instance_id,
            n_attempt=n_attempt,
            patch_content=patch_content,
            patch_applied=results.get("patch_applied", False),
            f2p_success_list=f2p_success,
            f2p_failure_list=f2p_failure,
            p2p_success_list=p2p_success,
            p2p_failure_list=p2p_failure,
            eval_results=results,
        )
        _mark_report_completed(report, instance_id)
        if container_cleanup is not None and container_cleanup.shutdown_requested():
            logger.warning(
                "Evaluation interrupted during shutdown; not saving report."
            )
            return _interrupted_result("Interrupted during shutdown")
        save_report(report, log_dir)
        if container_cleanup is not None and container_cleanup.shutdown_requested():
            logger.warning(
                "Evaluation interrupted during shutdown; removing saved report."
            )
            try:
                report_path.unlink()
            except FileNotFoundError:
                pass
            return _interrupted_result("Interrupted during shutdown")

        resolved = report[instance_id]["resolved"]
        f2p_pass_rate = report[instance_id]["pass_rate"]

        logger.info(f"{'=' * 60}")
        logger.info(f"Evaluation completed for {instance_id}")
        logger.info(f"Resolved: {resolved}")
        logger.info(f"F2P Pass rate: {f2p_pass_rate}")
        logger.info(f"{'=' * 60}")

        # Save review codes if requested
        if review_codes:
            logger.info("Saving agent-generated code for review...")
            review_labels = None
            if container_cleanup is not None:
                review_labels = container_cleanup.container_labels(
                    instance_id,
                    n_attempt,
                    purpose="review",
                )
            if level == 1:
                save_review_codes_level1(
                    instance,
                    patch_content,
                    log_dir,
                    docker_image,
                    logger,
                    labels=review_labels,
                )
            elif level == 2:
                save_review_codes_level2(
                    instance,
                    patch_content,
                    log_dir,
                    docker_image,
                    logger,
                    labels=review_labels,
                )
            logger.info("Review codes saved.")

        return {
            "instance_id": instance_id,
            "n_attempt": n_attempt,
            "completed": True,
            "resolved": resolved,
            "patch_applied": results.get("patch_applied", False),
            "report": report,
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())

        if container_cleanup is not None and container_cleanup.shutdown_requested():
            logger.warning(
                "Evaluation interrupted during shutdown; not saving failure report."
            )
            return _interrupted_result(f"Interrupted during shutdown: {e}")

        # Generate and save error report
        patch_content = pred.get(KEY_PREDICTION, "")
        error_report = generate_error_report(
            instance_id=instance_id,
            n_attempt=n_attempt,
            patch_content=patch_content,
            error=str(e),
            traceback_str=traceback.format_exc(),
        )
        _mark_report_completed(error_report, instance_id)
        save_report(error_report, log_dir)
        if container_cleanup is not None and container_cleanup.shutdown_requested():
            logger.warning(
                "Evaluation interrupted during shutdown; removing saved failure report."
            )
            try:
                report_path.unlink()
            except FileNotFoundError:
                pass
            return _interrupted_result(f"Interrupted during shutdown: {e}")

        return {
            "instance_id": instance_id,
            "n_attempt": n_attempt,
            "completed": True,
            "resolved": False,
            "patch_applied": False,
            "error": str(e),
            "report": error_report,
        }

    finally:
        # Cleanup container
        if container is not None:
            try:
                container_manager.cleanup_container(container)
            finally:
                if container_cleanup is not None:
                    container_cleanup.unregister(container)

        # Release GPU lease even if container creation failed.
        if gpu_lease is not None and gpu_scheduler is not None:
            try:
                gpu_scheduler.release(gpu_lease)
            except Exception as e:
                logger.warning(f"Error releasing GPU lease: {e}")

        if running_tasks_tracker is not None:
            running_tasks_tracker.mark_finished(instance_id, n_attempt)


def load_dataset_from_hf(
    console: Console,
    split: str,
    dataset: str,
    config_path: str | None = None,
    dataset_revision: str | None = None,
) -> pd.DataFrame:
    """
    Load dataset from HuggingFace using config.toml settings.

    Args:
        console: Rich console for output
        split: HuggingFace split name (e.g., "lite", "full")

    Returns:
        DataFrame with instances from the specified split
    """
    # Load configuration from config.toml
    if not dataset:
        raise ValueError("dataset must be a non-empty string")

    try:
        resolved_path = Path(config_path).expanduser() if config_path else None
        config_loader = InferConfigLoader(resolved_path)
        env_vars = config_loader.env_vars
        hf_token, hf_endpoint = configure_hf_access(env_vars)
        configured_revision = config_loader.get_dataset_revision()
    except FileNotFoundError:
        console.print("[yellow]Warning: config.toml not found, using defaults[/]")
        hf_token, hf_endpoint = configure_hf_access({})
        configured_revision = DEFAULT_DATASET_REVISION

    requested_revision = str(dataset_revision or "").strip()
    effective_revision = requested_revision or configured_revision

    console.print(f"[bold blue]Loading dataset from HuggingFace...[/]")
    console.print(f"[dim]Repository: {dataset}[/]")
    console.print(f"[dim]Split: {split}[/]")
    console.print(f"[dim]Version: {effective_revision}[/]")

    if hf_token:
        console.print("[dim]Using HuggingFace token from config[/]")

    try:
        resolved_revision = resolve_hf_dataset_revision(
            dataset,
            effective_revision,
            hf_token,
            hf_endpoint,
        )
        # Load the specified split
        hf_dataset = load_dataset(
            dataset,
            split=split,
            token=hf_token,
            revision=resolved_revision,
        )
        verify_loaded_dataset_revision(hf_dataset, resolved_revision)
        df = pd.DataFrame(hf_dataset)

        # Determine level from instance_id's last segment (e.g., "xxx.lv1" -> level 1)
        def get_level_from_instance_id(instance_id: str) -> int:
            last_segment = instance_id.split(".")[-1] if instance_id else ""
            if last_segment == "lv1":
                return 1
            elif last_segment == "lv2":
                return 2
            else:
                raise RuntimeError(f"Unknown level: {last_segment} in instance_id: {instance_id}")

        df['level'] = df['instance_id'].apply(get_level_from_instance_id)

        lv1_count = len(df[df['level'] == 1])
        lv2_count = len(df[df['level'] == 2])
        console.print(f"[green]Loaded {lv1_count} Level 1 + {lv2_count} Level 2 = {len(df)} instances from split '{split}'[/]")
        return df

    except Exception as e:
        console.print(f"[bold red]Error loading dataset: {e}[/]")
        console.print("[yellow]Troubleshooting:[/]")
        console.print("  1. Check HF_TOKEN in config.toml")
        console.print(f"  2. Check access: https://huggingface.co/datasets/{dataset}")
        console.print(f"  3. Verify split '{split}' exists in the dataset")
        raise


def resolve_eval_dataset_revision(
    cli_revision: str | None,
    run_metadata: dict | None,
) -> str | None:
    """Resolve the dataset revision for evaluation.

    An explicit CLI value wins. Otherwise, evaluation follows the revision
    recorded by inference. Runs created before revision tracking belong to the
    original v1.0 dataset. When no run metadata is available, the dataset
    loader falls back to the configured/default revision.
    """
    requested_revision = str(cli_revision or "").strip()
    if requested_revision:
        return requested_revision

    if run_metadata:
        recorded_revision = str(run_metadata.get("dataset_revision") or "").strip()
        return recorded_revision or "v1.0"

    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FeatureBench evaluation on predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.toml (used for HF_TOKEN/HF_ENDPOINT when loading dataset)",
    )
    parser.add_argument(
        "--predictions-path", "-p",
        type=str,
        required=True,
        help="Path to predictions JSONL file, or 'gold' to evaluate gold patch",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        nargs="+",
        dest="task_ids",
        help="Specific task IDs (instance IDs) to evaluate (optional)",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=4,
        help="Number of parallel workers",
        dest="n_concurrent",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override timeout for test execution (seconds). If not set, uses timeout_run from repo_settings",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1' or '2,3')",
    )
    parser.add_argument(
        "--review-codes",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=False,
        help="Save agent-generated code for review (true/false)",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=None,
        help="Proxy port for network access (uses host network if set)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="HuggingFace dataset repo name (e.g., 'LiberCoders/FeatureBench')",
    )

    parser.add_argument(
        "--data-version",
        dest="dataset_version",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset version (tag, branch, or commit SHA). "
            "Default: infer from run_metadata.json, then [dataset].revision, then v1.1"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="full",
        help="HuggingFace dataset split name (e.g., 'lite', 'full'). Default: 'full'",
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "Include predictions with success=false in the evaluation queue. "
            "By default, failed infer outputs are skipped when using runs/*/output.jsonl."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Force rerun task IDs (space-separated), or provide a .txt file with one task_id per line. "
            "These tasks are evaluated even if report.json already exists."
        ),
    )

    return parser.parse_args()


def print_summary(console: Console, summary_report: dict) -> None:
    """
    Print summary report to console using rich.

    Args:
        console: Rich console
        summary_report: Summary report dictionary
    """
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/]")
    console.print("[bold cyan]Evaluation Summary[/]")
    console.print("[bold cyan]" + "=" * 60 + "[/]")

    for attempt_key in sorted(summary_report.keys()):
        attempt_data = summary_report[attempt_key]
        console.print()
        console.print(f"[bold white][{attempt_key}][/]")
        console.print(f"  [white]Total:[/] [green]{attempt_data['total_instances']}[/]")
        console.print(f"  [white]Completed:[/] [green]{attempt_data['completed_instances']}[/]")
        console.print(f"  [white]Resolved:[/] [green]{attempt_data['resolved_instances']}[/] | [red]Unresolved: {attempt_data['unresolved_instances']}[/]")
        not_applied_empty = attempt_data.get('not_applied_patch_empty_instances', 0)
        not_applied_other = attempt_data.get('not_applied_patch_other_instances', 0)
        console.print(
            "  [white]Not applied patch:[/] "
            f"[yellow]empty:[/] [yellow]{not_applied_empty}[/]"
            f" | [yellow]other:[/] [yellow]{not_applied_other}[/]"
            f" | [red]Errors: {attempt_data['error_instances']}[/]"
        )
        console.print(f"  [white]Resolved rate:[/] [bold green]{attempt_data['resolved_rate'] * 100:.1f}%[/]")
        console.print(f"  [white]Avg F2P pass rate:[/] [bold green]{attempt_data['pass_rate'] * 100:.1f}%[/]")

    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/]")


def main():
    """Main entry point for FeatureBench evaluation."""
    args = parse_args()

    use_gold_patch = args.predictions_path == "gold"

    # Get output directory from predictions_path
    predictions_path = Path(args.predictions_path) if not use_gold_patch else Path("gold")
    output_dir = Path("./runs/gold") if use_gold_patch else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect white-box mode from FeatureBench infer metadata (best-effort).
    #
    # Rationale: harness is typically pointed at runs/<ts>/output.jsonl. In that folder,
    # featurebench.infer writes run_metadata.json which contains `white_box`. When enabled,
    # we keep FAIL_TO_PASS test files visible during Level 1 patch application to avoid
    # "patch touches deleted test file" apply failures.
    white_enabled = False
    run_metadata = None
    try:
        metadata_path = output_dir / "run_metadata.json"
        if not use_gold_patch and metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                run_metadata = meta
                white_enabled = meta.get("white_box") is True
    except Exception:
        white_enabled = False
        run_metadata = None

    dataset_revision = resolve_eval_dataset_revision(
        args.dataset_version,
        run_metadata,
    )

    # Print startup banner
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/]")
    console.print("[bold cyan]FeatureBench Evaluation[/]")
    if use_gold_patch:
        console.print(f"[white]Predictions:[/] [green]gold (from dataset 'patch' field)[/]")
    else:
        console.print(f"[white]Predictions:[/] [green]{predictions_path}[/]")
    console.print(f"[white]Output:[/] [green]{output_dir}[/]")
    console.print(f"[white]Dataset:[/] [green]{args.dataset}[/]")
    console.print(f"[white]Split:[/] [green]{args.split}[/]")
    console.print(f"[white]Workers:[/] [green]{args.n_concurrent}[/]")
    if args.timeout:
        console.print(f"[white]Timeout:[/] [green]{args.timeout}s (override)[/]")
    else:
        console.print(f"[white]Timeout:[/] [green]from repo_settings[/]")
    if args.gpu_ids:
        console.print(f"[white]GPUs:[/] [green]{args.gpu_ids}[/]")
    else:
        console.print(f"[white]GPUs:[/] [green]all available[/]")
    if args.proxy_port:
        console.print(f"[white]Proxy:[/] [green]http://{DOCKER_HOST_GATEWAY}:{args.proxy_port}[/]")
    if args.review_codes:
        console.print(f"[white]Review codes:[/] [yellow]ENABLED[/]")
    if white_enabled:
        console.print(f"[white]White-box (harness):[/] [yellow]ENABLED[/]")
    console.print("[bold cyan]" + "=" * 60 + "[/]")
    console.print()

    # Load dataset
    dataset = load_dataset_from_hf(
        console,
        args.split,
        args.dataset,
        args.config_path,
        dataset_revision,
    )

    # Load predictions
    console.print(f"[bold blue]Loading predictions...[/]")
    if use_gold_patch:
        if "patch" not in dataset.columns:
            raise RuntimeError("Dataset does not contain required 'patch' field for --predictions-path gold")
        before_gold_filter = len(dataset)
        dataset_gold = dataset[~dataset[KEY_INSTANCE_ID].astype(str).str.endswith("lv2")]
        filtered_lv2 = before_gold_filter - len(dataset_gold)
        if filtered_lv2:
            console.print(f"[yellow]Gold mode: filtered out {filtered_lv2} lv2 instances[/]")
        predictions = []
        for _, row in dataset_gold.iterrows():
            raw_patch = "" if pd.isna(row["patch"]) else str(row["patch"])
            # Preprocess HF patch for evaluation
            # Note: This is necessary because the dataset stores the corruption patch, but we need to evaluate the fix patch.
            processed_patch = preprocess_hf_patch(raw_patch, row.get("FAIL_TO_PASS"))
            predictions.append(
                {
                    KEY_INSTANCE_ID: row[KEY_INSTANCE_ID],
                    KEY_PREDICTION: processed_patch,
                }
            )
    else:
        predictions = get_predictions_from_file(args.predictions_path)

    # If predictions come from FeatureBench infer output.jsonl, they include a boolean `success`.
    # By default, skip failed infer results to avoid spending harness time on instances that
    # never produced a usable patch.
    if not use_gold_patch and not args.include_failed:
        before = len(predictions)
        predictions = [p for p in predictions if p.get("success") is not False]
        skipped = before - len(predictions)
        if skipped:
            console.print(f"[yellow]Skipped {skipped} failed predictions (success=false)[/]")

    console.print(f"[green]Loaded {len(predictions)} predictions[/]")

    # Filter by instance IDs if specified
    if args.task_ids:
        predictions = filter_predictions_by_ids(predictions, args.task_ids)
        console.print(f"[yellow]Filtered to {len(predictions)} predictions[/]")

    # Match predictions with dataset
    instances_to_eval = []
    missing_count = 0
    for pred in predictions:
        instance_id = pred[KEY_INSTANCE_ID]
        instance = get_instance_from_dataset(dataset, instance_id)
        if instance is None:
            missing_count += 1
            continue
        # Precompute docker runtime config once (avoids repeating repo_settings parsing in workers)
        try:
            repo_settings = parse_repo_settings(instance)
            docker_runtime_config = get_docker_runtime_config(repo_settings)
        except Exception:
            docker_runtime_config = {}
        instances_to_eval.append((instance, pred, docker_runtime_config))

    if missing_count > 0:
        console.print(f"[yellow]Warning: {missing_count} instances not found in dataset[/]")

    total_tasks = len(instances_to_eval)
    console.print(f"[bold blue]Tasks to evaluate: {total_tasks}[/]")
    console.print()

    if total_tasks == 0:
        console.print("[bold red]No tasks to evaluate![/]")
        return

    # Track success/failure counts
    resolved_count = 0
    failed_count = 0

    # Initialize GPU scheduler (best-effort) if there are any GPU tasks.
    gpu_scheduler: GpuScheduler | None = None
    try:
        need_gpu_any = any(cfg.get("need_gpu") for _, _, cfg in instances_to_eval)
    except Exception:
        need_gpu_any = False

    if need_gpu_any:
        gpu_pool: list[str] | None = None
        if args.gpu_ids:
            gpu_pool = parse_gpu_id_list(args.gpu_ids)
        else:
            gpu_pool = detect_host_gpu_ids()

        if gpu_pool:
            max_required = 1
            for _, _, cfg in instances_to_eval:
                if not cfg.get("need_gpu"):
                    continue
                n = cfg.get("number_once", 1)
                if isinstance(n, int) and n > max_required:
                    max_required = n

            if max_required > len(gpu_pool):
                raise RuntimeError(
                    f"GPU scheduling pool too small: max number_once={max_required} but pool={','.join(gpu_pool)}"
                )

            gpu_scheduler = GpuScheduler(gpu_pool)
            console.print(
                f"[bold cyan]GPU scheduling enabled[/]: pool=[green]{','.join(gpu_pool)}[/] "
                f"(each GPU eval task gets number_once GPUs)"
            )
        else:
            console.print(
                "[bold yellow]GPU scheduling disabled[/]: failed to detect GPU pool (nvidia-smi unavailable). "
                "Falling back to previous behavior (GPU tasks may all default to GPU0)."
            )

    force_rerun_ids = _load_force_rerun_ids(args.force_rerun)
    eval_run_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{os.getpid()}"
    container_cleanup = EvalContainerCleanup(eval_run_id, console)

    # Run evaluations in parallel with rich progress
    results = []
    running_tasks_tracker = RunningTasksTracker()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[green]{task.fields[resolved]}✓[/] [red]{task.fields[failed]}✗[/]"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=10,
    )

    task_progress = progress.add_task(
        "[cyan]Evaluating...",
        total=total_tasks,
        resolved=0,
        failed=0
    )

    with Live(
        Group(progress, RunningTasksView(running_tasks_tracker)),
        console=console,
        refresh_per_second=10,
    ):
        executor = ThreadPoolExecutor(max_workers=args.n_concurrent)
        futures = {}
        try:
            for instance, pred, docker_runtime_config in instances_to_eval:
                if container_cleanup.shutdown_requested():
                    break
                future = executor.submit(
                    run_instance,
                    instance,
                    pred,
                    output_dir,
                    args.timeout,
                    args.gpu_ids,
                    args.review_codes,
                    args.proxy_port,
                    white_enabled,
                    docker_runtime_config,
                    gpu_scheduler,
                    force_rerun_ids,
                    running_tasks_tracker,
                    container_cleanup,
                )
                futures[future] = instance[KEY_INSTANCE_ID]

            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.get("resolved"):
                        resolved_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    failed_count += 1
                    results.append({
                        "instance_id": instance_id,
                        "completed": False,
                        "resolved": False,
                        "patch_applied": False,
                        "error": str(e),
                    })

                progress.update(
                    task_progress,
                    advance=1,
                    resolved=resolved_count,
                    failed=failed_count
                )
        except KeyboardInterrupt:
            container_cleanup.request_shutdown()
            for pending_future in futures:
                pending_future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            console.print("[bold yellow]Interrupted; cleaning FeatureBench eval containers...[/]")
            container_cleanup.cleanup_active_containers("keyboard interrupt")
            return 130
        finally:
            if not container_cleanup.shutdown_requested():
                executor.shutdown(wait=True)

    # Generate and save summary report
    console.print()
    console.print("[bold blue]Generating summary report...[/]")

    summary_report = generate_summary_report(results, output_dir)

    summary_report_path = output_dir / "report.json"
    with open(summary_report_path, "w", encoding=UTF8) as f:
        json.dump(summary_report, f, indent=4)

    console.print(f"[green]Report saved to: {summary_report_path}[/]")

    # Print summary
    print_summary(console, summary_report)


if __name__ == "__main__":
    raise SystemExit(main())
