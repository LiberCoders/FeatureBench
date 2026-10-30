"""
Thread-safe output management for inference results.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from featurebench.infer.models import InferResult, RunMetadata, TaskPaths
from featurebench.utils.identifiers import normalize_instance_id


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class OutputManager:
    """
    Thread-safe output manager for inference results.

    Serializes writes and replaces output files atomically.
    """

    def __init__(self, output_dir: Path, logger: logging.Logger | None = None):
        """
        Initialize the output manager.

        Args:
            output_dir: Output directory (timestamp folder)
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.output_jsonl_path = self.output_dir / "output.jsonl"
        self.metadata_path = self.output_dir / "run_metadata.json"

        # Run summary path - uses current run timestamp to avoid overwriting on resume
        self._run_timestamp_current = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        self.run_summary_path = self.output_dir / f"run_summary_{self._run_timestamp_current}.json"

        # Thread-safe lock for file operations
        self._file_lock = threading.Lock()

        # Separate lock for run summary to avoid blocking main output
        self._summary_lock = threading.Lock()

        # In-memory run summary for current session
        self._run_summary: dict[str, list[str]] = {
            "timestamp": self._run_timestamp_current,
            "success": [],
            "failure": [],
        }

    def _write_result_atomic(self, result: InferResult) -> None:
        """
        Write a result to output.jsonl atomically using file locking.

        If a record with the same instance_id and n_attempt exists,
        it will be replaced with the new result.

        Args:
            result: Inference result to write
        """
        with self._file_lock:
            existing_lines = []
            replaced = False

            for data, line in self._load_result_records():
                if (
                    data["instance_id"] == result.instance_id
                    and data["n_attempt"] == result.n_attempt
                ):
                    replaced = True
                    continue
                existing_lines.append(line)

            temporary_path = self.output_jsonl_path.with_suffix(".jsonl.tmp")
            try:
                with temporary_path.open("w", encoding="utf-8") as f:
                    for line in existing_lines:
                        f.write(line + "\n")
                    f.write(result.to_jsonl() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                temporary_path.replace(self.output_jsonl_path)
            finally:
                temporary_path.unlink(missing_ok=True)

            action = "Replaced" if replaced else "Wrote"
            self.logger.debug(
                f"{action} result for {result.instance_id} attempt {result.n_attempt}"
            )

        # Update run summary (thread-safe, separate lock)
        self._update_run_summary(result)

    def write_result(self, result: InferResult) -> None:
        """
        Write a result to output.jsonl.

        Args:
            result: Inference result to write
        """
        self._write_result_atomic(result)

    def _update_run_summary(self, result: InferResult) -> None:
        """
        Update the run summary with a result.

        This method is thread-safe and updates both in-memory summary
        and the summary file atomically.

        Args:
            result: Inference result
        """
        task_paths = TaskPaths(self.output_dir, result.instance_id, result.n_attempt)
        infer_log_path = str(task_paths.infer_log_path)

        with self._summary_lock:
            # Remove from both lists first (in case of re-run/replacement)
            if infer_log_path in self._run_summary["success"]:
                self._run_summary["success"].remove(infer_log_path)
            if infer_log_path in self._run_summary["failure"]:
                self._run_summary["failure"].remove(infer_log_path)

            if result.success:
                self._run_summary["success"].append(infer_log_path)
            else:
                self._run_summary["failure"].append(infer_log_path)

            self._save_run_summary()

    def _save_run_summary(self) -> None:
        """
        Save the current run summary to file.

        Must be called while holding _summary_lock.
        """
        _write_json_atomic(self.run_summary_path, self._run_summary)

    def get_run_summary(self) -> dict[str, Any]:
        """
        Get the current run summary.

        Returns:
            Dictionary with timestamp, success list, and failure list
        """
        with self._summary_lock:
            return dict(self._run_summary)

    def save_metadata(self, metadata: RunMetadata) -> None:
        """Atomically save validated run metadata to run_metadata.json."""
        with self._file_lock:
            self._write_metadata(metadata)
            self.logger.info(f"Saved run metadata to {self.metadata_path}")

    def _write_metadata(self, metadata: RunMetadata) -> None:
        _write_json_atomic(self.metadata_path, metadata.to_dict())

    def update_metadata_end_time(self) -> None:
        """Update the end_time in run_metadata.json."""
        with self._file_lock:
            if not self.metadata_path.exists():
                return
            metadata = RunMetadata.load(self.metadata_path)
            metadata.end_time = datetime.now().isoformat()
            self._write_metadata(metadata)
            self.logger.info("Updated end_time in metadata")

    def save_patch(self, task_paths: TaskPaths, patch: str) -> None:
        """
        Save a patch.diff file for a task.

        Args:
            task_paths: TaskPaths instance containing all path information
            patch: Patch content
        """
        # Ensure directories exist
        task_paths.ensure_dirs()

        if not patch or not patch.strip():
            self.logger.warning(
                f"patch for {task_paths.task_id} attempt {task_paths.attempt} is empty"
            )

        temporary_path = task_paths.patch_path.with_suffix(
            f"{task_paths.patch_path.suffix}.tmp"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as f:
                f.write(patch)
                f.flush()
                os.fsync(f.fileno())
            temporary_path.replace(task_paths.patch_path)
            self.logger.debug(f"Saved patch for {task_paths.task_id} attempt {task_paths.attempt}")
        except OSError as exc:
            raise RuntimeError(f"failed to save patch for {task_paths.task_id}") from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    def get_task_paths(self, task_id: str, attempt: int) -> TaskPaths:
        """
        Get TaskPaths instance for a task.

        Args:
            task_id: Task instance ID
            attempt: Attempt number

        Returns:
            TaskPaths instance with all path information
        """
        return TaskPaths(self.output_dir, task_id, attempt)

    def load_completed_tasks(self) -> set[tuple[str, int]]:
        """
        Load completed tasks from output.jsonl.

        Returns:
            Set of (instance_id, n_attempt) tuples that have been completed
        """
        completed: set[tuple[str, int]] = set()

        for data, _ in self._load_result_records():
            if data["success"]:
                completed.add((data["instance_id"], data["n_attempt"]))

        return completed

    def _load_result_records(self) -> list[tuple[dict[str, Any], str]]:
        """Load and validate durable inference result records."""
        if not self.output_jsonl_path.exists():
            return []

        records: list[tuple[dict[str, Any], str]] = []
        seen_keys: set[tuple[str, int]] = set()
        with self.output_jsonl_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {self.output_jsonl_path} at line {line_number}"
                    ) from exc
                if not isinstance(data, dict):
                    raise ValueError(
                        f"expected a JSON object in {self.output_jsonl_path} "
                        f"at line {line_number}"
                    )

                try:
                    instance_id = normalize_instance_id(data.get("instance_id"))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid instance_id in {self.output_jsonl_path} at line {line_number}: "
                        f"{exc}"
                    ) from exc
                data["instance_id"] = instance_id
                n_attempt = data.get("n_attempt")
                if isinstance(n_attempt, bool) or not isinstance(n_attempt, int) or n_attempt < 1:
                    raise ValueError(
                        f"invalid n_attempt in {self.output_jsonl_path} at line {line_number}"
                    )
                if not isinstance(data.get("model_patch"), str):
                    raise ValueError(
                        f"invalid model_patch in {self.output_jsonl_path} at line {line_number}"
                    )
                if not isinstance(data.get("success"), bool):
                    raise ValueError(
                        f"invalid success in {self.output_jsonl_path} at line {line_number}"
                    )

                result_key = (instance_id, n_attempt)
                if result_key in seen_keys:
                    raise ValueError(
                        f"duplicate result for instance_id={instance_id!r}, "
                        f"n_attempt={n_attempt} in {self.output_jsonl_path}"
                    )
                seen_keys.add(result_key)
                records.append((data, line.rstrip("\n")))

        return records
