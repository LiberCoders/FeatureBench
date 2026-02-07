"""
Thread-safe output management for inference results.
"""

import fcntl
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from featurebench.infer.models import InferResult, RunMetadata, TaskPaths


class OutputManager:
    """
    Thread-safe output manager for inference results.
    
    Uses file locking to ensure atomic writes to output.jsonl,
    and a queue for handling concurrent write requests.
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
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
        self._run_summary: Dict[str, List[str]] = {
            "timestamp": self._run_timestamp_current,
            "success": [],
            "failure": []
        }
        
        # Write queue for serializing writes
        self._write_queue: Queue = Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
    
    def start(self) -> None:
        """Start the background writer thread."""
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self.logger.debug("Output writer thread started")
    
    def stop(self) -> None:
        """Stop the background writer thread and flush remaining items."""
        self._stop_writer.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=30)
        
        # Flush any remaining items in queue
        self._flush_queue()
        self.logger.debug("Output writer thread stopped")
    
    def _writer_loop(self) -> None:
        """Background writer loop that processes the queue."""
        while not self._stop_writer.is_set():
            try:
                result = self._write_queue.get(timeout=1.0)
                self._write_result_atomic(result)
                self._write_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in writer loop: {e}")
    
    def _flush_queue(self) -> None:
        """Flush all remaining items in the queue."""
        while True:
            try:
                result = self._write_queue.get_nowait()
                self._write_result_atomic(result)
                self._write_queue.task_done()
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"Error flushing queue: {e}")
    
    def _write_result_atomic(self, result: InferResult) -> None:
        """
        Write a result to output.jsonl atomically using file locking.
        
        If a record with the same instance_id and n_attempt exists,
        it will be replaced with the new result.
        
        Args:
            result: Inference result to write
        """
        with self._file_lock:
            try:
                # Read existing lines and filter out duplicates
                existing_lines = []
                replaced = False
                
                if self.output_jsonl_path.exists():
                    with open(self.output_jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    # Check if this is a duplicate (same instance_id and n_attempt)
                                    if (data.get("instance_id") == result.instance_id and 
                                        data.get("n_attempt", 1) == result.n_attempt):
                                        # Skip this line (will be replaced)
                                        replaced = True
                                        continue
                                except json.JSONDecodeError:
                                    pass
                                existing_lines.append(line.rstrip('\n'))
                
                # Write back all lines plus the new result
                with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        for line in existing_lines:
                            f.write(line + "\n")
                        f.write(result.to_jsonl() + "\n")
                        f.flush()
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                if replaced:
                    self.logger.debug(f"Replaced existing result for {result.instance_id} attempt {result.n_attempt}")
                else:
                    self.logger.debug(f"Wrote result for {result.instance_id} attempt {result.n_attempt}")
                
            except Exception as e:
                self.logger.error(f"Failed to write result for {result.instance_id}: {e}")
        
        # Update run summary (thread-safe, separate lock)
        self._update_run_summary(result)
    
    def write_result(self, result: InferResult) -> None:
        """
        Queue a result for writing to output.jsonl.
        
        This method is thread-safe and returns immediately.
        The actual write happens in the background writer thread.
        
        Args:
            result: Inference result to write
        """
        self._write_queue.put(result)
    
    def write_result_sync(self, result: InferResult) -> None:
        """
        Write a result synchronously (blocking).
        
        Use this for immediate writes when needed.
        
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
            try:
                # Remove from both lists first (in case of re-run/replacement)
                if infer_log_path in self._run_summary["success"]:
                    self._run_summary["success"].remove(infer_log_path)
                if infer_log_path in self._run_summary["failure"]:
                    self._run_summary["failure"].remove(infer_log_path)
                
                # Add to appropriate list
                if result.success:
                    self._run_summary["success"].append(infer_log_path)
                else:
                    self._run_summary["failure"].append(infer_log_path)
                
                # Save to file atomically
                self._save_run_summary()
                
            except Exception as e:
                self.logger.error(f"Failed to update run summary: {e}")
    
    def _save_run_summary(self) -> None:
        """
        Save the current run summary to file.
        
        Must be called while holding _summary_lock.
        """
        try:
            # Write to a temp file first, then rename for atomicity
            temp_path = self.run_summary_path.with_suffix('.tmp')
            
            with open(temp_path, "w", encoding="utf-8") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self._run_summary, f, ensure_ascii=False, indent=2)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            temp_path.rename(self.run_summary_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save run summary: {e}")
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get the current run summary.
        
        Returns:
            Dictionary with timestamp, success list, and failure list
        """
        with self._summary_lock:
            return dict(self._run_summary)
    
    def save_metadata(self, metadata: RunMetadata) -> None:
        """
        Save run metadata to run_metadata.json.
        
        Args:
            metadata: Run metadata
        """
        with self._file_lock:
            try:
                with open(self.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved run metadata to {self.metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")
    
    def update_metadata_end_time(self) -> None:
        """Update the end_time in run_metadata.json."""
        with self._file_lock:
            try:
                if self.metadata_path.exists():
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    data["end_time"] = datetime.now().isoformat()
                    
                    with open(self.metadata_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    self.logger.info("Updated end_time in metadata")
            except Exception as e:
                self.logger.error(f"Failed to update metadata end_time: {e}")
    
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
        
        try:
            with open(task_paths.patch_path, "w", encoding="utf-8") as f:
                f.write(patch)
            self.logger.debug(f"Saved patch for {task_paths.task_id} attempt {task_paths.attempt}")
        except Exception as e:
            self.logger.error(f"Failed to save patch for {task_paths.task_id}: {e}")
    
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
    
    def load_existing_results(self) -> List[str]:
        """
        Load existing results from output.jsonl.
        
        Returns:
            List of instance IDs that have been processed
        """
        processed_ids = []
        
        if self.output_jsonl_path.exists():
            try:
                with open(self.output_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            processed_ids.append(data.get("instance_id"))
            except Exception as e:
                self.logger.warning(f"Error loading existing results: {e}")
        
        return processed_ids
    
    def load_completed_tasks(self) -> set:
        """
        Load completed tasks from output.jsonl.
        
        Returns:
            Set of (instance_id, n_attempt) tuples that have been completed
        """
        completed = set()
        
        if self.output_jsonl_path.exists():
            try:
                with open(self.output_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            instance_id = data.get("instance_id")
                            n_attempt = data.get("n_attempt", 1)
                            if instance_id and data.get("success", False):
                                completed.add((instance_id, n_attempt))
            except Exception as e:
                self.logger.warning(f"Error loading completed tasks: {e}")
        
        return completed

