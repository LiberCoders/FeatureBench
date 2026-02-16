import json
import difflib
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Set, Union
from filelock import FileLock
import queue
import threading
import atexit
import copy
import traceback
from featurebench.utils.config import Config
from featurebench.mask.signature_extractor import extract_signature


class StorageManager:
	"""Storage manager for local disk persistence."""
	
	# Class-level queue and worker thread (shared by all instances)
	_save_queue = queue.Queue()
	_save_worker_thread = None
	_shutdown_event = threading.Event()
	_worker_lock = threading.Lock()  # Guard _save_worker_thread initialization
	
	def __init__(self, config: Config):
		self.config = config
		self.output_dir = config.actual_output_dir  # Use timestamped output dir
		self.output_dir.mkdir(parents=True, exist_ok=True)
		# File lock for files_status.json
		self._files_status_lock_file = self.output_dir / "metadata_outputs" / ".files_status.lock"
		self._files_status_lock_file.parent.mkdir(parents=True, exist_ok=True)
		# File lock for data_status.json
		self._data_status_lock_file = self.output_dir / "metadata_outputs" / ".data_status.lock"
		self._data_status_lock_file.parent.mkdir(parents=True, exist_ok=True)
		
		# Start background save worker (once)
		self._ensure_save_worker_started()
	
	@classmethod
	def _ensure_save_worker_started(cls):
		"""Ensure the background save worker is started (thread-safe)."""
		with cls._worker_lock:
			if cls._save_worker_thread is None or not cls._save_worker_thread.is_alive():
				cls._shutdown_event.clear()
				cls._save_worker_thread = threading.Thread(
					target=cls._save_worker,
					daemon=False,  # Non-daemon to ensure queued tasks complete
					name="StorageManager-SaveWorker"
				)
				cls._save_worker_thread.start()
				# Register exit handler to wait for queue completion
				atexit.register(cls._shutdown_save_worker)
	
	@classmethod
	def _save_worker(cls):
		"""Background worker: serialize all save operations."""
		while not cls._shutdown_event.is_set():
			try:
				# Get a save task from the queue (0.1s timeout)
				task = cls._save_queue.get(timeout=0.1)
				if task is None:  # None signals shutdown
					cls._save_queue.task_done()  # Required so join() can finish
					break
				
				# Execute save operation
				try:
					task['func'](*task['args'], **task['kwargs'])
				except Exception as e:
					# Log error without killing the worker
					print(f"❌ Save operation failed: {e}")
				finally:
					cls._save_queue.task_done()
			except queue.Empty:
				continue
	
	@classmethod
	def _shutdown_save_worker(cls):
		"""Shut down background save worker."""
		# Prevent duplicate shutdown
		if cls._shutdown_event.is_set():
			return
		
		cls._shutdown_event.set()
		cls._save_queue.put(None)  # Send shutdown signal
		if cls._save_worker_thread and cls._save_worker_thread.is_alive():
			cls._save_worker_thread.join(timeout=5)  # Wait up to 5 seconds
			if cls._save_worker_thread.is_alive():
				print("⚠️ Background worker did not stop within 5s; exiting")
			else:
				pass
		
	def save_config(self, config: Config) -> None:
		"""Save config metadata to metadata.json."""
		# Read and save config file content separately
		config_content = None
		if config.config_path.exists():
			with open(config.config_path, 'r', encoding='utf-8') as f:
				config_content = f.read()
			
			# Save config content to a separate file
			config_backup_file = self.output_dir / "metadata_outputs" / f"config_backup{config.config_path.suffix}"
			config_backup_file.parent.mkdir(parents=True, exist_ok=True)
			with open(config_backup_file, 'w', encoding='utf-8') as f:
				f.write(config_content)
		
		metadata = {
			"config_path": str(config.config_path),
			"output_dir": str(config.output_dir),  # Root output dir provided by user
			"actual_output_dir": str(config.actual_output_dir),  # Actual run output dir
			"resume": config.resume,
			"seed": config.seed,
			"debug_end_stage": config.debug_end_stage,
			"debug_cache_overrides": config.debug_cache_overrides,
			"debug_repo": config.debug_repo,
			"debug_sample": config.debug_sample,
			"log_level": config.log_level,
			"gpu_ids": config.gpu_ids,
			"cmd": config.cmd,  # Full command line
		}
		
		metadata_file = self.output_dir / "metadata_outputs" / "metadata.json"
		# Atomic write: write temp file then rename (avoid Ctrl+C corruption)
		temp_file = metadata_file.with_suffix('.json.tmp')
		with open(temp_file, 'w', encoding='utf-8') as f:
			json.dump(metadata, f, indent=2, ensure_ascii=False)
		temp_file.replace(metadata_file)
	
	def load_files_status(self, specs_name: str) -> dict:
		"""
		Load test file status (from files_status.json).
		
		Args:
			specs_name: Repo spec name
			
		Returns:
			File status data for the repo (p2p_files and f2p_files), or None if missing
		"""
		files_status_file = self.output_dir / "metadata_outputs" / "files_status.json"
		
		if not files_status_file.exists():
			return None
		
		# Guard file reads with a file lock (cross-process safe)
		with FileLock(self._files_status_lock_file, thread_local=False):
			try:
				with open(files_status_file, 'r', encoding='utf-8') as f:
					data = json.load(f)
			except json.JSONDecodeError as e:
				raise RuntimeError(
					f"Failed to parse files_status JSON {files_status_file}: "
					f"line {e.lineno}, col {e.colno} - {e.msg}"
				)
			except Exception as e:
				raise RuntimeError(f"Failed to read files_status file {files_status_file}: {e}")
		
		return data.get(specs_name)
	
	def load_all_test_results(self, repo_names: List[str]) -> Dict[str, Dict[str, List]]:
		"""
		Load test results for the specified repos from disk.

		Args:
			repo_names: Repo name list

		Returns:
			Test results dict: {specs_name: {p2p_files: [...], f2p_files: [...]}}
		"""
		test_results = {}
		for specs_name in repo_names:
			data = self.load_files_status(specs_name)
			if data:
				test_results[specs_name] = data
		return test_results
	
	def save_files_status(
		self,
		specs_name: str,
		p2p_files_with_info: list = None,
		f2p_files_with_info: list = None,
		update_single_file: dict = None
	) -> None:
		"""
		Save test file status locally (updates files_status.json).
		[Async] Enqueue the save task; the background worker processes it serially.

		Supports two modes:
		1. Full save: provide full p2p_files_with_info and f2p_files_with_info (used by test_scanner)
		2. Incremental update: provide update_single_file, format: {'file_path': '...', 'updates': {'passed': True, ...}}

		Args:
			specs_name: Repo spec name
			p2p_files_with_info: Detailed p2p test file list (full save)
			f2p_files_with_info: Detailed f2p test file list (full save)
			update_single_file: Single-file incremental update info
		"""
		if update_single_file:
			# Incremental mode: only pass required updates
			task = {
				'func': self._do_update_single_file_status,
				'args': (specs_name, update_single_file),
				'kwargs': {}
			}
		else:
			# Full save mode: deep copy to avoid mutation
			p2p_snapshot = copy.deepcopy(p2p_files_with_info) if p2p_files_with_info else []
			f2p_snapshot = copy.deepcopy(f2p_files_with_info) if f2p_files_with_info else []
			
			task = {
				'func': self._do_save_files_status,
				'args': (specs_name, p2p_snapshot, f2p_snapshot),
				'kwargs': {}
			}
		
		self._save_queue.put(task)
	
	def _do_update_single_file_status(
		self,
		specs_name: str,
		update_info: dict
	) -> None:
		"""
		Incrementally update a single file status (called by worker thread).
		
		Args:
			specs_name: Repo spec name
			update_info: Update info, format: {'file_path': '...', 'updates': {'passed': True, ...}}
		"""
		metadata_dir = self.output_dir / "metadata_outputs"
		metadata_dir.mkdir(parents=True, exist_ok=True)
		files_status_file = metadata_dir / "files_status.json"
		
		file_path = update_info['file_path']
		updates = update_info['updates']
		
		# Guard read-modify-write with a file lock
		with FileLock(self._files_status_lock_file, thread_local=False, timeout=30):
			try:
				# Load existing data
				existing_data = {}
				if files_status_file.exists():
					try:
						with open(files_status_file, 'r', encoding='utf-8') as f:
							existing_data = json.load(f)
					except json.JSONDecodeError as e:
						print(f"Warning: failed to parse files_status.json; recreating: {e}")
						existing_data = {}
				
				# Ensure repo data structure exists
				if specs_name not in existing_data:
					existing_data[specs_name] = {
						"p2p_files": [],
						"f2p_files": []
					}
				
				# Update matching entry in p2p_files (if any)
				p2p_files = existing_data[specs_name].get('p2p_files', [])
				for file_info in p2p_files:
					if file_info.get('file_path') == file_path:
						# Found file; update fields
						file_info.update(updates)
						break
				
				# Update matching entry in f2p_files (if any)
				f2p_files = existing_data[specs_name].get('f2p_files', [])
				for file_info in f2p_files:
					if file_info.get('file_path') == file_path:
						# Found file; update fields
						file_info.update(updates)
						break
				
				# Atomic write
				temp_file = files_status_file.with_suffix('.json.tmp')
				with open(temp_file, 'w', encoding='utf-8') as f:
					json.dump(existing_data, f, indent=2, ensure_ascii=False)
				temp_file.replace(files_status_file)
				
			except Exception as e:
				print(f"Error: failed to update file status for {specs_name} {file_path}: {e}")
				traceback.print_exc()
				raise
	
	def _do_save_files_status(
		self,
		specs_name: str,
		p2p_files_with_info: list,
		f2p_files_with_info: list
	) -> None:
		"""
		Execute file status save (called by worker thread).
		
		Args:
			specs_name: Repo spec name
			p2p_files_with_info: Detailed p2p test file list
			f2p_files_with_info: Detailed f2p test file list
		"""
		metadata_dir = self.output_dir / "metadata_outputs"
		metadata_dir.mkdir(parents=True, exist_ok=True)

		# Update files_status.json
		files_status_file = metadata_dir / "files_status.json"
		
		# Guard read-modify-write with a file lock (process/thread safe)
		with FileLock(self._files_status_lock_file, thread_local=False, timeout=30):
			try:
				# If file exists, read existing data first
				existing_data = {}
				if files_status_file.exists():
					try:
						with open(files_status_file, 'r', encoding='utf-8') as f:
							existing_data = json.load(f)
					except json.JSONDecodeError as e:
						# File may be corrupted; log and continue (will overwrite)
						print(f"Warning: failed to parse files_status.json; recreating: {e}")
						existing_data = {}
				
				# Update or add data for this repo
				existing_data[specs_name] = {
					"p2p_files": p2p_files_with_info,
					"f2p_files": f2p_files_with_info,
				}
				
				# Atomic write: temp file then rename (avoid Ctrl+C corruption)
				temp_file = files_status_file.with_suffix('.json.tmp')
				with open(temp_file, 'w', encoding='utf-8') as f:
					json.dump(existing_data, f, indent=2, ensure_ascii=False)
				temp_file.replace(files_status_file)
				
			except Exception as e:
				print(f"Error: failed to save files_status for {specs_name}: {e}")
				traceback.print_exc()
				raise
	
	def wait_for_save_completion(self, shutdown: bool = False):
		"""
		Wait for all save tasks to complete.
		Call before exit or when data must be fully persisted.

		Args:
			shutdown: Whether to shut down the background worker after completion (default False)
		"""
		self._save_queue.join()  # Block until all queued tasks finish
		# Only shut down worker when shutdown=True
		if shutdown:
			self._shutdown_save_worker()
	
	def clear_repo_files_status(self, specs_name: str) -> None:
		"""
		Clear files_status data for the specified repo.

		Used when test discovery runs without cache and old data must be removed.

		Args:
			specs_name: Repo spec name
		"""
		files_status_file = self.output_dir / "metadata_outputs" / "files_status.json"
		
		if not files_status_file.exists():
			return
		
		# Guard read-modify-write with a file lock (cross-process safe)
		with FileLock(self._files_status_lock_file, thread_local=False):
			# Load existing data
			with open(files_status_file, 'r', encoding='utf-8') as f:
				existing_data = json.load(f)
			
			# Delete data for the specified repo
			if specs_name in existing_data:
				del existing_data[specs_name]
			
			# Write back file (atomic)
			temp_file = files_status_file.with_suffix('.json.tmp')
			with open(temp_file, 'w', encoding='utf-8') as f:
				json.dump(existing_data, f, indent=2, ensure_ascii=False)
			temp_file.replace(files_status_file)
	
	def load_data_status(self, repo: str, file_path: str) -> Optional[dict]:
		"""
		Load data item processing status (from data_status.json).
		
		Args:
			repo: Repo name
			file_path: Test file path (container path)
			
		Returns:
			Processing status for the data item, or None if missing
			Status contains: success, error, timestamp, etc.
		"""
		data_status_file = self.output_dir / "metadata_outputs" / "data_status.json"
		
		if not data_status_file.exists():
			return None
		
		# Guard file reads with a file lock (cross-process safe)
		with FileLock(self._data_status_lock_file, thread_local=False):
			try:
				with open(data_status_file, 'r', encoding='utf-8') as f:
					data = json.load(f)
			except json.JSONDecodeError as e:
				raise RuntimeError(
					f"Failed to parse data_status JSON {data_status_file}: "
					f"line {e.lineno}, col {e.colno} - {e.msg}"
				)
			except Exception as e:
				raise RuntimeError(f"Failed to read data_status file {data_status_file}: {e}")
		
		# Two-level structure: repo -> file_path
		if repo not in data:
			return None
		return data[repo].get(file_path)
	
	def save_data_status(
		self,
		repo: str,
		file_path: str,
		success_lv1: bool,
		success_lv2: bool,
		error: Optional[Union[str, List[str]]] = None,
		test_count: Optional[int] = None,
		test_count_run: Optional[int] = None,
		last_modified: Optional[str] = None,
		first_commit: Optional[str] = None,
		deleted_lines: Optional[int] = None,
		mask_file_count: Optional[int] = None,
		mask_object_count: Optional[int] = None,
		case_dirs: Optional[Dict[str, Optional[str]]] = None,
		lv1_post_rate: Optional[float] = None,
		lv2_post_rate: Optional[float] = None,
	) -> None:
		"""
		Save data item processing status locally (updates data_status.json).
		[Async] Enqueue the save task; the background worker processes it serially.

		Args:
			repo: Repo name
			file_path: Test file path (container path)
			success_lv1: Whether Level1 processing succeeded
			success_lv2: Whether Level2 processing succeeded
			error: Error list (empty means no error)
			test_count: Test count (discovery stage)
			test_count_run: Test count (run stage)
			last_modified: Last modified time
			first_commit: First commit time
			deleted_lines: Deleted lines count
			mask_file_count: Number of files affected by mask
			mask_object_count: Number of objects affected by mask
			case_dirs: Level1/Level2 case directory paths
			lv1_post_rate: Level1 F2P post-validation pass rate
			lv2_post_rate: Level2 F2P post-validation pass rate
			error: Error list (empty means no error)
		"""
		# Pass copies to avoid caller-side mutations
		normalized_error = self._normalize_error_messages(error)
		normalized_case_dirs = self._normalize_case_dirs(case_dirs)
		# Build save task
		task = {
			'func': self._do_save_data_status,
			'args': (
				repo,
				file_path,
				success_lv1,
				success_lv2,
				normalized_error,
				test_count,
				test_count_run,
				last_modified,
				first_commit,
				deleted_lines,
				mask_file_count,
				mask_object_count,
				normalized_case_dirs,
				lv1_post_rate,
				lv2_post_rate,
			),
			'kwargs': {}
		}
		self._save_queue.put(task)
	
	def _do_save_data_status(
		self,
		repo: str,
		file_path: str,
		success_lv1: bool,
		success_lv2: bool,
		error: Optional[Union[str, List[str]]] = None,
		test_count: Optional[int] = None,
		test_count_run: Optional[int] = None,
		last_modified: Optional[str] = None,
		first_commit: Optional[str] = None,
		deleted_lines: Optional[int] = None,
		mask_file_count: Optional[int] = None,
		mask_object_count: Optional[int] = None,
		case_dirs: Optional[Dict[str, Optional[str]]] = None,
		lv1_post_rate: Optional[float] = None,
		lv2_post_rate: Optional[float] = None,
	) -> None:
		"""
		Execute data status save (called by worker thread).

		Args:
			repo: Repo name
			file_path: Test file path (container path)
			success_lv1: Whether Level1 processing succeeded
			success_lv2: Whether Level2 processing succeeded
			error: Error list (empty means no error)
			test_count: Test count
			test_count_run: Test count (run stage)
			last_modified: Last modified time
			first_commit: First commit time
			deleted_lines: Deleted lines count
			mask_file_count: Number of files affected by mask
			mask_object_count: Number of objects affected by mask
			case_dirs: Level1/Level2 case directory paths
			lv1_post_rate: Level1 F2P post-validation pass rate
			lv2_post_rate: Level2 F2P post-validation pass rate
			error: Error list (empty means no error)
		"""
		error_messages = self._normalize_error_messages(error)
		case_dirs_payload = self._normalize_case_dirs(case_dirs)
		metadata_dir = self.output_dir / "metadata_outputs"
		metadata_dir.mkdir(parents=True, exist_ok=True)

		data_status_file = metadata_dir / "data_status.json"
		
		# Guard read-modify-write with a file lock (cross-process safe)
		with FileLock(self._data_status_lock_file, thread_local=False, timeout=30):
			try:
				# If file exists, read existing data first
				existing_data = {}
				if data_status_file.exists():
					try:
						with open(data_status_file, 'r', encoding='utf-8') as f:
							existing_data = json.load(f)
					except json.JSONDecodeError as e:
						print(f"Warning: failed to parse data_status.json; recreating: {e}")
						existing_data = {}
				
				# Two-level structure: repo -> file_path
				if repo not in existing_data:
					existing_data[repo] = {}
				
				# Update or add status for this item
				existing_data[repo][file_path] = {
					"success_lv1": success_lv1,
					"success_lv2": success_lv2,
					"error": error_messages,
					"timestamp": datetime.now().isoformat(),
					"test_count": test_count,
					"test_count_run": test_count_run,
					"last_modified": last_modified,
					"first_commit": first_commit,
					"deleted_lines": deleted_lines,
					"mask_file_count": mask_file_count,
					"mask_object_count": mask_object_count,
					"case_dirs": case_dirs_payload,
					"lv1_post_rate": lv1_post_rate,
					"lv2_post_rate": lv2_post_rate,
				}
				
				# Atomic write: temp file then rename (avoid Ctrl+C corruption)
				temp_file = data_status_file.with_suffix('.json.tmp')
				with open(temp_file, 'w', encoding='utf-8') as f:
					json.dump(existing_data, f, indent=2, ensure_ascii=False)
				temp_file.replace(data_status_file)
				
			except Exception as e:
				print(f"Error: failed to save data status for {repo} {file_path}: {e}")
				traceback.print_exc()
				raise

	@staticmethod
	def _normalize_error_messages(value: Optional[Union[str, List[str]]]) -> List[str]:
		"""Normalize error payloads into a clean list of strings."""
		if value is None:
			return []
		if isinstance(value, list):
			result: List[str] = []
			for item in value:
				if item is None:
					continue
				text = str(item).strip()
				if text:
					result.append(text)
			return result
		text = str(value).strip()
		return [text] if text else []

	@staticmethod
	def _normalize_case_dirs(value: Optional[Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
		"""Normalize case dir payload into a simple lv1/lv2 mapping of strings."""
		template = {'lv1': None, 'lv2': None}
		if not value:
			return dict(template)
		result = dict(template)
		for key in result.keys():
			if key not in value:
				continue
			path_value = value.get(key)
			if path_value is None:
				result[key] = None
			else:
				result[key] = str(path_value)
		return result
	
	@staticmethod
	def _get_test_file_hash_key(test_file_path: str) -> str:
		"""
		Generate unique test file identifier: test_file_name.hash[:8]

		Used to avoid collisions for same-name tests under a repo.

		Args:
			test_file_path: Test file path (container path, e.g., /testbed/tests/test_model.py)

		Returns:
			str: Unique identifier in format "test_file_name.hash[:8]"
		"""
		test_file_name = Path(test_file_path).stem  # Strip .py suffix
		
		# Hash the relative test file path (same logic as case_converter.py)
		# Strip /testbed/ prefix to get a relative path
		if test_file_path.startswith('/testbed/'):
			relative_path = test_file_path[9:]  # Strip '/testbed/'
		else:
			relative_path = test_file_path
		
		path_hash = hashlib.md5(relative_path.encode()).hexdigest()[:8]
		
		return f"{test_file_name}.{path_hash}"

	@staticmethod
	def _format_full_id_for_display(full_id: str) -> str:
		"""
		Format full object ID into a readable string.

		Args:
			full_id: Full object ID, format: "/testbed/file.py::Class.method::123"

		Returns:
			str: Readable string, e.g. "Class.method (line 123)"
		"""
		parts = full_id.split('::')
		if len(parts) == 3:
			return f"{parts[1]} (line {parts[2]})"
		elif len(parts) >= 2:
			return parts[1]
		else:
			return full_id  # Fallback: return raw string
	
	@staticmethod
	def _format_ids_for_display(obj_ids: List[str]) -> Set[str]:
		"""
		Format full object ID list into a readable set of strings.

		Args:
			obj_ids: Full object IDs, e.g. ["/testbed/file.py::Class.method::123", ...]

		Returns:
			Set[str]: Readable strings, e.g. {'Class.method (line 123)', ...}
			Includes line numbers to disambiguate objects with the same name.
		"""
		return {StorageManager._format_full_id_for_display(obj_id) for obj_id in obj_ids}
	
	def create_dynamic_trace_result_path(
		self,
		specs_name: str,
		test_file: str
	) -> tuple[Path, Path]:
		"""
		Create dynamic trace result paths (normal and collect phases).

		Args:
			specs_name: Repo spec name
			test_file: Test file path (container path)

		Returns:
			(Normal trace result path, collect trace result path)
		"""
		# Create output dir: metadata_outputs/dynamic_trace/{specs_name}/
		dynamic_trace_dir = self.output_dir / "metadata_outputs" / "dynamic_trace" / specs_name
		dynamic_trace_dir.mkdir(parents=True, exist_ok=True)
		
		# Build result filename
		# Use full path to avoid collisions (strip leading slash, replace / with _)
		# Example: /testbed/tests/unit/test_foo.py -> testbed_tests_unit_test_foo.json
		safe_filename = test_file.lstrip('/').replace('/', '_').replace('.py', '.json')
		
		# Normal trace result path
		result_path = dynamic_trace_dir / safe_filename
		
		# Collect trace result path
		collect_safe_filename = safe_filename.replace('.json', '_collect.json')
		collect_result_path = dynamic_trace_dir / collect_safe_filename
		
		return result_path, collect_result_path
	
	def create_code_classification_result_path(
		self,
		specs_name: str,
		test_file: str
	) -> Path:
		"""
		Create code classification result path.

		Args:
			specs_name: Repo spec name
			test_file: Test file path (container path)

		Returns:
			Full path to code classification result
		"""
		# Create output dir: metadata_outputs/code_classification/{specs_name}/
		code_classification_dir = self.output_dir / "metadata_outputs" / "classification" / specs_name
		code_classification_dir.mkdir(parents=True, exist_ok=True)
		
		# Build result filename
		# Use full path to avoid collisions (strip leading slash, replace / with _)
		# Example: /testbed/tests/unit/test_foo.py -> testbed_tests_unit_test_foo.json
		safe_filename = test_file.lstrip('/').replace('/', '_').replace('.py', '.json')
		result_path = code_classification_dir / safe_filename
		
		return result_path

	def create_masked_files_path(
		self,
		repo: str,
		test_file: str
	) -> Path:
		"""
		Create masked files directory path.

		Args:
			repo: Repo name
			test_file: Test file path (container path, e.g., /testbed/tests/test_model.py)

		Returns:
			Full path to masked files directory
		"""
		# Use hash key to avoid same-name test collisions
		test_file_hash_key = self._get_test_file_hash_key(test_file)
		
		# Create output dir: metadata_outputs/masked_files/{repo}/{test_file_hash_key}/
		masked_files_dir = self.output_dir / "metadata_outputs" / "masked_files" / repo / test_file_hash_key
		masked_files_dir.mkdir(parents=True, exist_ok=True)
		
		return masked_files_dir
	
	def save_masked_files(
		self,
		repo: str,
		test_file: str,
		mask_results: dict,
		logger
	) -> None:
		"""
		Save masked files to output directory.

		Args:
			repo: Repo name
			test_file: Test file path (container path)
			mask_results: Mask result dict {file_path: MaskResult}
			logger: Logger
		"""
		# Create output dir (create_masked_files_path computes hash key)
		masked_files_dir = self.create_masked_files_path(repo=repo, test_file=test_file)
		
		# Clear old files in the directory (avoid stale cache)
		if masked_files_dir.exists():
			for old_file in masked_files_dir.glob('*.py'):
				old_file.unlink()
		
		# Save each masked file
		success_count = 0
		for file_path, mask_result in mask_results.items():
			if mask_result.success:
				# Build filename: convert path to a safe name
				# Example: /testbed/src/utils/helper.py -> src-utils-helper.py
				relative_path = file_path.replace('/testbed/', '', 1)
				safe_filename = relative_path.replace('/', '-')
				
				masked_file_path = masked_files_dir / safe_filename
				
				# Write masked code
				with open(masked_file_path, 'w', encoding='utf-8') as f:
					f.write(mask_result.masked_code)
					
				success_count += 1
	
	def save_mask_diffs(
		self,
		repo: str,
		test_file: str,
		mask_results: Dict,
		repo_manager,
		logger
	) -> int:
		"""
		Generate and save diffs before/after masking.

		Args:
			repo: Repo name
			test_file: Test file path (container path)
			mask_results: Mask result dict {file_path: MaskResult}
			repo_manager: Repo manager (for path conversion)
			logger: Logger

		Returns:
			int: Total deleted line count
		"""
		# Use hash key to avoid same-name test collisions
		test_file_hash_key = self._get_test_file_hash_key(test_file)
		
		# Create diff output dir: debug_outputs/mask_diff/{repo}/{test_file_hash_key}/
		diff_dir = self.output_dir / "debug_outputs" / "mask_diff" / repo / test_file_hash_key
		diff_dir.mkdir(parents=True, exist_ok=True)
		
		# Clear old diff files (avoid stale cache)
		if diff_dir.exists():
			for old_file in diff_dir.glob('*.diff'):
				old_file.unlink()
		
		diff_stats = {}

		# Generate diff for each file
		for file_path, mask_result in mask_results.items():
			if not mask_result.success:
				logger.warning(f"Skipping failed mask result: {file_path}")
				continue
			
			try:
				# Read original file content
				host_path = repo_manager.convert_container_path_to_local(repo, file_path)
				with open(host_path, 'r', encoding='utf-8') as f:
					original_code = f.read()
				
				# Generate unified diff
				original_lines = original_code.splitlines(keepends=True)
				masked_lines = mask_result.masked_code.splitlines(keepends=True)
				
				diff_lines = list(
					difflib.unified_diff(
					original_lines,
					masked_lines,
					fromfile=f"a{file_path}",
					tofile=f"b{file_path}",
					lineterm=''
				)
				)
				deleted_line_count = sum(
					1
					for line in diff_lines
					if line.startswith('-') and not line.startswith('---')
				)
				
				# Build diff filename
				relative_path = file_path.replace('/testbed/', '', 1)
				safe_filename = relative_path.replace('/', '-')
				diff_file_path = diff_dir / f"{safe_filename}.diff"
				
				# Write diff (with stats and metadata)
				with open(diff_file_path, 'w', encoding='utf-8') as f:
					# Write metadata header
					f.write(f"# Mask Diff Report\n")
					f.write(f"# Repo: {repo}\n")
					f.write(f"# Test file: {test_file}\n")
					f.write(f"# Original file: {file_path}\n")
					f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
					f.write(f"#\n")
					
					# Extract qualified_name from full IDs for display
					top_obj_names = self._format_ids_for_display(mask_result.top_objects)
					specific_obj_names = self._format_ids_for_display(mask_result.specific_objects)
					
					f.write(f"# Removed top objects ({len(top_obj_names)}):\n")
					for obj in sorted(top_obj_names):
						f.write(f"#   - {obj}\n")
					f.write(f"#\n")
					f.write(f"# Removed specific objects ({len(specific_obj_names)}):\n")
					for obj in sorted(specific_obj_names):
						f.write(f"#   - {obj}\n")
					f.write(f"#\n")
					f.write(f"# Stats:\n")
					f.write(f"#   Original lines: {len(original_lines)}\n")
					f.write(f"#   Masked lines: {len(masked_lines)}\n")
					f.write(f"#   Deleted lines: {deleted_line_count}\n")
					f.write(f"#\n")
					f.write(f"{'=' * 80}\n\n")
					
					# Write unified diff
					diff_content = ''.join(diff_lines)
					if diff_content.strip():
						f.write(diff_content)
					else:
						f.write("# No changes detected (file not modified)\n")

				diff_stats[file_path] = {"deleted_lines": deleted_line_count}
				
				logger.debug(f"Generated diff file: {diff_file_path}")
				
			except Exception as e:
				logger.error(f"Failed to generate diff file for {file_path}: {e}")
		
		# Generate summary report
		self._generate_mask_diff_summary(
			diff_dir=diff_dir,
			repo=repo,
			test_file=test_file,
			mask_results=mask_results,
			repo_manager=repo_manager,
			diff_stats=diff_stats
		)
		
		# Return total deleted line count
		total_deleted_lines = sum(stats.get("deleted_lines", 0) for stats in diff_stats.values())
		return total_deleted_lines
	
	def _generate_mask_diff_summary(
		self,
		diff_dir: Path,
		repo: str,
		test_file: str,
		mask_results: Dict,
		repo_manager,
		diff_stats: Dict[str, Dict[str, int]]
	) -> None:
		"""
		Generate mask diff summary report.

		Args:
			diff_dir: Diff output directory
			repo: Repo name
			test_file: Test file path
			mask_results: Mask result dict
			repo_manager: Repo manager (for path conversion)
		"""
		summary_path = diff_dir / "SUMMARY.md"
		
		with open(summary_path, 'w', encoding='utf-8') as f:
			f.write(f"# Mask Diff Summary\n\n")
			f.write(f"**Repo**: `{repo}`  \n")
			f.write(f"**Test file**: `{test_file}`  \n")
			f.write(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
			
			# Compute overall stats
			total_files = len(mask_results)
			success_files = sum(1 for r in mask_results.values() if r.success)
			
			# Count objects that should be removed (top + specific)
			# Note: full IDs need deduplication before counting
			total_should_remove = sum(
				len(set(r.top_objects + r.specific_objects))
				for r in mask_results.values()
			)
			
			total_deleted_lines = sum(stats.get("deleted_lines", 0) for stats in diff_stats.values())

			f.write(f"## Overall statistics\n\n")
			f.write(f"- **Processed files**: {total_files}\n")
			f.write(f"- **Succeeded**: {success_files}\n")
			f.write(f"- **Failed**: {total_files - success_files}\n")
			f.write(f"- **Total objects to remove**: {total_should_remove}\n")
			f.write(f"- **Total deleted lines**: {total_deleted_lines}\n\n")
			
			f.write(f"## File details\n\n")
			
			# Generate per-file detailed reports
			all_passed = True
			for file_path, mask_result in sorted(mask_results.items()):
				relative_path = file_path.replace('/testbed/', '', 1)
				safe_filename = relative_path.replace('/', '-')
				
				f.write(f"### 📄 `{file_path}`\n\n")
				
				if not mask_result.success:
					f.write(f"**Status**: ❌ Failed  \n")
					f.write(f"**Error**: {mask_result.error_message}  \n\n")
					f.write("---\n\n")
					all_passed = False
					continue
				
				# Get validation result
				validation_result = mask_result.validation_result
				
				# Extract validation result
				expected_to_remove = validation_result.expected_to_remove
				actually_removed = validation_result.actually_removed
				wrongly_removed = validation_result.wrongly_removed
				missed_removal = validation_result.missed_removal
				is_perfect = validation_result.is_perfect
				
				if not is_perfect:
					all_passed = False
				
				# Write status
				if is_perfect:
					f.write(f"**Status**: ✅ Perfect match  \n")
				else:
					f.write(f"**Status**: ⚠️ Partial match  \n")
				
				# Build original file link (file:// URI)
				local_path = repo_manager.convert_container_path_to_local(repo, file_path)
				file_uri = f"file://{local_path}"
				
				# Build masked file path (create_masked_files_path computes hash key)
				masked_files_dir = self.create_masked_files_path(repo=repo, test_file=test_file)
				masked_file_path = masked_files_dir / safe_filename
				masked_file_uri = f"file://{masked_file_path}"
				
				f.write(f"**Original file**: [`{relative_path}`]({file_uri})  \n")
				f.write(f"**Masked file**: [`{safe_filename}`]({masked_file_uri})  \n")
				f.write(f"**Diff file**: [`{safe_filename}.diff`](./{safe_filename}.diff)  \n\n")
				
				# Summary table (object-level, no type split)
				f.write(f"#### Stats\n\n")
				f.write(f"| Expected | Removed | Wrongly removed | Missed | Status |\n")
				f.write(f"|----------|---------|-----------------|--------|--------|\n")
				f.write(f"| {len(expected_to_remove)} | {len(actually_removed)} | {len(wrongly_removed)} | {len(missed_removal)} | {'✅' if is_perfect else '⚠️'} |\n\n")
				
				# Detailed list: objects that should be removed
				if expected_to_remove:
					f.write(f"**Expected to remove ({len(expected_to_remove)})**:  \n")
					for full_id in sorted(expected_to_remove):
						status = "✅" if full_id in actually_removed else "❌"
						display_name = self._format_full_id_for_display(full_id)
						f.write(f"- {status} `{display_name}`\n")
					f.write("\n")
			
				# Show missed removals
				if missed_removal:
					f.write(f"**⚠️ Missed removals ({len(missed_removal)})**:  \n")
					for full_id in sorted(missed_removal):
						display_name = self._format_full_id_for_display(full_id)
						f.write(f"- ❌ `{display_name}`\n")
					f.write("\n")
			
				# Show wrongly removed objects
				if wrongly_removed:
					f.write(f"**🚨 Wrongly removed ({len(wrongly_removed)})**:  \n")
					for full_id in sorted(wrongly_removed):
						display_name = self._format_full_id_for_display(full_id)
						f.write(f"- 🔴 `{display_name}`\n")
					f.write("\n")
			
				# Group top and specific (extract qualified_name)
				top_obj_names = self._format_ids_for_display(mask_result.top_objects)
				specific_obj_names = self._format_ids_for_display(mask_result.specific_objects)
			
				if top_obj_names:
					f.write(f"<details>\n")
					f.write(f"<summary>Top objects ({len(top_obj_names)})</summary>\n\n")
					for obj in sorted(top_obj_names):
						f.write(f"- `{obj}`\n")
					f.write(f"\n</details>\n\n")
			
				if specific_obj_names:
					f.write(f"<details>\n")
					f.write(f"<summary>Specific objects ({len(specific_obj_names)})</summary>\n\n")
					for obj in sorted(specific_obj_names):
						f.write(f"- `{obj}`\n")
					f.write(f"\n</details>\n\n")
			
				f.write("---\n\n")
			
			# Summary
			f.write(f"## Validation results\n\n")
			if all_passed:
				f.write(f"### ✅ All passed\n\n")
				f.write(f"All mask operations are correct; all required objects were removed without omissions.\n\n")
			else:
				f.write(f"### ⚠️ Issues found\n\n")
				f.write(f"Some files have missed removals or failed deletions; see details above.\n\n")
				f.write(f"Check each diff file for concrete code changes.\n\n")
	
	def save_llm_analysis_top_batch(
		self,
		specs_name: str,
		object_ids: List[str],
		test_file_path: str,
		prompt: str,
		llm_response: str
	) -> Path:
		"""
		Save LLM analysis report to debug_outputs/llm_top_classification/{specs_name}/{test_file_hash_key}/.
		Filename is object_id1-object_id2-...-object_idn.md with qualified names.

		Args:
			specs_name: Repo spec name
			object_ids: Object IDs (e.g., ['/testbed/file.py::ObjectName.method::123', ...])
			test_file_path: Test file path (container path)
			prompt: Prompt
			llm_response: Full LLM response

		Returns:
			Path: Saved analysis report path
		"""
		try:
			# Use hash key to avoid same-name test collisions
			test_file_hash_key = self._get_test_file_hash_key(test_file_path)
			
			# Create output dir: debug_outputs/llm_top_classification/{specs_name}/{test_file_hash_key}/
			llm_output_dir = self.output_dir / 'debug_outputs' / 'llm_top_classification' / specs_name / test_file_hash_key
			llm_output_dir.mkdir(parents=True, exist_ok=True)
			
			# Extract object name from object_id
			# Format: /path/to/file.py::ObjectName.method::line_number
			# Use segment [1] split by :: (ObjectName.method)
			object_names = []
			for object_id in object_ids:
				object_names.append(object_id.split('::')[1])

			safe_object_name = '-'.join(object_names)
			
			# If length > 100, truncate and append -truncated
			if len(safe_object_name) > 100:
				safe_object_name = safe_object_name[:100] + '-truncated' if safe_object_name[99] != '-' else safe_object_name[:99] + '-truncated'

			analysis_log_path = llm_output_dir / f"{safe_object_name}.md"
			
			# Write analysis log (Markdown)
			with open(analysis_log_path, "w", encoding="utf-8") as f:
				f.write(f"# LLM Analysis Report\n\n")
				f.write(f"## Basic Info\n\n")
				f.write(f"- **Repo**: `{specs_name}`\n")
				f.write(f"- **Test file**: `{test_file_path}`\n")
				f.write(f"- **Object ID**: \n{'\n'.join([f'- `{object_id}`' for object_id in object_ids])}\n")
				f.write(f"- **Analyzed at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
				f.write(f"## Prompt \n\n")
				f.write(prompt)
				f.write(f"\n\n---\n")
				f.write(f"## Full LLM Response\n\n")
				f.write(llm_response)
				f.write(f"\n\n---\n")
				f.write(f"*Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
			
			return analysis_log_path
			
		except Exception as e:
			raise Exception(f"Error saving LLM batch analysis report: {e}")

	def save_llm_top_docstring(
		self,
		specs_name: str,
		test_file_path: str,
		file_path: str,
		llm_docstring_result: Dict[str, str] # Format: {full_id: docstring}
	) -> None:
		"""
		Save LLM-generated docstrings for top objects to metadata_outputs/llm_docstring/{specs_name}/{test_file_hash_key}/{file_path}.json

		Args:
			specs_name: Repo spec name
			test_file_path: Test file path (container path)
			file_path: File path (container path)
			llm_docstring_result: {full_id: docstring}

		Returns:
			None
		"""
		try:
			# Use hash key to avoid same-name test collisions
			test_file_hash_key = self._get_test_file_hash_key(test_file_path)
			# Convert file_path to a safe filename (strip slashes, use underscores)
			safe_file_name = file_path.lstrip('/').replace('/', '_')
			json_path = self.output_dir / 'metadata_outputs' / 'llm_docstring' / specs_name / test_file_hash_key / f'{safe_file_name}.json'
			
			# Ensure parent directory exists
			json_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Write JSON file
			with open(json_path, "w", encoding="utf-8") as f:
				json.dump(llm_docstring_result, f, indent=4, ensure_ascii=False)
			
			return None
			
		except Exception as e:
			raise Exception(f"Error saving LLM docstring cache: {e}")

	def load_llm_top_docstring(
		self,
		specs_name: str,
		test_file_path: str,
		file_path: str
	) -> Dict[str, str]:
		"""
		Load LLM-generated docstrings from metadata_outputs/llm_docstring/{specs_name}/{test_file_hash_key}/{file_path}.json
		Returns a dict like {full_id: docstring}.
		"""
		# Use hash key to avoid same-name test collisions
		test_file_hash_key = self._get_test_file_hash_key(test_file_path)
		# Convert file_path to a safe filename (strip slashes, use underscores)
		safe_file_name = file_path.lstrip('/').replace('/', '_')
		json_path = self.output_dir / 'metadata_outputs' / 'llm_docstring' / specs_name / test_file_hash_key / f'{safe_file_name}.json'
		if not json_path.exists():
			return {}
		try:
			with open(json_path, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception as e:
			return {}

	def save_llm_task_statement(
		self,
		specs_name: str,
		test_file_path: str,
		llm_task_statement: str
	) -> None:
		"""
		Save LLM-generated task statement to metadata_outputs/llm_task_statement/{specs_name}/{test_file_hash_key}.txt

		Args:
			specs_name: Repo spec name
			test_file_path: Test file path (container path)
			llm_task_statement: LLM-generated task statement

		Returns:
			None
		"""
		try:
			# Use hash key to avoid same-name test collisions
			test_file_hash_key = self._get_test_file_hash_key(test_file_path)
			task_statement_path = self.output_dir / 'metadata_outputs' / 'llm_task_statement' / specs_name / f'{test_file_hash_key}.txt'
			
			# Ensure parent directory exists
			task_statement_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Write file
			with open(task_statement_path, "w", encoding="utf-8") as f:
				f.write(llm_task_statement)
			
			return None
			
		except Exception as e:
			raise Exception(f"Error saving LLM task statement cache: {e}")

	def save_llm_task_statement_prompt(
		self,
		specs_name: str,
		test_file_path: str,
		prompt: str,
	) -> Path:
		"""Save prompt used for LLM task statements (debugging)."""
		try:
			key = self._get_test_file_hash_key(test_file_path)
			dir_path = self.output_dir / "debug_outputs" / "llm_task_statement" / specs_name
			dir_path.mkdir(parents=True, exist_ok=True)
			prompt_path = dir_path / f"{key}.txt"
			with open(prompt_path, "w", encoding="utf-8") as f:
				f.write(prompt)
			return prompt_path
		except Exception as e:  # noqa: BLE001
			raise Exception(f"Error saving LLM task statement prompt: {e}")

	def load_llm_task_statement(
		self,
		specs_name: str,
		test_file_path: str,
	) -> str:
		"""
		Load LLM-generated task statement from metadata_outputs/llm_task_statement/{specs_name}/{test_file_hash_key}.txt
		Returns the task statement string.
		"""
		# Use hash key to avoid same-name test collisions
		test_file_hash_key = self._get_test_file_hash_key(test_file_path)
		task_statement_path = self.output_dir / 'metadata_outputs' / 'llm_task_statement' / specs_name / f'{test_file_hash_key}.txt'
		if not task_statement_path.exists():
			return ''
		try:
			with open(task_statement_path, "r", encoding="utf-8") as f:
				return f.read()
		except Exception as e:
			return ''
