"""Dynamic tracer manager for running traces in Docker containers."""

import json
import logging
import subprocess
import os
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

from acebench.utils.logger import print_build_report, create_dynamic_tracer_logger
from acebench.utils.command_utils import apply_uv_run_prefix
from acebench.classification.file_analyzer import FunctionClassVisitor


class DynamicTracer:
	"""Run dynamic tracing in Docker containers and record results."""
	
	def __init__(
		self,
		config,
		repo_manager,
		image_manager,
		storage_manager,
		logger: Optional[logging.Logger] = None
	):
		"""
		Initialize the dynamic tracer.
		
		Args:
			config: Config object (for logs_dir, etc.)
			repo_manager: Repo manager
			image_manager: Image manager
			storage_manager: Storage manager
			logger: Logger instance
		"""
		self.config = config
		self.repo_manager = repo_manager
		self.image_manager = image_manager
		self.storage_manager = storage_manager
		self.logs_dir = config.logs_dir
		self.logger = logger if logger is not None else logging.getLogger(__name__)
		
		# Load test results from file
		repo_names = list(self.repo_manager.loaded_repos.keys())
		self.test_results: Dict[str, Dict[str, List]] = self.storage_manager.load_all_test_results(repo_names)
		
		# Store dynamic trace results: {specs_name: {test_file: trace_result_path}}
		self.trace_results: Dict[str, Dict[str, str]] = {}

		# Track missing files in container to avoid repeated copy attempts
		self._missing_container_files: Set[Tuple[str, str]] = set()
		# Track known non-Python files to avoid repeated warnings
		self._non_python_files: Set[str] = set()
		# Track ignored files (e.g., .venv) to avoid repeated warnings
		self._ignored_paths: Set[str] = set()
	
	def run(self, max_workers: Optional[int] = None) -> None:
		"""
		Run dynamic tracing in parallel.
		
		Args:
			max_workers: Max parallel workers; auto-select if None
		"""
		self.logger.info("")
		self.logger.info("=" * 60)
		self.logger.info("Starting dynamic tracing stage (only for passed files) ...")
		self.logger.info("=" * 60)
		
		# Set default parallelism
		if max_workers is None:
			max_workers = min(len(self.repo_manager.loaded_repos), os.cpu_count() or 1, 5)
		
		self.logger.info(f"Running dynamic tracing for {len(self.repo_manager.loaded_repos)} repos with {max_workers} workers")
		
		# Count passed test files (trace only passed tests)
		passed_files_by_repo = {}
		total_test_files = 0
		for specs_name in self.repo_manager.loaded_repos.keys():
			if specs_name in self.test_results:
				p2p_files = self.test_results[specs_name].get('p2p_files', [])
				passed_files = [f for f in p2p_files if f.get('passed', False)]
				if self.config.debug_sample:
					passed_files = [
						f for f in passed_files
						if self.config.is_sample_selected(f.get('file_path', ''))
					]
				passed_files_by_repo[specs_name] = passed_files
				total_test_files += len(passed_files)
		
		self.logger.info(f"Total test files: {total_test_files}")
		
		# Failed repo info
		failed_repos: List[Tuple[str, str]] = []
		
		# Shared progress bar
		pbar = tqdm(total=total_test_files, desc="Dynamic tracing", unit="file")
		
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			# Submit all dynamic tracing tasks
			future_to_repo = {}
			for specs_name, repo_info in self.repo_manager.loaded_repos.items():
				specs = repo_info['specs']
				# Override cache config with config.get_cache_config (debug supported)
				use_cache = self.config.get_cache_config('dynamic', specs.get('dynamic_cache', True))
				
				future = executor.submit(
					self._trace_repo_tests,
					specs_name,
					specs,
					use_cache,
					pbar
				)
				future_to_repo[future] = specs_name
			
			# Track progress
			completed_repos = 0
			total_repos = len(self.repo_manager.loaded_repos)
			
			# Handle completed tasks
			for future in as_completed(future_to_repo):
				specs_name = future_to_repo[future]
				completed_repos += 1
				
				try:
					traced_count = future.result()
					
					pbar.set_postfix_str(f"Latest success: {specs_name}")
					# tqdm.write(f"✅ {specs_name}: tracing done (files={traced_count}) (repo progress: {completed_repos}/{total_repos})")
					
				except Exception as e:
					# Only collect failures; errors already logged inside
					failed_repos.append((specs_name, str(e)))
					
					pbar.set_postfix_str(f"Latest failure: {specs_name}")
		
		pbar.close()
		
		# Wait for all save tasks to finish
		self.storage_manager.wait_for_save_completion(shutdown=False)
		
		# Reload latest data for report generation
		repo_names = list(self.test_results.keys())
		self.test_results = self.storage_manager.load_all_test_results(repo_names)
		
		# Update passed_files_by_repo
		for specs_name, latest_data in self.test_results.items():
			p2p_files = latest_data.get('p2p_files', [])
			passed_files_by_repo[specs_name] = [f for f in p2p_files if f.get('passed', False)]
		
		# Print dynamic tracing report
		success_info = {}
		for specs_name in self.trace_results.keys():
			passed_p2p_files = passed_files_by_repo.get(specs_name, [])
			total = len(passed_p2p_files)
			
			# Count success (dynamic_trace_file present) vs failure (None)
			traced = sum(1 for f in passed_p2p_files if f.get('dynamic_trace_file'))
			failed = sum(1 for f in passed_p2p_files if 'dynamic_trace_file' in f and f['dynamic_trace_file'] is None)
			
			success_info[specs_name] = {
				'Traced': traced,
				'Failed': failed,
				'Total': total,
				'Success rate': f"{traced/total*100:.1f}%" if total > 0 else "N/A"
			}
		
		print_build_report(
			logger=self.logger,
			total_repos=list(self.repo_manager.loaded_repos.items()),
			failed_repos=failed_repos,
			success_info=success_info,
			operation_name="Dynamic tracing"
		)
		
		# Raise if any repo failed
		if failed_repos:
			failed_names = [name for name, _ in failed_repos]
			raise RuntimeError(f"Dynamic tracing failed for repos: {', '.join(failed_names)}")
		
		self.logger.info("Dynamic tracing stage completed")
	
	def _trace_repo_tests(
		self,
		specs_name: str,
		specs: Dict,
		use_cache: bool = True,
		pbar: Optional[tqdm] = None
	) -> int:
		"""
		Run dynamic tracing for all test files in a repo.
		
		Args:
			specs_name: Repo spec name
			specs: Repo specs config
			use_cache: Whether to use cache
			pbar: Progress bar
			
		Returns:
			int: number of traced files
		"""
		# tqdm.write(f"🏃 Start dynamic tracing for {specs_name}...")
		
		# Get test file list
		if specs_name not in self.test_results:
			tqdm.write(f"❌ {specs_name}: test discovery results not found; run discovery first")
			raise RuntimeError(f"{specs_name}: test discovery results not found; run discovery first")
		
		# Trace only test files that passed
		all_p2p_files = self.test_results[specs_name].get('p2p_files', [])
		p2p_files = [f for f in all_p2p_files if f.get('passed', False)]
		if self.config.debug_sample:
			p2p_files = [
				f for f in p2p_files
				if self.config.is_sample_selected(f.get('file_path', ''))
			]
		
		if not p2p_files:
			if self.config.debug_sample:
				tqdm.write(f"🔍 {specs_name}: no traceable files after debug_sample filter; skipping")
				return 0
			tqdm.write(f"⚠️ {specs_name}: no passed p2p test files found")
			raise RuntimeError(f"⚠️ {specs_name}: no passed p2p test files found")
		
		# Get dynamic tracing command and timeout
		dynamic_cmd = specs.get("test_dynamic_cmd")
		timeout_dynamic = specs.get("timeout_dynamic", 600)
		if isinstance(timeout_dynamic, (int, float)) and timeout_dynamic < 0:
			# Allow timeout_dynamic = -1 to mean no limit
			timeout_dynamic = None
		
		# Initialize results dict
		self.trace_results[specs_name] = {}
		
		# Set parallelism
		max_workers = min(len(p2p_files), os.cpu_count() or 1, 10)
		
		traced_count = 0
		
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			future_to_file = {}
			
			for file_info in p2p_files:
				file_path = file_info['file_path']
				
				# Check cache: if dynamic_trace_file exists and has value, skip
				if use_cache and 'dynamic_trace_file' in file_info:
					# Trace result already exists; skip
					self.trace_results[specs_name][file_path] = file_info['dynamic_trace_file']
					traced_count += 1
					# Update progress bar
					if pbar:
						pbar.update(1)
					continue
				
				future = executor.submit(
					self._trace_single_test,
					specs_name,
					file_path,
					dynamic_cmd,
					timeout_dynamic
				)
				future_to_file[future] = file_info
			
			# Collect results
			for future in as_completed(future_to_file):
				file_info = future_to_file[future]
				file_path = file_info['file_path']
				
				try:
					result_path = future.result()
					
					if result_path:
						self.trace_results[specs_name][file_path] = result_path
						traced_count += 1
					
				except Exception:
					# Errors already logged in _trace_single_test; only record status
					result_path = None
				
				# Save directly to file (incremental); background thread updates p2p/f2p
				self.storage_manager.save_files_status(
					specs_name=specs_name,
					update_single_file={
						'file_path': file_path,
						'updates': {'dynamic_trace_file': result_path}
					}
				)
				
				# Update progress bar
				if pbar:
					pbar.update(1)
		
		return traced_count
	
	def _trace_single_test(
		self,
		specs_name: str,
		test_file: str,
		dynamic_cmd: str,
		timeout: int
	) -> Optional[str]:
		"""
		Run dynamic tracing for a single test file.
		
		Args:
			specs_name: Repo spec name
			test_file: Test file path (container path)
			dynamic_cmd: Dynamic trace command args
			timeout: Timeout in seconds
			
		Returns:
			Optional[str]: Trace result path, or None on failure
		"""
		container_id = None
		is_timeout = False
		specs = self.repo_manager.loaded_repos.get(specs_name, {}).get("specs", {})
		
		# Create log file paths (normal and collect)
		log_file_path, collect_log_file_path = create_dynamic_tracer_logger(
			specs_name=specs_name,
			test_file=test_file,
			logs_dir=self.logs_dir
		)
		
		# Resolve dynamic tracing script directory
		dynamic_script_dir = Path(__file__).parent.parent / "resources" / "scripts" / "dynamic_script"

		try:
			# Use storage_manager to create result paths (normal and collect)
			result_path, collect_result_path = self.storage_manager.create_dynamic_trace_result_path(
				specs_name=specs_name,
				test_file=test_file
			)
			
			# tqdm.write(f"🏃 {specs_name}: trace {test_file} | 📝 log: {log_file_path} | 💾 result: {result_path}")
			
			# 1. Start container
			container_id = self.image_manager.run_container(
				specs_name=specs_name,
				working_dir="/testbed",
				prepare_env=True
			)
			
			# 2. Copy dynamic tracing scripts into container
			for script_file in dynamic_script_dir.glob("*.py"):
				self.image_manager.copy_to_container(
					container_id,
					str(script_file),
					f"/tmp/{script_file.name}"
				)
			
			# 3. Run normal tracing (pytest execution)
			trace_cmd = f"python /tmp/dynamic_script.py {test_file} /testbed -o /tmp/trace_result.json --pytest-args {dynamic_cmd}"
			trace_cmd = apply_uv_run_prefix(trace_cmd, specs)

			result = self.image_manager.exec_in_container(
				container_id,
				trace_cmd,
				timeout=timeout,
				log_file_path=log_file_path
			)
			
			if result.returncode != 0:
				raise RuntimeError(f"Dynamic tracing command exit code {result.returncode}")

			# 4. Copy normal trace result file from container
			self.image_manager.copy_from_container(
				container_id,
				"/tmp/trace_result.json",
				str(result_path)
			)

			# 5. Convert result_path object IDs to qualified names
			self._convert_trace_to_qualified_names(result_path, specs_name, container_id)
			
			# 6. Run collect-phase tracing (pytest --collect-only)
			collect_trace_cmd = f"python /tmp/dynamic_script.py {test_file} /testbed -o /tmp/trace_result_collect.json --pytest-args '--collect-only'"
			collect_trace_cmd = apply_uv_run_prefix(collect_trace_cmd, specs)
			
			collect_result = self.image_manager.exec_in_container(
				container_id,
				collect_trace_cmd,
				timeout=timeout,
				log_file_path=str(collect_log_file_path)
			)
			
			if collect_result.returncode != 0:
				raise RuntimeError(f"Collect-phase tracing command exit code {collect_result.returncode}")

			# 7. Copy collect trace result file from container
			self.image_manager.copy_from_container(
				container_id,
				"/tmp/trace_result_collect.json",
				str(collect_result_path)
			)
			
			# 8. Convert collect_result_path object IDs to qualified names
			self._convert_trace_to_qualified_names(collect_result_path, specs_name, container_id)

			# tqdm.write(f"✅ {specs_name}: trace done {test_file} | 💾 result: {result_path}, collect: {collect_result_path}")
			
			return str(result_path)
		
		except subprocess.TimeoutExpired:
			is_timeout = True
			error_msg = f"Dynamic tracing timed out ({timeout}s), logs: {log_file_path}, {collect_log_file_path}"
			tqdm.write(f"⏱️ {specs_name}: {error_msg}")
			raise RuntimeError(error_msg)
		
		except Exception as e:
			error_msg = f"Dynamic tracing error - {test_file}: {e}, logs: {log_file_path}, {collect_log_file_path}"
			tqdm.write(f"❌ {specs_name}: {error_msg}")
			raise RuntimeError(error_msg) from e
		
		finally:
			# Cleanup container (force kill on timeout)
			if container_id:
				self.image_manager.stop_container(container_id, force=is_timeout)
				# tqdm.write(f"🗑️  {specs_name}: cleaned container {container_id[:12]}")
	
	def _convert_trace_to_qualified_names(
		self,
		result_path: str,
		specs_name: str,
		container_id: Optional[str] = None
	) -> None:
		"""
		Convert object IDs in trace results from simple to qualified names.
		
		Args:
			result_path: Dynamic trace result path
			specs_name: Repo spec name
		"""
		try:
			# Read dynamic trace results
			with open(result_path, 'r', encoding='utf-8') as f:
				trace_data = json.load(f)
			
			objects = trace_data.get('objects', {})
			if not objects:
				return
			
			# Convert object IDs to qualified names
			new_objects = self._convert_to_qualified_names(objects, specs_name, container_id)
			
			# Update and save
			trace_data['objects'] = new_objects
			with open(result_path, 'w', encoding='utf-8') as f:
				json.dump(trace_data, f, indent=2, ensure_ascii=False)
				
		except Exception as e:
			# Conversion failure should not block main flow; log warning only
			tqdm.write(f"⚠️ {specs_name}: Failed to convert qualified names: {e}")
	
	def _convert_to_qualified_names(
		self,
		objects: Dict[str, Dict[str, Any]],
		specs_name: str,
		container_id: Optional[str] = None
	) -> Dict[str, Dict[str, Any]]:
		"""
		Convert object IDs from simple to qualified names.
		
		Trace format: /path/file.py::SimpleName::LineNumber
		Converted:   /path/file.py::QualifiedName::LineNumber
		
		Example: /path/file.py::forward::120 -> /path/file.py::MyModel.forward::120
		
		Args:
			objects: Dynamic trace objects dict
			specs_name: Repo spec name
			
		Returns:
			Objects dict with qualified names
		"""
		# Group objects by file
		objects_by_file = {}
		for obj_id, obj_info in objects.items():
			file_path = obj_info['file']
			if file_path not in objects_by_file:
				objects_by_file[file_path] = []
			objects_by_file[file_path].append((obj_id, obj_info))
		
		# Objects after replacing with qualified names
		new_objects = {}

		# Process file by file
		for file_path, file_objects in objects_by_file.items():
			# Get local path for this file
			host_path = self.repo_manager.convert_container_path_to_local(specs_name, file_path)
			host_path_str = str(host_path)
			
			try:
				# e.g., Class.Func::lineno -> (start_line, end_line, type)
				definitions = self._analyze_single_file(
					host_path=host_path_str,
					container_path=file_path,
					specs_name=specs_name,
					container_id=container_id
				)
				if not definitions:
					# File parse failed; keep as-is
					for obj_id, obj_info in file_objects:
						new_objects[obj_id] = obj_info
					continue
				
				# Build mapping: (simple_name, lineno) -> qualified_name
				name_lineno_map = {}
				for qualified_key in definitions.keys():
					# qualified_key format: "qualified_name::lineno"
					parts = qualified_key.rsplit('::', 1)
					if len(parts) == 2:
						qualified_name, lineno_str = parts
						try:
							lineno = int(lineno_str)
							simple_name = qualified_name.split('.')[-1]
							name_lineno_map[(simple_name, lineno)] = qualified_name
						except ValueError:
							pass
				
				# Convert each object ID
				for obj_id, obj_info in file_objects:
					# obj_id format: "/path/file.py::SimpleName::LineNumber"
					parts = obj_id.split('::')
					if len(parts) >= 3:
						file_part = parts[0]
						simple_name = parts[1]
						lineno_str = parts[2]
						try:
							lineno = int(lineno_str)
							# Look up qualified name
							if (simple_name, lineno) in name_lineno_map:
								qualified_name = name_lineno_map[(simple_name, lineno)]
								new_obj_id = f"{file_part}::{qualified_name}::{lineno_str}"
								# Update obj_info name to qualified name
								new_obj_info = dict(obj_info)
								new_obj_info['name'] = qualified_name
								new_objects[new_obj_id] = new_obj_info
							else:
								# Mapping missing; keep as-is
								new_objects[obj_id] = obj_info
						except ValueError:
							# Line number parse failed; keep as-is
							new_objects[obj_id] = obj_info
					else:
						# Invalid format; keep as-is
						new_objects[obj_id] = obj_info
			except Exception as e:
				# File processing failed; keep as-is
				if self.logger:
					self.logger.warning(f"Failed to convert qualified names for file {file_path}: {e}")
				for obj_id, obj_info in file_objects:
					new_objects[obj_id] = obj_info
		
		# Build global mapping from old IDs to new IDs
		old_to_new_id_map = {}
		for new_id in new_objects.keys():
			# Try to derive old ID from new ID
			parts = new_id.split('::')
			if len(parts) >= 3:
				file_part = parts[0]
				qualified_name = parts[1]
				lineno_str = parts[2]
				simple_name = qualified_name.split('.')[-1]
				old_id = f"{file_part}::{simple_name}::{lineno_str}"
				old_to_new_id_map[old_id] = new_id
		
		# Update all object edges, replacing old IDs with new IDs
		for obj_id, obj_info in new_objects.items():
			new_edges = []
			for edge_id in obj_info.get('edges', []):
				# If edge_id is mapped, use new ID; else keep as-is
				new_edge_id = old_to_new_id_map.get(edge_id, edge_id)
				new_edges.append(new_edge_id)
			obj_info['edges'] = new_edges
		
		return new_objects
	
	def _analyze_single_file(
		self,
		host_path: str,
		container_path: str,
		specs_name: str,
		container_id: Optional[str] = None
	) -> Optional[Dict]:
		"""
		Analyze a single file and extract class/function definitions.
		
		Args:
			host_path: File path on host
			container_path: File path in container
			specs_name: Repo spec name
			container_id: Current container ID (optional)
			
		Returns:
			Definitions dict: {qualified_name::lineno: (start_line, end_line, type)}
		"""
		host_path_obj = Path(host_path)

		# Skip .venv files entirely
		ignore_key = container_path or str(host_path_obj)
		if self._is_ignored_path(ignore_key):
			if ignore_key not in self._ignored_paths:
				self._ignored_paths.add(ignore_key)
				# tqdm.write(f"⚠️ {specs_name}: Skipping .venv file: {ignore_key}")
			return None
		missing_key = (specs_name, container_path)
		# If file missing, copy from container (avoid repeated attempts)
		if (not host_path_obj.exists()
			and container_id
			and container_path
			and missing_key not in self._missing_container_files):
			copied = self._ensure_file_from_container(
				container_id=container_id,
				container_path=container_path,
				host_path=host_path_obj,
				specs_name=specs_name
			)
			if not copied:
				self._missing_container_files.add(missing_key)
				tqdm.write(f"❌ Failed to read file {host_path}: missing and cannot copy from container")
				return None
		elif not host_path_obj.exists() and missing_key in self._missing_container_files:
			# Already confirmed unavailable; return to avoid repeated logs
			return None

		# Do not parse AST for non-.py files
		if host_path_obj.suffix.lower() != '.py':
			cache_key = container_path or str(host_path_obj)
			if cache_key not in self._non_python_files:
				self._non_python_files.add(cache_key)
				tqdm.write(f"⚠️ {specs_name}: Skipping AST analysis for non-.py file: {container_path or host_path_obj}")
			return None
		
		try:
			with open(host_path_obj, 'r', encoding='utf-8') as f:
				content = f.read()
		except Exception as e:
			if missing_key not in self._missing_container_files:
				tqdm.write(f"❌ Failed to read file {host_path}: {e}")
				self._missing_container_files.add(missing_key)
			return None
			
		try:
			tree = ast.parse(content)
		except SyntaxError as e:
			tqdm.write(f"❌ Failed to parse AST for {host_path}: {e}")
			return None
			
		visitor = FunctionClassVisitor()
		visitor.visit(tree)
		
		return visitor.definitions

	def _ensure_file_from_container(
		self,
		container_id: str,
		container_path: str,
		host_path: Path,
		specs_name: str
	) -> bool:
		"""Copy container files to host for later AST analysis."""
		try:
			host_path.parent.mkdir(parents=True, exist_ok=True)
			self.image_manager.copy_from_container(
				container_id=container_id,
				src_path=container_path,
				dest_path=str(host_path)
			)
			self._missing_container_files.discard((specs_name, container_path))
			return True
		except subprocess.CalledProcessError as e:
			return False
		except Exception as e:
			return False

	def _is_ignored_path(self, path_str: str) -> bool:
		"""Return True if the path should be ignored (e.g., .venv)."""
		try:
			parts = Path(path_str).parts
			return ".venv" in parts
		except Exception:
			return "/.venv/" in path_str
