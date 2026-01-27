"""Test discovery manager for scanning tests in Docker containers."""

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile

from acebench.utils.logger import print_build_report, create_test_scanner_logger
from acebench.utils.command_utils import apply_uv_run_prefix


class TestScanner:
	"""Scan project test files in Docker containers."""
	
	def __init__(
		self,
		config,
		repo_manager,
		image_manager,
		storage_manager,
		logger: Optional[logging.Logger] = None
	):
		"""
		Initialize the test scanner.
		
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
		
		# Store test discovery results: {specs_name: {p2p_files, f2p_files}}
		self.test_results: Dict[str, Dict[str, List]] = {}
	
	def run(self, max_workers: Optional[int] = None) -> None:
		"""
		Run test discovery in parallel across loaded repos.
		
		Args:
			max_workers: Max parallel workers; auto-select if None
		"""
		self.logger.info("")
		self.logger.info("=" * 60)
		self.logger.info("Starting test discovery stage...")
		self.logger.info("=" * 60)
		
		# Clear data for repos that disable cache
		repos_to_clear = []
		for specs_name, repo_info in self.repo_manager.loaded_repos.items():
			specs = repo_info['specs']
			use_cache = self.config.get_cache_config('scanner', specs.get('scan_cache', True))
			if not use_cache:
				repos_to_clear.append(specs_name)
		
		if repos_to_clear:
			self.logger.info(f"Cache disabled for repos, clearing data: {', '.join(repos_to_clear)}")
			for specs_name in repos_to_clear:
				self.storage_manager.clear_repo_files_status(specs_name)
		
		# Set default parallelism
		if max_workers is None:
			max_workers = min(len(self.repo_manager.loaded_repos), os.cpu_count() or 1, 5)  # Max 5 parallel
		
		self.logger.info(
			f"Scanning {len(self.repo_manager.loaded_repos)} repos with {max_workers} parallel workers"
		)
		
		# Failed repo info
		failed_repos: List[Tuple[str, str]] = []  # (repo_name, error_message)
		
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			# Submit all discovery tasks
			future_to_repo = {}
			for specs_name, repo_info in self.repo_manager.loaded_repos.items():
				specs = repo_info['specs']
				# Override cache config with config.get_cache_config (debug supported)
				use_cache = self.config.get_cache_config('scanner', specs.get('scan_cache', True))
				
				future = executor.submit(
					self._scan_tests,
					specs_name,
					specs,
					use_cache
				)
				future_to_repo[future] = specs_name
			
			# Track progress
			completed_count = 0
			total_count = len(self.repo_manager.loaded_repos)
			
			# Use progress bar for completed tasks
			with tqdm(total=total_count, desc="Test discovery", unit="repo") as pbar:
				for future in as_completed(future_to_repo):
					specs_name = future_to_repo[future]
					completed_count += 1
					
					try:
						p2p_files_with_info, f2p_files_with_info = future.result()
						
						# Save results
						self.test_results[specs_name] = {
							'p2p_files': p2p_files_with_info,
							'f2p_files': f2p_files_with_info
						}
						
						# Update progress bar
						pbar.update(1)
						pbar.set_postfix_str(f"Latest completed: {specs_name}")
						tqdm.write(
							f"✅ {specs_name}: test discovery complete "
							f"(P2P={len(p2p_files_with_info)}, F2P={len(f2p_files_with_info)}) "
							f"({completed_count}/{total_count})"
						)
						
					except Exception as e:
						# Only collect failures; errors already logged inside
						failed_repos.append((specs_name, str(e)))
						
						# Update progress bar
						pbar.update(1)
						pbar.set_postfix_str(f"Latest failed: {specs_name}")
		
		# Print discovery report and convert result format
		success_info = {}
		for specs_name, result in self.test_results.items():
			success_info[specs_name] = {
				'P2P file count': len(result['p2p_files']),
				'F2P file count': len(result['f2p_files'])
			}
		print_build_report(
			logger=self.logger,
			total_repos=list(self.repo_manager.loaded_repos.items()),
			failed_repos=failed_repos,
			success_info=success_info,
			operation_name="Test discovery"
		)
		
		# Raise if any repo failed
		if failed_repos:
			failed_names = [name for name, _ in failed_repos]
			raise RuntimeError(f"Test discovery failed for repos: {', '.join(failed_names)}")
		
		self.logger.info("Test discovery stage completed")
	
	def _scan_tests(
		self,
		specs_name: str,
		specs: Dict,
		use_cache: bool = True
	) -> Tuple[List[Dict], List[Dict]]:
		"""
		Scan and filter test files.
		
		Args:
			specs_name: Repo spec name
			specs: Repo specs config
			use_cache: Whether to use cache (default True)
			
		Returns:
			(p2p_files_with_info, f2p_files_with_info)
			- p2p_files_with_info: detailed info list for p2p tests
			- f2p_files_with_info: detailed info list for f2p tests
		"""
		tqdm.write(f"🏃 Starting test discovery for {specs_name}...")
		
		# Check cache usage
		if use_cache:
			cached_result = self.storage_manager.load_files_status(specs_name)
			if cached_result:
				tqdm.write(f"⏭️ {specs_name}: using cached test discovery results")
				return cached_result['p2p_files'], cached_result['f2p_files']
		
		# Run test discovery
		# Data format: {file_path: {test_count: int, test_points: List[str]}}
		test_files_info, log_file_path = self._run_test_discovery(specs_name, specs)

		# Filter test files
		p2p_files_with_info, f2p_files_with_info = self._filter_test_files(specs_name, specs, test_files_info, log_file_path)
		
		# Save results to local storage
		self.storage_manager.save_files_status(specs_name, p2p_files_with_info, f2p_files_with_info)
		
		return p2p_files_with_info, f2p_files_with_info
	
	def _run_test_discovery(
		self,
		specs_name: str,
		specs: Dict,
	) -> Tuple[Dict[str, Dict], Path]:
		"""
		Run test discovery inside a Docker container.
		
		Args:
			specs_name: Repo spec name
			specs: Repo specs config
			
		Returns:
			(test_files_info, log_file_path)
			- test_files_info: {file_path: {test_count: int, test_points: List[str]}}
			- log_file_path: Log file path
		"""
		# Get discovery command and timeout
		test_scanner_cmd = apply_uv_run_prefix(specs.get("test_scanner_cmd"), specs)
		timeout_scanner = specs.get("timeout_scanner", 300)
		if isinstance(timeout_scanner, (int, float)) and timeout_scanner < 0:
			# Allow timeout_scanner = -1 to mean no limit
			timeout_scanner = None
		
		# Get discovery script path
		scanner_script_path = Path(__file__).parent.parent / "resources" / "scripts" / "scanner_script.py"
		
		# Use temp directory for results
		with tempfile.TemporaryDirectory(prefix="test_scanner_") as temp_dir:
			result_file = Path(temp_dir) / "test_results.json"
			
			container_id = None
			is_timeout = False  # Mark timeout
			try:
				# Create log file
				log_file_path = create_test_scanner_logger(specs_name, logs_dir=self.logs_dir)
				tqdm.write(f"📝 {specs_name}: test scan logs saved to {log_file_path}")
				
				# 1. Start container
				container_id = self.image_manager.run_container(
					specs_name=specs_name,
					working_dir="/testbed",
					prepare_env=True
				)
				
				# 2. Copy discovery script into container
				self.image_manager.copy_to_container(
					container_id,
					str(scanner_script_path),
					"/tmp/scanner_script.py"
				)
				
				# 3. Run discovery script (streaming logs enabled)
				cmd_args = " ".join([f'"{arg}"' for arg in test_scanner_cmd])
				discovery_cmd = f"python /tmp/scanner_script.py /tmp/test_results.json {cmd_args}"
				
				result = self.image_manager.exec_in_container(
					container_id,
					discovery_cmd,
					timeout=timeout_scanner,
					log_file_path=log_file_path  # Enable streaming logs
				)
				
				if result.returncode != 0:
					tqdm.write(
						f"⚠️ {specs_name}: test discovery command exited with code {result.returncode}, "
						f"see log: {log_file_path}"
					)
				
				# 4. Copy results from container
				self.image_manager.copy_from_container(
					container_id,
					"/tmp/test_results.json",
					str(result_file)
				)

				if not result_file.exists():
					error_msg = f"Test discovery result file not found, log: {log_file_path}"
					raise FileNotFoundError(error_msg)
				
				with open(result_file, 'r', encoding='utf-8') as f:
					test_files_info = json.load(f)
				
				return test_files_info, log_file_path
				
			except subprocess.TimeoutExpired:
				is_timeout = True  # Mark timeout
				error_msg = f"Test discovery container timed out ({timeout_scanner}s), log: {log_file_path}"
				tqdm.write(f"❌ {specs_name}: {error_msg}")
				raise RuntimeError(error_msg)
			except Exception as e:
				# If error message lacks log path, append it
				error_str = str(e)
				if str(log_file_path) not in error_str:
					error_msg = f"{error_str}, log: {log_file_path}"
					tqdm.write(f"❌ {specs_name}: test discovery failed - {error_msg}")
					raise RuntimeError(error_msg) from e
				else:
					# Error message already has log path; log and re-raise
					tqdm.write(f"❌ {specs_name}: test discovery failed - {error_str}")
					raise RuntimeError(error_str) from e
			finally:
				# 6. Cleanup container (force kill on timeout)
				if container_id:
					self.image_manager.stop_container(container_id, force=is_timeout)
					# tqdm.write(f"🗑️  {specs_name}: cleaned container {container_id[:12]}")

	
	def _filter_test_files(
		self,
		specs_name: str,
		specs: Dict,
		test_files_info: Dict[str, Dict],
		log_file_path: Path
	) -> Tuple[List[Dict], List[Dict]]:
		"""
		Filter test files.
		
		Args:
			specs_name: Repo spec name
			specs: Repo specs config
			test_files_info: Test file info
			log_file_path: Log file path
			
		Returns:
			(p2p_files_with_info, f2p_files_with_info)
			- p2p_files_with_info: list of all test files with details
			- f2p_files_with_info: list of f2p files with details
		"""
		# Write filtering info to log file
		def log_to_file(message: str):
			"""Append a message to the log file."""
			with open(log_file_path, 'a', encoding='utf-8') as f:
				f.write(message + '\n')
		
		# Get filter parameters
		min_test_num = specs.get('min_test_num')
		max_f2p_num = specs.get('max_f2p_num')
		max_p2p_num = specs.get('max_p2p_num', -1)  # Default -1 means no limit
		start_time_str = specs.get('start_time', None)
		
		log_to_file("\n" + "=" * 80)
		log_to_file("Start filtering test files")
		log_to_file("=" * 80)
		log_to_file("Filter parameters:")
		log_to_file(f"  - min_test_num: {min_test_num}")
		log_to_file(f"  - max_f2p_num: {max_f2p_num}")
		log_to_file(f"  - max_p2p_num: {max_p2p_num}")
		log_to_file(f"  - start_time: {start_time_str}")
		log_to_file(f"Total discovered test files: {len(test_files_info)}")
		log_to_file("")
		
		# Convert start_time to timestamp
		start_timestamp = None
		if start_time_str:
			try:
				start_dt = datetime.strptime(start_time_str, "%Y-%m-%d")
				start_timestamp = start_dt.timestamp()
			except ValueError:
				error_msg = f"Invalid start_time format: {start_time_str}, expected YYYY-MM-DD"
				tqdm.write(f"❌ {specs_name}: {error_msg}")
				log_to_file(f"❌ Error: {error_msg}")
				raise ValueError(error_msg)

		# All test files (p2p) and eligible files (for f2p)
		p2p_files_with_info = []
		f2p_candidates = []
		filtered_by_test_count = 0  # Filtered out by test count
		filtered_by_time = 0  # Filtered out by time
		
		for file_path, info in test_files_info.items():
			test_count = info.get('test_count', 0)
			test_points = info.get('test_points', [])
			
			# Get file timestamp
			file_timestamp = self.repo_manager.get_file_git_last_modified_time(specs_name, file_path)
			file_time_str = "N/A"
			if file_timestamp is not None:
				file_time_str = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d %H:%M:%S")
			
			# All discovered test files go into p2p
			p2p_files_with_info.append({
				"file_path": file_path,
				"test_count": test_count,
				"last_modified": file_time_str,
				"test_points": test_points
			})
			
			# Filter f2p candidate files
			# 1. Filter by test point count
			if test_count < min_test_num:
				filtered_by_test_count += 1
				continue
			
			# 2. Filter by time (if start_time specified)
			if start_timestamp:
				if file_timestamp is None or file_timestamp < start_timestamp:
					filtered_by_time += 1
					continue
			
			f2p_candidates.append((file_path, test_count, file_time_str, test_points))
		
		# Record filter stats
		log_to_file("Filter stats:")
		log_to_file(f"  - P2P test files: {len(p2p_files_with_info)}")
		log_to_file(f"  - F2P candidate files: {len(f2p_candidates)}")
		log_to_file(
			f"  - Filtered by test count: {filtered_by_test_count} (test_count < {min_test_num})"
		)
		log_to_file(f"  - Filtered by time: {filtered_by_time} (before {start_time_str})")
		log_to_file("")
		
		# Sort F2P candidates by test point count
		f2p_candidates.sort(key=lambda x: x[1], reverse=True)
		
		# Take top max_f2p_num (no limit if -1)
		if max_f2p_num == -1:
			selected_f2p_candidates = f2p_candidates
			log_to_file(
				f"F2P limit: max_f2p_num=-1, no limit; selecting all {len(f2p_candidates)} candidates"
			)
		else:
			selected_f2p_candidates = f2p_candidates[:max_f2p_num]
			log_to_file(
				f"F2P limit: max_f2p_num={max_f2p_num}, selecting top {len(selected_f2p_candidates)} files"
			)
		log_to_file("")
		
		# Enforce constraint: max_p2p_num >= len(selected_f2p_candidates)
		if max_p2p_num != -1 and max_p2p_num < len(selected_f2p_candidates):
			error_msg = (
				f"Invalid config: max_p2p_num ({max_p2p_num}) < F2P file count "
				f"({len(selected_f2p_candidates)}), violates f2p ⊆ p2p"
			)
			tqdm.write(f"❌ {specs_name}: {error_msg}")
			log_to_file(f"❌ Error: {error_msg}")
			raise ValueError(error_msg)
		
		# Build P2P set, ensure f2p ⊆ p2p
		# 1. Convert f2p paths to set
		f2p_file_paths = set(item[0] for item in selected_f2p_candidates)
		
		# 2. Group: f2p files and non-f2p files
		f2p_files_in_p2p = []  # f2p files (must be in p2p)
		non_f2p_files = []  # non-f2p files (fill to max_p2p_num)
		
		for item in p2p_files_with_info:
			if item['file_path'] in f2p_file_paths:
				f2p_files_in_p2p.append(item)
			else:
				non_f2p_files.append(item)
		
		# 3. Apply P2P count limit
		if max_p2p_num == -1:
			# No limit; p2p includes all files
			final_p2p_files = p2p_files_with_info
			log_to_file(
				f"P2P limit: max_p2p_num=-1, no limit; keeping all {len(p2p_files_with_info)} files"
			)
		else:
			# Ensure all f2p files are in p2p first
			final_p2p_files = f2p_files_in_p2p.copy()
			
			# Compute how many non-f2p files to add
			remaining_slots = max_p2p_num - len(f2p_files_in_p2p)
			
			if remaining_slots > 0:
				# Sort non-f2p files by test point count
				non_f2p_files.sort(key=lambda x: x['test_count'], reverse=True)
				# Fill up to max_p2p_num
				final_p2p_files.extend(non_f2p_files[:remaining_slots])
				log_to_file(f"P2P limit: max_p2p_num={max_p2p_num}")
				log_to_file(f"  - F2P files: {len(f2p_files_in_p2p)}")
				log_to_file(f"  - Added from non-F2P: {min(remaining_slots, len(non_f2p_files))}")
				log_to_file(f"  - Final P2P files: {len(final_p2p_files)}")
			else:
				log_to_file(
					f"P2P limit: max_p2p_num={max_p2p_num}, equals F2P file count, no fill needed"
				)
		
		p2p_files_with_info = final_p2p_files
		log_to_file("")
		
		# Build detailed info list
		f2p_files_with_info = [
			{
				"file_path": item[0],
				"test_count": item[1],
				"last_modified": item[2],
				"test_points": item[3]
			}
			for item in selected_f2p_candidates
		]
		
		# Record final results
		log_to_file("=" * 80)
		log_to_file("Filter results:")
		log_to_file(f"  - Final P2P files: {len(p2p_files_with_info)}")
		log_to_file(f"  - Final F2P files: {len(f2p_files_with_info)}")
		log_to_file("=" * 80)
		
		if f2p_files_with_info:
			log_to_file("\nFinal selected F2P test files:")
			for idx, item in enumerate(f2p_files_with_info, 1):
				log_to_file(f"  {idx}. {Path(item['file_path']).name}")
				log_to_file(f"      Path: {item['file_path']}")
				log_to_file(f"      Test count: {item['test_count']}")
				log_to_file(f"      Last modified: {item['last_modified']}")
				log_to_file("")
		if p2p_files_with_info:
			log_to_file("\nFinal selected P2P test files:")
			for idx, item in enumerate(p2p_files_with_info, 1):
				log_to_file(f"  {idx}. {Path(item['file_path']).name}")
				log_to_file(f"      Path: {item['file_path']}")
				log_to_file(f"      Test count: {item['test_count']}")
				log_to_file(f"      Last modified: {item['last_modified']}")
				log_to_file("")
		
		return p2p_files_with_info, f2p_files_with_info
