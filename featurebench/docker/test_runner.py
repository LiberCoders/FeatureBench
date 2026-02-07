"""Test runner manager for running tests in Docker containers."""

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from featurebench.utils.logger import print_build_report, create_test_runner_logger
from featurebench.utils.command_utils import apply_uv_run_prefix
from featurebench.utils.parser.pytest_parser import PytestParser


class TestRunner:
	"""Run test files in Docker containers and record results."""
	
	def __init__(
		self,
		config,
		repo_manager,
		image_manager,
		storage_manager,
		logger: Optional[logging.Logger] = None
	):
		"""
		Initialize the test runner.
		
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
		
		# Load test discovery results from file:
		# {specs_name: {p2p_files: List[Dict], f2p_files: List[Dict]}}
		# Each Dict contains {file_path, test_count, last_modified, passed}
		repo_names = list(self.repo_manager.loaded_repos.keys())
		self.test_results: Dict[str, Dict[str, List]] = self.storage_manager.load_all_test_results(repo_names)
	
	def run(self, max_workers: Optional[int] = None) -> None:
		"""
		Run tests in parallel across all repos.
		
		Args:
			max_workers: Max parallel workers; auto-select if None
		"""
		self.logger.info("")
		self.logger.info("=" * 60)
		self.logger.info("Starting test run phase...")
		self.logger.info("=" * 60)
		
		# Set default parallelism
		if max_workers is None:
			max_workers = min(len(self.repo_manager.loaded_repos), os.cpu_count() or 1, 5)  # Max 5 parallel
		
		self.logger.info(f"Running tests for {len(self.repo_manager.loaded_repos)} repos with {max_workers} parallel workers")
		
		# Count total test files
		total_test_files = 0
		for specs_name in self.repo_manager.loaded_repos.keys():
			if specs_name in self.test_results:
				p2p_files = self.test_results[specs_name].get('p2p_files', [])
				if self.config.debug_sample:
					p2p_files = [
						f for f in p2p_files
						if self.config.is_sample_selected(f.get('file_path', ''))
					]
				total_test_files += len(p2p_files)
		
		self.logger.info(f"Total test files: {total_test_files}")
		
		# Failed repo info
		failed_repos: List[Tuple[str, str]] = []  # (repo_name, error_message)
		
		# Shared progress bar
		pbar = tqdm(total=total_test_files, desc="Test run", unit="file")
		
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			# Submit all test run tasks
			future_to_repo = {}
			for specs_name, repo_info in self.repo_manager.loaded_repos.items():
				specs = repo_info['specs']
				# Override cache config with config.get_cache_config (debug supported)
				use_cache = self.config.get_cache_config('runner', specs.get('test_cache', True))
				
				future = executor.submit(
					self._run_repo_tests,
					specs_name,
					specs,
					use_cache,
					pbar  # Pass progress bar
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
					passed, failed, total = future.result()
					
					pbar.set_postfix_str(f"Latest completed: {specs_name}")
					# tqdm.write(f"✅ {specs_name}: tests done (passed={passed}/{total}, failed={failed}) (repo progress: {completed_repos}/{total_repos})")
					
				except Exception as e:
					# Only collect failure info; errors already logged inside
					failed_repos.append((specs_name, str(e)))
					
					pbar.set_postfix_str(f"Latest failed: {specs_name}")
		
		pbar.close()
		
		# Wait for all save tasks to finish
		self.storage_manager.wait_for_save_completion(shutdown=False)
		
		# Reload latest data for report generation
		repo_names = list(self.test_results.keys())
		self.test_results = self.storage_manager.load_all_test_results(repo_names)
		
		# Print test run report
		success_info = {}
		for specs_name, result in self.test_results.items():
			p2p_files = result.get('p2p_files', [])
			total = len(p2p_files)
			passed = sum(1 for f in p2p_files if f.get('passed', False))
			failed = total - passed
			
			success_info[specs_name] = {
				'passed': passed,
				'failed': failed,
				'total': total,
				'pass_rate': f"{passed/total*100:.1f}%" if total > 0 else "N/A"
			}
		print_build_report(
			logger=self.logger,
			total_repos=list(self.repo_manager.loaded_repos.items()),
			failed_repos=failed_repos,
			success_info=success_info,
			operation_name="Test run"
		)
		
		# Raise if any repo failed
		if failed_repos:
			failed_names = [name for name, _ in failed_repos]
			raise RuntimeError(f"Repos with failed test runs: {', '.join(failed_names)}")
		
		self.logger.info("Test run phase completed")
	
	def _run_repo_tests(
		self,
		specs_name: str,
		specs: Dict,
		use_cache: bool = True,
		pbar: Optional[tqdm] = None
	) -> Tuple[int, int, int]:
		"""
		Run all test files for a single repo.
		
		Args:
			specs_name: Repo spec name
			specs: Repo specs config
			use_cache: Whether to use cache (default True)
			pbar: Progress bar for overall updates
			
		Returns:
			(passed, failed, total)
		"""
		# tqdm.write(f"🏃 Start running tests for {specs_name}...")
		
		# Read test file info from test_results
		if specs_name not in self.test_results:
			tqdm.write(f"❌ {specs_name}: test discovery results not found; run test discovery first")
			raise RuntimeError(f"{specs_name}: test discovery results not found; run test discovery first")
		
		test_files_info = self.test_results[specs_name]
		p2p_files = test_files_info.get('p2p_files', [])
		if self.config.debug_sample:
			p2p_files = [
				f for f in p2p_files
				if self.config.is_sample_selected(f.get('file_path', ''))
			]
		f2p_files = test_files_info.get('f2p_files', [])
		
		if not p2p_files:
			if self.config.debug_sample:
				tqdm.write(f"🔍 {specs_name}: no p2p tests after debug_sample filter, skipping")
				return 0, 0, 0
			tqdm.write(f"⚠️ {specs_name}: no p2p test files found")
			raise RuntimeError(f"{specs_name}: no p2p test files found")
		
		# Get test command and timeout
		test_cmd = apply_uv_run_prefix(specs.get("test_cmd"), specs)
		timeout_one = specs.get("timeout_one")
		if timeout_one is not None:
			test_cmd = f"{test_cmd} --timeout={timeout_one}"
		timeout_run = specs.get("timeout_run", 600)
		if isinstance(timeout_run, (int, float)) and timeout_run < 0:
			# Allow timeout_run = -1 to mean no limit
			timeout_run = None
		
		# Run all p2p test files (parallel)
		passed = 0
		failed = 0
		total = len(p2p_files)
		
		# Set parallelism for test files within this repo
		max_workers = min(total, os.cpu_count() or 1, 10)  # Max 10 parallel
		
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			future_to_file = {}
			for file_info in p2p_files:
				file_path = file_info['file_path']
				
				# Check cache
				if use_cache and 'passed' in file_info:
					# Result already exists; skip
					if file_info['passed']:
						passed += 1
					else:
						failed += 1
					# Update progress bar
					if pbar:
						pbar.update(1)
					continue
				
				future = executor.submit(
					self._run_single_test,
					specs_name,
					file_path,
					test_cmd,
					timeout_run
				)
				future_to_file[future] = file_info
			
			# Collect results
			for future in as_completed(future_to_file):
				file_info = future_to_file[future]
				file_path = file_info['file_path']
				
				try:
					passed_result, test_count_run = future.result()
					
					if passed_result:
						passed += 1
					else:
						failed += 1
					
				except Exception as e:
					# Errors already logged in _run_single_test; only record here
					passed_result = False
					test_count_run = None
					failed += 1
				
				# Save directly to file (incremental); background thread updates p2p/f2p
				self.storage_manager.save_files_status(
					specs_name=specs_name,
					update_single_file={
						'file_path': file_path,
						'updates': {
							'passed': passed_result,
							'test_count_run': test_count_run
						}
					}
				)
				
				# Update progress bar
				if pbar:
					pbar.update(1)
		
		return passed, failed, total
	
	def _run_single_test(
		self,
		specs_name: str,
		file_path: str,
		test_cmd: str,
		timeout: int
	) -> Tuple[bool, Optional[int]]:
		"""
		Run a single test file inside a container.
		
		Args:
			specs_name: Repo spec name
			file_path: Test file path (container path)
			test_cmd: Test command
			timeout: Timeout in seconds
		
		Returns:
			tuple[bool, Optional[int]]: (passed, test count actually run)
		"""
		container_id = None
		is_timeout = False  # Mark timeout status
		
		# Create log file path
		log_file_path = create_test_runner_logger(
			specs_name=specs_name,
			test_file=file_path,
			logs_dir=self.logs_dir
		)
		
		# tqdm.write(f"🏃 {specs_name}: run test file {file_path} (log: {log_file_path})")
		
		try:
			# Start container
			container_id = self.image_manager.run_container(
				specs_name=specs_name,
				working_dir="/testbed",
				prepare_env=True
			)
			
			# Build full test command
			full_cmd = f"{test_cmd} {file_path}"
			
			# Execute tests (log_file_path enables streaming logs)
			result = self.image_manager.exec_in_container(
				container_id,
				full_cmd,
				timeout=timeout,
				log_file_path=log_file_path  # Enable streaming log capture
			)
			
			# Read test output and parse with PytestParser
			with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
				test_output = f.read()
			pytest_result = PytestParser.parse_output(test_output)
			passed_result = (
				pytest_result.pass_rate == 1.0
				and pytest_result.return_code == 0
				and pytest_result.failed == 0
				and pytest_result.error == 0
			)

			# Print info by result type
			if passed_result:
				# tqdm.write(f"✅ {specs_name}: test {file_path} passed (log: {log_file_path})")
				pass
			else:
				# Failure cases by result type
				if pytest_result.skipped > 0 and pytest_result.total == pytest_result.skipped:
					# All skipped
					tqdm.write(f"⏭️ {specs_name}: test file {file_path} - all skipped (log: {log_file_path})")
				elif pytest_result.failed > 0 or pytest_result.error > 0:
					# Has failed or error
					tqdm.write(f"❌ {specs_name}: test file {file_path} - not all passed (log: {log_file_path})")
				elif pytest_result.deselected > 0 and pytest_result.total == 0:
					# All deselected
					tqdm.write(f"⏭️ {specs_name}: test file {file_path} - all deselected (log: {log_file_path})")
				elif pytest_result.xfailed > 0 or pytest_result.xpassed > 0:
					# Only xfailed/xpassed/skipped
					tqdm.write(f"⏭️ {specs_name}: test file {file_path} - only xfailed / xpassed / skipped (log: {log_file_path})")
				else:
					# Other cases
					tqdm.write(f"❓ {specs_name}: test file {file_path} - other cases (log: {log_file_path})")
			
			return passed_result, pytest_result.total
			
		except subprocess.TimeoutExpired:
			is_timeout = True  # Mark timeout
			error_msg = f"Test file {file_path} timed out ({timeout}s), log: {log_file_path}"
			tqdm.write(f"⏱️ {specs_name}: {error_msg}")
			raise RuntimeError(error_msg)
		
		except Exception as e:
			error_msg = f"Test file {file_path} failed to run: {e}, log: {log_file_path}"
			tqdm.write(f"❌ {specs_name}: {error_msg}")
			raise RuntimeError(error_msg) from e
		
		finally:
			# Cleanup container (force kill on timeout)
			if container_id:
				self.image_manager.stop_container(container_id, force=is_timeout)
				# tqdm.write(f"🗑️  {specs_name}: cleaned container {container_id[:12]}")
