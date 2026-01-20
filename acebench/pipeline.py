import sys
import os
import signal
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from acebench.utils.config import Config
from acebench.utils.storage import StorageManager
from acebench.utils.logger import configure_logging
from acebench.utils.repo_manager import RepoManager
from acebench.utils.utils import (
	collect_data_items,
	print_data_processing_statistics,
	initialize_gpu_tracking,
)
from acebench.docker.image_manager import ImageManager
from acebench.docker.test_scanner import TestScanner
from acebench.docker.test_runner import TestRunner
from acebench.docker.dynamic_tracer import DynamicTracer
from acebench.classification.llm_top_classifier import LLMTopClassifier
from acebench.classification.p2p_chooser import P2PChooser
from acebench.classification.code_classifier import CodeClassifier
from acebench.mask.mask_generator import MaskGenerator
from acebench.post_verify.f2p_validator import F2PValidator
from acebench.post_verify.p2p_validator import P2PValidator
from acebench.conversion.case_converter import CaseConverter

@dataclass
class DataItem:
	"""Data item representing a full processing unit."""
	repo: str                              # Repo name
	specs: dict                            # Repo specs config
	repo_root: Path                        # Repo root path
	file_path: str                         # f2p test file path (container path)
	p2p_list: List[str]                    # p2p test file paths (non-empty, container paths)
	dynamic_trace_file: str                # f2p dynamic trace path (host)
	dynamic_trace_files: List[str]         # p2p dynamic trace paths (host)
	updated_top_objects: List[str]         # Updated top objects
	all_top_objects_candidates: List[str]  # All top object candidates
	p2p_test_points: Optional[float] = None  # Sum of p2p test points (test_count_run)
	test_count: Optional[int] = None       # Test count
	test_count_run: Optional[int] = None   # Executed test count
	last_modified: Optional[str] = None    # Last modified time
	first_commit: Optional[str] = None     # First commit time


def process_data_item(
	data_item: DataItem,
	config: Config,
	repo_manager: RepoManager,
	image_manager: ImageManager,
	storage: StorageManager,
	logger
) -> dict:
	"""
	Run the full pipeline for a single data item.
	
	Args:
		data_item: Data item to process
		config: Config object
		repo_manager: Repo manager
		image_manager: Image manager
		storage: Storage manager
		logger: Logger
		
	Returns:
		dict: Processing result
	"""
	result = {
		'repo': data_item.repo,
		'file_path': data_item.file_path,
		'success_lv1': False,
		'success_lv2': False,
		'cached': False,
		'error': [],
		'case_dirs': {
			'lv1': None,
			'lv2': None,
		},
		'deleted_lines': None,
		'mask_file_count': None,
		'mask_object_count': None,
		'test_count': getattr(data_item, 'test_count', None),
		'test_count_run': getattr(data_item, 'test_count_run', None),
		'lv1_post_rate': None,
		'lv2_post_rate': None,
	}

	def _add_error(message: Optional[str], scope: str = "Both") -> None:
		"""Append a scoped, normalized error message to the list."""
		if not message:
			return
		text = str(message).strip()
		if not text:
			return
		valid_scopes = {"Both", "Level1", "Level2"}
		if scope not in valid_scopes:
			raise ValueError(f"Invalid error scope: {scope}")
		result['error'].append(f"[{scope}] {text}")

	def _append_existing_error(message: Optional[str]) -> None:
		if not message:
			return
		text = str(message).strip()
		if text:
			result['error'].append(text)

	def _extend_errors(messages) -> None:
		"""Extend the error list with preformatted string or iterable inputs."""
		if not messages:
			return
		if isinstance(messages, str):
			_append_existing_error(messages)
		elif isinstance(messages, list):
			for msg in messages:
				_append_existing_error(msg)
		else:
			_append_existing_error(messages)

	def _error_contains(keyword: str) -> bool:
		return any(keyword in err for err in result['error'])

	deleted_lines: Optional[int] = None
	mask_file_count: Optional[int] = None
	mask_object_count: Optional[int] = None
	lv1_post_rate: Optional[float] = None
	lv2_post_rate: Optional[float] = None
	
	# Get test counts and Git timestamps
	test_count = result['test_count']
	test_count_run = result['test_count_run']
	last_modified = getattr(data_item, 'last_modified', None)
	first_commit = getattr(data_item, 'first_commit', None)
	
	try:
		# Check cache setting (debug overrides supported)
		use_cache = config.get_cache_config('data', data_item.specs.get("data_cache", True))
		if use_cache:
			# Try loading status from cache
			cached_status = storage.load_data_status(data_item.repo, data_item.file_path)
			# Cache hit: return directly
			if cached_status:
				result['cached'] = True
				if cached_status.get('success_lv1') == True:
					result['success_lv1'] = True
				if cached_status.get('success_lv2') == True:
					result['success_lv2'] = True
				result['deleted_lines'] = cached_status.get('deleted_lines')
				result['mask_file_count'] = cached_status.get('mask_file_count')
				result['mask_object_count'] = cached_status.get('mask_object_count')
				result['error'] = []
				_extend_errors(cached_status.get('error'))
				case_dirs_cached = cached_status.get('case_dirs') or {}
				result['case_dirs']['lv1'] = case_dirs_cached.get('lv1')
				result['case_dirs']['lv2'] = case_dirs_cached.get('lv2')
				if 'test_count' in cached_status:
					result['test_count'] = cached_status.get('test_count')
				if 'test_count_run' in cached_status:
					result['test_count_run'] = cached_status.get('test_count_run')
				result['lv1_post_rate'] = cached_status.get('lv1_post_rate')
				result['lv2_post_rate'] = cached_status.get('lv2_post_rate')
				# If cache shows both levels succeeded, return directly
				if (cached_status.get('success_lv1') == True) and (cached_status.get('success_lv2') == True):
					return result
				# Otherwise return with error info
				else:
					return result
		
		# Classification
		code_classifier = CodeClassifier(
			config=config,
			repo_manager=repo_manager,
			storage_manager=storage,
			data_item=data_item,
			logger=logger
		)
		classification_summary = code_classifier.run()
		if len(classification_summary['top_objects']) == 0:
			_add_error("No top objects found; cannot continue", scope="Both")
			result['deleted_lines'] = None
			# Save failure status to cache
			storage.save_data_status(
				data_item.repo, data_item.file_path, 
				success_lv1=False, success_lv2=False, error=list(result['error']),
				test_count=test_count,
				test_count_run=test_count_run,
				last_modified=last_modified,
				first_commit=first_commit,
				deleted_lines=None,
				mask_file_count=None,
				mask_object_count=None,
				case_dirs=result['case_dirs'],
				lv1_post_rate=lv1_post_rate,
				lv2_post_rate=lv2_post_rate,
			)
			return result
		
		# mask
		mask_generator = MaskGenerator(
			config=config,
			repo_manager=repo_manager,
			image_manager=image_manager,
			storage_manager=storage,
			data_item=data_item,
			classification_summary=classification_summary,
			logger=logger
		)
		mask_results, is_passed, deleted_lines = mask_generator.run()
		mask_file_count = len(mask_results) if mask_results else 0
		mask_object_count = 0
		if mask_results:
			for mask_result in mask_results.values():
				top_objects = getattr(mask_result, 'top_objects', []) or []
				specific_objects = getattr(mask_result, 'specific_objects', []) or []
				mask_object_count += len(top_objects) + len(specific_objects)
		result['mask_file_count'] = mask_file_count
		result['mask_object_count'] = mask_object_count
			
		if not is_passed:
			_add_error("Mask failed", scope="Both")
			result['deleted_lines'] = deleted_lines
			# Save failure status to cache
			storage.save_data_status(
				data_item.repo, data_item.file_path, 
				success_lv1=False, success_lv2=False, error=list(result['error']),
				test_count=test_count,
				test_count_run=test_count_run,
				last_modified=last_modified,
				first_commit=first_commit,
				deleted_lines=deleted_lines,
				mask_file_count=mask_file_count,
				mask_object_count=mask_object_count,
				case_dirs=result['case_dirs'],
				lv1_post_rate=lv1_post_rate,
				lv2_post_rate=lv2_post_rate,
			)
			return result

		# Exit if mask produced no top objects
		if mask_results:
			all_empty_top = True
			for mask_result in mask_results.values():
				top_objects = getattr(mask_result, 'top_objects', []) or []
				if top_objects:
					all_empty_top = False
					break
			if all_empty_top:
				_add_error("Mask results contain no top objects; cannot continue", scope="Both")
				result['deleted_lines'] = deleted_lines
				storage.save_data_status(
					data_item.repo, data_item.file_path,
					success_lv1=False, success_lv2=False, error=list(result['error']),
					test_count=test_count,
					test_count_run=test_count_run,
					last_modified=last_modified,
					first_commit=first_commit,
					deleted_lines=deleted_lines,
					mask_file_count=mask_file_count,
					mask_object_count=mask_object_count,
					case_dirs=result['case_dirs'],
					lv1_post_rate=lv1_post_rate,
					lv2_post_rate=lv2_post_rate,
				)
				return result

		# F2P post-check
		f2p_validator = F2PValidator(
			config=config,
			repo_manager=repo_manager,
			image_manager=image_manager,
			storage_manager=storage,
			data_item=data_item,
			mask_results=mask_results,
			logger=logger
		)
		f2p_result, is_passed = f2p_validator.run()
		lv1_post_rate = getattr(f2p_result, 'pass_rate', None)
		result['lv1_post_rate'] = lv1_post_rate
		f2p_error_message = f2p_result.error_message
		if not is_passed:
			if (f2p_error_message or '').startswith('F2P post-validation pass rate too high'):
				# If F2P post rate is too high, record error and continue
				_add_error(f2p_error_message, scope="Level1")
			else:
				_add_error(f2p_error_message, scope="Level1")
				result['deleted_lines'] = deleted_lines
				# Save failure status to cache
				storage.save_data_status(
					data_item.repo, data_item.file_path, 
					success_lv1=False, success_lv2=False, error=list(result['error']),
					test_count=test_count,
					test_count_run=test_count_run,
					last_modified=last_modified,
					first_commit=first_commit,
					deleted_lines=deleted_lines,
					mask_file_count=mask_file_count,
					mask_object_count=mask_object_count,
					case_dirs=result['case_dirs'],
					lv1_post_rate=lv1_post_rate,
					lv2_post_rate=lv2_post_rate,
				)
				return result

		# P2P post-check
		p2p_validator = P2PValidator(
			config=config,
			repo_manager=repo_manager,
			image_manager=image_manager,
			storage_manager=storage,
			data_item=data_item,
			mask_results=mask_results,
			logger=logger
		)
		
		p2p_all_passed = True
		
		with ThreadPoolExecutor(max_workers=min(len(data_item.p2p_list), 5)) as p2p_executor:
			# Submit all p2p test tasks
			p2p_futures = {
				p2p_executor.submit(p2p_validator.run, p2p_file): p2p_file
				for p2p_file in data_item.p2p_list
			}
			# Collect results
			for future in as_completed(p2p_futures):
				p2p_file = p2p_futures[future]
				try:
					p2p_result = future.result()
					
					# No test succeeded, break
					if not p2p_result.success:
						p2p_all_passed = False
						_add_error(
							f"P2P test execution failed - {p2p_file}: {p2p_result.error_message}; cannot generate Level1 data",
							scope="Level1"
						)
						break
					
					# Pass rate < 100%, break
					if p2p_result.pass_rate != 1.0:
						p2p_all_passed = False
						_add_error(
							f"P2P tests not fully passing - {p2p_file}: pass rate {p2p_result.pass_rate:.2%} (expected 100%), log: {p2p_result.log_file}; cannot generate Level1 data",
							scope="Level1"
						)
						break
						
				except Exception as e:
					p2p_all_passed = False
					_add_error(
						f"P2P validation error - {p2p_file}: {e}; cannot generate Level1 data",
						scope="Level1"
					)
					break
		
		# If any p2p test fails, lv1 cannot be generated; pipeline still proceeds to lv2
		# Generate data
		case_converter = CaseConverter(
			config=config,
			repo_manager=repo_manager,
			storage_manager=storage,
			image_manager=image_manager,
			data_item=data_item,
			classification_summary=classification_summary,
			mask_results=mask_results,
			deleted_lines=deleted_lines,
			mask_file_count=mask_file_count,
			mask_object_count=mask_object_count,
			generate_level1=p2p_all_passed,
			generate_level2=True,
			logger=logger
		)
		lv1_error_message, success_lv1, lv2_result, success_lv2, is_llm_error = case_converter.run()
		case_dirs = getattr(case_converter, 'case_dirs', {}) or {}
		result['case_dirs']['lv1'] = case_dirs.get(1)
		result['case_dirs']['lv2'] = case_dirs.get(2)
		
		if success_lv1 and not _error_contains('F2P post-validation pass rate too high'):
			result['success_lv1'] = True
		else:
			result['success_lv1'] = False
			if lv1_error_message:
				_add_error(lv1_error_message, scope="Level1")
		
		if success_lv2:
			result['success_lv2'] = True
		else:
			result['success_lv2'] = False
			_add_error(lv2_result.error_message, scope="Level2")

		lv2_post_rate = getattr(lv2_result, 'pass_rate', None) if lv2_result else None
		result['lv2_post_rate'] = lv2_post_rate

		# Do not cache on LLM error so it can be retried
		if is_llm_error:
			tqdm.write(f"🚨 LLM error; not saving cache state: {data_item.repo}/{data_item.file_path}")
		else:
			storage.save_data_status(
				data_item.repo, data_item.file_path, 
				success_lv1=result['success_lv1'], success_lv2=result['success_lv2'], error=list(result['error']),
				test_count=test_count,
				test_count_run=test_count_run,
				last_modified=last_modified,
				first_commit=first_commit,
				deleted_lines=deleted_lines,
				mask_file_count=mask_file_count,
				mask_object_count=mask_object_count,
				case_dirs=result['case_dirs'],
				lv1_post_rate=lv1_post_rate,
				lv2_post_rate=lv2_post_rate,
			)

		result['deleted_lines'] = deleted_lines

	except Exception as e:
		tqdm.write(f"❌ Failed to process data item {data_item.repo}/{data_item.file_path}: {e}")
		_add_error(str(e), scope="Both")
		result['success_lv1'] = False
		result['success_lv2'] = False
		
		# Save failure status to cache
		storage.save_data_status(
			data_item.repo, data_item.file_path, 
			success_lv1=False, success_lv2=False, error=list(result['error']),
			test_count=test_count,
			test_count_run=test_count_run,
			last_modified=last_modified,
			first_commit=first_commit,
			deleted_lines=deleted_lines,
			mask_file_count=mask_file_count,
			mask_object_count=mask_object_count,
			case_dirs=result['case_dirs'],
			lv1_post_rate=lv1_post_rate,
			lv2_post_rate=lv2_post_rate,
		)
		result['deleted_lines'] = deleted_lines

	return result


def run_data_processing(
	data_items: List[DataItem],
	total_count: int,
	config: Config,
	repo_manager: RepoManager,
	image_manager: ImageManager,
	storage: StorageManager,
	logger,
	max_workers: Optional[int] = None
):
	"""
	Process all data items in parallel.
	
	Args:
		data_items: List of data items to process
		total_count: Total number of items
		config: Config object
		repo_manager: Repo manager
		image_manager: Image manager
		storage: Storage manager
		logger: Logger
		max_workers: Max workers; auto-select if None
	"""
	logger.info("")
	logger.info("=" * 60)
	logger.info("Starting data processing stage...")
	logger.info("=" * 60)
	logger.info(f"Total data items: {total_count}")
	
	# Set worker count
	if max_workers is None:
		max_workers = min(total_count, os.cpu_count() or 1, 10)
	logger.info(f"Processing {total_count} data items with {max_workers} workers")
	
	# Track progress with a progress bar
	pbar = tqdm(total=total_count, desc="Data processing", unit="item")
	
	# Process all data items in parallel
	processing_results = []
	failed_items = []
	
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Submit all data processing tasks
		future_to_item = {
			executor.submit(process_data_item, item, config, repo_manager, image_manager, storage, logger): item 
			for item in data_items
		}
		
		# Handle completed tasks
		for future in as_completed(future_to_item):
			data_item = future_to_item[future]
			
			try:
				result = future.result()
				processing_results.append(result)
				
				if result['success_lv1'] and result['success_lv2']:
					tqdm.write(f"😄 {result['repo']}/{result['file_path']}: Succeeded")
				else:
					error_lines = result['error'] if result['error'] else ['Unknown error']
					formatted = '\n'.join(f"    - {line}" for line in error_lines)
					tqdm.write(f"😭 {result['repo']}/{result['file_path']}: Failed\n{formatted}")
					failed_items.append(data_item)
					
			except Exception as e:
				# Only collect exceptions; errors already logged in process_data_item
				failed_items.append(data_item)
			
			pbar.update(1)
	
	pbar.close()

	# Wait for all data status saves to finish
	storage.wait_for_save_completion(shutdown=False)

	# Print detailed summary stats
	print_data_processing_statistics(processing_results, logger)
	logger.info("=" * 60)


def main():
	# Get config object
	config = Config._init_from_cli()
	# Get logger
	logger = configure_logging(config)
	# Initialize GPU usage tracking
	initialize_gpu_tracking(logger)

	# Get storage manager
	storage = StorageManager(config)
	# Persist config info locally
	storage.save_config(config)

	# Get repo manager
	repo_manager = RepoManager(repos_dir=config.repos_dir, logger=logger)
	# Load required repos locally
	repo_manager.load(config)
	
	# Debug: check whether to stop after repo stage
	if config.should_stop_after('repo'):
		logger.info("🛑 Debug mode: stop after repo stage")
		return 0

	# Get image manager
	image_manager = ImageManager(config=config, logger=logger)
	# Prepare Docker images for repos (parallel builds if multiple repos)
	repo_count = len(repo_manager.loaded_repos)
	if repo_count > 1:
		image_manager.prepare_images_parallel(repo_manager, max_workers=None)	# None = auto worker count
	else:
		image_manager.prepare_images(repo_manager)
	
	# Debug: check whether to stop after image stage
	if config.should_stop_after('image'):
		logger.info("🛑 Debug mode: stop after image stage")
		return 0

	# Get test scanner manager
	test_scanner = TestScanner(
		config=config,
		repo_manager=repo_manager,
		image_manager=image_manager,
		storage_manager=storage,
		logger=logger
	)
	# Run test discovery (parallel)
	test_scanner.run(max_workers=None)
	
	# Debug: check whether to stop after scanner stage
	if config.should_stop_after('scanner'):
		logger.info("🛑 Debug mode: stop after scanner stage")
		storage.wait_for_save_completion(shutdown=True)
		return 0

	# Get test runner manager
	test_runner = TestRunner(
		config=config,
		repo_manager=repo_manager,
		image_manager=image_manager,
		storage_manager=storage,
		logger=logger
	)
	# Run tests (parallel across repos, then across files)
	test_runner.run(max_workers=None)
	
	# Debug: check whether to stop after runner stage
	if config.should_stop_after('runner'):
		logger.info("🛑 Debug mode: stop after runner stage")
		storage.wait_for_save_completion(shutdown=True)
		return 0

	# Get dynamic tracer manager
	dynamic_tracer = DynamicTracer(
		config=config,
		repo_manager=repo_manager,
		image_manager=image_manager,
		storage_manager=storage,
		logger=logger
	)
	# Run dynamic tracing (parallel across repos, then files)
	dynamic_tracer.run(max_workers=None)
	
	# Debug: check whether to stop after dynamic stage
	if config.should_stop_after('dynamic'):
		logger.info("🛑 Debug mode: stop after dynamic stage")
		storage.wait_for_save_completion(shutdown=True)
		return 0

	# Get top classifier
	llm_top_classifier = LLMTopClassifier(
		config=config,
		repo_manager=repo_manager,
		storage_manager=storage,
		logger=logger
	)
	# Run top classifier (parallel)
	llm_top_classifier.run(max_workers=None)
	
	# Debug: check whether to stop after top stage
	if config.should_stop_after('top'):
		logger.info("🛑 Debug mode: stop after top stage")
		storage.wait_for_save_completion(shutdown=True)
		return 0

	# Get p2p chooser
	p2p_chooser = P2PChooser(
		config=config,
		repo_manager=repo_manager,
		storage_manager=storage,
		logger=logger
	)
	# Run p2p chooser (parallel)
	p2p_chooser.run(max_workers=None)

	# Collect f2p files with complete data and build DataItems
	data_items, total_count, repo_counts, repo_total_f2p_counts = collect_data_items(
		test_results=storage.load_all_test_results(list(repo_manager.loaded_repos.keys())),
		loaded_repos=repo_manager.loaded_repos,
		logger=logger,
		config=config,
		DataItem=DataItem,
		repo_manager=repo_manager
	)

	# Debug: check whether to stop after p2p stage
	if config.should_stop_after('p2p'):
		logger.info("🛑 Debug mode: stop after p2p stage")
		storage.wait_for_save_completion(shutdown=True)
		return 0

	# Data stage: run full processing for each DataItem
	run_data_processing(
		data_items,
		total_count,
		config,
		repo_manager,
		image_manager,
		storage,
		logger,
		max_workers=None
	)

	storage.wait_for_save_completion(shutdown=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
