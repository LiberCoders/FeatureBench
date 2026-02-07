"""Utility functions module."""
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import threading
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Union

# ===== GPU management globals (in-memory) =====
_gpu_usage_counter: Dict[int, int] = {}  # GPU ID -> usage count
_gpu_usage_lock = threading.Lock()  # Thread lock for GPU counter


LV1_ERROR_MARKER = "F2P post-validation pass rate too high"
LV2_ERROR_MARKER = "Level2 post-validation pass rate too high"
ALLOWED_ERROR_MARKERS = (LV1_ERROR_MARKER, LV2_ERROR_MARKER)


def ensure_list(value) -> List[str]:
	"""Normalize error or string values into a list of non-empty strings."""
	if not value:
		return []
	if isinstance(value, list):
		return [str(item) for item in value if item]
	return [str(value)]


def split_errors_by_scope(errors: Sequence[str]) -> Dict[str, List[str]]:
	"""Group error strings by explicit [Level] prefixes."""
	grouped: Dict[str, List[str]] = {"Both": [], "Level1": [], "Level2": []}
	for raw in errors or []:
		text = str(raw).strip()
		if not text:
			continue
		scope = "Both"
		if text.startswith("["):
			close_idx = text.find("]")
			if close_idx > 1:
				candidate = text[1:close_idx]
				if candidate in grouped:
					scope = candidate
					text = text[close_idx + 1:].lstrip()
		if text:
			grouped[scope].append(text)
	return grouped


def resolve_numeric(value: Optional[object]) -> Optional[float]:
	"""Safely coerce ints/floats/strings into float values."""
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str) and value.strip():
		try:
			return float(value)
		except ValueError:
			return None
	return None


def parse_user_datetime(label: str, value: Optional[str]) -> Optional[datetime]:
	"""Parse ISO-8601 user input and raise ValueError with context on failure."""
	if value is None:
		return None
	try:
		return datetime.fromisoformat(value)
	except ValueError as exc:
		raise ValueError(f"{label} must be ISO 8601, got {value}") from exc


def parse_payload_datetime(value: Optional[str]) -> Optional[datetime]:
	"""Parse ISO-8601 payload timestamps; swallow errors by returning None."""
	if value is None:
		return None
	try:
		return datetime.fromisoformat(value)
	except ValueError:
		return None


def has_blocking_error(errors: Sequence[str]) -> bool:
	"""Return True if any error is unrelated to post-rate thresholds."""
	for error in errors:
		if not any(marker in error for marker in ALLOWED_ERROR_MARKERS):
			return True
	return False


def resolve_post_rate(payload: Dict, level: int) -> Optional[float]:
	"""Retrieve lv1/lv2 post rates from a payload dict, coercing to float."""
	rate_key = f"lv{level}_post_rate"
	value = payload.get(rate_key)
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str) and value.strip():
		try:
			return float(value)
		except ValueError:
			return None
	return None


@dataclass
class MetadataFilters:
	min_deleted_lines: Optional[int] = None
	min_commit_time: Optional[datetime] = None
	min_update_time: Optional[datetime] = None
	min_test_count: Optional[int] = None
	min_test_count_run: Optional[int] = None


@dataclass
class CaseRecord:
	repo: str
	file_path: str
	level: str
	source_dir: Path
	reason: str
	pass_rate: Optional[float] = None
	new_threshold: Optional[float] = None
	status: str = ""
	deleted_lines: Optional[float] = None
	mask_file_count: Optional[int] = None
	mask_object_count: Optional[int] = None
	test_count: Optional[float] = None
	test_count_run: Optional[float] = None
	first_commit: Optional[datetime] = None
	last_modified: Optional[datetime] = None


def meets_metadata_filters(
	filters: Optional[MetadataFilters],
	deleted_lines: Optional[float],
	first_commit: Optional[datetime],
	last_modified: Optional[datetime],
	test_count: Optional[float],
	test_count_run: Optional[float],
) -> bool:
	if not filters:
		return True

	if filters.min_deleted_lines is not None:
		if deleted_lines is None or deleted_lines < filters.min_deleted_lines:
			return False

	if filters.min_commit_time is not None:
		if first_commit is None or first_commit < filters.min_commit_time:
			return False

	if filters.min_update_time is not None:
		if last_modified is None or last_modified < filters.min_update_time:
			return False

	if filters.min_test_count is not None:
		if test_count is None or test_count < filters.min_test_count:
			return False

	if filters.min_test_count_run is not None:
		if test_count_run is None or test_count_run < filters.min_test_count_run:
			return False

	return True


def build_candidates(
	data: Dict[str, Dict[str, Dict]],
	lv1_threshold: Optional[float],
	lv2_threshold: Optional[float],
	metadata_filters: Optional[MetadataFilters] = None,
) -> List[CaseRecord]:
	"""Construct CaseRecord entries using the shared filtering logic."""
	candidates: List[CaseRecord] = []
	for repo, entries in data.items():
		for file_path, payload in entries.items():
			case_dirs = payload.get("case_dirs") or {}
			errors = ensure_list(payload.get("error"))
			grouped_errors = split_errors_by_scope(errors)

			# If there is a Both-level error, skip the file
			if grouped_errors["Both"]:
				continue

			lv1_blocked = has_blocking_error(grouped_errors["Level1"])
			lv2_blocked = has_blocking_error(grouped_errors["Level2"])

			deleted_lines = resolve_numeric(payload.get("deleted_lines"))
			mask_file_count = resolve_numeric(payload.get("mask_file_count"))
			mask_object_count = resolve_numeric(payload.get("mask_object_count"))
			test_count = resolve_numeric(payload.get("test_count"))
			test_count_run = resolve_numeric(payload.get("test_count_run"))
			first_commit = parse_payload_datetime(payload.get("first_commit"))
			last_modified = parse_payload_datetime(payload.get("last_modified"))

			if not meets_metadata_filters(
				metadata_filters,
				deleted_lines,
				first_commit,
				last_modified,
				test_count,
				test_count_run,
			):
				continue

			if lv1_threshold is not None and not lv1_blocked:
				dir_lv1 = case_dirs.get("lv1")
				if dir_lv1:
					pass_rate = resolve_post_rate(payload, 1)
					if pass_rate is not None and pass_rate <= lv1_threshold:
						reason = (
							f"Level1 pass rate {pass_rate:.2%} <= target {lv1_threshold:.2%}"
						)
						candidates.append(
							CaseRecord(
								repo=repo,
								file_path=file_path,
								level="lv1",
								source_dir=Path(dir_lv1),
								reason=reason,
								pass_rate=pass_rate,
								new_threshold=lv1_threshold,
								status="lv1-threshold",
								deleted_lines=deleted_lines,
								mask_file_count=mask_file_count,
								mask_object_count=mask_object_count,
								test_count=test_count,
								test_count_run=test_count_run,
								first_commit=first_commit,
								last_modified=last_modified,
							)
						)

			if lv2_threshold is not None and not lv2_blocked:
				dir_lv2 = case_dirs.get("lv2")
				if dir_lv2:
					pass_rate = resolve_post_rate(payload, 2)
					if pass_rate is not None and pass_rate <= lv2_threshold:
						reason = (
							f"Level2 pass rate {pass_rate:.2%} <= target {lv2_threshold:.2%}"
						)
						candidates.append(
							CaseRecord(
								repo=repo,
								file_path=file_path,
								level="lv2",
								source_dir=Path(dir_lv2),
								reason=reason,
								pass_rate=pass_rate,
								new_threshold=lv2_threshold,
								status="lv2-threshold",
								deleted_lines=deleted_lines,
								mask_file_count=mask_file_count,
								mask_object_count=mask_object_count,
								test_count=test_count,
								test_count_run=test_count_run,
								first_commit=first_commit,
								last_modified=last_modified,
							)
						)
	return candidates


@dataclass
class CaseSummary:
	total_candidates: int
	lv1_candidates: int
	lv2_candidates: int
	repo_stats: Dict[str, Dict[str, float]]
	overall_metrics: Dict[str, float]


def compute_case_summary(records: Sequence[CaseRecord]) -> CaseSummary:
	repo_stats: Dict[str, Dict[str, float]] = {}
	overall_metrics = {
		"deleted_sum": 0.0,
		"deleted_samples": 0,
		"mask_file_sum": 0.0,
		"mask_file_samples": 0,
		"mask_object_sum": 0.0,
		"mask_object_samples": 0,
		"test_count_sum": 0.0,
		"test_count_samples": 0,
		"test_count_run_sum": 0.0,
		"test_count_run_samples": 0,
	}
	lv1_total = 0
	lv2_total = 0

	for record in records:
		stats = repo_stats.setdefault(
			record.repo,
			{
				"total": 0,
				"lv1": 0,
				"lv2": 0,
				"deleted_sum": 0.0,
				"deleted_samples": 0,
				"mask_file_sum": 0.0,
				"mask_file_samples": 0,
				"mask_object_sum": 0.0,
				"mask_object_samples": 0,
				"test_count_sum": 0.0,
				"test_count_samples": 0,
				"test_count_run_sum": 0.0,
				"test_count_run_samples": 0,
			},
		)

		stats["total"] += 1
		if record.level.lower() == "lv1":
			stats["lv1"] += 1
			lv1_total += 1
		else:
			stats["lv2"] += 1
			lv2_total += 1

		if record.deleted_lines is not None:
			stats["deleted_sum"] += record.deleted_lines
			stats["deleted_samples"] += 1
			overall_metrics["deleted_sum"] += record.deleted_lines
			overall_metrics["deleted_samples"] += 1

		if record.mask_file_count is not None:
			stats["mask_file_sum"] += record.mask_file_count
			stats["mask_file_samples"] += 1
			overall_metrics["mask_file_sum"] += record.mask_file_count
			overall_metrics["mask_file_samples"] += 1

		if record.mask_object_count is not None:
			stats["mask_object_sum"] += record.mask_object_count
			stats["mask_object_samples"] += 1
			overall_metrics["mask_object_sum"] += record.mask_object_count
			overall_metrics["mask_object_samples"] += 1

		if record.test_count is not None:
			stats["test_count_sum"] += record.test_count
			stats["test_count_samples"] += 1
			overall_metrics["test_count_sum"] += record.test_count
			overall_metrics["test_count_samples"] += 1

		if record.test_count_run is not None:
			stats["test_count_run_sum"] += record.test_count_run
			stats["test_count_run_samples"] += 1
			overall_metrics["test_count_run_sum"] += record.test_count_run
			overall_metrics["test_count_run_samples"] += 1

	return CaseSummary(
		total_candidates=len(records),
		lv1_candidates=lv1_total,
		lv2_candidates=lv2_total,
		repo_stats=repo_stats,
		overall_metrics=overall_metrics,
	)


def format_average(total: float, samples: int) -> str:
	"""Format an average value or return '-' when no samples are available."""
	if samples <= 0:
		return "-"
	return f"{total / samples:.1f}"


def get_all_available_gpus() -> List[int]:
	"""
	Get all available GPU IDs (from the initialized counter).
	
	Returns:
		GPU ID list (sorted ascending); returns empty if uninitialized
	"""
	global _gpu_usage_counter
	
	with _gpu_usage_lock:
		return sorted(_gpu_usage_counter.keys())


def initialize_gpu_tracking(logger) -> None:
	"""
	Initialize GPU usage tracking (in-memory).

	Detect available GPUs and initialize usage counters to 0.

	Args:
		logger: Logger instance
	"""
	global _gpu_usage_counter
	
	try:
		# Use nvidia-smi to detect available GPUs
		result = subprocess.run(
			['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
			capture_output=True,
			text=True,
			timeout=5
		)
		
		if result.returncode == 0:
			gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
			if gpu_ids:
				# Initialize GPU usage counter (thread-safe)
				with _gpu_usage_lock:
					_gpu_usage_counter = {gpu_id: 0 for gpu_id in gpu_ids}
				logger.info(f"✅ Initialized GPU tracking: detected {len(gpu_ids)} GPUs: {gpu_ids}")
			else:
				logger.info("No available GPUs detected; skipping GPU tracking initialization")
		else:
			logger.debug(f"nvidia-smi failed; skipping GPU tracking initialization: {result.stderr}")
	except FileNotFoundError:
		logger.debug("nvidia-smi not found; skipping GPU tracking initialization")
	except Exception as e:
		logger.warning(f"Error initializing GPU tracking: {e}")


def select_and_allocate_gpu(candidate_gpus: List[int], logger, count: int = 1) -> List[int]:
	"""Allocate least-used GPUs; supports multiple allocations."""
	global _gpu_usage_counter

	if not candidate_gpus or count <= 0:
		return []

	selected: List[int] = []

	try:
		with _gpu_usage_lock:
			# Initialize counter on first use
			if not _gpu_usage_counter:
				_gpu_usage_counter = {gpu_id: 0 for gpu_id in candidate_gpus}

			# Ensure candidate GPU exists in counter
			for gpu_id in candidate_gpus:
				_gpu_usage_counter.setdefault(gpu_id, 0)

			logger.debug(f"[GPU Select] Candidates: {candidate_gpus}")
			logger.debug(f"[GPU Select] Current usage: {dict(sorted(_gpu_usage_counter.items()))}")

			available = sorted(set(candidate_gpus))
			for _ in range(count):
				best_gpu = None
				best_usage = float('inf')
				for gpu_id in available:
					usage = _gpu_usage_counter.get(gpu_id, 0)
					if usage < best_usage:
						best_gpu = gpu_id
						best_usage = usage

				if best_gpu is None:
					break

				_gpu_usage_counter[best_gpu] = _gpu_usage_counter.get(best_gpu, 0) + 1
				selected.append(best_gpu)
				available.remove(best_gpu)

			if len(selected) < count:
				logger.warning(
					f"[GPU Select] Requested {count} GPUs but only allocated {len(selected)}"
				)
	except Exception as e:
		logger.warning(f"Error selecting GPUs: {e}")
		if not selected:
			fallback = candidate_gpus[: min(count, len(candidate_gpus))]
			for gid in fallback:
				_gpu_usage_counter[gid] = _gpu_usage_counter.get(gid, 0) + 1
			logger.debug(f"[GPU Select] Using fallback strategy, selected GPUs: {fallback}")
			return fallback

	if selected:
		logger.debug(f"[GPU Select] ✅ Selected GPUs: {selected}")
	return selected


def release_gpu(gpu_id: Optional[Union[int, List[int]]], logger) -> None:
	"""Release GPU usage counters; supports multiple IDs."""
	global _gpu_usage_counter

	if gpu_id is None:
		return

	gpu_ids = gpu_id if isinstance(gpu_id, (list, tuple, set)) else [gpu_id]

	try:
		with _gpu_usage_lock:
			for gid in gpu_ids:
				if gid in _gpu_usage_counter and _gpu_usage_counter[gid] > 0:
					_gpu_usage_counter[gid] -= 1
					logger.debug(
						f"[GPU Release] GPU {gid} usage count: {_gpu_usage_counter[gid] + 1} -> {_gpu_usage_counter[gid]}"
					)
				else:
					logger.warning(f"[GPU Release] GPU {gid} count is 0 or missing; cannot release")
	except Exception as e:
		logger.warning(f"Error releasing GPU {gpu_id}: {e}")

def _serialize_data_item_for_cache(data_item) -> Dict:
	return {
		"repo": data_item.repo,
		"file_path": data_item.file_path,
		"p2p_list": list(data_item.p2p_list),
		"dynamic_trace_file": data_item.dynamic_trace_file,
		"dynamic_trace_files": list(data_item.dynamic_trace_files),
		"updated_top_objects": list(data_item.updated_top_objects),
		"all_top_objects_candidates": list(data_item.all_top_objects_candidates),
		"p2p_test_points": data_item.p2p_test_points,
		"test_count": data_item.test_count,
		"test_count_run": data_item.test_count_run,
		"last_modified": data_item.last_modified,
		"first_commit": data_item.first_commit,
	}


def _build_data_item_from_cache(entry: Dict, DataItem, loaded_repos: Dict):
	repo = entry.get("repo")
	repo_meta = loaded_repos.get(repo)
	if not repo_meta:
		return None
	return DataItem(
		repo=repo,
		specs=repo_meta.get('specs', {}),
		repo_root=repo_meta.get('local_path'),
		file_path=entry.get('file_path'),
		p2p_list=entry.get('p2p_list', []),
		dynamic_trace_file=entry.get('dynamic_trace_file'),
		dynamic_trace_files=entry.get('dynamic_trace_files', []),
		updated_top_objects=entry.get('updated_top_objects', []),
		all_top_objects_candidates=entry.get('all_top_objects_candidates', []),
		p2p_test_points=entry.get('p2p_test_points'),
		test_count=entry.get('test_count'),
		test_count_run=entry.get('test_count_run'),
		last_modified=entry.get('last_modified'),
		first_commit=entry.get('first_commit'),
	)


def _try_load_data_items_cache(cache_path: Path, loaded_repos: Dict, DataItem, logger):
	if not cache_path.exists():
		return None
	try:
		with cache_path.open('r', encoding='utf-8') as f:
			payload = json.load(f)
	except Exception as exc:
		logger.warning(f"Failed to read data_items cache; ignoring cache: {exc}")
		return None

	cached_repos = set(payload.get('repo_names') or [])
	current_repos = set(loaded_repos.keys())
	if not cached_repos or not (cached_repos & current_repos):
		return None

	repo_counts = payload.get('repo_counts', {})
	repo_total_f2p_counts = payload.get('repo_total_f2p_counts', {})
	skip_reasons = payload.get('skip_reasons', {})
	data_items_payload = payload.get('data_items', [])
	data_items_by_repo: Dict[str, List] = defaultdict(list)
	for entry in data_items_payload:
		repo_name = entry.get('repo')
		if not repo_name:
			return None
		item = _build_data_item_from_cache(entry, DataItem, loaded_repos)
		if item is None:
			return None
		data_items_by_repo[repo_name].append(item)
	return data_items_by_repo, repo_counts, repo_total_f2p_counts, skip_reasons


def _update_cache_for_repo(
	cache_path: Path,
	repo_name: str,
	repo_data_items: List,
	repo_count: int,
	repo_total_f2p_count: int,
	repo_skip_reasons: Dict[str, List[str]],
	logger,
) -> None:
	payload = {
		'repo_names': [],
		'repo_counts': {},
		'repo_total_f2p_counts': {},
		'skip_reasons': {},
		'data_items': [],
	}
	if cache_path.exists():
		try:
			with cache_path.open('r', encoding='utf-8') as f:
				payload = json.load(f)
		except Exception as exc:
			logger.warning(f"Failed to read data_items cache; recreating: {exc}")
			payload = {
				'repo_names': [],
				'repo_counts': {},
				'repo_total_f2p_counts': {},
				'skip_reasons': {},
				'data_items': [],
			}

	serialized_items = [_serialize_data_item_for_cache(item) for item in repo_data_items]
	payload['data_items'] = [
		entry for entry in payload.get('data_items', [])
		if entry.get('repo') != repo_name
	]
	payload['data_items'].extend(serialized_items)
	payload.setdefault('repo_counts', {})[repo_name] = repo_count
	payload.setdefault('repo_total_f2p_counts', {})[repo_name] = repo_total_f2p_count
	payload.setdefault('skip_reasons', {})[repo_name] = repo_skip_reasons
	existing_names = set(payload.get('repo_names', []))
	existing_names.add(repo_name)
	payload['repo_names'] = sorted(existing_names)
	temp_path = cache_path.with_suffix('.tmp')
	with temp_path.open('w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)
	temp_path.replace(cache_path)


def _log_data_collection_summary(logger, total_count: int, repo_counts: Dict[str, int], skip_reasons: Dict[str, Dict[str, List[str]]]) -> None:
	logger.info("=" * 60)
	logger.info("Dataset statistics:")
	logger.info("=" * 60)
	logger.info(f"  Total data items: {total_count}")
	for repo, count in repo_counts.items():
		logger.info(f"  {repo}: {count} items")

	logger.info("")
	logger.info("Skip reason summary:")
	for repo, reasons in skip_reasons.items():
		total_skipped = sum(len(files) for files in reasons.values())
		if total_skipped <= 0:
			continue
		logger.info("")
		logger.info(f"{repo}: skipped {total_skipped} files")
		if reasons['test_failed']:
			reason_count = len(reasons['test_failed'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - tests failed: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['test_failed']:
			# 	logger.info(f"      {fp}")
		if reasons['no_p2p_list']:
			reason_count = len(reasons['no_p2p_list'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing p2p_list: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_p2p_list']:
			# 	logger.info(f"      {fp}")
		if reasons['no_dynamic_trace']:
			reason_count = len(reasons['no_dynamic_trace'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing dynamic_trace_file: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_dynamic_trace']:
			# 	logger.info(f"      {fp}")
		if reasons['no_all_top_objects_candidates']:
			reason_count = len(reasons['no_all_top_objects_candidates'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing all_top_objects_candidates: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_all_top_objects_candidates']:
			# 	logger.info(f"      {fp}")
		if reasons['no_top_objects']:
			reason_count = len(reasons['no_top_objects'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing top_objects: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_top_objects']:
			# 	logger.info(f"      {fp}")
		if reasons['no_updated_top_objects']:
			reason_count = len(reasons['no_updated_top_objects'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing updated_top_objects: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_updated_top_objects']:
			# 	logger.info(f"      {fp}")
		if reasons['no_p2p_traces']:
			reason_count = len(reasons['no_p2p_traces'])
			reason_percentage = reason_count / total_skipped * 100
			logger.info(f"  - missing p2p trace files: {reason_count} ({reason_percentage:.1f}%)")
			# for fp in reasons['no_p2p_traces']:
			# 	logger.info(f"      {fp}")


def _build_skip_reason_bucket() -> Dict[str, List[str]]:
	return {
		'test_failed': [],
		'no_p2p_list': [],
		'no_dynamic_trace': [],
		'no_all_top_objects_candidates': [],
		'no_top_objects': [],
		'no_updated_top_objects': [],
		'no_p2p_traces': [],
	}


def collect_data_items(test_results: Dict, loaded_repos: Dict, logger, config, DataItem, repo_manager) -> tuple[List, int, Dict[str, int], Dict[str, int]]:
	"""
	Collect all eligible data items from p2p_chooser test results.
	
	Args:
		test_results: p2p_chooser.test_results with all repo test results
		loaded_repos: repo_manager.loaded_repos with all loaded repos
		logger: Logger instance
		config: Config instance (for debug_sample, etc.)
		DataItem: DataItem class for creating data items
		repo_manager: Repo manager for Git time metadata
		
	Returns:
		tuple: (data_items, total_count, repo_counts, repo_total_f2p_counts)
			- data_items: List of DataItem instances
			- total_count: Total number of data items
			- repo_counts: Dict of data item counts per repo
			- repo_total_f2p_counts: Dict of total f2p file counts per repo
	"""
	logger.info("")
	logger.info("Starting data collection...")
	
	metadata_dir = config.actual_output_dir / "metadata_outputs"
	metadata_dir.mkdir(parents=True, exist_ok=True)
	cache_path = metadata_dir / "data_items_cache.json"

	cached_repo_items: Dict[str, List] = {}
	cached_repo_counts: Dict[str, int] = {}
	cached_repo_total_f2p_counts: Dict[str, int] = {}
	cached_skip_reasons: Dict[str, Dict[str, List[str]]] = {}

	# Top-level keys in data_items_cache.json: repo_names, repo_counts, repo_total_f2p_counts, skip_reasons, data_items
	cached_payload = _try_load_data_items_cache(cache_path, loaded_repos, DataItem, logger)
	if cached_payload:
		(
			cached_repo_items,
			cached_repo_counts,
			cached_repo_total_f2p_counts,
			cached_skip_reasons,
		) = cached_payload

	data_items = []
	total_count = 0
	repo_counts = {}
	repo_total_f2p_counts = {}  # Total f2p file counts per repo
	skip_reasons = {}  # Skip reason counters
	debug_sample_enabled = bool(getattr(config, 'debug_sample', None))
	
	for specs_name, repo_info in loaded_repos.items():
		if specs_name not in test_results:
			continue

		specs = repo_info.get('specs', {})
		use_cache_for_repo = bool(config.get_cache_config('p2p', specs.get('p2p_cache', True)))

		if use_cache_for_repo and specs_name in cached_repo_items:
			cached_items = cached_repo_items[specs_name]
			# If debug_sample is enabled, re-filter cached results to avoid full runs
			if debug_sample_enabled:
				filtered_items = []
				for item in cached_items:
					file_path = None
					try:
						file_path = getattr(item, 'file_path', None)
					except Exception:
						file_path = None
					if file_path is None and isinstance(item, dict):
						file_path = item.get('file_path', None)
					if file_path and config.is_sample_selected(file_path):
						filtered_items.append(item)
			else:
				filtered_items = cached_items

			repo_counts[specs_name] = cached_repo_counts.get(specs_name, len(filtered_items))
			# repo_total_f2p_counts reflects considered items; use filtered count with debug_sample
			if debug_sample_enabled:
				repo_total_f2p_counts[specs_name] = len(filtered_items)
			else:
				repo_total_f2p_counts[specs_name] = cached_repo_total_f2p_counts.get(specs_name, 0)
			skip_reasons[specs_name] = cached_skip_reasons.get(specs_name, _build_skip_reason_bucket())
			data_items.extend(filtered_items)
			total_count += len(filtered_items)
			continue
		
		repo_counts[specs_name] = 0
		skip_reasons[specs_name] = _build_skip_reason_bucket()
		repo_data_items: List = []
		
		f2p_files = test_results[specs_name].get('f2p_files', [])
		p2p_files = test_results[specs_name].get('p2p_files', [])
		p2p_test_count_run_map = {
			info.get('file_path'): info.get('test_count_run')
			for info in p2p_files
			if isinstance(info, dict) and info.get('file_path')
		}
		if debug_sample_enabled:
			filtered_total_candidates = [
				f for f in f2p_files
				if config.is_sample_selected(f.get('file_path', ''))
			]
			repo_total_f2p_counts[specs_name] = len(filtered_total_candidates)
		else:
			repo_total_f2p_counts[specs_name] = len(f2p_files)
		
		for file_info in f2p_files:
			file_path = file_info.get('file_path', 'unknown')
			if debug_sample_enabled and not config.is_sample_selected(file_path):
				continue
			
			# Only collect files that passed tests
			if not file_info.get('passed', False):
				skip_reasons[specs_name]['test_failed'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: tests failed")
				continue
			
			# dynamic_trace_file is required
			dynamic_trace_file = file_info.get('dynamic_trace_file')
			if not dynamic_trace_file:
				skip_reasons[specs_name]['no_dynamic_trace'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: missing dynamic_trace_file")
				continue

			# all_top_objects_candidates is required
			all_top_objects_candidates = file_info.get('all_top_objects_candidates')
			if not all_top_objects_candidates:
				skip_reasons[specs_name]['no_all_top_objects_candidates'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: missing all_top_objects_candidates")
				continue

			# top_objects is required
			top_objects = file_info.get('top_objects')
			if not top_objects:
				skip_reasons[specs_name]['no_top_objects'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: missing top_objects")
				continue

			# updated_top_objects is required
			updated_top_objects = file_info.get('updated_top_objects')
			if not updated_top_objects:
				skip_reasons[specs_name]['no_updated_top_objects'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: missing updated_top_objects")
				continue
			
			# p2p_list is required and must be non-empty
			p2p_list = file_info.get('p2p_list')
			if not p2p_list:
				skip_reasons[specs_name]['no_p2p_list'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: empty or missing p2p_list")
				continue
			
			# Collect dynamic_trace_file for all p2p files
			dynamic_trace_files = []
			for p2p_file_path in p2p_list:
				# Find corresponding file info in p2p_files
				for p2p_info in p2p_files:
					if p2p_info['file_path'] == p2p_file_path:
						p2p_trace_file = p2p_info.get('dynamic_trace_file')
						if p2p_trace_file:
							dynamic_trace_files.append(p2p_trace_file)
						break
			
			# Skip if p2p_list is non-empty but no dynamic_trace_files found
			if not dynamic_trace_files:
				skip_reasons[specs_name]['no_p2p_traces'].append(file_path)
				logger.debug(f"Skip {specs_name}/{file_path}: missing dynamic_trace_file for p2p files")
				continue
			
			# Count p2p test points (run stage)
			p2p_test_points: Optional[float] = 0.0
			p2p_test_points_found = False
			for p2p_file_path in p2p_list:
				value = p2p_test_count_run_map.get(p2p_file_path)
				if value is None:
					continue
				try:
					p2p_test_points += float(value)
					p2p_test_points_found = True
				except (TypeError, ValueError):
					continue
			if not p2p_test_points_found:
				p2p_test_points = None

			# Get repo specs config
			repo_specs = repo_info.get('specs', {})
			
			# Get Git time metadata
			last_modified_timestamp = repo_manager.get_file_git_last_modified_time(specs_name, file_path)
			first_commit_timestamp = repo_manager.get_file_git_first_commit_time(specs_name, file_path)
			
			# Convert to ISO format strings
			last_modified = datetime.fromtimestamp(last_modified_timestamp).isoformat() if last_modified_timestamp else None
			first_commit = datetime.fromtimestamp(first_commit_timestamp).isoformat() if first_commit_timestamp else None
			
			# Build DataItem
			data_item = DataItem(
				repo=specs_name,
				specs=repo_specs,
				repo_root=repo_info['local_path'],
				file_path=file_info['file_path'],
				p2p_list=p2p_list,
				dynamic_trace_file=dynamic_trace_file,
				dynamic_trace_files=dynamic_trace_files,
				updated_top_objects=updated_top_objects,
				all_top_objects_candidates=all_top_objects_candidates,
				p2p_test_points=p2p_test_points,
				test_count=file_info.get('test_count'),  # Test point count
				test_count_run=file_info.get('test_count_run'),  # Executed test point count
				last_modified=last_modified,             # Last modified time
				first_commit=first_commit                # First commit time
			)
			
			data_items.append(data_item)
			repo_data_items.append(data_item)
			total_count += 1
			repo_counts[specs_name] += 1

		# Update cache (memory + disk)
		cached_repo_items[specs_name] = list(repo_data_items)
		cached_repo_counts[specs_name] = repo_counts[specs_name]
		cached_repo_total_f2p_counts[specs_name] = repo_total_f2p_counts[specs_name]
		cached_skip_reasons[specs_name] = skip_reasons[specs_name]
		_update_cache_for_repo(
			cache_path,
			specs_name,
			repo_data_items,
			repo_counts[specs_name],
			repo_total_f2p_counts[specs_name],
			skip_reasons[specs_name],
			logger,
		)
	
	_log_data_collection_summary(logger, total_count, repo_counts, skip_reasons)

	return data_items, total_count, repo_counts, repo_total_f2p_counts



def print_data_processing_statistics(
	processing_results: List[Dict],
	logger,
) -> None:

	def _collect_error_text(entry: Dict) -> str:
		value = entry.get("error")
		if value is None:
			return ""
		if isinstance(value, list):
			messages = [str(item).strip() for item in value if item]
			return "\n".join([m for m in messages if m])
		return str(value).strip()

	def _collect_scoped_errors(entry: Dict) -> Dict[str, List[str]]:
		return split_errors_by_scope(ensure_list(entry.get("error")))

	def _classify_error(message: str) -> str:
		if "Missing top objects" in message:
			return "Missing top objects"
		if "Mask generation failed" in message:
			return "Mask generation failed"
		if "Mask result missing top objects" in message:
			return "Mask result missing top objects"
		if "F2P" in message:
			return "F2P post-validation failed"
		if "P2P" in message:
			return "P2P post-validation failed"
		if "Level2 post-validation pass rate too high" in message:
			return "Level2 post-validation failed"
		if "LLM error" in message:
			return "LLM error"
		if "Processing timeout" in message:
			return "Processing timeout"
		return "Other"

	repo_stats = defaultdict(lambda: {
		"total": 0,
		"success_lv1": 0,
		"success_lv2": 0,
		"cached": 0,
		"failed": 0,
		"failed_lv1": 0,
		"failed_lv2": 0,
		"error_types": {
			"Level1": defaultdict(int),
			"Level2": defaultdict(int),
		},
		"deleted_lines_sum": 0,
		"deleted_lines_count": 0,
		"mask_file_total": 0,
		"mask_file_samples": 0,
		"mask_object_total": 0,
		"mask_object_samples": 0,
		"test_count_sum": 0,
		"test_count_samples": 0,
	})
	success_records: List[CaseRecord] = []
	success_cached_count = 0

	for result in processing_results:
		repo = result.get("repo", "unknown_repo")
		stats = repo_stats[repo]
		stats["total"] += 1
		if result.get("cached"):
			stats["cached"] += 1
		if result.get("success_lv1"):
			stats["success_lv1"] += 1
		if result.get("success_lv2"):
			stats["success_lv2"] += 1
		lv1_failed = not result.get("success_lv1")
		lv2_failed = not result.get("success_lv2")
		if lv1_failed or lv2_failed:
			stats["failed"] += 1
		if lv1_failed:
			stats["failed_lv1"] += 1
		if lv2_failed:
			stats["failed_lv2"] += 1
		if lv1_failed or lv2_failed:
			scoped_errors = _collect_scoped_errors(result)
			default_error = _collect_error_text(result) or "Unknown error"
			for level, failed_flag in (("Level1", lv1_failed), ("Level2", lv2_failed)):
				if not failed_flag:
					continue
				messages = list(scoped_errors[level]) + list(scoped_errors["Both"])
				if not messages:
					messages = [default_error]
				for message in messages:
					error_type = _classify_error(message)
					stats["error_types"][level][error_type] += 1

		deleted_lines = result.get("deleted_lines")
		if isinstance(deleted_lines, (int, float)):
			stats["deleted_lines_sum"] += deleted_lines
			stats["deleted_lines_count"] += 1

		mask_file_count = result.get("mask_file_count")
		if isinstance(mask_file_count, int):
			stats["mask_file_total"] += mask_file_count
			stats["mask_file_samples"] += 1

		mask_object_count = result.get("mask_object_count")
		if isinstance(mask_object_count, int):
			stats["mask_object_total"] += mask_object_count
			stats["mask_object_samples"] += 1

		test_count_value = result.get("test_count")
		if isinstance(test_count_value, (int, float)):
			stats["test_count_sum"] += test_count_value
			stats["test_count_samples"] += 1

		case_dirs = result.get("case_dirs") or {}
		file_path = result.get("file_path", "")
		if (result.get("success_lv1") or result.get("success_lv2")) and result.get("cached"):
			success_cached_count += 1

		if result.get("success_lv1") and case_dirs.get("lv1"):
			success_records.append(
				CaseRecord(
					repo=repo,
					file_path=file_path,
					level="lv1",
					source_dir=Path(case_dirs["lv1"]),
					reason="pipeline-success",
					pass_rate=result.get("lv1_post_rate"),
					status="lv1",
					deleted_lines=deleted_lines if isinstance(deleted_lines, (int, float)) else None,
					mask_file_count=mask_file_count if isinstance(mask_file_count, int) else None,
					mask_object_count=mask_object_count if isinstance(mask_object_count, int) else None,
					test_count=test_count_value if isinstance(test_count_value, (int, float)) else None,
					test_count_run=result.get("test_count_run"),
				)
			)
		if result.get("success_lv2") and case_dirs.get("lv2"):
			success_records.append(
				CaseRecord(
					repo=repo,
					file_path=file_path,
					level="lv2",
					source_dir=Path(case_dirs["lv2"]),
					reason="pipeline-success",
					pass_rate=result.get("lv2_post_rate"),
					status="lv2",
					deleted_lines=deleted_lines if isinstance(deleted_lines, (int, float)) else None,
					mask_file_count=mask_file_count if isinstance(mask_file_count, int) else None,
					mask_object_count=mask_object_count if isinstance(mask_object_count, int) else None,
					test_count=test_count_value if isinstance(test_count_value, (int, float)) else None,
					test_count_run=result.get("test_count_run"),
				)
			)

	_log_success_case_summary(logger, success_records, repo_stats, success_cached_count)


def _log_success_case_summary(
	logger,
	records: Sequence[CaseRecord],
	repo_processing_stats: Dict[str, Dict[str, Union[int, float, Dict]]],
	success_cached_count: int,
) -> None:
	logger.info("=" * 60)
	logger.info("Statistics:")
	logger.info("=" * 60)

	overall_total = sum(stats.get("total", 0) for stats in repo_processing_stats.values())
	summary = compute_case_summary(records)
	overall = summary.overall_metrics
	logger.info(f"  Total data items: {overall_total}")
	logger.info(f"  Level1 processed successfully: {summary.lv1_candidates}")
	logger.info(f"  Level2 processed successfully: {summary.lv2_candidates}")
	logger.info(f"  Cache hits: {success_cached_count}")
	logger.info(f"  Avg deleted lines: {format_average(overall['deleted_sum'], overall['deleted_samples'])}")
	logger.info(f"  Avg mask files: {format_average(overall['mask_file_sum'], overall['mask_file_samples'])}")
	logger.info(f"  Avg mask objects: {format_average(overall['mask_object_sum'], overall['mask_object_samples'])}")
	logger.info(f"  Avg test points: {format_average(overall['test_count_sum'], overall['test_count_samples'])}")

	repo_names = sorted(set(repo_processing_stats.keys()) | set(summary.repo_stats.keys()))
	if not repo_names:
		return

	logger.info("")
	logger.info("Per-repo case statistics:")
	logger.info("-" * 60)
	for repo in repo_names:
		processing = repo_processing_stats.get(repo, {})
		total = processing.get("total", 0)
		lv1_success = processing.get("success_lv1", 0)
		lv2_success = processing.get("success_lv2", 0)
		logger.info("")
		logger.info(f"{repo}:")
		logger.info(f"  Total: {total}")
		if total:
			logger.info(f"  Level1 processed successfully: {lv1_success} ({lv1_success / total * 100:.1f}%)")
			logger.info(f"  Level2 processed successfully: {lv2_success} ({lv2_success / total * 100:.1f}%)")
		else:
			logger.info("  Level1 processed successfully: 0")
			logger.info("  Level2 processed successfully: 0")

		success_stats = summary.repo_stats.get(repo)
		if success_stats:
			logger.info(f"  Avg deleted lines: {format_average(success_stats['deleted_sum'], success_stats['deleted_samples'])}")
			logger.info(f"  Avg mask files: {format_average(success_stats['mask_file_sum'], success_stats['mask_file_samples'])}")
			logger.info(f"  Avg mask objects: {format_average(success_stats['mask_object_sum'], success_stats['mask_object_samples'])}")
			logger.info(f"  Avg test points: {format_average(success_stats['test_count_sum'], success_stats['test_count_samples'])}")
		else:
			logger.info("  Avg deleted lines: -")
			logger.info("  Avg mask files: -")
			logger.info("  Avg mask objects: -")
			logger.info("  Avg test points: -")

		failed = processing.get("failed", 0)
		error_types = processing.get("error_types", {}) or {}
		failed_lv1 = processing.get("failed_lv1", 0)
		failed_lv2 = processing.get("failed_lv2", 0)
		if failed:
			for level, failed_count in (("Level1", failed_lv1), ("Level2", failed_lv2)):
				level_errors = (error_types.get(level) or {}) if isinstance(error_types, dict) else {}
				if not failed_count or not level_errors:
					continue
				logger.info(f"  {level} failure type distribution:")
				total_level_errors = sum(level_errors.values()) or failed_count
				for error_type, count in sorted(level_errors.items(), key=lambda x: -x[1]):
					logger.info(
						f"    - {error_type}: {count} ({count / total_level_errors * 100:.1f}%)"
					)

