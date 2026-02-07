import logging
import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from featurebench.utils.config import Config


def make_json_serializable(obj):
	"""Recursively convert objects into JSON-serializable values."""
	if isinstance(obj, dict):
		return {key: make_json_serializable(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [make_json_serializable(item) for item in obj]
	elif isinstance(obj, (str, int, float, bool)) or obj is None:
		return obj
	else:
		# For other types (e.g., URL objects), convert to string.
		return str(obj)


def save_llm_api_call(
	msg_list: List[Dict[str, str]], 
	completion, 
	start_time: float, 
	end_time: float, 
	test_file: str,
	llm_model: str,
	llm_client,
	logs_dir: Optional[Path] = None,
	logger: Optional[logging.Logger] = None
) -> None:
	"""
	Save an LLM API call record to a JSONL file.
	
	Args:
		msg_list: Message list
		completion: LLM completion object
		start_time: Start time
		end_time: End time
		test_file: Test file path
		llm_model: LLM model name
		llm_client: LLM client object
		logs_dir: Log directory; uses default location when None
		logger: Logger instance
	"""
	if logger is None:
		logger = logging.getLogger(__name__)
	
	try:
		# Create log directory
		if logs_dir:
			log_dir = Path(logs_dir) / 'llm_api_calls'
		else:
			current_file = Path(__file__)
			log_dir = current_file.parent.parent / "logs" / 'llm_api_calls'
		
		log_dir.mkdir(parents=True, exist_ok=True)
		
		# Build log file path
		model_name_safe = llm_model.replace('/', '_').replace('-', '_')
		llm_out_path = log_dir / f"api_call_{model_name_safe}.jsonl"
		
		# Build log payload
		try:
			# Safely serialize the completion object
			if hasattr(completion, 'model_dump'):
				resp_data = completion.model_dump()
				# Recursively handle non-JSON-serializable objects
				resp_data = make_json_serializable(resp_data)
			else:
				resp_data = str(completion)
		except Exception as e:
			logger.warning(f"Failed to serialize completion object: {e}; using string form")
			resp_data = str(completion)
		
		line_data = {
			"provider": str(getattr(llm_client, 'base_url', 'unknown')),
			"model_name": llm_model,
			"elapsed_sec": end_time - start_time,
			"test_file": test_file,
			"msg_list": msg_list,
			"resp": resp_data,
		}
		
		# Write to log file
		try:
			line = f"{json.dumps(line_data, ensure_ascii=False)}\n"
		except TypeError as e:
			logger.warning(f"JSON serialization failed: {e}; using fallback serializer")
			# Fallback: drop the potentially problematic resp field
			line_data_safe = {k: v for k, v in line_data.items() if k != 'resp'}
			line_data_safe['resp'] = "Unserializable response object"
			line = f"{json.dumps(line_data_safe, ensure_ascii=False)}\n"
		
		with open(llm_out_path, "a+", encoding="utf-8") as f:
			f.write(line)
		
		logger.debug(f"API call record saved to: {llm_out_path}")
		
	except Exception as e:
		logger.error(f"Failed to save API call record: {e}")


class DualOutput:
	"""Dual output that writes to both terminal and file."""
	
	def __init__(self, terminal, log_file):
		self.terminal = terminal
		self.log_file = log_file
	
	def write(self, message):
		self.terminal.write(message)
		self.log_file.write(message)
		self.log_file.flush()  # Ensure immediate flush to file
	
	def flush(self):
		self.terminal.flush()
		self.log_file.flush()


def configure_logging(config: Config):
	"""Configure logging with dual output to terminal and file."""
	# Basic logging configuration
	logging.basicConfig(
		level=getattr(logging, config.log_level, logging.INFO),
		format="%(asctime)s [%(levelname)s]: %(message)s",
		datefmt="%H:%M:%S"
	)
	
	# Enable real-time dual output
	setup_real_time_logging(config.actual_output_dir)
	
	return logging.getLogger(__name__)


def setup_real_time_logging(log_dir):
	"""Enable real-time dual output and capture terminal output."""
	
	# Create debug_outputs/terminal directory structure
	terminal_log_dir = log_dir / "debug_outputs" / "terminal"
	os.makedirs(terminal_log_dir, exist_ok=True)
	
	# Generate timestamp-based filename
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file_path = terminal_log_dir / f"terminal_{timestamp}.log"
	
	log_fd = open(log_file_path, 'w', encoding='utf-8')
	
	# Create dual output objects
	dual_stdout = DualOutput(sys.stdout, log_fd)
	dual_stderr = DualOutput(sys.stderr, log_fd)
	
	# Redirect stdout and stderr
	sys.stdout = dual_stdout
	sys.stderr = dual_stderr
	
	# Add a file handler to the root logger so logging output is saved
	root_logger = logging.getLogger()
	
	# Create a file handler using the same file as log_fd
	file_handler = logging.StreamHandler(log_fd)
	file_handler.setLevel(logging.DEBUG)  # Capture all log levels
	
	# Set formatter to match existing format
	formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
	file_handler.setFormatter(formatter)
	
	# Attach file handler to root logger
	root_logger.addHandler(file_handler)
	
	print(f"=== Terminal output is saved in real time to: {log_file_path} ===")
	
	return str(log_file_path)


def create_docker_build_logger(image_type: str, image_name: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for Docker build steps.
	
	Args:
		image_type: Image type, 'base' or 'instance'
		image_name: Image name
		featurebench_root: FeatureBench root path; auto-detected when None
		
	Returns:
		Path to the log file
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	else:
		logs_dir = logs_dir

	# Create log directory structure
	logs_dir.mkdir(parents=True, exist_ok=True)
	build_logs_dir = logs_dir / image_type
	build_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename: image_name_timestamp.log
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	# Replace special characters in image name with safe ones
	safe_image_name = image_name.replace(":", "_").replace("/", "_")
	log_filename = f"{safe_image_name}_{timestamp}.log"
	log_file_path = build_logs_dir / log_filename
	
	return log_file_path


def run_subprocess_with_logging(
	command: list, 
	log_file_path: Path, 
	logger: logging.Logger = None,
	parallel_mode: bool = False,
	specs_name: str = None,
	build_type: str = None,
	is_rebuild: bool = False
) -> int:
	"""
	Run subprocess and stream output to a log file.
	
	Args:
		command: Command list to execute
		log_file_path: Log file path
		logger: Logger instance
		parallel_mode: Whether running in parallel mode (no console logs)
		specs_name: Repo spec name (used in parallel mode output)
		build_type: Build type ('base' or 'instance', used in parallel mode output)
		is_rebuild: Whether this is a forced rebuild
		
	Returns:
		Process return code
	"""
	if not parallel_mode:
		logger.info(f"Starting command: {' '.join(command)}")
		logger.info(f"Build log will be saved to: {log_file_path}")
	else:
		if build_type == 'base':
			if is_rebuild:
				tqdm.write(f"🔄 {specs_name}: Forcing base image rebuild, log: {log_file_path}")
			else:
				tqdm.write(f"🏗️ {specs_name}: Building base image, log: {log_file_path}")
		elif build_type == 'instance':
			if is_rebuild:
				tqdm.write(f"🔄 {specs_name}: Forcing instance image rebuild, log: {log_file_path}")
			else:
				tqdm.write(f"🔨 {specs_name}: Building instance image, log: {log_file_path}")
	
	try:
		with open(log_file_path, 'w', encoding='utf-8') as log_file:
			# Write log header
			log_file.write(f"=== Docker Build Log ===\n")
			log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			log_file.write(f"Command: {' '.join(command)}\n")
			log_file.write(f"{'='*50}\n\n")
			log_file.flush()
			
			# Start subprocess
			process = subprocess.Popen(
				command,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
				universal_newlines=True
			)
			
			# Stream and record output
			for line in iter(process.stdout.readline, ''):
				if line.strip():  # Skip empty lines
					# Write to log file
					log_file.write(line)
					log_file.flush()
			
			# Wait for process to finish
			process.wait()
			
			# Write log footer
			log_file.write(f"\n{'='*50}\n")
			log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			log_file.write(f"Return code: {process.returncode}\n")
			
			return process.returncode
			
	except Exception as e:
		if 'process' in locals():
			process.terminate()
		if not parallel_mode:
			logger.error(f"Exception while running command: {e}")
		raise


def print_build_report(
	logger: logging.Logger,
	total_repos: list,
	failed_repos: list,
	success_info: dict,
	operation_name: str = "Operation"
) -> None:
	"""
	Print a report.
	
	Args:
		logger: Logger instance
		total_repos: List of all repos
		failed_repos: Failed repos [(repo_name, error_message), ...]
		success_info: Success info {repo_name: {key1: value1, key2: value2}}
		operation_name: Operation name for the report title
	"""
	total_count = len(total_repos)
	success_count = total_count - len(failed_repos)
	
	logger.info("-" * 60)
	logger.info(f"{operation_name} report")
	logger.info("-" * 60)
	logger.info(f"Total: {total_count} repos")
	logger.info(f"Succeeded: {success_count}")
	logger.info(f"Failed: {len(failed_repos)}")
	
	if failed_repos:
		logger.info("")
		logger.info("Failed repo details:")
		for i, (repo_name, error_msg) in enumerate(failed_repos, 1):
			logger.error(f"  {i}. {repo_name}: {error_msg}")
	
	if success_count > 0:
		logger.info("")
		logger.info("Successful repos:")
		for specs_name, info in success_info.items():
			logger.info(f"  • {specs_name}")
			for key, value in info.items():
				logger.info(f"    {key}: {value}")
	
	logger.info("-" * 60)


def create_test_scanner_logger(specs_name: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for test scanning.
	
	Args:
		specs_name: Repo spec name
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	else:
		# If log dir is provided, create logs subdir under it
		logs_dir = logs_dir
	
	# Create test_scanner log directory
	test_scanner_logs_dir = logs_dir / "test_scanner"
	test_scanner_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename: specs_name_timestamp.log
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{specs_name}_{timestamp}.log"
	log_file_path = test_scanner_logs_dir / log_filename
	
	return log_file_path


def create_test_runner_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for test runs.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/test_runner/{specs_name}/
	test_runner_logs_dir = logs_dir / "test_runner" / specs_name
	test_runner_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	# Example: /testbed/test/chunked_loss/test_cosine_loss.py -> test_cosine_loss
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = test_runner_logs_dir / log_filename
	
	return log_file_path


def create_dynamic_tracer_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> tuple[Path, Path]:
	"""
	Create a dedicated log file for dynamic tracing (normal and collect stages).
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		(Normal trace log path, collect trace log path)
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/dynamic_trace/{specs_name}/
	dynamic_trace_logs_dir = logs_dir / "dynamic_trace" / specs_name
	dynamic_trace_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	# Example: /testbed/test/chunked_loss/test_cosine_loss.py -> test_cosine_loss
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Normal trace log
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = dynamic_trace_logs_dir / log_filename
	
	# Collect trace log
	collect_log_filename = f"{test_filename}_{timestamp}_collect.log"
	collect_log_file_path = dynamic_trace_logs_dir / collect_log_filename
	
	return log_file_path, collect_log_file_path


def create_mask_generator_logger(specs_name: str, test_file: str, container_id: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for mask generator pytest collect.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		container_id: Docker container ID (full or short)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/mask_generator/{specs_name}/
	mask_generator_logs_dir = logs_dir / "mask_generator" / specs_name
	mask_generator_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	# Example: /testbed/test/chunked_loss/test_cosine_loss.py -> test_cosine_loss
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%H%M%S")
	container_short_id = container_id[:12] if len(container_id) >= 12 else container_id
	log_filename = f"{test_filename}_pytest_collect_{container_short_id}_{timestamp}.log"
	log_file_path = mask_generator_logs_dir / log_filename
	
	return log_file_path


def create_p2p_chooser_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> logging.Logger:
	"""
	Create a dedicated logger for p2p selection and write to the specified log file.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		logging.Logger: Configured logger instance
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/p2p_choose/{specs_name}/
	p2p_choose_logs_dir = logs_dir / "p2p_choose" / specs_name
	p2p_choose_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	# Example: /testbed/test/chunked_loss/test_cosine_loss.py -> test_cosine_loss
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = p2p_choose_logs_dir / log_filename
	
	# Create a unique logger name
	logger_name = f"p2p_chooser_{specs_name}_{test_filename}_{timestamp}"
	logger = logging.getLogger(logger_name)
	
	# Clear any existing handlers
	logger.handlers.clear()
	
	# Create file handler
	file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
	file_handler.setLevel(logging.DEBUG)
	
	# Set formatter (minimal: time and message)
	formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
	file_handler.setFormatter(formatter)
	
	# Add handler to logger
	logger.addHandler(file_handler)
	logger.setLevel(logging.DEBUG)
	
	# Prevent propagation to root logger to avoid duplicates
	logger.propagate = False
	
	# Write log header
	logger.info("=== P2P Selection Log ===")
	logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	logger.info(f"Repo: {specs_name}")
	logger.info(f"Test file: {test_file}")
	logger.info("=" * 50)
	logger.info("")
	
	# Attach log file path for later use
	logger.log_file_path = log_file_path
	
	return logger


def run_command_with_streaming_log(
	command: list,
	log_file_path: Path,
	timeout: int = None,
	log_header: dict = None
) -> subprocess.CompletedProcess:
	"""
	Execute a command and stream logs to a file.
	
	Args:
		command: Command list to execute
		log_file_path: Log file path
		timeout: Timeout in seconds
		log_header: Log header dict, e.g., {'Container ID': 'xxx', 'Command': 'xxx'}
		
	Returns:
		subprocess.CompletedProcess instance
	"""
	with open(log_file_path, 'w', encoding='utf-8') as log_file:
		# Write log header
		log_file.write(f"=== Test Run Log ===\n")
		log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		
		# Write custom header info
		if log_header:
			for key, value in log_header.items():
				log_file.write(f"{key}: {value}\n")
		
		log_file.write(f"{'='*50}\n\n")
		log_file.flush()
		
		try:
			# Use subprocess.run to redirect output to file
			# This avoids blocking and preserves timeout behavior
			result = subprocess.run(
				command,
				stdout=log_file,  # Pass file object; subprocess writes streaming
				stderr=subprocess.STDOUT,  # Merge stderr into stdout
				text=True,
				timeout=timeout
			)
			
			# Write log footer
			log_file.write(f"\n{'='*50}\n")
			log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			log_file.write(f"Return code: {result.returncode}\n")
			log_file.write(f"Test result: {'PASS ✅' if result.returncode == 0 else 'FAIL ❌'}\n")
			log_file.flush()
			
			return result
			
		except subprocess.TimeoutExpired as e:
			log_file.write(f"\n{'='*50}\n")
			log_file.write(f"Execution timed out: {timeout}s\n")
			log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			log_file.flush()
			raise
		except Exception as e:
			log_file.write(f"\n{'='*50}\n")
			log_file.write(f"Execution error: {str(e)}\n")
			log_file.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			log_file.flush()
			raise


def create_f2p_validator_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for F2P validation.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/f2p_validate/{specs_name}/
	f2p_validate_logs_dir = logs_dir / "f2p_validate" / specs_name
	f2p_validate_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = f2p_validate_logs_dir / log_filename
	
	return log_file_path

def create_level2_validator_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for Level2 validation.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/level2_validate/{specs_name}/
	level2_validate_logs_dir = logs_dir / "level2_validate" / specs_name
	level2_validate_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = level2_validate_logs_dir / log_filename
	
	return log_file_path

def create_p2p_validator_logger(specs_name: str, test_file: str, logs_dir: Path = None) -> Path:
	"""
	Create a dedicated log file for P2P validation.
	
	Args:
		specs_name: Repo spec name
		test_file: Test file path (container path)
		logs_dir: Log directory path; uses default if None
		
	Returns:
		Log file path
	"""
	if logs_dir is None:
		# Use default featurebench/logs if no log dir is provided
		current_file = Path(__file__)
		logs_dir = current_file.parent.parent / "logs"
	
	# Create log dir structure: logs/p2p_validate/{specs_name}/
	p2p_validate_logs_dir = logs_dir / "p2p_validate" / specs_name
	p2p_validate_logs_dir.mkdir(parents=True, exist_ok=True)
	
	# Build log filename from test file path
	test_filename = Path(test_file).stem
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_filename = f"{test_filename}_{timestamp}.log"
	log_file_path = p2p_validate_logs_dir / log_filename
	
	return log_file_path
