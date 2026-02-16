import argparse
import sys
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import tomllib


class Config:
	"""Configuration container for CLI arguments."""

	def __init__(
		self,
		config_path: Path,
		output_dir: Path,
		actual_output_dir: Path,
		resume: Optional[str] = None,
		seed: Optional[int] = None,
		debug_end_stage: Optional[str] = None,
		debug_cache_overrides: Optional[Dict[str, bool]] = None,
		debug_repo: Optional[List[str]] = None,
		debug_sample: Optional[List[str]] = None,
		log_level: str = "INFO",
		logs_dir: Optional[Path] = None,
		repos_dir: Optional[Path] = None,
		gpu_ids: Optional[List[int]] = None,
		cmd: Optional[str] = None,
		env_vars: Optional[Dict[str, str]] = None,
		llm_config: Optional[Dict[str, Any]] = None,
	):
		self.config_path = config_path
		self.output_dir = output_dir  # Root output dir provided by user
		self.actual_output_dir = actual_output_dir  # Output dir for this run
		self.resume = resume
		self.seed = seed
		self.debug_end_stage = debug_end_stage  # Stop after this stage when debugging
		self.debug_cache_overrides = debug_cache_overrides or {}  # Per-stage cache overrides
		self.debug_repo = debug_repo
		self.debug_sample = debug_sample
		self.log_level = log_level
		# Log directory; defaults to actual_output_dir/logs when None
		self.logs_dir = logs_dir if logs_dir is not None else actual_output_dir / "logs"
		self.repos_dir = repos_dir  # Repo storage dir; default location if None
		self.gpu_ids = gpu_ids  # GPU rank IDs from CLI (e.g., [0, 1, 2]); None means unset
		self.cmd = cmd  # Full command line
		self.env_vars = env_vars or {}  # Environment variable config
		self.llm_config = llm_config or {}  # LLM config

	@staticmethod
	def _validate_arguments(args, actual_output_dir: Path):
		"""Validate CLI arguments."""
		# Validate --resume
		if args.resume:
			if not actual_output_dir.exists():
				raise FileNotFoundError(
					f"Resume directory does not exist: {actual_output_dir}\n"
					f"Please check --resume; expected format: YYYY-MM-DD/HH-MM-SS"
				)
			if not actual_output_dir.is_dir():
				raise NotADirectoryError(
					f"Resume path is not a directory: {actual_output_dir}"
				)
		
		# Ensure config file exists
		config_path = Path(args.config_path)
		if not config_path.exists():
			raise FileNotFoundError(f"Config file does not exist: {config_path}")
		if not config_path.is_file():
			raise IsADirectoryError(f"Config path is a directory, not a file: {config_path}")

	@staticmethod
	def _parse_debug_config(debug_str: str) -> Tuple[Optional[str], Dict[str, bool]]:
		"""
		Parse debug configuration string.
		
		Args:
			debug_str: Debug config string, e.g. "end=scanner, repo=False, scanner=False"
			
		Returns:
			tuple: (debug_end_stage, debug_cache_overrides)
		"""
		# Valid stage names
		valid_stages = {"repo", "image", "scanner", "runner", "dynamic", "top", "p2p", "data"}
		
		debug_end_stage = None
		debug_cache_overrides = {}
		
		# Split options
		parts = [part.strip() for part in debug_str.split(",")]
		
		for part in parts:
			if "=" not in part:
				raise ValueError(f"Invalid debug argument format: '{part}'. Expected 'key=value'.")
			
			key, value = part.split("=", 1)
			key = key.strip()
			value = value.strip()
			
			if key == "end":
				# Parse end
				if value not in valid_stages:
					raise ValueError(
						f"Invalid value for debug 'end': '{value}'. "
						f"Must be one of: {', '.join(sorted(valid_stages))}"
					)
				debug_end_stage = value
			elif key in valid_stages:
				# Parse per-stage cache overrides
				if value.lower() == "true":
					debug_cache_overrides[key] = True
				elif value.lower() == "false":
					debug_cache_overrides[key] = False
				else:
					raise ValueError(
						f"Invalid value for debug '{key}': '{value}'. Must be 'True' or 'False'."
					)
			else:
				raise ValueError(
					f"Unknown key in debug config: '{key}'. "
					f"Expected 'end' or one of: {', '.join(sorted(valid_stages))}"
				)
		
		return debug_end_stage, debug_cache_overrides

	@staticmethod
	def _parse_gpu_ids(gpu_ids_str: Optional[str]) -> Optional[List[int]]:
		"""Parse --gpu-ids value (comma-separated integers)."""
		if gpu_ids_str is None:
			return None

		parts = [part.strip() for part in gpu_ids_str.split(",")]
		parts = [part for part in parts if part]
		if not parts:
			raise ValueError("Invalid --gpu-ids: empty value; expected format like '0,1,2'")

		parsed_ids: List[int] = []
		seen = set()
		for part in parts:
			if not part.isdigit():
				raise ValueError(
					f"Invalid GPU id '{part}' in --gpu-ids; expected non-negative integers like '0,1,2'"
				)
			gpu_id = int(part)
			if gpu_id in seen:
				continue
			seen.add(gpu_id)
			parsed_ids.append(gpu_id)

		return parsed_ids
	
	def get_cache_config(self, stage: str, default: bool = True) -> bool:
		"""
		Get cache configuration for a stage.
		
		Args:
			stage: Stage name (repo/image/scanner/runner/dynamic/top/p2p/data)
			default: Default value (from specs config)
			
		Returns:
			bool: Whether cache is enabled
		"""
		# Use override when present
		if stage in self.debug_cache_overrides:
			return self.debug_cache_overrides[stage]
		# Otherwise fall back to default
		return default

	def is_sample_selected(self, file_path: str) -> bool:
		"""Check whether a file matches debug_sample filter."""
		if not self.debug_sample:
			return True
		try:
			file_stem = Path(file_path).stem
		except Exception:
			file_stem = Path(str(file_path)).stem
		return file_stem in set(self.debug_sample)
	
	def should_stop_after(self, stage: str) -> bool:
		"""
		Check whether execution should stop after a stage.
		
		Args:
			stage: Stage name
			
		Returns:
			bool: Whether to stop
		"""
		return self.debug_end_stage == stage

	@staticmethod
	def _parse_global_configs(global_config_path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
		"""
		Parse global config file.
		
		Args:
			global_config_path: Global config file path
			
		Returns:
			tuple: (env_vars, llm_config)
				- env_vars: Environment variables dict
				- llm_config: LLM config dict with base + model config
		"""
		if not global_config_path.exists():
			raise FileNotFoundError(f"Global config file does not exist: {global_config_path}")
		
		# Read TOML config
		# tomllib (Python 3.11+) needs binary mode; toml/tomli use text mode
		try:
			# Try tomllib (Python 3.11+)
			with open(global_config_path, "rb") as f:
				config_data = tomllib.load(f)
		except (AttributeError, TypeError):
			# Fall back to toml/tomli
			with open(global_config_path, "r", encoding="utf-8") as f:
				config_data = tomllib.load(f)
		
		# Parse env vars
		env_vars = config_data.get("env_vars", {})
		
		# Parse base llm_config
		llm_base_config = config_data.get("llm_config", {})
		llm_name = llm_base_config.get("llm_name")
		
		# Initialize llm_config
		llm_config = {
			"llm_name": llm_name,
			"llm_temperature": llm_base_config.get("llm_temperature", 0.0),
			"llm_max_tokens": llm_base_config.get("llm_max_tokens", 4096),
		}
		
		# If llm_name is set, load details from llm.<name>
		if llm_name:
			llm_specific_config = config_data.get("llm", {}).get(llm_name)
			if llm_specific_config:
				llm_config.update({
					"model": llm_specific_config.get("model"),
					"api_key": llm_specific_config.get("api_key"),
					"base_url": llm_specific_config.get("base_url"),
				})
				# api_version is optional (Azure only)
				if "api_version" in llm_specific_config:
					llm_config["api_version"] = llm_specific_config["api_version"]
				# backend is optional (labels vLLM or other local backends)
				if "backend" in llm_specific_config:
					llm_config["backend"] = llm_specific_config["backend"]
			else:
				raise ValueError(
					f"LLM '{llm_name}' not found in the global config.\n"
					f"Please check the [llm.{llm_name}] section exists."
				)
		
		return env_vars, llm_config

	@classmethod
	def _init_from_cli(cls) -> "Config":
		"""Initialize Config from CLI args."""
		# Capture full command line
		cmd = " ".join(sys.argv)
		
		parser = argparse.ArgumentParser(
			prog="featurebench-pipeline",
			description="FeatureBench main pipeline entry"
		)
		parser.add_argument(
			"--config-path",
			required=True,
			help="Path to config file under constants/",
		)
		parser.add_argument(
			"--output-dir",
			required=True,
			help="Output root for this run (a timestamp subdir will be created)",
		)
		parser.add_argument(
			"--resume",
			default=None,
			help="Resume strategy: None or 2025-09-21/14-23-45",
		)
		parser.add_argument(
			"--seed",
			type=int,
			default=None,
			help="Random seed for reproducibility",
		)
		parser.add_argument(
			"--debug",
			default=None,
			help=(
				"Debug config, format: 'end=STAGE, stage1=True/False, stage2=True/False'\n"
				"- end: stop after a stage; options: repo/image/scanner/runner/dynamic/top/p2p\n"
				"- stage=True/False: override cache for a stage; options: scanner/runner/dynamic/top/p2p/data\n"
				"Example: 'end=scanner, repo=False, scanner=False'"
			),
		)
		parser.add_argument(
			"--debug-repo",
			nargs="*",
			default=None,
			help="Only run specified repos (names, multiple allowed), e.g., huggingface/diffusers django/django",
		)
		parser.add_argument(
			"--debug-sample",
			nargs="*",
			default=None,
			help="Only run specified test samples (test file stem, multiple allowed), e.g., test1 test2",
		)
		parser.add_argument(
			"--log-level",
			default="INFO",
			choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
			help="Log level",
		)
		parser.add_argument(
			"--logs-dir",
			default=None,
			help="Logs directory path; defaults to logs subdir under the run output",
		)
		parser.add_argument(
			"--repos-dir",
			default=None,
			help="Repo storage directory; defaults to featurebench/resources/repos",
		)
		parser.add_argument(
			"--gpu-ids",
			default=None,
			help="Comma-separated GPU rank IDs for data pipeline, e.g. 6,7",
		)
		parser.add_argument(
			"--global-config-path",
			default="config.toml",
			help="Global config file path with user token and LLM config; default: config.toml",
		)
		args = parser.parse_args()
		
		# Create timestamped output dir
		root_dir = Path(args.output_dir)
		
		if args.resume:
			# For resume, use the specified timestamp dir
			actual_output_dir = root_dir / args.resume
		else:
			# Create a new timestamp dir
			now = datetime.now()
			date_str = now.strftime("%Y-%m-%d")
			time_str = now.strftime("%H-%M-%S")
			actual_output_dir = root_dir / date_str / time_str
		
		# Validate arguments
		cls._validate_arguments(args, actual_output_dir)
		
		# Set global random seed for reproducibility
		if args.seed is not None:
			random.seed(args.seed)

		# Parse global config
		global_config_path = Path(args.global_config_path)
		env_vars, llm_config = cls._parse_global_configs(global_config_path)

		# Parse debug options
		debug_end_stage = None
		debug_cache_overrides = {}
		if args.debug:
			debug_end_stage, debug_cache_overrides = cls._parse_debug_config(args.debug)
		gpu_ids = cls._parse_gpu_ids(args.gpu_ids)

		return cls(
			config_path=Path(args.config_path),
			output_dir=root_dir,
			actual_output_dir=actual_output_dir,
			resume=args.resume,
			seed=args.seed,
			debug_end_stage=debug_end_stage,
			debug_cache_overrides=debug_cache_overrides,
			debug_repo=list(args.debug_repo) if args.debug_repo else None,
			debug_sample=list(args.debug_sample) if args.debug_sample else None,
			log_level=args.log_level.upper(),
			logs_dir=Path(args.logs_dir) if args.logs_dir else None,
			repos_dir=Path(args.repos_dir) if args.repos_dir else None,
			gpu_ids=gpu_ids,
			cmd=cmd,
			env_vars=env_vars,
			llm_config=llm_config,
		)
