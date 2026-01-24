"""
Configuration loader for the inference module.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import toml


class InferConfigLoader:
    """Load and manage inference configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the config file. If None, searches for config.toml
                        in the project root.
        """
        if config_path is None:
            # Search for config.toml in project root
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                config_file = current_dir / "config.toml"
                if config_file.exists():
                    config_path = config_file
                    break
                current_dir = current_dir.parent
        
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(
                "config.toml not found. Please create a config.toml file in the project root."
            )
        
        self.config_path = config_path
        self._config = toml.load(config_path)

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Return the raw parsed config.toml dictionary."""
        return self._config
    
    @property
    def env_vars(self) -> Dict[str, str]:
        """Get global environment variables."""
        return self._config.get("env_vars", {})
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get global LLM configuration."""
        return self._config.get("llm_config", {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "claude_code", "openhands")
            
        Returns:
            Agent configuration dictionary
        """
        infer_config = self._config.get("infer_config", {})
        agent_config = infer_config.get(agent_name, {})
        return agent_config

    def get_cache_dir(self) -> Optional[Path]:
        """Get host cache directory for downloads (mounted to /download)."""
        cache_cfg = self._config.get("infer", {}) or {}
        path = cache_cfg.get("download_cache_dir")
        if path:
            return Path(path).expanduser()
        return None

    def get_no_proxy_hosts(self) -> Optional[str]:
        """Get comma-separated NO_PROXY host list from [infer].

        Supports either a string or a list in config.toml:

        - [infer].no_proxy_hosts = "yunwu.ai,api3.wlai.vip"
        - [infer].no_proxy_hosts = ["https://yunwu.ai/v1", "api3.wlai.vip"]

        Values may be full URLs; we normalize to hostnames.
        """
        infer_cfg = self._config.get("infer", {}) or {}
        raw = infer_cfg.get("no_proxy_hosts")
        if raw is None:
            return None

        items: List[str] = []
        if isinstance(raw, list):
            for v in raw:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    items.append(s)
        else:
            s = str(raw).strip()
            if not s:
                return None
            # Allow either comma-separated or whitespace-separated.
            for part in s.replace("\n", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                for tok in part.split():
                    tok = tok.strip()
                    if tok:
                        items.append(tok)

        normalized: List[str] = []
        for item in items:
            v = item.strip()
            if not v:
                continue

            host: Optional[str] = None
            if "://" in v:
                parsed = urlparse(v)
                host = parsed.hostname
            else:
                # Allow "host/path" input; keep only host.
                host = v.split("/", 1)[0].strip()

            if host:
                normalized.append(host)

        # Dedupe while preserving order.
        seen = set()
        uniq: List[str] = []
        for h in normalized:
            if h not in seen:
                seen.add(h)
                uniq.append(h)

        return ",".join(uniq) if uniq else None

    def get_agent_env_vars(self, agent_name: str) -> Dict[str, str]:
        """
        Get environment variables for a specific agent.
        Merges global env_vars with agent-specific ones.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Merged environment variables dictionary
        """
        # Start with global env vars
        env_vars = dict(self.env_vars)
        
        # Merge agent-specific config
        agent_config = self.get_agent_config(agent_name)
        for key, value in agent_config.items():
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            env_vars[key] = str(value)

        # [infer] defaults (applied only if not explicitly set elsewhere)
        no_proxy_hosts = self.get_no_proxy_hosts()
        if no_proxy_hosts:
            env_vars.setdefault("ACE_NO_PROXY_HOSTS", no_proxy_hosts)

        self._apply_agent_defaults_and_validate(agent_name, env_vars)
        
        return env_vars

    def _apply_agent_defaults_and_validate(self, agent_name: str, env_vars: Dict[str, str]) -> None:
        """Apply minimal defaults and validate enum-like agent options.

        Intentionally selective: we only validate a small set of known fields to
        fail fast on typos without over-restricting free-form settings.
        """
        if agent_name != "codex":
            return

        allowed_reasoning_effort = {"low", "medium", "high"}

        # Default: if left empty/unset, use "medium".
        raw = env_vars.get("CODEX_REASONING_EFFORT") or env_vars.get("MODEL_REASONING_EFFORT")
        if raw is None or not str(raw).strip():
            env_vars["CODEX_REASONING_EFFORT"] = "medium"
            return

        value = str(raw).strip().lower()
        env_vars["CODEX_REASONING_EFFORT"] = value
    
    def get_llm_by_name(self, llm_name: str) -> Optional[Dict[str, Any]]:
        """
        Get LLM configuration by name.
        
        Args:
            llm_name: Name of the LLM configuration
            
        Returns:
            LLM configuration dictionary or None if not found
        """
        llm_configs = self._config.get("llm", {})
        return llm_configs.get(llm_name)


class DatasetLoader:
    """Load dataset from HuggingFace."""
    
    def __init__(self, config_loader: InferConfigLoader):
        """
        Initialize the dataset loader.
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader
        self._setup_hf_env()
    
    def _setup_hf_env(self) -> None:
        """Set up HuggingFace environment variables."""
        env_vars = self.config.env_vars
        
        if env_vars.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = env_vars["HF_ENDPOINT"]
        
        if env_vars.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = env_vars["HF_TOKEN"]
    
    def load_dataset(
        self,
        dataset: str,
        split: Optional[str] = None,
        levels: Optional[List[int]] = None,
        task_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset: HuggingFace dataset repo name (e.g., "LiberCoders/ACE-Bench"). Required.
            split: HuggingFace split name (e.g., "lite", "full"). Required.
            levels: List of levels to filter (1, 2). If None, loads all levels.
                   Level is determined by the last segment of instance_id (e.g., "lv1", "lv2").
            task_ids: Specific task IDs to filter. If None, loads all tasks.
            
        Returns:
            List of task instances as dictionaries
        """
        # Suppress verbose HTTP request logs from httpx and other libraries
        # but keep progress bars visible
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        
        from datasets import load_dataset
        
        env_vars = self.config.env_vars
        hf_token = env_vars.get("HF_TOKEN")

        dataset_name = str(dataset or "").strip()
        if not dataset_name:
            raise ValueError("dataset parameter is required. Please specify --dataset (e.g., 'LiberCoders/ACE-Bench')")
        
        if split is None:
            raise ValueError("split parameter is required. Please specify --split (e.g., 'lite', 'full')")
        
        all_instances = []
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                token=hf_token
            )
            
            for item in dataset:
                item_dict = dict(item)
                
                # Determine level from instance_id's last segment (e.g., "xxx.lv1" -> level 1)
                instance_id = item_dict.get("instance_id", "")
                last_segment = instance_id.split(".")[-1] if instance_id else ""
                if last_segment == "lv1":
                    level = 1
                elif last_segment == "lv2":
                    level = 2
                else:
                    # raise error if level is not 1 or 2
                    raise RuntimeError(f"Unknown level: {last_segment} in instance_id: {instance_id}")
                
                item_dict["level"] = level
                
                # Filter by levels if specified
                if levels is not None and level not in levels:
                    continue
                
                # Filter by task_ids if specified
                if task_ids is not None and instance_id not in task_ids:
                    continue
                
                all_instances.append(item_dict)
                    
        except Exception as e:
            print(f"Warning: Failed to load split '{split}': {e}")
        
        return all_instances
