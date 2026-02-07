"""LLM caller base class with unified init/call interface."""

import logging
import time
from typing import Dict, Any, Optional, List, Union
import requests
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm

from featurebench.llm.llm_exceptions import LLMException, LLMAPIException, LLMInitException


class LLMCaller:
    """LLM caller base class for LLM operations."""
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLM caller.
        
        Args:
            config: Config object containing llm_config
            logger: Logger instance
        """
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Initialize LLM-related attributes
        self.llm_config = config.llm_config or {}
        self.llm_clients: List[Any] = []  # Clients for multiple base_url values
        self.llm_client = None  # Backward-compat: first client
        self.llm_model = None
        self.is_gpt5 = False
        self.is_vllm = False  # Whether this is a local vLLM service
        self.base_urls: List[str] = []  # All base_url values
        
        # Auto-init client if LLM config is provided
        if self.llm_config:
            self._setup_llm_client()
    
    def _setup_llm_client(self) -> None:
        """Initialize LLM client(s), supporting multiple base_url values."""
        if not self.llm_config:
            raise LLMInitException("LLM config not provided")
        
        if OpenAI is None:
            raise LLMInitException("Please install openai: pip install openai")
        
        try:
            # Silence httpx/openai INFO logs to avoid noisy HTTP request/retry output
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("openai._base_client").setLevel(logging.WARNING)
            
            # Read LLM parameters from config
            api_key = self.llm_config.get('api_key')
            base_url_config = self.llm_config.get('base_url')
            self.llm_model = self.llm_config.get('model')
            api_version = self.llm_config.get('api_version')
            backend = self.llm_config.get('backend', '').lower()
            
            # Detect vLLM local service
            self.is_vllm = backend == 'vllm'
            
            # Handle base_url: string or list
            if isinstance(base_url_config, str):
                self.base_urls = [base_url_config]
            elif isinstance(base_url_config, list):
                self.base_urls = base_url_config
            else:
                self.base_urls = []
            
            # vLLM mode: auto-detect model name; api_key optional (depends on deployment)
            if self.is_vllm:
                if not self.base_urls:
                    raise LLMInitException("vLLM config incomplete: missing base_url")
                # If user provides api_key, pass through; otherwise use default
                if not api_key:
                    api_key = "EMPTY"
                # Auto-detect model name from vLLM service
                if not self.llm_model:
                    self.llm_model = self._fetch_vllm_model_name(self.base_urls[0], api_key)
                self.logger.debug(f"Detected vLLM config: backend={backend}, model={self.llm_model}")
            else:
                if not api_key or not self.base_urls or not self.llm_model:
                    raise LLMInitException("LLM config incomplete: missing api_key/base_url/model")
            
            # Detect GPT-5 model (vLLM not affected)
            model_for_check = self.llm_model
            if model_for_check.startswith('azure/'):
                model_for_check = model_for_check[6:]
            self.is_gpt5 = not self.is_vllm and ('gpt-5' in model_for_check.lower() or 'gpt5' in model_for_check.lower())
            
            # Create client for each base_url
            for base_url in self.base_urls:
                client = self._create_client(api_key, base_url, api_version)
                self.llm_clients.append(client)
            
            # Backward-compat: llm_client points to the first client
            if self.llm_clients:
                self.llm_client = self.llm_clients[0]
            
            # Normalize model name (strip azure/ prefix)
            if api_version and self.llm_model.startswith('azure/'):
                self.llm_model = self.llm_model[6:]
                self.logger.debug(f"Azure prefix detected; normalized model: {self.llm_model}")
            
            if self.is_gpt5:
                self.logger.debug("Detected GPT-5 model; using special parameter config")
            
            self.logger.debug(f"Using base_url list: {self.base_urls}")
            
        except LLMInitException:
            # Re-raise LLM-related exceptions
            raise
        except Exception as e:
            raise LLMInitException(f"Failed to initialize LLM client: {e}")
    
    def _create_client(self, api_key: str, base_url: str, api_version: Optional[str]) -> Any:
        """Create a single LLM client."""
        # Read timeout from config (default 180s)
        timeout = self.llm_config.get('timeout', 180)
        
        if api_version:
            # Azure OpenAI client
            if AzureOpenAI is None:
                self.logger.warning("AzureOpenAI not found; using standard OpenAI client")
                return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            else:
                return AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=api_version,
                    timeout=timeout
                )
        else:
            # Standard OpenAI client
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    
    def _fetch_vllm_model_name(self, base_url: str, api_key: Optional[str] = None) -> str:
        """
        Auto-fetch model name from a vLLM service.
        
        Args:
            base_url: vLLM base_url, e.g. "http://localhost:8080/v1"
            
        Returns:
            Model name
            
        Raises:
            LLMInitException: If model name cannot be fetched
        """
        try:
            # Build /v1/models endpoint URL
            models_url = base_url.rstrip('/') + '/models'
            
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.get(models_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = data.get('data', [])
            
            if not models:
                raise LLMInitException(f"vLLM service ({base_url}) returned no models")
            
            # Use the first model ID
            model_name = models[0].get('id')
            if not model_name:
                raise LLMInitException(f"vLLM service ({base_url}) returned invalid model data")
            
            self.logger.debug(f"Auto-detected model name from vLLM service: {model_name}")
            return model_name
            
        except requests.RequestException as e:
            raise LLMInitException(f"Failed to connect to vLLM service ({base_url}): {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise LLMInitException(f"Failed to parse vLLM model list: {e}")
    
    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Call the LLM API.
        
        Args:
            messages: Message list, e.g. [{"role": "user", "content": "..."}]
            temperature: Temperature parameter (ignored by GPT-5)
            max_tokens: Max output tokens
            **kwargs: Other parameters
            
        Returns:
            LLM response object
        """
        if not self.llm_client or not self.llm_model:
            raise LLMInitException("LLM client is not initialized")
        
        # Build API parameters
        api_params = {
            'model': self.llm_model,
            'messages': messages
        }
        
        # Add other parameters
        api_params.update(kwargs)
        
        # Set token/temperature params by model/service type
        if self.is_vllm:
            # vLLM uses standard OpenAI params; ensure compatibility
            if temperature is not None:
                api_params['temperature'] = temperature
            elif 'temperature' not in api_params:
                api_params['temperature'] = self.llm_config.get('llm_temperature', 0.0)
            
            # max_tokens: -1 means no limit
            if max_tokens is not None and max_tokens > 0:
                api_params['max_tokens'] = max_tokens
            elif 'max_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_tokens'] = config_max_tokens
                
        elif self.is_gpt5:
            # GPT-5 uses max_completion_tokens and ignores temperature
            # max_tokens: -1 means no limit
            if max_tokens is not None and max_tokens > 0:
                api_params['max_completion_tokens'] = max_tokens
            elif 'max_completion_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_completion_tokens'] = config_max_tokens
        else:
            # Other models use standard params
            if temperature is not None:
                api_params['temperature'] = temperature
            elif 'temperature' not in api_params:
                # Read defaults from config
                api_params['temperature'] = self.llm_config.get('llm_temperature', 0.0)
            
            # max_tokens: -1 means no limit
            if max_tokens is not None and max_tokens > 0:
                api_params['max_tokens'] = max_tokens
            elif 'max_tokens' not in api_params:
                config_max_tokens = self.llm_config.get('llm_max_tokens', -1)
                if config_max_tokens > 0:
                    api_params['max_tokens'] = config_max_tokens
        
        # Call API with retries: 3 rounds over all base_url values
        max_rounds = 3
        retry_delay = 1  # seconds
        last_exception = None
        
        for round_idx in range(max_rounds):
            for client_idx, client in enumerate(self.llm_clients):
                try:
                    return client.chat.completions.create(**api_params)
                except Exception as e:
                    last_exception = e
                    base_url = self.base_urls[client_idx] if client_idx < len(self.base_urls) else "unknown"
                    tqdm.write(f"Round {round_idx + 1}, base_url[{client_idx}] ({base_url}) call failed: {e}")
                    # Wait after each failure
                    time.sleep(retry_delay)
        
        raise LLMAPIException(f"LLM API call failed: {last_exception}")
    
    def get_llm_response_text(self, completion) -> str:
        """
        Extract text content from an LLM response.
        
        Args:
            completion: LLM response object
            
        Returns:
            Response text content
        """
        return completion.choices[0].message.content.strip()

