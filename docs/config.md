# ACE-Bench Configuration Guide: config.toml

ACE-Bench has three major workflows:

- **harness (evaluation)**: runs tests on inference outputs and produces a report.
- **infer (inference)**: runs different agents on ACE-Bench benchmark tasks to generate patches.
- **data pipeline (dataset generation)**: builds ACE-Bench-style tasks automatically from real repositories.

The repo-root `config.toml` is the single configuration file that manages what all three workflows need at runtime.

This guide uses the example configuration [config_example.toml](../config_example.toml) as the source of truth. It explains every field and how to use it, organized into **harness / infer / data pipeline**, and provides examples.

## Quickstart

Copy the example config:

```bash
cp config_example.toml config.toml
```

### [env_vars] (global environment variables used by harness / infer / pipeline)

`[env_vars]` is the global environment variable section. Values here will be injected into every runnable container.

Example:

```toml
[env_vars]
HF_ENDPOINT = ""
HF_TOKEN = ""
GITHUB_TOKEN = ""
```

---

## A. Harness

The core idea of harness is: read the `output.jsonl` produced by infer, apply each patch to the corresponding repository image, run tests, and generate evaluation results.

When harness loads the HuggingFace dataset during evaluation, it uses `HF_ENDPOINT` from `[env_vars]`.

---

## B. Infer

The core idea of infer is: run an agent inside a container (e.g. `claude_code/openhands/codex/...`) and let it generate a patch for each task.

How infer reads `config.toml`:

- Load global `[env_vars]` first
- Then load `[infer]` (inference runtime settings)
- Then load the current agent’s `[infer_config.<agent>]`
- Merge `[env_vars]` and `[infer_config.<agent>]`, and inject them into the inference container as environment variables

### B1. [infer]

```toml
[infer]
download_cache_dir = "/abs/path/to/ACE-Bench/download_cache"
```

- `download_cache_dir`: host-side cache directory that will be mounted into the container at `/download`.
  - Purpose: speed up agent installs/downloads (e.g. npm cache).
  - Empty means no mount (each container downloads on its own; usually slower).

### B2. [infer_config.openhands]

```toml
[infer_config.openhands]
LLM_API_KEY = ""
LLM_BASE_URL = ""     # Optional
LLM_API_VERSION = ""  # Optional: Azure only
OPENHANDS_VERSION = "0.62.0" # Optional: pin OpenHands version (empty usually means default/latest)
SAVE_COMPLETIONS = false      # Optional: whether to save LLM completions (true/false)
INFER_LOG_RENDER_MODE = "compact" # Optional: compact|full for infer.log rendering

LLM_REASONING_EFFORT = ""     # Optional: Reasoning effort for OpenAI o-series models
OPENHANDS_MAX_ITERATIONS = "" # Optional: OpenHands agent max iterations (step limit). Upstream default is 500.

```

Example 1: OpenHands + OpenAI

```toml
[infer_config.openhands]
LLM_API_KEY = "<your-openai-api-key>"
LLM_BASE_URL = ""            # Fill if needed
LLM_API_VERSION = ""
OPENHANDS_VERSION = "0.62.0"
SAVE_COMPLETIONS = false
```

Example 2: OpenHands + Azure OpenAI

```toml
[infer_config.openhands]
LLM_API_KEY = "<your-azure-api-key>"
LLM_BASE_URL = "https://<resource>.openai.azure.com"
LLM_API_VERSION = "2024-xx-xx"
OPENHANDS_VERSION = "0.62.0"
SAVE_COMPLETIONS = false
```

### B3. [infer_config.claude_code]

```toml
[infer_config.claude_code]
ANTHROPIC_API_KEY = ""    # Required
ANTHROPIC_BASE_URL = ""   # Optional
```

### B4. [infer_config.gemini_cli]

```toml
[infer_config.gemini_cli]
GEMINI_API_KEY = ""         # Required
GOOGLE_GEMINI_BASE_URL = "" # Optional
```

### B5. [infer_config.codex]

```toml
[infer_config.codex]
OPENAI_API_KEY = ""         # Required
OPENAI_BASE_URL = ""        # Optional
CODEX_REASONING_EFFORT = "" # Optional: empty defaults to medium
```

Example 1: Codex + OpenAI

```toml
[infer_config.codex]
OPENAI_API_KEY = "<your-openai-api-key>"
OPENAI_BASE_URL = ""
CODEX_REASONING_EFFORT = "medium"
```

Example 2: Codex + Azure OpenAI

```toml
[infer_config.codex]
OPENAI_API_KEY = "<your-azure-api-key>"
OPENAI_BASE_URL = "https://<resource>.openai.azure.com"
CODEX_REASONING_EFFORT = "high"
```

---

## C. Data Pipeline

the core idea of the data pipeline is: call an LLM + access repositories/platforms to automatically generate or process data (tasks, prompts, test files, etc.).

It mainly depends on two types of config:

- `[env_vars]`: platform/repo access related
- `[llm_config]` + `[llm.<name>]`: select an LLM configuration for the pipeline

### C1. [llm_config]

```toml
[llm_config]
llm_name = ""         # Select the name of one [llm.<name>]
llm_temperature = 0.0  # Default temperature (some models/backends may ignore it, e.g. some GPT-5 paths)
llm_max_tokens = 65536 # Max output tokens; set -1 to "not send max_tokens/max_completion_tokens", backend decides the default behavior
timeout = 180          # Request timeout (seconds)
```

### C2. [llm.\<name>]

Example (Azure OpenAI):

```toml
[llm.azure-o3-fy1]
model = "azure/o3-fy1"
api_key = "..."
base_url = "https://<resource>.openai.azure.com"
api_version = "2025-04-01-preview"
```

Example (Anthropic):

```toml
[llm.claude-sonnet-4-20250514]
model = "claude-sonnet-4-20250514"
api_key = "..."
base_url = [
  "https://...",
  "https://..."
]   # Supports multiple URLs; will retry the list on failure
```

Example (local vLLM):

```toml
[llm.local]
backend = "vllm"
base_url = "http://localhost:8080/v1"
```
