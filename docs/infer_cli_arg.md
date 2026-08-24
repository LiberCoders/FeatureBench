# FeatureBench Infer CLI Arguments

This document describes all CLI arguments supported by `fb infer`.

## 1 Basic Usage

```bash
fb infer \
  --data-version v1.1 \
  --agent gemini_cli \
  --model gemini-3-pro-preview
```

## 2 Resume Mode

```bash
fb infer \
  --resume runs/2025-12-02__16-06-04
```

In resume mode, most arguments are loaded from `run_metadata.json`. Only a few
flags can override metadata (see the argument list below).

## 3 Argument Reference

### Core

- `--config-path`  
  Path to `config.toml`.  
  If not provided, uses default discovery (searching upward from `featurebench/infer`). 

- `--agent, -a`  
  Agent to use: `claude_code`, `gemini_cli`, `openhands`, `codex`, `mini_swe_agent`.  
  Required unless `--resume` is used.

- `--model, -m`  
  Model name (e.g., `claude-sonnet-4-20250514`, `gemini-3-pro-preview`).  
  For `openhands` or `mini_swe_agent`, use `provider/model` format.  
  Required unless `--resume` is used. 

- `--api-key`  
  Override agent API key (takes precedence over config).  
  Saved into `run_metadata.json`; resume uses metadata unless overridden again.

- `--base-url`  
  Override agent base URL (takes precedence over config).  
  Saved into `run_metadata.json`; resume uses metadata unless overridden again.

- `--version`  
  Override agent version (takes precedence over config).  
  Saved into `run_metadata.json`; resume uses metadata unless overridden again.

- `--dataset`  
  HuggingFace dataset repo name (e.g., `LiberCoders/FeatureBench`).  
  Default: `LiberCoders/FeatureBench` in non-resume mode.

- `--data-version`
  Hugging Face dataset tag, branch, or commit SHA.
  Overrides `[dataset].revision` from `config.toml`; default: `v1.1`.
  Saved into `run_metadata.json`; resume mode keeps the recorded dataset version.
  This is separate from `--version`, which controls the agent software version.

- `--split`  
  Dataset split name (e.g., `lite`, `full`).  
  Default: `full` in non-resume mode. In resume mode, uses metadata.

- `--level`  
  Filter tasks by level (`1` or `2`).  
  Default: all levels.

- `--task-id, -t`  
  Only process specified task IDs (space-separated).  
  Default: all tasks.

- `--n-attempts`  
  Number of attempts per task.  
  Default: `1`.

- `--n-concurrent`  
  Number of concurrent tasks.  
  Default: `1` in non-resume mode.  
  Resume mode: can override metadata if explicitly provided.

- `--output-dir, -o`  
  Output directory root.  
  Default: `runs`.  
  Resume mode: ignored (uses the resume directory).

- `--timeout`  
  Timeout per task (seconds).  
  Default: `3600`.  
  Resume mode: can override metadata if explicitly provided.

- `--resume`  
  Resume from a previous run directory (e.g., `runs/2025-12-02__16-06-04`).  
  Most arguments are loaded from `run_metadata.json`.

- `--force-rerun`  
  Force rerun specific task IDs even if they were completed.  
  Accepts space-separated task IDs or a `.txt` file path (one task_id per line).

### Networking / Runtime

- `--proxy-port`  
  Proxy port for container network (host gateway) (e.g., `--proxy-port 7890`).  
  Default: `None`.  
  Resume mode: can override metadata if explicitly provided.

- `--runtime-proxy`  
  Enable or disable `HTTP_PROXY/HTTPS_PROXY` at agent runtime.  
  Choices: `on`, `off`.  
  Default: `on` when `--proxy-port` is provided, otherwise `off`.  
  Resume mode: can override metadata if explicitly provided.

- `--gpu-ids`  
  Comma-separated GPU IDs (e.g., `0,1,2,3`).  
  Default: all available.  
  Resume mode: can override metadata if explicitly provided.

- `--force-timeout`  
  If a task run times out (`infer.log` contains `[TIMEOUT after ... seconds]`), treat that attempt as successful instead of failed.  
  Default: disabled.  
  Resume mode: can override metadata if explicitly provided.

### Prompt Control (For ablation experiment)

- `--without`  
  Remove the `## Interface Descriptions` section from the prompt.  
  Resume mode: ignored (uses metadata).

- `--white`  
  Enable white-box mode (expose FAIL_TO_PASS test file path in prompt).  
  Resume mode: ignored (uses metadata).

### OpenHands Only

- `--native-tool-calling`  
  Force native tool calling on (`LLM_NATIVE_TOOL_CALLING=true`).  
  Resume mode: ignored (uses metadata).

- `--no-native-tool-calling`  
  Force native tool calling off (`LLM_NATIVE_TOOL_CALLING=false`).  
  Resume mode: ignored (uses metadata).

- `--send-reasoning-content`
  Send prior assistant reasoning back to the model in subsequent OpenHands requests using both `reasoning_content` and `reasoning` fields.
  Useful for thinking models whose chat template supports reasoning history.
  Resume mode: ignored (uses metadata).

- `--litellm-extra-body`  
  Pass a JSON object to OpenHands `LLM.litellm_extra_body`, for example `--litellm-extra-body '{"enable_thinking": true}'`.  
  Resume mode: ignored (uses metadata).

- `--session-cache`
  Enable backend KV-cache session reuse for OpenHands by sending a unique `X-Session-Id` header for each task attempt and releasing it after the run with `DELETE /session_release?session_id=...`.
  Resume mode: ignored (uses metadata).

- `--max-iters`  
  Maximum iterations for OpenHands (`OPENHANDS_MAX_ITERATIONS`).  
  Default: no override (OpenHands default applies).  
  Resume mode: ignored (uses metadata).

## Section 4: Output Directory Structure

```
runs/{timestamp}/
├── output.jsonl                  # Inference results (one JSON per line)
├── run_metadata.json             # Run configuration and metadata
├── run_summary_{timestamp}.json  # Run summary of success and failure
└── run_outputs/ 
    └── {task_id}/
        └── attempt-{n}/
            ├── infer.log         # Agent execution log
            ├── run.log           # Runtime log
            └── patch.diff        # Generated patch
```
