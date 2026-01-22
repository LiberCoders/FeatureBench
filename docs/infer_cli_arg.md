# ACE-Bench Infer CLI Arguments

This document describes all CLI arguments supported by `acebench.infer.run_infer`.

## Basic Usage

```bash
python -m acebench.infer.run_infer \
  --agent gemini_cli \
  --model gemini-3-pro-preview
```

## Resume Mode

```bash
python -m acebench.infer.run_infer \
  --resume runs/2025-12-02__16-06-04
```

In resume mode, most arguments are loaded from `run_metadata.json`. Only a few
flags can override metadata (see the argument list below).

## Argument Reference

### Core

- `--agent, -a`  
  Agent to use: `claude_code`, `gemini_cli`, `openhands`, `codex`.  
  Required unless `--resume` is used.

- `--model, -m`  
  Model name (e.g., `claude-sonnet-4-20250514`, `gemini-3-pro-preview`).  
  Required unless `--resume` is used.

- `--dataset`  
  HuggingFace dataset repo name (e.g., `LiberCoders/ACE-Bench`).  
  Default: `LiberCoders/ACE-Bench` in non-resume mode.

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
  Default: `7200`.  
  Resume mode: can override metadata if explicitly provided.

- `--resume`  
  Resume from a previous run directory (e.g., `runs/2025-12-02__16-06-04`).  
  Most arguments are loaded from `run_metadata.json`.

- `--force-rerun`  
  Force rerun specific task IDs even if they were completed.  
  Accepts space-separated task IDs or a `.txt` file path (one task_id per line).

### Networking / Runtime

- `--proxy-port`  
  Proxy port for container network (host gateway).  
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

### Prompt Control (For ablation experiment)

- `--without`  
  Remove the `## Interface Descriptions` section from the prompt.  
  Resume mode: ignored (uses metadata).

- `--white`  
  Enable white-box mode (expose FAIL_TO_PASS test file path in prompt).  
  Resume mode: ignored (uses metadata).

### OpenHands Only

- `--native-tool-calling`  
  Force native tool calling (`LLM_NATIVE_TOOL_CALLING=true`).  
  Resume mode: ignored (uses metadata).

- `--max-iters`  
  Maximum iterations for OpenHands (`OPENHANDS_MAX_ITERATIONS`).  
  Default: no override (OpenHands default applies).  
  Resume mode: ignored (uses metadata).

- `--force-timeout`  
  When resuming OpenHands runs, treat attempts with existing `infer.log` TIMEOUT markers as completed so they are not rerun.  
  Default: disabled.  
  Resume mode: can override metadata if explicitly provided.
