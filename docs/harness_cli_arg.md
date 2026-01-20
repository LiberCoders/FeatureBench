# ACE-Bench Harness CLI Arguments

This document describes all CLI arguments supported by `acebench.harness.run_evaluation`.

## Basic Usage

```bash
python -m acebench.harness.run_evaluation \
  --predictions-path runs/2026-01-17__12-54-55/output.jsonl
```

## Argument Reference

### Core

- `--predictions-path, -p`  
  Path to predictions JSONL file (typically `runs/<timestamp>/output.jsonl`).  
  Required.

- `--task-id`  
  Specific task IDs (instance IDs) to evaluate. Accepts space-separated values.  
  Default: all tasks in predictions.

- `--n-concurrent`  
  Number of parallel workers.  
  Default: `4`.

- `--timeout`  
  Override timeout for test execution (seconds).  
  Default: `None` (uses `timeout_run` from repo_settings).

- `--gpu-ids`  
  Comma-separated GPU IDs (e.g., `0,1,2,3`).  
  Default: all available.

- `--proxy-port`  
  Proxy port for container network (host gateway).  
  Default: `None`.

- `--review-codes`  
  Save agent-generated code for review.  
  Accepts `true/false`, `1/0`, `yes/no`.  
  Default: `false`.

- `--dataset`  
  HuggingFace dataset repo name (e.g., `LiberCoders/ACE-Bench`).  
  Default: `LiberCoders/ACE-Bench`.

- `--split`  
  Dataset split name (e.g., `lite`, `full`).  
  Default: `full`.

### Filtering / Control

- `--include-failed`  
  Include predictions with `success=false` from `output.jsonl`.  
  Default: skip failed predictions.

- `--force-rerun`  
  Force rerun specified task IDs even if `report.json` already exists.  
  Accepts space-separated task IDs or a `.txt` file path (one task_id per line).
