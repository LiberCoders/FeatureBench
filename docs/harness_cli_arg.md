# FeatureBench Harness CLI Arguments

This document describes all CLI arguments supported by `fb eval`.

## 1 Basic Usage

```bash
fb eval \
  -p runs/2026-01-17__12-54-55/output.jsonl
```

## 2 Argument Reference

### Core

- `--config-path`  
  Path to `config.toml` (used for `HF_TOKEN` / `HF_ENDPOINT` when loading dataset).  
  If not provided, uses default discovery (searching upward from `featurebench/infer`).

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
  Proxy port for container network (host gateway) (e.g., `--proxy-port 7890`).  
  Default: `None`.

- `--review-codes`  
  Save agent-generated code for review.  
  Accepts `true/false`, `1/0`, `yes/no`.  
  Default: `false`.

- `--dataset`  
  HuggingFace dataset repo name (e.g., `LiberCoders/FeatureBench`).  
  Default: `LiberCoders/FeatureBench`.

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

## Section 3: Output Directory Structure

```
runs/{timestamp}/
├── report.json               # Evaluation summary
└── eval_outputs/
    └── {instance_id}/
        └── attempt-{n}/
            ├── run_instance.log  # Evaluation log
            ├── test_output.txt   # Test execution output
            ├── patch.diff        # Applied patch
            └── report.json       # Instance evaluation result
```

## Section 4: Summarize results

Summarize `eval_outputs` and generate a CSV report.

```bash
python featurebench/scripts/cal_eval_outputs.py --path <eval_outputs_dir> --attempt-mode <attempt_mode>
```

`<attempt_mode>` can be `best`, `worst`, or a number (e.g., `1`, `2`, `3`). Default: `best`.

When `<attempt_mode>` is a number `k` and `--merge` is enabled, attempts 1..k are merged:
- pass_rate: average over the first k attempts
- resolved: pass@k style (success if any of the first k succeeds)
- prompt_tokens/completion_tokens: sum over the first k, then take the mean
