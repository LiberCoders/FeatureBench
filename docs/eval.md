# Eval

This document describes all CLI arguments supported by `fb eval`.

## Basic usage

```bash
fb eval \
  -p runs/2026-01-17__12-54-55/output.jsonl
```

## Arguments

### Core

- `--config-path`
  Path to `config.toml` (used for `HF_TOKEN` / `HF_ENDPOINT` when loading dataset).
  If not provided, uses `config.toml` in the current working directory.

- `--predictions-path, -p`
  Path to predictions JSONL file (typically `runs/<timestamp>/output.jsonl`).
  Required. Use `-p gold` to verify Level 1 gold patches; results are written to
  `runs/gold/`.

- `--task-id`
  Specific task IDs (instance IDs) to evaluate. Accepts space-separated values.
  Default: all tasks in predictions.

- `--n-concurrent`
  Number of parallel workers.
  Default: `4`.

- `--timeout`
  Override timeout for test execution (seconds).
  When omitted, uses `timeout_run` from the repository settings.

- `--gpu-ids`
  Comma-separated GPU IDs (e.g., `0,1,2,3`).
  Default: all available.

- `--proxy-port`
  Proxy port for container network (host gateway) (e.g., `--proxy-port 7890`).
  Disabled by default.

- `--review-codes`
  Save agent-generated code for review.
  Accepts `true/false`, `1/0`, `yes/no`.
  Default: `false`.

- `--dataset`
  Hugging Face dataset repo name (e.g., `LiberCoders/FeatureBench`).
  Default: `LiberCoders/FeatureBench`.

- `--data-version`
  Hugging Face dataset tag, branch, or commit SHA.
  Normally inferred from the inference run's `run_metadata.json`.
  If unavailable, falls back to `[dataset].revision` or `v1.1`.

- `--split`
  Dataset split name (e.g., `lite`, `full`).
  Default: `full`.

### Filtering and control

- `--include-failed`
  Include predictions with `success=false` from `output.jsonl`.
  Default: skip failed predictions.

- `--force-rerun`
  Force rerun specified task IDs even if `report.json` already exists.
  Accepts space-separated task IDs or a `.txt` file path (one task_id per line).

## Output

```text
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

## Summarize results

Summarize `eval_outputs` and generate a CSV report.

```bash
python -m featurebench.scripts.cal_eval_outputs --path <eval_outputs_dir> --attempt-mode <attempt_mode>
```

`<attempt_mode>` can be `best`, `worst`, or a number such as `1`, `2`, or `3`.
The default is `best`.

When `<attempt_mode>` is a number `k` and `--merge` is enabled, attempts 1 through `k` are merged:

- `pass_rate`: average over the first `k` attempts
- `resolved`: pass@k (success if any of the first `k` attempts succeeds)
- `prompt_tokens` and `completion_tokens`: sum over the first `k` attempts, then take the mean
