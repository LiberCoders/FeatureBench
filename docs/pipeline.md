# FeatureBench Data Pipeline (Dataset Generation)

This document describes the data pipeline entrypoint `featurebench.pipeline`, its
CLI arguments, and the repo specs defined in `python.py`.

## Pipeline Stages

- **repo**: clone repositories from specs, checkout commits.
- **image**: build base and instance Docker images per repo.
- **scanner**: discover tests via `pytest --collect-only`; produce P2P and F2P candidates.
- **runner**: run P2P tests; mark passed files for later stages.
- **dynamic**: run dynamic tracing on passed tests.
- **top**: LLM-based top-object selection from trace data.
- **p2p**: choose P2P lists per F2P file (and update top objects).
- **data**: per-file processing (mask -> post-verify -> case generation).

The **data** stage performs:
`CodeClassifier` -> `MaskGenerator` -> `F2PValidator` -> `P2PValidator` -> `CaseConverter`.

## Argument Reference

### Core

- `--config-path`  
  Path to the repo specs Python module (e.g. `featurebench/resources/constants/python.py`).

- `--global-config-path`  
  Path to `config.toml` for global env vars and LLM config.  
  Default: `config.toml`.

- `--output-dir`  
  Output root directory. The pipeline creates a timestamped subdirectory
  `YYYY-MM-DD/HH-MM-SS` under this root.

- `--resume`  
  Resume a previous run using `<output-dir>/<YYYY-MM-DD>/<HH-MM-SS>`.

- `--seed`  
  Random seed for reproducibility (sampling and selection logic).

- `--debug-repo`  
  Run only the listed spec names (e.g. `SPECS_LITGPT`).  
  Accepts multiple values.

- `--debug-sample`  
  Run only listed test file stems (no `.py` suffix).  
  Example: `--debug-sample test_config test_modeling`.

- `--log-level`  
  Logging verbosity: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.  
  Default: `INFO`.

- `--logs-dir`  
  Optional override for logs root.  
  Default: `<actual_output_dir>/logs`.

- `--repos-dir`  
  Optional override for local repo storage.  
  Default: `featurebench/resources/repos`.

- `--gpu-ids`  
  Comma-separated GPU rank IDs for data pipeline scheduling (e.g. `6,7,8`).  
  Required when any repo uses `docker_specs.run_args.cuda_visible_num`.

## Advanced Debugging (`--debug`)

`--debug` takes a comma-separated list of `key=value` items:

- `end=<stage>`: stop after this stage.  
  Stages: `repo`, `image`, `scanner`, `runner`, `dynamic`, `top`, `p2p`, `data`.
- `<stage>`=`True/False`: override cache usage for that stage.  
  Stages: `scanner`, `runner`, `dynamic`, `top`, `p2p`, `data`.

Examples:

```bash
# Stop after scanner, and disable scanner cache.
--debug "end=scanner, scanner=False"

# Use cached results everywhere except data stage.
--debug "data=False"
```

Notes:
- Cache overrides map to `Config.get_cache_config` and take precedence over
  per-repo specs (e.g. `scan_cache`, `dynamic_cache`).
- `end=data` is effectively the same as running to completion because data is
  the last stage.

## Repo Specs Reference

The `--config-path` file defines one or more `SPECS_*` dictionaries. Each
dictionary describes how to build, run, and filter a repo.

### Repo Source

- `repository`  
  Repo slug in `owner/name` form (e.g. `huggingface/accelerate`).

- `commit`  
  Commit SHA or tag to check out. If omitted, the current HEAD is used and
  written back into the specs at runtime.

- `clone_method`  
  `https` or `ssh`. Default: `https`.

- `base_url`  
  Base Git hosting URL. Default: `https://github.com`.

### Image Build and Install

- `base_image`  
  Base image template name (matches `featurebench/resources/dockerfiles/<name>.py`).

- `rebuild_base_image`  
  Force rebuild of the base image.

- `rebuild_instance_image`  
  Force rebuild of the instance image.

- `custom_instance_image_build`  
  Extra Dockerfile commands injected into instance image build.

- `pre_install`  
  Shell commands run before project installation inside the instance image.

- `install`  
  Shell command used to install the project (e.g. `pip install -e .`).

- `pip_packages`  
  Extra pip packages installed into the testbed environment.

### Docker Runtime (`docker_specs`)

- `docker_specs.run_args.cuda_visible_num`  
  Candidate pool size for GPU scheduling.  
  Type: positive integer or `None` (`None` means no GPU allocation in data pipeline).

- `docker_specs.run_args.number_once`  
  Actual number of GPUs allocated per container run.  
  Must satisfy: `cuda_visible_num >= number_once` (when `cuda_visible_num` is not `None`).

- `docker_specs.run_args.shm_size`  
  Shared memory size passed to Docker (`--shm-size`).

- `docker_specs.run_args.cap_add`  
  Docker capabilities to add (list of strings).

- `docker_specs.custom_docker_args`  
  Extra Docker args; supports `-e`, `-v`, and `-ee`.

### Test Discovery (Scanner)

- `test_scanner_cmd`  
  Command array for test discovery (passed to `scanner_script.py`).

- `timeout_scanner`  
  Timeout for test discovery, in seconds. `-1` means no timeout.

- `scan_cache`  
  Cache discovery results in `metadata_outputs/files_status.json`.

- `start_time`  
  Filter tests by last modified date (`YYYY-MM-DD`).

- `min_test_num`  
  Minimum number of tests for a file to qualify as F2P.

- `max_f2p_num`  
  Cap F2P candidates per repo. `-1` means no limit.

- `max_p2p_num`  
  Cap P2P candidates per repo. Must be `-1` or >= F2P count.

### Test Execution

- `test_cmd`  
  Base test command used for running tests.

- `timeout_run`  
  Timeout for running a test file (seconds).  
  Used by runner and post-verify stages.

- `timeout_one`  
  Intended per-test timeout (appended as `--timeout=<value>`).

- `test_cache`  
  Cache P2P execution results in `metadata_outputs/files_status.json`.

### Dynamic Tracing

- `test_dynamic_cmd`  
  Extra pytest args for dynamic tracing.

- `timeout_dynamic`  
  Timeout for dynamic tracing, in seconds. `-1` means no timeout.

- `dynamic_cache`  
  Cache dynamic trace results under `metadata_outputs/dynamic_trace/`.

### Top Selection (LLM)

- `llm_cache`  
  Cache LLM top-object selection results.

- `batchsize_top`  
  Number of files grouped per LLM call.

- `max_depth_top`  
  Max dependency depth when expanding top-object candidates.

### P2P Selection

- `p2p_cache`  
  Cache P2P selection results.

- `min_p2p_files`  
  Minimum P2P files selected per F2P case.

- `max_p2p_files`  
  Maximum P2P files selected per F2P case.

- `max_code_line_lower_bound`  
  Lower bound of code line budget for masking.

- `max_code_line_upper_bound`  
  Upper bound of code line budget for masking.

### Data Pipeline and Case Generation

- `data_cache`  
  Cache data-stage results in `metadata_outputs/data_status.json`.

- `timeout_collect`  
  Timeout for `pytest --collect-only` checks in mask generation.

- `f2p_pass_rate_threshold`  
  Max allowed pass rate in F2P post-verify.

- `llm_prompt_for_case`  
  If true, LLM generates docstrings and task statements for cases.

### Metadata for Rendering

- `library_name`  
  Library name displayed in generated tasks.

- `black_links`  
  Reference links included in task rendering.

## Output Directory Structure

The pipeline writes to `<output-dir>/<YYYY-MM-DD>/<HH-MM-SS>/`:

```
<run_root>/
├── cases/
│   └── {owner}__{repo}.{commit8}/
│       └── {test}.{hash}.lv{1,2}/
│           ├── config.yaml
│           ├── instance.json
│           ├── patch.diff            # lv1 only
│           ├── problem_statement.md
│           └── test_patch.diff
├── metadata_outputs/
│   ├── config_backup.py
│   ├── metadata.json
│   ├── files_status.json
│   ├── data_status.json
│   ├── data_items_cache.json
│   ├── classification/{SPECS_NAME}/
│   ├── dynamic_trace/{SPECS_NAME}/
│   ├── masked_files/{repo}/{test_file_hash}/
│   ├── llm_docstring/{SPECS_NAME}/{test_file_hash}/
│   └── llm_task_statement/{SPECS_NAME}/{test_file_hash}.txt
├── logs/
│   ├── base/                         # base image builds
│   ├── instance/                     # instance image builds
│   ├── test_scanner/
│   ├── test_runner/{SPECS_NAME}/
│   ├── dynamic_trace/{SPECS_NAME}/
│   ├── mask_generator/{SPECS_NAME}/
│   ├── p2p_choose/{SPECS_NAME}/
│   ├── f2p_validate/{SPECS_NAME}/
│   ├── p2p_validate/{SPECS_NAME}/
│   ├── level2_validate/{SPECS_NAME}/
│   └── llm_api_calls/
└── debug_outputs/
    ├── terminal/
    ├── mask_diff/{repo}/{test_file_hash}/
    ├── llm_top_classification/{SPECS_NAME}/{test_file_hash}/
    └── llm_task_statement/{SPECS_NAME}/
```
