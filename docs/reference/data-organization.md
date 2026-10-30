# Data organization

FeatureBench uses JSONL as the contract between data-factory stages. Each line is one
complete record.

```text
Discover run/
├── repositories.jsonl
├── run.yaml
└── .featurebench/

Build run/
├── environments.jsonl
├── failures.jsonl
├── run.yaml
└── .featurebench/

Generate run/
├── tasks.jsonl
├── failures.jsonl
├── run.yaml
└── .featurebench/
```

`repositories.jsonl` is accepted by `fb build`. `environments.jsonl` is accepted by
`fb generate`. `tasks.jsonl` is the final dataset.

## Repository record

Each line in `repositories.jsonl` contains a GitHub repository snapshot.

| Field | Description |
| --- | --- |
| `discovery_run_id` | Discover run that created the record. |
| `github_id`, `node_id` | Stable GitHub identifiers. |
| `full_name`, `name`, `owner` | Repository identity. |
| `clone_url`, `html_url`, `api_url` | Repository URLs. |
| `default_branch` | Default branch observed during discovery. |
| `description`, `topics` | Repository metadata. |
| `stars`, `forks`, `open_issues`, `size_kb` | Repository statistics. |
| `language`, `license` | Primary language and license. |
| `visibility`, `private`, `archived`, `fork` | Repository status. |
| `created_at`, `updated_at`, `pushed_at`, `discovered_at` | Timestamps. |

## Environment record

Each line in `environments.jsonl` describes one validated environment.

| Field | Description |
| --- | --- |
| `environment_id` | Stable environment identifier. |
| `repository` | Canonical repository URL. |
| `commit` | Exact 40-character Git commit. |
| `language` | Detected primary language. |
| `gpu_requirement` | `none`, `optional`, or `required`. |
| `image` | Immutable Docker image reference. |
| `dockerfile` | Self-contained Dockerfile used to build the image. |
| `runtime` | Activation, Runner, limits, and container resources. |
| `verified_test_points` | Tests that passed during baseline validation. |

Each verified test point contains an exact Runner `id` and its repository-relative
`path`.

## Task record

Each line in `tasks.jsonl` contains a common task envelope and an Adapter payload.

| Field | Description |
| --- | --- |
| `task_id` | Stable task identifier. |
| `environment_id` | Environment that produced the task. |
| `image` | Docker image used to run the task. |
| `runtime` | Runtime contract inherited from the Environment. |
| `adapter` | Self-contained Adapter-specific payload. |

### FeatureBench payload

The built-in FeatureBench Adapter writes these fields under `adapter`:

| Field | Description |
| --- | --- |
| `type` | Always `featurebench`. |
| `prompt` | Task statement shown to the coding Agent. |
| `gold_patch` | Patch that restores the masked production code. |
| `hidden_tests_patch` | Patch that restores hidden F2P tests. |
| `fail_to_pass` | Tests that must change from failing to passing. |
| `pass_to_pass` | Tests that must remain passing. |
| `test_timeout_seconds` | Timeout for one task test run. |
| `metrics.gold_patch` | Gold Patch line and file counts. |
| `metrics.tests` | F2P and P2P counts. |
| `metrics.top` | Top-object count and Gold Patch coverage. |

## Run files

`run.yaml` records the resumable command inputs. `.featurebench/` stores state,
events, diagnostics, invocation snapshots, and optional audit artifacts. These files
support operation and debugging; they are not stage inputs.
