# Generate

Generate benchmark tasks from validated environments through an Adapter. The built-in
FeatureBench Adapter writes accepted tasks to `tasks.jsonl`.

## Usage

Generate tasks from every environment available at startup:

```bash
fb generate outputs/build-YYYYMMDD-HHMMSS/environments.jsonl \
  --adapter featurebench
```

Process one environment:

```bash
fb generate outputs/build-YYYYMMDD-HHMMSS/environments.jsonl \
  --environment-id ENVIRONMENT_ID \
  --adapter featurebench
```

Choose the output directory or an explicit Adapter profile when needed:

```bash
fb generate outputs/build-YYYYMMDD-HHMMSS/environments.jsonl \
  --adapter featurebench \
  --config configs/default/adapters/featurebench.yaml \
  --output outputs/my-generation
```

`--limit` selects the first N environments in source-file order. It cannot be used
with `--environment-id`.

## Resume

Continue with the current Adapter configuration:

```bash
fb --resume outputs/generate-YYYYMMDD-HHMMSS
```

Retry all work without a durable success:

```bash
fb --resume outputs/generate-YYYYMMDD-HHMMSS --select non-success
```

Successful tasks are never regenerated. See [Resume](reference/resume.md) for more
selection options.

## Configuration

The default FeatureBench profile is
[`configs/default/adapters/featurebench.yaml`](../configs/default/adapters/featurebench.yaml).
It controls:

- task quality thresholds, including Gold Patch, F2P, and P2P limits;
- Top selection and Mask repair;
- model, concurrency, and timeout settings;
- maximum tasks per environment;
- task prompt metadata and audit retention.

The YAML file is fully commented and is the configuration reference.

Each task passes a final Gold-closure validation before it is written. See
[Data organization](reference/data-organization.md) for the output fields.
