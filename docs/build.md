# Build

Build a reproducible test environment for one or more repositories. Successful
environments are written to `environments.jsonl`.

## Usage

Add the model credential required by `configs/default/build.yaml` to `.env`:

```dotenv
OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://your-relay.example/v1
```

Build one repository:

```bash
fb build owner/repository
```

Build a specific branch, tag, or commit:

```bash
fb build owner/repository --ref main
```

Build repositories produced by Discover:

```bash
fb build outputs/discover-YYYYMMDD-HHMMSS/repositories.jsonl --limit 10
```

Use `--output` to choose the run directory and repeat `--config` to compose YAML
profiles:

```bash
fb build owner/repository \
  --config configs/default/build.yaml \
  --config configs/default/host.yaml \
  --output outputs/my-build
```

## Resume

Continue a run with its current configuration:

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS
```

Retry failed, rejected, or interrupted repositories as well as new ones:

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS --select non-success
```

See [Resume](reference/resume.md) for selection options.

## Rebuild a successful environment

Remove the existing result before rebuilding it:

```bash
fb build remove outputs/build-YYYYMMDD-HHMMSS owner/repository
fb --resume outputs/build-YYYYMMDD-HHMMSS --select non-success
```

The removed record is archived under `.featurebench/removals/`. Its local Docker
image is deleted.

## Configuration

The default profile is [`configs/default/build.yaml`](../configs/default/build.yaml).
It controls:

- repository checkout and Git behavior;
- Docker resources and timeouts;
- the environment-building model;
- accepted test coverage and retry limits;
- available language and runtime base images.

The YAML file is fully commented and is the configuration reference.

## Environment contract

- The repository is pinned to an exact commit.
- `/testbed` is the source of truth for project code.
- Tests must consume Agent edits from the current `/testbed` checkout, not a stale
  installed copy.
- Dependencies are installed while the image is built; runtime test execution is
  offline by default.
- Only environments that collect and pass enough real tests are written to
  `environments.jsonl`.

See [Data organization](reference/data-organization.md) for the output fields.
