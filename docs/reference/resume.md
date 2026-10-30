# Resume

Resume continues a Discover, Build, or Generate run in the same output directory.
Current YAML contents are reloaded each time.

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS
```

By default, only work with no previous terminal result is selected. Successful,
failed, rejected, and interrupted work is left unchanged.

## Retry non-successful work

Retry everything without a durable success:

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS --select non-success
```

Or select exact states:

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS \
  --select unseen+failed+interrupted
```

Available states are `unseen`, `failed`, `rejected`, and `interrupted`.
`non-success` is shorthand for all four. Successful outputs are never selected.

To rebuild a successful Environment, remove it first:

```bash
fb build remove outputs/build-YYYYMMDD-HHMMSS owner/repository
fb --resume outputs/build-YYYYMMDD-HHMMSS --select non-success
```

## Change configuration

Edit model, concurrency, timeout, source, and Adapter settings directly in the YAML
files recorded by `run.yaml`.

Use `fb configure` when changing the YAML paths or stage inputs:

```bash
fb configure outputs/build-YYYYMMDD-HHMMSS
fb --resume outputs/build-YYYYMMDD-HHMMSS
```

Each invocation stores its resolved settings and code provenance under
`.featurebench/invocations/`. Do not manually edit the stage JSONL files or
`.featurebench/state.sqlite3`.
