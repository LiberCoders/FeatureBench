# Discover

Find GitHub repositories and write their metadata to `repositories.jsonl`.

## Usage

Add a GitHub token to `.env`:

```dotenv
GITHUB_TOKEN=your_token
```

Search with the filters in `configs/default/discover.yaml`:

```bash
fb discover
```

Read an explicit list instead. The file accepts one `owner/repository` or GitHub URL
per line:

```bash
fb discover repositories.txt
```

Choose the output directory or compose additional configuration profiles when needed:

```bash
fb discover \
  --config configs/default/discover.yaml \
  --config configs/default/host.yaml \
  --output outputs/my-discovery
```

## Resume

Continue discovery with the current configuration:

```bash
fb --resume outputs/discover-YYYYMMDD-HHMMSS
```

Existing repository records are preserved and deduplicated.

## Configuration

The default profile is
[`configs/default/discover.yaml`](../configs/default/discover.yaml). It controls:

- languages, licenses, keywords, stars, and activity dates;
- explicit target repositories;
- result limits and ordering;
- GitHub request concurrency, timeouts, and retries.

The YAML file is fully commented and is the configuration reference. When explicit
targets are provided, search filters are ignored.

The output can be passed directly to `fb build`:

```bash
fb build outputs/discover-YYYYMMDD-HHMMSS/repositories.jsonl
```
