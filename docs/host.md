# Host

`host.yaml` contains machine-level settings shared by Discover, Build, and Generate.
It is an optional profile, not a command.

```bash
fb build owner/repository \
  --config configs/default/build.yaml \
  --config configs/default/host.yaml
```

When any `--config` option is supplied, include both the stage profile and every
optional profile you want to use.

Configuration file: [`configs/default/host.yaml`](../configs/default/host.yaml)

| Field | Description |
| --- | --- |
| `workspace` | Directory for staging and run-owned workspaces. |
| `refresh_per_second` | Maximum terminal dashboard refresh rate. |
| `jsonl_fsync_interval` | Maximum appended records between durability flushes. |
| `env_file` | Dotenv file used for GitHub and model credentials. |

Relative paths are resolved from the current working directory.
