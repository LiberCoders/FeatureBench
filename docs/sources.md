# Sources

`sources.yaml` configures download mirrors and their canonical fallbacks. It is an
optional profile shared by Build and other networked stages.

```bash
fb build owner/repository \
  --config configs/default/build.yaml \
  --config configs/default/sources.yaml
```

When any `--config` option is supplied, include both the stage profile and every
optional profile you want to use.

Configuration file: [`configs/default/sources.yaml`](../configs/default/sources.yaml)

## Endpoints

Each service has an ordered endpoint list. Exactly one endpoint must be canonical and
must appear last:

```yaml
sources:
  services:
    pypi:
      endpoints:
        - name: tuna
          url: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
        - name: official
          url: https://pypi.org/simple
          canonical: true
```

| Field | Description |
| --- | --- |
| `name` | Unique endpoint name within the service. |
| `url` | Public HTTP or HTTPS endpoint. |
| `canonical` | Marks the official fallback endpoint. |
| `forward_credentials` | Allows credentials only when the canonical endpoint needs them. |

FeatureBench temporarily cools down unhealthy mirrors and always keeps the canonical
endpoint as the final fallback. Set `sources.enabled: false` to use canonical endpoints
only.

## Built-in services

The default profile covers GitHub, container registries, APT, PyPI, Hugging Face,
Node.js, Go, Rust, Maven, RubyGems, Composer, Dart, Elixir, and CRAN. It also includes
separate services for large assets such as PyTorch CPU wheels, Playwright, Rustup, uv,
Conda, and Miniforge.

Common controls include download timeouts, retries, endpoint cooldowns, health-state
storage, and optional source metrics. The checked-in YAML is fully commented and is
the complete service list and configuration reference.
