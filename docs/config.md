# Configuration

FeatureBench uses separate configuration for benchmark workflows and the data factory.

## Inference and evaluation

`fb infer` and `fb eval` read `config.toml` from the current working directory unless
`--config-path` selects another file. In a source checkout, start from the checked-in example:

```bash
cp config_example.toml config.toml
```

The TOML file contains three kinds of settings:

- `[env_vars]` contains values injected into benchmark containers.
- `[dataset]` pins the Hugging Face dataset revision used by inference and evaluation.
- `[infer]` and `[infer_config.<agent>]` configure the selected inference runtime.

Command-line options override the matching TOML values. See
[Inference CLI](infer.md) and [Evaluation CLI](eval.md) for the complete
argument references.

## Data factory

`fb discover`, `fb build`, and `fb generate` use YAML instead of `config.toml`. Without
`--config`, each command uses its stage default from local `configs/default/` when available,
or from the installed package otherwise.

- `discover.yaml` controls repository discovery.
- `build.yaml` controls environment construction and validation.
- `adapters/featurebench.yaml` controls FeatureBench task generation.
- `host.yaml` and `sources.yaml` are optional overlays for machine-specific resources and
  package-source configuration.

Repeat `--config` to compose profiles. By default, credentials are read from `.env`, while
outputs and runtime state are written under the current working directory. Source checkouts can
use [`.env.example`](../.env.example) as the template.

```bash
fb build owner/project \
  --config configs/default/build.yaml \
  --config configs/default/host.yaml
```

See [Host configuration](host.md), [Source configuration](sources.md), and
[Run configuration](configure.md) for usage details.
