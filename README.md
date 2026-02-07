<p align="center">
  <img src="docs/pics/logo.png" style="height: 10em" alt="logo" />
</p>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://hub.docker.com/u/libercoders"><img src="https://img.shields.io/badge/DockerHub-Images-blue.svg" alt="DockerHub"></a>
  <a href="https://huggingface.co/datasets/LiberCoders/ACE-Bench"><img src="https://img.shields.io/badge/HuggingFace-datasets-yellow.svg" alt="HuggingFace"></a>
  <a href="https://LiberCoders.github.io/ACE-Bench/"><img src="https://img.shields.io/badge/Leaderboard-view-purple.svg" alt="Leaderboard"></a>
</p>

---

ACE-Bench is a test-driven data generation and evaluation pipeline for feature-level coding benchmarks.
It provides a unified CLI to run inference, evaluation, and dataset generation.

## 📰 News

🎁 **2026.02.06**: We now support one-click inference for mainstream agent frameworks, including **OpenHands, Claude Code, Codex, Gemini CLI, and mini-swe-agent**. All supported agent frameworks can be found [here](acebench/infer/agents/). We have also open-sourced the ACE-Bench **data pipeline**.

## 🚀 Quickstart

**Prerequisites:**
- `uv` for Python environment management
- `docker` for reproducible builds and evaluation

```bash
# pypi
pip install ace-bench

# local
git clone https://github.com/LiberCoders/ACE-Bench.git
cd ACE-Bench
uv sync
```

**Configure:**
```bash
cp config_example.toml config.toml
```
See [docs/config.md](docs/config.md) for a comprehensive reference (harness, infer, data pipeline) with examples.

**Optional: pre-pull images to reduce network variance:**
```bash
ace pull --mode lite                 # lite split image list (13 images)
ace pull --mode full                 # full split image list (24 images)
ace pull --mode /path/to/images.txt  # one image name per line

# full list: acebench/resources/constants/full_images.txt
# lite list: acebench/resources/constants/lite_images.txt
```

**Run inference:**
```bash
ace infer \
    --config-path config.toml \
    --agent mini_swe_agent \
    --model openai/qwen3-coder-480b-a35b-instruct \
    --split lite
```

**Run evaluation:**
```bash
ace eval \
    -p runs/<timestamp>/output.jsonl \
    --split lite
```

## 🧭 CLI Overview

`ace` provides three core commands:
- `ace infer` runs `acebench.infer.run_infer` (docs: [docs/infer_cli_arg.md](docs/infer_cli_arg.md))
- `ace eval` runs `acebench.harness.run_evaluation` (docs: [docs/harness_cli_arg.md](docs/harness_cli_arg.md))
- `ace data` runs `acebench.pipeline` (docs: [docs/pipeline.md](docs/pipeline.md))

## ✍️ Citation

If you found ACE-Bench useful, please cite us as:

```bibtex
xxx
```

## 📧 Contact

If you have any questions, feel free to contact [qixingzhou1125@gmail.com](mailto:qixingzhou1125@gmail.com) or [zjcheng2022@gmail.com](mailto:zjcheng2022@gmail.com).
