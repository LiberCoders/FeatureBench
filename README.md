<p align="center">
  <img src="docs/pics/logo.png" style="height: 10em" alt="logo" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.10975"><img src="https://img.shields.io/badge/arXiv-2602.10975-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://hub.docker.com/u/libercoders"><img src="https://img.shields.io/badge/DockerHub-Images-blue.svg" alt="DockerHub"></a>
  <a href="https://huggingface.co/datasets/LiberCoders/FeatureBench"><img src="https://img.shields.io/badge/HuggingFace-datasets-yellow.svg" alt="HuggingFace"></a>
  <a href="https://LiberCoders.github.io/FeatureBench/"><img src="https://img.shields.io/badge/Leaderboard-view-purple.svg" alt="Leaderboard"></a>
</p>

---

FeatureBench is a test-driven data generation and evaluation pipeline for feature-level coding benchmarks.
It provides a unified CLI to run inference, evaluation, and dataset generation.

## 📰 News

🎁 **2026.02.06**: We now support one-click inference for mainstream agent frameworks, including **OpenHands, Claude Code, Codex, Gemini CLI, and mini-swe-agent**. All supported agent frameworks can be found [here](featurebench/infer/agents/). We have also open-sourced the FeatureBench **data pipeline**.

## 🚀 Quickstart

**Prerequisites:**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment management
- [docker](https://docs.docker.com/engine/install/) for reproducible builds and evaluation

```bash
# pypi
pip install featurebench
# or uv add featurebench

# local
git clone https://github.com/LiberCoders/FeatureBench.git
cd FeatureBench
uv sync
```

**Configure:**
```bash
cp config_example.toml config.toml
```
See [docs/config.md](docs/config.md) for a comprehensive reference (harness, infer, data pipeline) with examples.

**Optional: pre-pull images to reduce network variance:**
```bash
fb pull --mode lite                 # lite split image list (13 images)
fb pull --mode full                 # full split image list (24 images)
fb pull --mode /path/to/images.txt  # one image name per line

# full list: featurebench/resources/constants/full_images.txt
# lite list: featurebench/resources/constants/lite_images.txt
```

**Run inference:**
```bash
fb infer \
    --config-path config.toml \
    --agent mini_swe_agent \
    --model openai/qwen3-coder-480b-a35b-instruct \
    --split lite
```

**Run evaluation:**
```bash
fb eval \
    -p runs/<timestamp>/output.jsonl \
    --split lite
```

## 🧭 CLI Overview

`fb` provides three core commands:
- `fb infer` runs `featurebench.infer.run_infer` (docs: [docs/infer_cli_arg.md](docs/infer_cli_arg.md))
- `fb eval` runs `featurebench.harness.run_evaluation` (docs: [docs/harness_cli_arg.md](docs/harness_cli_arg.md))
- `fb data` runs `featurebench.pipeline` (docs: [docs/pipeline.md](docs/pipeline.md))

## ✍️ Citation

If you found FeatureBench useful, please cite us as:

```bibtex
@misc{zhou2026featurebenchbenchmarkingagenticcoding,
      title={FeatureBench: Benchmarking Agentic Coding for Complex Feature Development}, 
      author={Qixing Zhou and Jiacheng Zhang and Haiyang Wang and Rui Hao and Jiahe Wang and Minghao Han and Yuxue Yang and Shuzhe Wu and Feiyang Pan and Lue Fan and Dandan Tu and Zhaoxiang Zhang},
      year={2026},
      eprint={2602.10975},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.10975}, 
}
```

## 📧 Contact

If you have any questions, feel free to contact [qixingzhou1125@gmail.com](mailto:qixingzhou1125@gmail.com) or [zjcheng2022@gmail.com](mailto:zjcheng2022@gmail.com).
