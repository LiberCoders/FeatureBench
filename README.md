<p align="center">
  <img src="docs/pics/logo.png" style="height: 10em" alt="logo" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.10975"><img src="https://img.shields.io/badge/arXiv-2602.10975-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://hub.docker.com/u/libercoders"><img src="https://img.shields.io/badge/DockerHub-Images-blue.svg" alt="DockerHub"></a>
  <a href="https://huggingface.co/datasets/LiberCoders/FeatureBench"><img src="https://img.shields.io/badge/HuggingFace-datasets-yellow.svg" alt="HuggingFace"></a>
  <a href="https://libercoders.github.io/FeatureBench/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-view-purple.svg" alt="Leaderboard"></a>
</p>

---

FeatureBench is a test-driven data generation and evaluation pipeline for feature-level coding benchmarks.
It provides a unified CLI to run inference, evaluation, and dataset generation.

## 📰 News

📊 **2026.05.18**: We added **lite split** evaluation results for frontier models including **GPT-5.5, Claude Opus 4.7, DeepSeek-V4, GLM-5.1, Kimi-2.6, Mimo-V2.5-Pro**, and more to the leaderboard.

🚀 **2026.03.27**: We released the **fast split** containing 100 instances (a subset of full split). These instances require no GPU and are optimized for rapid evaluation. On an Intel Xeon Platinum 8457C with 944GB RAM, the average evaluation time per instance using gold patches is **57.2 seconds**.

🎁 **2026.02.06**: We now support one-click inference for mainstream agent frameworks, including **OpenHands, Claude Code, Codex, Gemini CLI, and mini-swe-agent**. All supported agent frameworks can be found [here](featurebench/infer/agents/). We have also open-sourced the FeatureBench **data pipeline**.

<!-- LEADERBOARD_START -->
## 🏆 Leaderboard

[Full interactive leaderboard](https://libercoders.github.io/FeatureBench/leaderboard/) with tabs, filters, and sorting.

<details open>
<summary><b>Lite split results, ranked by %PASSED</b></summary>

| Rank | Model | Scaffold | %PASSED | %RESOLVED |
|---:|---|---|---:|---:|
| 1 | Claude Opus 4.7 | OpenHands | 78.2 | 46.7 |
| 2 | GPT-5.5 | OpenHands | 69.8 | 26.7 |
| 3 | Claude Opus 4.6 | OpenHands | 69.5 | 20 |
| 4 | Claude Opus 4.5 | OpenHands | 67.2 | 20 |
| 5 | GPT-5.4 | OpenHands | 66.2 | 23.3 |
| 6 | GPT-5.1-Codex | Codex | 60.2 | 20 |
| 7 | DeepSeek-V4-Pro | OpenHands | 59.6 | 26.7 |
| 8 | Claude Opus 4.5 | Claude Code | 59.1 | 20 |
| 9 | Kimi-2.6 | OpenHands | 49.4 | 20 |
| 10 | Mimo-V2.5-Pro | OpenHands | 47.8 | 13.3 |
| 11 | Gemini-3-Pro-Preview | OpenHands | 45.1 | 10 |
| 12 | GLM-5.1 | OpenHands | 44.2 | 13.3 |
| 13 | Gemini-3-Pro-Preview | Gemini-CLI | 43.4 | 10 |
| 14 | DeepSeek-V4-Flash | OpenHands | 41.9 | 16.7 |
| 15 | MiniMax M2.1 | Mini-SWE-Agent | 41.9 | 10 |
| 16 | GLM 4.7 | Mini-SWE-Agent | 41.2 | 6.7 |
| 17 | Qwen3-Coder-480B-A35B-Instruct | OpenHands | 38.3 | 6.7 |
| 18 | DeepSeek V3.2 | OpenHands | 35.9 | 6.7 |
| 19 | Qwen3-Coder-30B-A3B-Instruct | OpenHands | 23 | 3.3 |

</details>
<!-- LEADERBOARD_END -->

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
source .venv/bin/activate
```

**Configure:**
```bash
cp config_example.toml config.toml
```
See [docs/config.md](docs/config.md) for a comprehensive reference (harness, infer, data pipeline) with examples.

**Optional: pre-pull images to reduce network variance:**
```bash
fb pull --mode lite                 # lite split image list (13 images)
fb pull --mode fast                 # fast split image list (18 images)
fb pull --mode full                 # full split image list (24 images)
fb pull --mode /path/to/images.txt  # one image name per line

# full list: featurebench/resources/constants/full_images.txt
# lite list: featurebench/resources/constants/lite_images.txt
# fast list: featurebench/resources/constants/fast_images.txt
```

**Run inference:**
```bash
fb infer \
    --config-path config.toml \
    --agent mini_swe_agent \
    --model openai/qwen3-coder-480b-a35b-instruct \
    --split fast
```

**Run evaluation:**
```bash
fb eval \
    -p runs/<timestamp>/output.jsonl \
    --split fast
    # use -p gold to verify the gold patches
```

## 🧭 CLI Overview

`fb` provides three core commands:
- `fb infer` runs `featurebench.infer.run_infer` (docs: [docs/infer_cli_arg.md](docs/infer_cli_arg.md))
- `fb eval` runs `featurebench.harness.run_evaluation` (docs: [docs/harness_cli_arg.md](docs/harness_cli_arg.md))
- `fb data` runs `featurebench.pipeline` (docs: [docs/pipeline.md](docs/pipeline.md))

## ✍️ Citation

If you found FeatureBench useful, please cite us as:

```bibtex
@article{zhou2026featurebench,
  title={FeatureBench: Benchmarking Agentic Coding for Complex Feature Development},
  author={Zhou, Qixing and Zhang, Jiacheng and Wang, Haiyang and Hao, Rui and Wang, Jiahe and Han, Minghao and Yang, Yuxue and Wu, Shuzhe and Pan, Feiyang and Fan, Lue and others},
  journal={arXiv preprint arXiv:2602.10975},
  year={2026}
}
```

## 📧 Contact

If you have any questions, feel free to contact [qixingzhou1125@gmail.com](mailto:qixingzhou1125@gmail.com) or [zjcheng2022@gmail.com](mailto:zjcheng2022@gmail.com).
