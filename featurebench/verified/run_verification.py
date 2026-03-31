"""CLI to run verification workflow.

启动示例：
python -m featurebench.verified.run_verification \
    --tasks-dir cases_filter \
    --output-dir outputs/verified \
    --max-workers 5 \
    --model claude-sonnet-4-5-20250929 \
    --port 7899 \
    --n-rounds 3 \
    --resume

--tasks-dir 可以传具体的一个 task dir, 也可以传包含多个 task dir 的上级目录
--max-workers 并行任务数
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from featurebench.infer.config import InferConfigLoader
from featurebench.infer.container import ContainerManager
from featurebench.verified.runner import verify_tasks, postprocess_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ACE-Bench verification over task folders")
    p.add_argument("--tasks-dir", type=Path, required=True, help="Directory containing task folders")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write verified outputs")
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--port", type=int, help="HTTP(S) proxy port, e.g. 7890 (fallback; config.toml overrides)")
    p.add_argument("--model", type=str, required=True, help="Override ANTHROPIC_MODEL for Claude Code (required)")
    p.add_argument("--n-rounds", type=int, default=1, help="Run each task multiple rounds; >=1")
    p.add_argument("--resume", action="store_true", help="If set, skip tasks whose outputs already exist under --output-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 仅注入配置中声明的变量，避免携带宿主机杂项环境
    env: dict[str, str] = {}

    def reset_dir(path: Path) -> None:
        """Remove a directory if it exists, then recreate it empty."""
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.toml"

    config_loader = InferConfigLoader(config_path)  # 配置加载器

    # 全局 env_vars（含 HF/代理等）
    env.update(config_loader.env_vars)  # 合并全局环境变量

    # Claude Code agent env（model/base_url 等）
    agent_env = config_loader.get_agent_env_vars("claude_code")  # 获取 agent 环境
    env.update(agent_env)  # 合并 agent 变量

    # 下载缓存挂载时，显式告知 agent 复用 /download
    download_cache = config_loader.get_cache_dir()  # 下载缓存目录
    if download_cache:
        env.setdefault("AGENT_DOWNLOAD_CACHE", "/download")

    # 命令行优先覆盖模型
    if args.model:
        env["ANTHROPIC_MODEL"] = args.model

    # 只把声明的关键变量写回进程环境, 相当于在进程里进行 source 了
    os.environ.update(env)

    # 使用内置缓存仓库作为只读源，后续复制到临时目录再 checkout
    base_repo_dir = project_root / "featurebench" / "resources" / "repos"

    # 输出目录：所有任务放入 all/ 下
    verified_root = args.output_dir
    tasks_root = verified_root / "all"
    tasks_root.mkdir(parents=True, exist_ok=True)

    cm = ContainerManager(env_vars=env)  # 初始化容器管理器

    proxy_port = args.port  # 代理端口

    # 并行执行所有任务的验证，生成 summary.jsonl
    summary = verify_tasks(
        tasks_dir=args.tasks_dir,       # cases
        output_root=tasks_root,         # 输出目录（all/）
        base_repo_dir=base_repo_dir,    # repo cache
        env=env,                        # 环境变量
        cm=cm,                          # 容器管理器
        download_cache=download_cache,  # 下载缓存
        proxy_port=proxy_port,          # 代理端口
        max_workers=args.max_workers,   # 并行数
        n_rounds=max(1, args.n_rounds), # 每个任务运行轮数
        resume=args.resume,             # 断点续跑：已存在的任务目录跳过
    )

    # 后处理：分拣无效结果、按完备比例排序
    failed_dir = verified_root / "failed"  # 无效结果目录
    sorted_dir = verified_root / "sorted"  # 排序结果目录
    reset_dir(failed_dir)
    reset_dir(sorted_dir)
    post = postprocess_results(tasks_root, failed_dir, sorted_dir)  # 后处理

    print("verification finished")
    print(f"summary log: {summary['summary_log']}")
    print(f"sorted dir: {post['sorted']}")
    print(f"failed dir: {post['failed']}")


if __name__ == "__main__":
    main()
