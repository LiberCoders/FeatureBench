"""Post-process incomplete verification results by asking agent to add missing info.

启动示例：
python -m featurebench.verified.post_additions \
    --tasks-dir cases_filter \
    --input-dir outputs/verified \
    --max-workers 1 \
    --model claude-sonnet-4-5-20250929 \
    --port 7899 \
    --resume

--input-dir 只能是 outputs/verified 这种层级
--max-workers 并行任务数
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import List

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
    console = Console()
except Exception:  # pragma: no cover - optional rich
    _RICH_AVAILABLE = False
    console = None

from featurebench.infer.config import InferConfigLoader
from featurebench.infer.container import ContainerManager
from featurebench.verified.runner import _detect_level, parse_verified_file, reconstruct_task, run_claude


def load_add_prompt(level: str) -> str:
    mapping = {
        "lv1": "add_prompt_lv1.txt",
        "lv2": "add_prompt_lv2.txt",
    }
    name = mapping.get(level.lower(), mapping["lv1"])
    template_path = Path(__file__).with_name("prompts") / name
    return template_path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-process incomplete tasks and generate add.txt")
    p.add_argument("--tasks-dir", type=Path, required=True, help="Root to locate original cases (single path)")
    p.add_argument("--input-dir", type=Path, required=True, help="Root dir of verified outputs (e.g., outputs/verified)")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--port", type=int, help="HTTP(S) proxy port, e.g. 7890 (fallback; config.toml overrides)")
    p.add_argument("--model", type=str, required=True, help="Override ANTHROPIC_MODEL for Claude Code (required)")
    p.add_argument("--resume", action="store_true", help="Skip tasks that already have add.txt")
    p.add_argument("--verbose", action="store_true", help="Log each task status (default: quiet like runner.py)")
    return p.parse_args()


def find_case_dir(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate
    # fallback: rglob once per root
    for root in roots:
        match = next(root.rglob(name), None)
        if match:
            return match
    return None


def collect_incomplete_objects(verified_path: Path, top_objects: List[str]) -> tuple[list[dict], float]:  # 收集不完备对象
    def _normalize_obj(obj: str) -> str:
        try:
            p = Path(obj)
            if p.is_absolute() and str(p).startswith("/testbed"):
                return str(Path("/workspace/codebase") / p.relative_to("/testbed"))
            return str(p)
        except Exception:
            return obj

    parsed = parse_verified_file(verified_path)  # 解析验证结果
    incomplete = []  # 存放不完备项
    for entry, top in zip(parsed.entries, top_objects, strict=False):  # 对齐遍历结果与对象名
        if entry.result.strip() != "完备":  # 仅保留未完备项
            incomplete.append(
                {
                    "obj": _normalize_obj(top),  # 对象名称（路径归一化，/testbed -> /workspace/codebase）
                    "reasons": entry.reasons,  # 未完备原因
                }
            )
    ratio = parsed.completeness_ratio()  # 完备度比例
    return incomplete, ratio  # 返回未完备列表与比例


def build_add_prompt(level: str, incomplete: list[dict]) -> str:
    template = load_add_prompt(level)
    lines: list[str] = []
    for idx, item in enumerate(incomplete, start=1):
        reasons = "; ".join(item.get("reasons", [])) or "未提供理由"
        lines.append(f"{idx}. {item['obj']}\n   理由: {reasons}")
    tasks_block = "\n\n".join(lines) if lines else "- 无"
    return template.strip() + "\n\n不完备对象列表（来自 verified.txt）：\n" + tasks_block + "\n"


def validate_add_output(incomplete: list[dict], add_path: Path) -> tuple[bool, str]:
    """Ensure add.txt mentions every incomplete object (case-insensitive substring match)."""
    if not add_path.exists():
        return False, "add.txt not produced"
    text = add_path.read_text(encoding="utf-8").lower()
    missing = []
    for item in incomplete:
        obj = str(item.get("obj", "")).strip()
        if not obj:
            continue
        if obj.lower() not in text:
            missing.append(obj)
    if missing:
        return False, f"missing objects: {missing}"
    return True, ""


def process_task(task_dir: Path, case_dir: Path, base_repo_dir: Path, env: dict, cm, download_cache: Path | None, proxy_port: int | None, resume: bool) -> str:  # 处理单个任务的补充流程
    verified_path = task_dir / "verified_result.txt"  # 已验证结果文件
    inst_path = task_dir / "instance.json"            # 实例配置
    ps_path = task_dir / "problem_statement.md"       # 问题描述

    if not verified_path.exists() or not inst_path.exists() or not ps_path.exists():
        return "missing required files"  # 缺少必要文件直接返回
    
    instance = json.loads(inst_path.read_text(encoding="utf-8"))  # 读取实例配置
    top_objects = instance.get("top_objects", [])                 # 获取检测对象列表

    incomplete, ratio = collect_incomplete_objects(verified_path, top_objects)  # 统计不完备对象
    if ratio >= 1.0 or not incomplete:
        return "complete"  # 已完备则无需处理
    
    add_dst = task_dir / "add.txt"  # 目标输出路径
    if resume and add_dst.exists():
        return "skipped (resume)"  # 续跑且已存在则跳过

    workspace = Path(tempfile.mkdtemp(prefix="post_add_"))  # 临时工作目录
    try:
        shutil.copy(ps_path, workspace / "problem_statement.md")       # 拷贝问题描述
        shutil.copy(verified_path, workspace / "verified_result.txt")  # 拷贝验证结果

        case = reconstruct_task(case_dir, workspace, base_repo_dir)  # 重建任务上下文
        level = _detect_level(case_dir, case.instance)  # 识别级别
        if level == "lv2":
            try:
                shutil.rmtree(case.mask_dir, ignore_errors=True)  # lv2 不用 mask，清理
            except Exception:
                pass
            case.mask_dir = case.codebase_dir  # lv2 直接用代码库

        prompt = build_add_prompt(level=level, incomplete=incomplete)  # 构建补充提示词

        add_path = workspace / "add_tmp.txt"  # 临时输出路径
        image_name = case.instance.get("image_name")  # 获取镜像名
        if not image_name:
            return "missing image_name"  # 缺镜像名则终止
        image_name = image_name.lower()  # 规范化大小写

        run_claude(
            mask_dir=case.mask_dir,  # 掩码或代码目录
            prompt=prompt,           # 提示词
            output_file=add_path,    # 输出文件
            env=env,                 # 环境变量
            image_name=image_name,   # 容器镜像
            cm=cm,                   # 容器管理器
            download_cache=download_cache,  # 下载缓存
            proxy_port=proxy_port,          # 代理端口
            log_file=task_dir / "claude_stream_add.log",  # 日志文件写入任务目录
            op="add",                # 补充信息模式
        )

        valid, msg = validate_add_output(incomplete, add_path)  # 校验补充输出是否覆盖所有不完备对象
        if not valid:
            return f"invalid add output: {msg}"
        
        shutil.copy(add_path, add_dst)  # 复制结果到目标
        return "ok"  # 标记成功
    
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"  # 捕获异常返回
    finally:
        shutil.rmtree(workspace, ignore_errors=True)  # 清理临时目录


def main() -> None:
    args = parse_args()
    show_each = args.verbose

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.toml"

    config_loader = InferConfigLoader(config_path)

    env: dict[str, str] = {}
    env.update(config_loader.env_vars)
    env.update(config_loader.get_agent_env_vars("claude_code"))

    download_cache = config_loader.get_cache_dir()
    if download_cache:
        env.setdefault("AGENT_DOWNLOAD_CACHE", "/download")
    if args.model:
        env["ANTHROPIC_MODEL"] = args.model
    os.environ.update(env)

    base_repo_dir = project_root / "featurebench" / "resources" / "repos"
    verified_root = args.input_dir / "all"
    task_dirs = [p for p in verified_root.iterdir() if p.is_dir()]

    case_roots = [args.tasks_dir]

    cm = ContainerManager(env_vars=env)
    proxy_port = args.port

    results: list[tuple[str, str]] = []
    write_lock = threading.Lock()

    # 初始化摘要与错误记录文件（与 runner.py 保持一致，但命名为 *_add）
    summary_add = verified_root.parent / "summary_add.jsonl"
    error_add = verified_root.parent / "error_add.txt"
    summary_add.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        summary_add.write_text("", encoding="utf-8")
    else:
        if not summary_add.exists():
            summary_add.write_text("", encoding="utf-8")

    if not args.resume:
        error_add.write_text("", encoding="utf-8")
    else:
        if not error_add.exists():
            error_add.write_text("", encoding="utf-8")

    def _run(task_path: Path) -> tuple[str, str]:  # 处理单个任务目录
        name = task_path.name  # 任务名与案例目录同名
        case_dir = find_case_dir(name, case_roots)  # 查找原始案例目录
        if not case_dir:
            return name, "case dir not found"  # 找不到源案例则跳过
        status = process_task(
            task_dir=task_path,           # 已验证结果所在目录
            case_dir=case_dir,            # 原始案例文件目录
            base_repo_dir=base_repo_dir,  # 仓库缓存根目录
            env=env,                      # 运行所需环境变量
            cm=cm,                        # 共享的容器管理器
            download_cache=download_cache,# 下载缓存挂载路径
            proxy_port=proxy_port,        # 可选代理端口
            resume=args.resume,           # 若已有 add.txt 则跳过
        )
        return name, status

    progress_cm = None
    task_progress = None
    if _RICH_AVAILABLE:
        progress_cm = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
    else:
        progress_cm = None

    context_mgr = progress_cm if progress_cm else nullcontext()

    with context_mgr:
        if progress_cm:
            task_progress = progress_cm.add_task("post-add", total=len(task_dirs))

        ok_count = 0
        skip_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_map = {ex.submit(_run, td): td for td in task_dirs}
            for fut in as_completed(future_map):
                name, status = fut.result()  # 获取结果
                results.append((name, status))
                if status == "ok":
                    ok_count += 1
                elif status.startswith("complete") or status.startswith("skipped"):
                    skip_count += 1
                else:
                    error_count += 1

                if show_each:
                    if console:
                        style = "green" if status == "ok" else "yellow" if status.startswith("complete") or status.startswith("skipped") else "red"
                        console.log(f"{name}: {status}", style=style)
                    else:
                        print(f"{name}: {status}")
                if progress_cm and task_progress is not None:
                    desc = f"ok:{ok_count} skip:{skip_count} err:{error_count}"
                    progress_cm.update(task_progress, description=desc)
                    progress_cm.advance(task_progress)

                # 同步写入 summary_add.jsonl / error_add.txt，逻辑参考 runner.py
                with write_lock:
                    try:
                        existing: list[dict] = []
                        if summary_add.exists() and summary_add.stat().st_size > 0:
                            for line in summary_add.read_text(encoding="utf-8").splitlines():
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                if obj.get("task") == name:
                                    continue
                                existing.append(obj)
                        existing.append({"task": name, "status": status})
                        summary_add.write_text(
                            "\n".join(json.dumps(o, ensure_ascii=False) for o in existing) + "\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                    try:
                        lines = []
                        if error_add.exists() and error_add.stat().st_size > 0:
                            for line in error_add.read_text(encoding="utf-8").splitlines():
                                if line.startswith("task="):
                                    prefix = line.split(" ", 1)[0]
                                    if prefix == f"task={name}":
                                        continue
                                lines.append(line)
                        # 仅当状态不是 ok/complete/skipped(resume) 时记录错误
                        if not (status == "ok" or status.startswith("complete") or status.startswith("skipped")):
                            lines.append(f"task={name} error={status}")
                        error_add.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    except Exception:
                        pass

    # 简单总结
    ok = sum(1 for _, s in results if s == "ok")  # 成功条目数量
    skipped = sum(1 for _, s in results if s.startswith("complete") or s.startswith("skipped"))  # 跳过或已完备数量
    errors = [r for r in results if r[1].startswith("error") or "not found" in r[1]]  # 错误列表
    if _RICH_AVAILABLE and console:
        table = Table(title="Post-add summary")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        table.add_row("ok", str(ok))
        table.add_row("skipped", str(skipped))
        table.add_row("errors", str(len(errors)))
        console.print(table)
    else:
        print(f"done. ok={ok}, skipped={skipped}, errors={len(errors)}")

    if errors:
        if console:
            console.rule("errors")
            for name, status in errors:
                console.print(f"{name}: {status}", style="red")
        else:
            for name, status in errors:
                print(f"error {name}: {status}")  # 逐条打印错误


if __name__ == "__main__":
    main()
