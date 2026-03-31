"""Verification runner: reconstruct, prompt, validate, postprocess."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import urllib.request
from collections import Counter
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - rich optional
    _RICH_AVAILABLE = False

from featurebench.infer.agents.claude_code import ClaudeCodeAgent


LEVEL_RE = re.compile(r"\.lv(\d+)\b", re.I)


# --------------------------- Reconstruction ---------------------------


class ReconstructionError(RuntimeError):
    """Raised when reconstruction fails."""


@dataclass
class ReconstructedCase:
    task_dir: Path
    codebase_dir: Path
    mask_dir: Path
    problem_statement: Path
    instance: dict
    top_objects: List[str]
    f2p_test: str


def _run(cmd: Sequence[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise ReconstructionError(
            f"Command failed: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )


def _ensure_repo(repo: str, commit: str, base_repo_dir: Path, workspace_cache: Path) -> Path:
    """Copy cached repo into a temp working clone and checkout commit (does not touch cache)."""
    repo_name = repo.split('/')[-1]
    source_repo = base_repo_dir / repo_name
    if not source_repo.exists():
        raise ReconstructionError(f"Cached repo not found: {source_repo}")

    workspace_cache.mkdir(parents=True, exist_ok=True)
    dest = workspace_cache / f"{repo_name}_work"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source_repo, dest)

    # 如果缓存里已经有该 commit，则直接 checkout，避免不必要的 fetch
    try:
        _run(["git", "rev-parse", "--verify", commit], cwd=dest)
        _run(["git", "checkout", commit], cwd=dest)
        return dest
    except ReconstructionError:
        pass

    print('aaaa')
    _run(["git", "fetch", "--all", "--tags"], cwd=dest)
    print('bbbb')
    _run(["git", "checkout", commit], cwd=dest)
    print('cccc')
    return dest


def _apply_patch_file(patch_file: Path, target_dir: Path) -> None:
    if not patch_file.exists() or patch_file.stat().st_size == 0:
        return
    # `git apply` can treat missing final newline as a corrupt patch (EOF).
    try:
        data = patch_file.read_bytes()
        if data and not data.endswith(b"\n"):
            patch_file.write_bytes(data + b"\n")
    except Exception:
        # Best-effort; if we can't rewrite the patch file, fall back to apply.
        pass
    _run(["git", "apply", "--whitespace=fix", str(patch_file)], cwd=target_dir)


def reconstruct_task(task_dir: Path, workspace: Path, base_repo_dir: Path) -> ReconstructedCase:
    # 拉起任务所需的工作目录：读取元数据、复制缓存仓库、应用 mask 与测试补丁
    task_dir = task_dir.resolve()
    instance_file = task_dir / "instance.json"
    problem_statement = task_dir / "problem_statement.md"
    patch_file = task_dir / "patch.diff"

    if not instance_file.exists():
        raise ReconstructionError(f"instance.json missing: {instance_file}")
    if not problem_statement.exists():
        raise ReconstructionError(f"problem_statement.md missing: {problem_statement}")

    with open(instance_file, "r", encoding="utf-8") as f:
        instance = json.load(f)

    repo = instance.get("repo") or instance.get("repository") or instance.get("repo_settings", {}).get("repository")
    commit = instance.get("base_commit") or instance.get("repo_settings", {}).get("commit")
    if not repo or not commit:
        raise ReconstructionError("repo or commit not found in instance.json")

    codebase_root = workspace / "codebase"
    mask_root = workspace / "maskcodebase"
    if codebase_root.exists():
        shutil.rmtree(codebase_root)
    if mask_root.exists():
        shutil.rmtree(mask_root)

    cached_repo = _ensure_repo(repo, commit, base_repo_dir, workspace)
    shutil.copytree(cached_repo, codebase_root, ignore=shutil.ignore_patterns(".git"))
    shutil.copytree(codebase_root, mask_root)
    # 不再需要工作副本，清理掉以免用户困惑
    shutil.rmtree(cached_repo, ignore_errors=True)

    _apply_patch_file(patch_file, mask_root)

    # 删除 maskcodebase 中的 FAIL_TO_PASS 测试文件，保持与 infer 侧一致
    f2p_tests = instance.get("FAIL_TO_PASS") or []
    for f2p in f2p_tests:
        try:
            p = Path(f2p)
            if p.is_absolute() and str(p).startswith("/testbed"):
                p = mask_root / p.relative_to("/testbed")
            if not p.is_absolute():
                p = (mask_root / p).resolve()
            if mask_root in p.parents and p.exists():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
        except Exception as e:
            # 删除失败，记录一条日志便于排查
            print(f"[reconstruct] failed to remove F2P path {f2p}: {e}")

    # 将 problem_statement.md 复制到本次临时工作目录根下，便于后续引用绝对路径
    problem_statement_copy = workspace / "problem_statement.md"
    shutil.copy(problem_statement, problem_statement_copy)

    top_objects = instance.get("top_objects") or []
    if not isinstance(top_objects, list):
        raise ReconstructionError("top_objects must be a list")
    f2p_test = f2p_tests[0] if f2p_tests else ""
    if f2p_test:
        test_path = Path(f2p_test)
        if not test_path.is_absolute():
            test_path = (codebase_root / test_path).resolve()
        f2p_test = str(test_path)

    return ReconstructedCase(
        task_dir=task_dir,
        codebase_dir=codebase_root,
        mask_dir=mask_root,
        problem_statement=problem_statement_copy,
        instance=instance,
        top_objects=top_objects,
        f2p_test=f2p_test,
    )


# --------------------------- Prompt & Agent ---------------------------


def load_prompt_template(level: str = "lv1") -> str:
    mapping = {
        "lv1": "prompt_template_lv1.txt",
        "lv2": "prompt_template_lv2.txt",
    }
    name = mapping.get(level.lower(), mapping["lv1"])
    template_path = Path(__file__).with_name("prompts") / name
    return template_path.read_text(encoding="utf-8")


def _detect_level(task_dir: Path, instance: dict) -> str:
    # 从 instance_id 或目录名推断 lv1/lv2，默认 lv1
    candidates = [instance.get("instance_id", ""), instance.get("id", ""), task_dir.name]
    for src in candidates:
        m = LEVEL_RE.search(str(src))
        if m:
            return f"lv{m.group(1)}"
    return "lv1"


def build_prompt(
    top_objects: List[str],
    codebase_dir: Path,
    mask_dir: Path,
    problem_statement: Path,
    f2p_test: str,
    container_root: Path | None = None,
    work_root: Path | None = None,
    level: str = "lv1",
) -> str:
    def _to_container(p: Path) -> Path:
        if container_root and work_root and p.is_relative_to(work_root):
            return container_root / p.relative_to(work_root)
        return p

    def _normalize_obj(obj: str) -> str:
        # Map legacy /testbed prefixes to the container-visible codebase path
        try:
            p = Path(obj)
            if p.is_absolute() and str(p).startswith("/testbed"):
                return str(Path("/workspace/codebase") / p.relative_to("/testbed"))
            return str(p)
        except Exception:
            return obj

    codebase_dir = _to_container(codebase_dir)
    mask_dir = _to_container(mask_dir)
    problem_statement = _to_container(problem_statement)

    top_lines = "\n".join(f"{idx+1}. {_normalize_obj(obj)}" for idx, obj in enumerate(top_objects))
    if f2p_test:
        f2p_path = Path(f2p_test)
        if not f2p_path.is_absolute():
            f2p_path = (codebase_dir / f2p_path).resolve()
        f2p_path = _to_container(f2p_path)
        f2p_display = str(f2p_path)
    else:
        f2p_display = "<unknown>"
    template = load_prompt_template(level=level)
    ans = template.format(
        top=top_lines,
        codebase=str(codebase_dir.resolve()),
        maskcodebase=str(mask_dir.resolve()),
        problem_statement=str(problem_statement.resolve()),
        f2p_test_file_path=f2p_display,
    )
    return ans


def run_claude(
    mask_dir: Path,                       # 宿主机上的 mask 代码目录
    prompt: str,                          # 要发送给 Claude 的提示词
    output_file: Path,                    # 宿主机上期望写出的 verified 结果文件路径
    env: dict[str, str],                  # 传递给容器的环境变量
    image_name: str,                      # 使用的容器镜像名
    cm,                                   # 容器管理器实例
    download_cache: Path | None = None,   # 下载缓存目录（挂载到 /download）
    proxy_port: int | None = None,        # 代理端口
    log_file: Path | None = None,         # agent 日志输出文件路径（宿主机）
    op: str = "verified",                # 操作类型：verified 或 add
) -> None:
    """Run Claude Code inside an isolated container, writing results to work_root."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    work_root = mask_dir.parent
    container_root = Path("/workspace")

    volumes: dict[str, dict] = {str(work_root): {"bind": str(container_root), "mode": "rw"}}
    if download_cache:
        download_cache.mkdir(parents=True, exist_ok=True)
        volumes[str(download_cache)] = {"bind": "/download", "mode": "rw"}

    container_name = f"featurebench-verify-{uuid.uuid4().hex[:8]}"
    container = cm.create_container(
        image_name=image_name,
        container_name=container_name,
        working_dir=str(container_root),
        use_host_network=False,  # 强制 bridge 模式，避免 host 网络
        proxy_port=proxy_port,
        volumes=volumes,
        docker_runtime_config={},
    )

    # 容器日志输出路径（宿主机）
    log_path = log_file or (work_root / "claude_stream.log")
    try:
        if log_path.exists():
            log_path.unlink()  # 清理旧日志，确保本次输出干净
    except Exception:
        pass

    # 清空容器内 /root 与 /testbed 的普通文件，并仅删除隐藏的 .git，避免遗留历史仓库
    cm.exec_command(
        container,
        "rm -rf /root/* /root/.git 2>/dev/null || true",
        log_file=log_path,
    )
    cm.exec_command(
        container,
        "rm -rf /testbed/* /testbed/.git 2>/dev/null || true",
        log_file=log_path,
    )

    # 激活 testbed conda 环境，保持与 infer 初始化一致
    activate_cmd = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
    exit_code, _ = cm.exec_command(container, activate_cmd, log_file=log_path)
    if exit_code != 0:
        raise RuntimeError("Failed to activate conda environment 'testbed' in container")
    
    agent = ClaudeCodeAgent(container_manager=cm, env_vars=env)
    try:
        if not agent.install(container, log_path):
            raise RuntimeError("Claude Code install failed in container")

        # 测试用:
        # prompt="""
        #     在工作目录下创建一个名为 verified_result.txt 的文件，并往里写入 "Hello, Claude!" 作为内容。然后打印这个文件的权限，然后退出
        # """
        run_cmd = agent.get_run_command(prompt)
        exit_code = cm.exec_command_stream(
            container,
            run_cmd,
            log_file=log_path,
            workdir=str(container_root),
            timeout=1800,  # 30 minutes
        )
        if exit_code != 0:
            raise RuntimeError(f"Claude run failed (exit {exit_code}), see {log_file}")

        if op not in {"verified", "add"}:
            raise ValueError(f"unsupported op: {op}")

        target_name = "verified_result.txt" if op == "verified" else "add.txt"
        produced = work_root / target_name
        if not produced.exists():
            raise RuntimeError(f"{target_name} not produced by claude agent in work root")

        # 放宽生成文件权限，防止 600 导致宿主机读/拷贝失败
        cm.exec_command(container, f"chmod 644 /workspace/{target_name} 2>/dev/null || true", log_file=log_path)

        # 避免同路径复制引发 SameFileError
        if produced.resolve() != output_file.resolve():
            shutil.copy(produced, output_file)
    finally:
        try:
            cm.stop_container(container, force=True)
        except Exception:
            pass


# --------------------------- Validation ---------------------------


@dataclass
class ParsedEntry:
    index: int
    obj: str
    result: str
    reasons: List[str]


@dataclass
class ValidationResult:
    entries: List[ParsedEntry]

    @staticmethod
    def _normalize_obj(obj: str) -> str:
        try:
            p = Path(obj)
            if p.is_absolute() and str(p).startswith("/testbed"):
                return str(Path("/workspace/codebase") / p.relative_to("/testbed"))
            return str(p)
        except Exception:
            return obj

    def is_valid(self, top_objects: List[str]) -> bool:
        expected = [self._normalize_obj(obj.strip()) for obj in top_objects]
        actual = [self._normalize_obj(entry.obj.strip()) for entry in self.entries]
        return Counter(actual) == Counter(expected)

    def completeness_ratio(self) -> float:
        if not self.entries:
            return 0.0
        complete = 0
        for e in self.entries:
            res = e.result.strip()
            if res == "完备":  # exact match to avoid counting "不完备"
                complete += 1
        return complete / len(self.entries)


ENTRY_RE = re.compile(
    r"编号:\s*(\d+)\s*对象:\s*(.*?)\s*判断结果:\s*(.*?)\s*判断理由:\s*(.*?)(?=\n编号:|\Z)",
    re.S,
)


def parse_verified_file(path: Path) -> ValidationResult:
    # 从大模型输出的 verified_result.txt 中解析编号/对象/判断结果
    text = path.read_text(encoding="utf-8")
    entries: List[ParsedEntry] = []
    for match in ENTRY_RE.finditer(text):
        idx = int(match.group(1))
        obj = match.group(2).strip()
        res = match.group(3).strip()
        reasons_block = match.group(4).strip()
        reason_lines = [ln.strip() for ln in reasons_block.splitlines() if ln.strip()]
        entries.append(ParsedEntry(idx, obj, res, reason_lines))
    return ValidationResult(entries=entries)


def build_verified_text(parsed: ValidationResult, ratio: float) -> str:
    lines: List[str] = [f"完备比例: {ratio:.3f}", ""]
    for entry in parsed.entries:
        lines.append(f"编号: {entry.index}")
        lines.append(f"对象: {entry.obj}")
        lines.append(f"判断结果: {entry.result}")
        lines.append("判断理由:")
        if entry.reasons:
            lines.extend(entry.reasons)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------- Post-process ---------------------------


def check_and_split(verified_root: Path, failed_dir: Path) -> Tuple[List[Path], List[Path]]:
    # 依据输出格式有效性将案例划分为 valid/invalid，并把失败案例归档到 failed_dir
    valid, invalid = [], []
    for sub in verified_root.iterdir():
        if not sub.is_dir():
            continue
        vf = sub / "verified_result.txt"
        inst = sub / "instance.json"
        if not vf.exists() or not inst.exists():
            invalid.append(sub)
            continue
        try:
            top = json.loads(inst.read_text(encoding="utf-8")).get("top_objects", [])
            parsed = parse_verified_file(vf)
            if parsed.is_valid(top):
                valid.append(sub)
            else:
                invalid.append(sub)
        except Exception:
            invalid.append(sub)

    failed_dir.mkdir(parents=True, exist_ok=True)
    for p in invalid:
        dest = failed_dir / p.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(p, dest)
    return valid, invalid


def sort_valid_cases(valid_cases: List[Path], dest_dir: Path) -> None:
    # 依据完备比例从高到低排序有效案例，生成带序号的副本
    scored = []
    for p in valid_cases:
        vf = p / "verified_result.txt"
        parsed = parse_verified_file(vf)
        scored.append((parsed.completeness_ratio(), p))
    scored.sort(key=lambda x: x[0], reverse=True)

    dest_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, path) in enumerate(scored, start=1):
        name = f"{rank:03d}_{path.name}"
        dest = dest_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)


def postprocess_results(verified_root: Path, failed_dir: Path, sorted_dir: Path) -> dict:
    valid, invalid = check_and_split(verified_root, failed_dir)
    sort_valid_cases(valid, sorted_dir)
    return {
        "valid": [str(p) for p in valid],
        "invalid": [str(p) for p in invalid],
        "sorted": str(sorted_dir),
        "failed": str(failed_dir),
    }


# --------------------------- Main verification ---------------------------


def _discover_tasks(tasks_dir: Path) -> List[Path]:
    # 支持两种输入：1) 直接传单个任务目录；2) 传包含多个任务子目录的父目录（递归）
    if (tasks_dir / "instance.json").exists():
        return [tasks_dir]

    seen = set()
    tasks: List[Path] = []
    for inst in tasks_dir.rglob("instance.json"):
        task_dir = inst.parent
        if task_dir in seen:
            continue
        seen.add(task_dir)
        tasks.append(task_dir)

    # 保持确定性顺序，方便调试
    tasks.sort()
    return tasks


def _verify_single(
    task_dir: Path,
    out_dir: Path,
    base_repo_dir: Path,
    env: dict[str, str],
    cm,
    download_cache: Path | None,
    proxy_port: int | None,
) -> Tuple[Path, ValidationResult, bool, str, float, str]:
    # 单个任务的端到端流程：重建 -> 生成提示 -> 调用大模型 -> 校验输出
    work_dir = Path(tempfile.mkdtemp(prefix="verified_"))  # 临时工作目录
    # 每次执行同一任务前先清理旧输出，避免续写
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    log_path = out_dir / "claude_stream.log"

    def _log(msg: str) -> None:
        # Simple append-only log to capture pre-agent steps
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    try:
        _log(f"start reconstruct: task_dir={task_dir}")
        case = reconstruct_task(task_dir, work_dir, base_repo_dir)  # 从缓存仓库重建完整/裁剪代码
        _log("reconstruct finished")
        level = _detect_level(task_dir, case.instance)
        if level == "lv2":
            # lv2 只暴露 problem_statement + codebase 给验证 agent
            try:
                shutil.rmtree(case.mask_dir, ignore_errors=True)
            except Exception:
                pass
            case.mask_dir = case.codebase_dir
        prompt = build_prompt(
            case.top_objects,           # 顶层对象列表
            case.codebase_dir,          # 原始代码目录
            case.mask_dir,              # mask 代码目录
            case.problem_statement,     # 题目描述路径
            case.f2p_test,              # FAIL_TO_PASS 测试路径
            container_root=Path("/workspace"),  # 容器内挂载根
            work_root=case.mask_dir.parent,     # 容器外工作根（用于路径映射）
            level=level,
        )
        _log("prompt built")

        # 立刻在任务根目录放置元数据，避免等所有轮次结束再写
        inst_dest = out_dir.parent / "instance.json"
        ps_dest = out_dir.parent / "problem_statement.md"
        if not inst_dest.exists():
            shutil.copy(case.task_dir / "instance.json", inst_dest)
        if not ps_dest.exists():
            shutil.copy(case.task_dir / "problem_statement.md", ps_dest)
        prompt_file = out_dir.parent / "prompt_used.txt"
        if not prompt_file.exists():
            prompt_file.write_text(prompt, encoding="utf-8")

        verified_path = out_dir / "verified_result.txt"  # agent 输出文件路径
        image_name = case.instance.get("image_name")
        if not image_name:
            raise RuntimeError("instance missing image_name")
        image_name = image_name.lower()

        run_claude(
            case.mask_dir,                        # 宿主机的 mask 目录
            prompt,                               # 任务描述
            verified_path,                        # 宿主机的输出文件路径
            env=env,                              # 环境变量
            image_name=image_name,                # 镜像名字
            cm=cm,                                # 容器管理器
            download_cache=download_cache,        # 下载缓存目录
            proxy_port=proxy_port,                # 代理端口
            log_file=out_dir / "claude_stream.log",  # 宿主机的日志输出路径
            op="verified",                      # 验证模式
        )
        _log("run_claude returned")
        
        parsed = parse_verified_file(verified_path)  # 解析模型输出
        norm_expected = [parsed._normalize_obj(obj.strip()) for obj in case.top_objects]
        norm_actual = [parsed._normalize_obj(e.obj.strip()) for e in parsed.entries]
        valid = Counter(norm_actual) == Counter(norm_expected)  # 校验对象集合（忽略顺序）

        err = ""
        if not valid:
            expected_count = len(norm_expected)
            actual_count = len(norm_actual)
            if expected_count != actual_count:
                err = f"format validation failed: entries count mismatch (expected {expected_count}, got {actual_count})"
            else:
                missing = list((Counter(norm_expected) - Counter(norm_actual)).elements())
                extra = list((Counter(norm_actual) - Counter(norm_expected)).elements())
                err_parts = []
                if missing:
                    err_parts.append(f"missing: {missing}")
                if extra:
                    err_parts.append(f"extra: {extra}")
                detail = "; ".join(err_parts) if err_parts else "objects mismatch"
                err = f"format validation failed: {detail}"

        ratio = parsed.completeness_ratio()  # 计算完备比例
        verified_text = build_verified_text(parsed, ratio)
        verified_path.write_text(verified_text, encoding="utf-8")
        return verified_path, parsed, valid, err, ratio, prompt  # 返回结果路径、校验标记与错误信息
    
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)  # 清理临时目录


def verify_tasks(
    tasks_dir: Path,
    output_root: Path,
    base_repo_dir: Path,
    env: dict[str, str],
    cm,
    download_cache: Path | None,
    proxy_port: int | None,
    max_workers: int = 2,
    n_rounds: int = 1,
    resume: bool = False,
) -> dict:
    tasks = _discover_tasks(tasks_dir)  # 扫描任务目录，找到含 instance.json 的案例
    results = []  # 汇总每个任务的执行结果
    write_lock = threading.Lock()  # 序列化 summary/error 更新，避免并发覆盖

    def _reuse_existing(task_path: Path, final_dir: Path) -> dict | None:
        # 仅当已有最终 verified_result 时才跳过重跑
        verified_file = final_dir / "verified_result.txt"
        inst_file = final_dir / "instance.json"
        if not verified_file.exists() or not inst_file.exists():
            return None
        try:
            top = json.loads(inst_file.read_text(encoding="utf-8")).get("top_objects", [])
            parsed = parse_verified_file(verified_file)
            valid = parsed.is_valid(top)
            ratio = parsed.completeness_ratio()
            err = "" if valid else "resume: existing result invalid"
            return {
                "task": str(task_path),
                "verified_file": str(verified_file),
                "valid": valid,
                "error": err,
                "ratio": ratio,
                "rounds": [
                    {
                        "round": 0,
                        "verified_file": str(verified_file),
                        "valid": valid,
                        "error": err,
                        "ratio": ratio,
                    }
                ],
            }
        except Exception as e:  # noqa: BLE001
            return {
                "task": str(task_path),
                "verified_file": str(verified_file),
                "valid": False,
                "error": f"resume reuse failed: {e}",
                "rounds": [],
            }
        
    output_root.mkdir(parents=True, exist_ok=True)  # 确保输出根目录存在

    def _run_task(task_path: Path) -> dict:
        final_dir = output_root / task_path.name

        # 断点续跑：若已有 verified_result，直接复用，不重跑
        if resume and final_dir.exists():
            reused = _reuse_existing(task_path, final_dir)
            if reused:
                return reused

        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        round_results = []
        for idx in range(1, n_rounds + 1):
            round_dir = final_dir / f"round_{idx}"
            try:
                vp, parsed, valid, err, ratio, prompt_text = _verify_single(
                    task_path,
                    round_dir,
                    base_repo_dir,
                    env,
                    cm,
                    download_cache,
                    proxy_port,
                )
                round_results.append({
                    "round": idx,
                    "verified_path": vp,
                    "parsed": parsed,
                    "valid": valid,
                    "error": err,
                    "ratio": ratio,
                    "dir": round_dir,
                    "prompt": prompt_text,
                })
            except Exception as e:  # noqa: BLE001
                try:
                    (round_dir / "claude_stream.log").open("a", encoding="utf-8").write(f"error: {e}\n")
                except Exception:
                    pass
                round_results.append({
                    "round": idx,
                    "verified_path": None,
                    "parsed": None,
                    "valid": False,
                    "error": str(e),
                    "ratio": 0.0,
                    "dir": round_dir,
                    "prompt": None,
                })

        valid_rounds = [r for r in round_results if r["valid"] and r["parsed"]]
        final_verified_path: Path | None = None
        final_valid = False
        final_err = ""
        final_ratio = 0.0

        if valid_rounds:  # 至少有一轮格式校验通过
            merged = valid_rounds[0]["parsed"]  # 以首个有效轮次为初始合并结果
            for r in valid_rounds[1:]:  # 遍历其余有效轮次
                other = r["parsed"]  # 当前轮次的解析结果
                for i, entry in enumerate(other.entries):  # entries 顺序与 top_objects 一致
                    if merged.entries[i].result.strip() == "完备" and entry.result.strip() != "完备":  # 若任一有效轮次标记为不完备则下调
                        merged.entries[i] = entry  # 用更严格的判定覆盖

            final_valid = True
            final_ratio = merged.completeness_ratio()
            final_verified_path = final_dir / "verified_result.txt"
            final_verified_path.write_text(build_verified_text(merged, final_ratio), encoding="utf-8")

            # 日志保留在各自轮次目录，不再复制到根目录
        else:
            final_valid = False
            final_err = "all rounds invalid"
            for r in round_results:
                if r["error"]:
                    final_err = r["error"]
                    break

        return {
            "task": str(task_path),
            "verified_file": str(final_verified_path) if final_verified_path else None,
            "valid": final_valid,
            "error": final_err,
            "ratio": final_ratio,
            "rounds": [
                {
                    "round": r["round"],
                    "verified_file": str(r["verified_path"]) if r["verified_path"] else None,
                    "valid": r["valid"],
                    "error": r["error"],
                    "ratio": r["ratio"],
                }
                for r in round_results
            ],
        }

    progress_cm = nullcontext()
    progress = None
    if _RICH_AVAILABLE:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        progress_cm = progress

    # 初始化总结记录
    summary_log = output_root.parent / "summary.jsonl"  # 增量 JSONL，便于长跑查看
    summary_log.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        summary_log.write_text("", encoding="utf-8")
    else:
        if not summary_log.exists():
            summary_log.write_text("", encoding="utf-8")

    # 初始化失败记录
    error_file = output_root.parent / "error.txt"
    if not resume:
        error_file.write_text("", encoding="utf-8")
    else:
        if not error_file.exists():
            error_file.write_text("", encoding="utf-8")

    with progress_cm:
        task_progress_id = None
        if progress:
            task_progress_id = progress.add_task("verify", total=len(tasks))

        processed = 0
        success = 0
        perfect = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:  # 线程池并发处理任务
            futures = {ex.submit(_run_task, t): t for t in tasks}  # 提交任务
            for fut in as_completed(futures):  # 逐个等待完成
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    task_path = futures[fut]
                    res = {
                        "task": str(task_path),
                        "verified_file": None,
                        "valid": False,
                        "error": str(e),
                        "rounds": [],
                        "ratio": 0.0,
                    }

                results.append(res)

                # 原子更新 summary.jsonl 与 error.txt：同一 task 仅保留最新记录
                with write_lock:
                    try:
                        # 重写 summary.jsonl，替换同 task 旧记录
                        existing: list[dict] = []
                        if summary_log.exists() and summary_log.stat().st_size > 0:
                            for line in summary_log.read_text(encoding="utf-8").splitlines():
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                if obj.get("task") == res.get("task"):
                                    continue
                                existing.append(obj)
                        existing.append(res)
                        summary_log.write_text(
                            "\n".join(json.dumps(o, ensure_ascii=False) for o in existing) + "\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                    try:
                        # 更新 error.txt：若本次成功则移除旧失败记录，若失败则保留去重后追加
                        lines = []
                        if error_file.exists() and error_file.stat().st_size > 0:
                            for line in error_file.read_text(encoding="utf-8").splitlines():
                                if line.startswith("task="):
                                    prefix = line.split(" ", 1)[0]
                                    if prefix == f"task={res.get('task','')}":
                                        continue
                                lines.append(line)
                        if not res.get("valid"):
                            lines.append(f"task={res.get('task','')} error={res.get('error','')}")
                        error_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    except Exception:
                        pass

                processed += 1
                if res.get("valid"):
                    success += 1
                    if abs(res.get("ratio", 0.0) - 1.0) < 1e-9:
                        perfect += 1
                fail = processed - success

                if progress and task_progress_id is not None:
                    desc = f"ok:{success} fail:{fail} perfect:{perfect}/{success or 1}"
                    progress.update(task_progress_id, description=desc)
                    progress.advance(task_progress_id, 1)

    return {"results": results, "summary_log": str(summary_log)}  # 返回结果与汇总文件位置
