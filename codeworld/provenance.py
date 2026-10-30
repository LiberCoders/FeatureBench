"""Resolved configuration and local pipeline identity for every durable run."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import Field

from codeworld.config import FactorySettings
from codeworld.models import StrictModel
from codeworld.storage.files import atomic_write_bytes


class RunProvenance(StrictModel):
    run_id: str = Field(min_length=1)
    created_at: datetime
    featurebench_version: str
    python_version: str
    platform: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    pipeline_patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pipeline_patch_path: str | None = None
    untracked_paths: tuple[str, ...] = ()
    resolved_settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_settings_path: str = "resolved-settings.json"


def _git_output(project_root: Path, *arguments: str) -> bytes | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def write_run_provenance(
    run_directory: str | Path,
    *,
    run_id: str,
    settings: FactorySettings,
    project_root: str | Path,
) -> Path:
    """Persist one immutable resolved-settings and tracked-pipeline identity."""
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    settings_bytes = settings.model_dump_json(indent=2, exclude_none=False).encode() + b"\n"
    project = Path(project_root).resolve()
    raw_commit = _git_output(project, "rev-parse", "HEAD")
    commit = raw_commit.decode(errors="replace").strip() if raw_commit else None
    status = _git_output(project, "status", "--porcelain=v1", "--untracked-files=all")
    patch = _git_output(project, "diff", "--binary", "--no-ext-diff", "HEAD")
    patch_path: Path | None = None
    patch_sha256: str | None = None
    if patch:
        patch_path = root / "pipeline.patch"
        atomic_write_bytes(patch_path, patch)
        patch_sha256 = hashlib.sha256(patch).hexdigest()
    status_lines = status.decode(errors="replace").splitlines() if status else []
    untracked = tuple(sorted(line[3:] for line in status_lines if line.startswith("?? ")))
    settings_path = root / "resolved-settings.json"
    atomic_write_bytes(settings_path, settings_bytes)
    try:
        package_version = version("featurebench")
    except PackageNotFoundError:
        package_version = "unknown"
    provenance = RunProvenance(
        run_id=run_id,
        created_at=datetime.now(UTC),
        featurebench_version=package_version,
        python_version=sys.version,
        platform=platform.platform(),
        git_commit=commit,
        git_dirty=bool(status_lines) if status is not None else None,
        pipeline_patch_sha256=patch_sha256,
        pipeline_patch_path=patch_path.name if patch_path is not None else None,
        untracked_paths=untracked,
        resolved_settings_sha256=hashlib.sha256(settings_bytes).hexdigest(),
    )
    destination = root / "provenance.json"
    provenance_bytes = provenance.model_dump_json(indent=2).encode() + b"\n"
    atomic_write_bytes(destination, provenance_bytes)
    return destination
