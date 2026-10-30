"""CLI invocations and durable run manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from codeworld.models import StrictModel
from codeworld.run_layout import RunLayout
from codeworld.storage.files import atomic_write_bytes

CommandName = Literal["discover", "build", "generate"]


def _default_config_root(project_root: Path) -> Path:
    local = project_root / "configs" / "default"
    if local.is_dir():
        return local

    package = Path(__file__).resolve().parents[1] / "configs" / "default"
    if package.is_dir():
        return package

    source_checkout = Path(__file__).resolve().parents[2] / "configs" / "default"
    return source_checkout


def default_config_paths(
    project_root: Path,
    command: CommandName,
    *,
    adapter: str | None = None,
) -> tuple[Path, ...]:
    """Return the stage-specific default profile for a command."""
    base = _default_config_root(project_root)
    if command == "generate":
        if adapter is None or not adapter.strip():
            raise ValueError("generate requires an Adapter type to select its default config")
        return (base / "adapters" / f"{adapter.strip()}.yaml",)
    return (base / f"{command}.yaml",)


def default_output_directory(project_root: Path, command: CommandName) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return project_root / "outputs" / f"{command}-{stamp}"


def project_relative(path: Path, project_root: Path) -> str:
    absolute = path if path.is_absolute() else project_root / path
    try:
        return str(absolute.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(absolute.resolve())


def validate_config_paths(paths: tuple[str | Path, ...], project_root: Path) -> tuple[Path, ...]:
    if not paths:
        raise ValueError("Config must contain at least one YAML file")
    normalized = tuple(
        Path(path) if Path(path).is_absolute() else project_root / path for path in paths
    )
    for path in normalized:
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Config is not a YAML file: {project_relative(path, project_root)}")
    return normalized


@dataclass(frozen=True, slots=True)
class CliInvocation:
    command: CommandName
    project_root: Path
    config_files: tuple[Path, ...]
    output_directory: Path
    repository_source: str | None = None
    environment_source: str | None = None
    environment_id: str | None = None
    adapter: str | None = None
    ref: str | None = None
    limit: int | None = None
    selection: str = "unseen"
    resume: bool = False
    run_id: str | None = None

    @property
    def repositories_path(self) -> Path:
        return RunLayout(self.output_directory).repositories

    @property
    def environments_path(self) -> Path:
        return RunLayout(self.output_directory).environments

    @property
    def failures_path(self) -> Path:
        return RunLayout(self.output_directory).failures

    @property
    def tasks_path(self) -> Path:
        return RunLayout(self.output_directory).tasks


class RunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    command: CommandName
    config_files: tuple[str, ...]
    output_directory: str
    repository_source: str | None = None
    environment_source: str | None = None
    environment_id: str | None = None
    adapter: str | None = None
    ref: str | None = None
    limit: int | None = Field(default=None, ge=1)
    selection: str = Field(min_length=1)

    @classmethod
    def from_invocation(cls, invocation: CliInvocation, run_id: str) -> RunManifest:
        return cls(
            run_id=run_id,
            command=invocation.command,
            config_files=tuple(
                project_relative(path, invocation.project_root) for path in invocation.config_files
            ),
            output_directory=project_relative(invocation.output_directory, invocation.project_root),
            repository_source=invocation.repository_source,
            environment_source=invocation.environment_source,
            environment_id=invocation.environment_id,
            adapter=invocation.adapter,
            ref=invocation.ref,
            limit=invocation.limit,
            selection=invocation.selection,
        )

    def invocation(self, project_root: Path, *, resume: bool) -> CliInvocation:
        config_files = tuple(
            Path(value) if Path(value).is_absolute() else project_root / value
            for value in self.config_files
        )
        output = Path(self.output_directory)
        if not output.is_absolute():
            output = project_root / output
        return CliInvocation(
            command=self.command,
            project_root=project_root,
            config_files=config_files,
            output_directory=output,
            repository_source=self.repository_source,
            environment_source=self.environment_source,
            environment_id=self.environment_id,
            adapter=self.adapter,
            ref=self.ref,
            limit=self.limit,
            selection=self.selection,
            resume=resume,
            run_id=self.run_id,
        )


def write_run_manifest(invocation: CliInvocation, run_id: str) -> Path:
    layout = RunLayout(invocation.output_directory)
    path = layout.manifest
    existing = layout.existing_run_entries()
    if not invocation.resume and existing:
        raise ValueError(
            "Output directory is not empty; choose a new directory or resume its run: "
            f"{path.parent}"
        )
    invocation.output_directory.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest.from_invocation(invocation, run_id)
    _atomic_yaml(path, manifest.model_dump(mode="json"))
    return path


def write_run_invocation_manifest(invocation: CliInvocation, run_id: str) -> Path:
    """Append one immutable manifest for an actual fresh or resumed execution."""
    root = RunLayout(invocation.output_directory).invocations
    root.mkdir(parents=True, exist_ok=True)
    sequences = tuple(
        int(candidate.name)
        for candidate in root.iterdir()
        if candidate.is_dir() and candidate.name.isdigit()
    )
    directory = root / f"{max(sequences, default=0) + 1:04d}"
    directory.mkdir(exist_ok=False)
    path = directory / "run.yaml"
    manifest = RunManifest.from_invocation(invocation, run_id)
    _atomic_yaml(path, manifest.model_dump(mode="json"))
    return path


def load_run_manifest(
    project_root: Path,
    output_directory: Path,
) -> tuple[Path, RunManifest]:
    """Load a run identity without requiring its referenced config files to remain valid."""
    directory = (
        output_directory if output_directory.is_absolute() else project_root / output_directory
    )
    path = RunLayout(directory).manifest
    if not path.is_file():
        raise ValueError(f"Not a FeatureBench run directory (missing run.yaml): {directory}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = RunManifest.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid run manifest {path}: {exc}") from exc
    invocation = manifest.invocation(project_root, resume=True)
    if invocation.output_directory.resolve() != directory.resolve():
        raise ValueError("Run directory was moved; run.yaml output_directory no longer matches")
    return path, manifest


def replace_run_manifest(
    path: Path,
    manifest: RunManifest,
    *,
    updates: Mapping[str, object] | None = None,
) -> RunManifest:
    """Atomically persist validated editable fields for one run manifest."""
    document = manifest.model_dump(mode="python")
    document.update(updates or {})
    updated = RunManifest.model_validate(document)
    _atomic_yaml(path, updated.model_dump(mode="json"))
    return updated


def validate_manifest_inputs(manifest: RunManifest, project_root: Path) -> None:
    """Validate editable execution inputs with the same semantics as the CLI."""

    if manifest.command == "discover":
        if manifest.repository_source is None:
            return
        source = Path(manifest.repository_source)
        source = source if source.is_absolute() else project_root / source
        if source.suffix.lower() != ".txt" or not source.is_file():
            raise ValueError(f"repository source is not a TXT file: {source}")
        return
    if manifest.command == "build":
        source_value = manifest.repository_source
        if source_value is None or not source_value.strip():
            raise ValueError("build source must be non-empty")
        if source_value.lower().endswith(".jsonl"):
            source = Path(source_value)
            source = source if source.is_absolute() else project_root / source
            if not source.is_file():
                raise ValueError(f"repository JSONL does not exist: {source}")
            if manifest.ref is not None:
                raise ValueError("ref only applies to one repository, not repository JSONL")
        return
    environment_value = manifest.environment_source
    if environment_value is None or not environment_value.strip():
        raise ValueError("generate environment must be non-empty")
    environment = Path(environment_value)
    environment = environment if environment.is_absolute() else project_root / environment
    if environment.suffix.lower() != ".jsonl" or not environment.is_file():
        raise ValueError(f"environment JSONL does not exist: {environment}")
    if manifest.environment_id is not None and manifest.limit is not None:
        raise ValueError("generate limit cannot be combined with environment_id")


def load_run_invocation(project_root: Path, output_directory: Path) -> CliInvocation:
    _, manifest = load_run_manifest(project_root, output_directory)
    invocation = manifest.invocation(project_root, resume=True)
    validate_config_paths(invocation.config_files, project_root)
    return invocation


def _atomic_yaml(path: Path, document: dict[str, Any]) -> None:
    payload = yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode("utf-8")
    atomic_write_bytes(path, payload)
