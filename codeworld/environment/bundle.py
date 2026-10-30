"""Read and write the complete external artifact bundle for one environment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.errors import ContractError
from codeworld.models import RunnerLimits, RuntimeResources, RuntimeSpec, StrictModel
from codeworld.storage.files import atomic_write_bytes

MANIFEST_NAME = "environment.json"
DOCKERFILE_NAME = "Dockerfile"
ACTIVATION_NAME = "activate"
RUNNER_NAME = "runner"


class ArtifactDigest(StrictModel):
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnvironmentManifest(StrictModel):
    environment_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    language: str = Field(min_length=1)
    gpu_requirement: Literal["none", "optional", "required"]
    image: str = Field(min_length=1)
    workdir: Literal["/testbed"] = "/testbed"
    limits: RunnerLimits
    resources: RuntimeResources
    verified_test_points: tuple[CollectedTestPoint, ...] = Field(min_length=1)
    dockerfile: ArtifactDigest
    activation: ArtifactDigest
    runner: ArtifactDigest


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_environment_bundle(destination: str | Path, record: EnvironmentRecord) -> Path:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    manifest = EnvironmentManifest(
        environment_id=record.environment_id,
        repository=record.repository,
        commit=record.commit,
        language=record.language,
        gpu_requirement=record.gpu_requirement,
        image=record.image,
        workdir=record.runtime.workdir,
        limits=record.runtime.limits,
        resources=record.runtime.resources,
        verified_test_points=record.verified_test_points,
        dockerfile=ArtifactDigest(file=DOCKERFILE_NAME, sha256=_digest(record.dockerfile)),
        activation=ArtifactDigest(file=ACTIVATION_NAME, sha256=_digest(record.runtime.activation)),
        runner=ArtifactDigest(file=RUNNER_NAME, sha256=_digest(record.runtime.runner)),
    )
    atomic_write_bytes(root / DOCKERFILE_NAME, record.dockerfile.encode("utf-8"))
    atomic_write_bytes(
        root / ACTIVATION_NAME,
        record.runtime.activation.encode("utf-8"),
        executable=True,
    )
    atomic_write_bytes(
        root / RUNNER_NAME,
        record.runtime.runner.encode("utf-8"),
        executable=True,
    )
    atomic_write_bytes(
        root / MANIFEST_NAME,
        (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    return root


def _read_checked(root: Path, artifact: ArtifactDigest) -> str:
    path = root / artifact.file
    if not path.resolve().is_relative_to(root.resolve()):
        raise ContractError(f"environment artifact path escapes its bundle: {artifact.file}")
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read environment artifact {path}: {exc}") from exc
    actual = _digest(content)
    if actual != artifact.sha256:
        raise ContractError(
            f"environment artifact digest mismatch for {artifact.file}: "
            f"expected {artifact.sha256}, got {actual}"
        )
    return content


def read_environment_bundle(source: str | Path) -> EnvironmentRecord:
    root = Path(source)
    try:
        manifest = EnvironmentManifest.model_validate_json(
            (root / MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"invalid environment bundle at {root}: {exc}") from exc
    dockerfile = _read_checked(root, manifest.dockerfile)
    activation = _read_checked(root, manifest.activation)
    runner = _read_checked(root, manifest.runner)
    return EnvironmentRecord(
        environment_id=manifest.environment_id,
        repository=manifest.repository,
        commit=manifest.commit,
        language=manifest.language,
        gpu_requirement=manifest.gpu_requirement,
        image=manifest.image,
        dockerfile=dockerfile,
        runtime=RuntimeSpec(
            workdir=manifest.workdir,
            activation=activation,
            runner=runner,
            limits=manifest.limits,
            resources=manifest.resources,
        ),
        verified_test_points=manifest.verified_test_points,
    )
