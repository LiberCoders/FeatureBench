"""Domain exceptions used across the generic data factory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FactoryError(RuntimeError):
    """Base class for actionable data-factory failures."""


class ConfigurationError(FactoryError):
    """Raised when the current configuration is invalid."""


class ContractError(FactoryError):
    """Raised when an environment or task violates its contract."""


class RunnerCollectionUnstableError(ContractError):
    """Raised when repeated Runner collection changes its test-ID-to-path mapping."""


class InfrastructureError(FactoryError):
    """Raised for Docker, storage, network, or worker infrastructure failures."""


class CommandTimeoutError(InfrastructureError):
    """Raised when a container command exceeds its deadline."""

    def __init__(
        self,
        message: str,
        *,
        argv: tuple[str, ...] = (),
        stdout: str = "",
        stderr: str = "",
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.argv = argv
        self.stdout = stdout
        self.stderr = stderr
        self.timeout_seconds = timeout_seconds

    @property
    def combined_output(self) -> str:
        return self.stdout + self.stderr


class RunnerTimeoutError(CommandTimeoutError):
    """Raised only when the image-owned Runner process exceeds its deadline."""


DOCKER_HARNESS_STEP_MARKER = "FEATUREBENCH_STEP_ORIGIN=harness"
DOCKER_HARNESS_FRAGMENT_BEGIN = f"# {DOCKER_HARNESS_STEP_MARKER}:begin"
DOCKER_HARNESS_FRAGMENT_END = f"# {DOCKER_HARNESS_STEP_MARKER}:end"


class DockerBuildOrigin(StrEnum):
    """Owner of one Docker build operation, derived from rendered-step provenance."""

    ARTIFACT = "artifact"
    HARNESS = "harness"
    PLATFORM = "platform"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DockerBuildStep:
    """The single BuildKit step that decided a Docker build outcome."""

    step_id: str
    stage: str
    instruction: str
    output: str
    origin: DockerBuildOrigin = DockerBuildOrigin.UNKNOWN

    @property
    def detail(self) -> str:
        return "\n".join(part for part in (self.instruction, self.output) if part)


class DockerBuildError(FactoryError):
    """One neutral Docker execution failure with adapter-derived step provenance."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...] = (),
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        duration_seconds: float | None = None,
        timeout_seconds: float | None = None,
        failed_step: DockerBuildStep | None = None,
    ) -> None:
        if (exit_code is None) == (timeout_seconds is None):
            raise ValueError("Docker build errors require exactly one exit code or timeout")
        if exit_code == 0:
            raise ValueError("a successful Docker build cannot produce DockerBuildError")
        self.argv = argv
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration_seconds = duration_seconds
        self.timeout_seconds = timeout_seconds
        self.failed_step = failed_step
        outcome = (
            f"timed out after {timeout_seconds:g}s"
            if timeout_seconds is not None
            else f"failed with exit code {exit_code}"
        )
        location = ""
        if failed_step is not None:
            location = f" in BuildKit stage {failed_step.stage!r}: {failed_step.instruction}"
        evidence = self.decisive_output.strip()[-5000:]
        super().__init__(
            f"Docker build {outcome}{location}" + (f"\n{evidence}" if evidence else "")
        )

    @property
    def timed_out(self) -> bool:
        return self.timeout_seconds is not None

    @property
    def origin(self) -> DockerBuildOrigin:
        if self.failed_step is None:
            return DockerBuildOrigin.UNKNOWN
        return self.failed_step.origin

    @property
    def combined_output(self) -> str:
        return self.stdout + self.stderr

    @property
    def process_output_tail(self) -> str:
        """Retain process-level errors that BuildKit does not prefix with a step ID."""
        return (self.stderr.strip() or self.stdout.strip())[-2900:]

    @property
    def decisive_output(self) -> str:
        """Combine precise step provenance with the bounded Docker process tail."""
        process_tail = self.process_output_tail
        if self.failed_step is None:
            return process_tail
        step_detail = self.failed_step.detail.strip()[-1900:]
        if not process_tail or process_tail == step_detail:
            return step_detail
        return f"BuildKit failed step:\n{step_detail}\nDocker process output tail:\n{process_tail}"


class SourceMaterializationError(InfrastructureError):
    """A Source adapter rendered an invalid build-only Dockerfile."""


class EnvironmentValidationError(ContractError):
    """A validation failure with an explicit phase and authorized repair fields."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        repair_fields: tuple[str, ...],
    ) -> None:
        if not phase or not repair_fields:
            raise ValueError("validation phase and repair fields must be non-empty")
        super().__init__(message)
        self.phase = phase
        self.repair_fields = repair_fields


class AgentTimeoutError(InfrastructureError):
    """Raised when one complete host-side Agent run exceeds its wall-clock deadline."""


class AgentContextWindowError(InfrastructureError):
    """Raised when a provider rejects an Agent request as larger than its context window."""


class AgentModelError(InfrastructureError):
    """Raised when an Agent model run fails after host-side recovery is exhausted."""


class NoWorkSelected(FactoryError):
    """Signal that an Adapter found no logical work matching the current selection."""


class RepositoryRejected(FactoryError):
    """A correctly analyzed repository that is outside the configured acceptance policy."""

    def __init__(self, category: str, reason: str) -> None:
        super().__init__(reason)
        self.category = category
        self.reason = reason


class RepositoryMaterializationRejected(RepositoryRejected):
    """A repository whose requested Git object definitively cannot be materialized."""

    def __init__(self, reason: str) -> None:
        super().__init__("repository_object_unavailable", reason)


class CandidateRejected(FactoryError):
    """Raised when a generation candidate cannot meet adapter quality gates."""
