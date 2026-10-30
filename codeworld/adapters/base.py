"""Adapter data-production contracts and runtime composition."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from codeworld.models import StrictModel, TaskRecord
from codeworld.selection import WorkSelection

if TYPE_CHECKING:
    from codeworld.environment.models import EnvironmentRecord
    from codeworld.events import EventSink
    from codeworld.progress import AdapterPhase, AdapterProgressReporter
    from codeworld.runtime.base import ContainerRuntimeProtocol


class AdapterSettings(StrictModel):
    """Execution controls shared by every Adapter-owned YAML schema."""

    environment_concurrency: int = Field(default=4, ge=1, le=32)


@dataclass(frozen=True, slots=True)
class GeneratorContext:
    """Host services and output-driven selection for one Adapter invocation."""

    runtime: ContainerRuntimeProtocol
    events: EventSink
    progress: AdapterProgressReporter
    work_dir: Path
    run_id: str
    selection: WorkSelection
    output_dir: Path | None = None
    completed_task_ids: frozenset[str] = frozenset()


@runtime_checkable
class AdapterGenerator(Protocol):
    """Produce canonical task candidates from one validated environment."""

    def generate(
        self, environment: EnvironmentRecord, context: GeneratorContext
    ) -> AsyncGenerator[TaskRecord, None]:
        """Generate only Adapter work whose durable state matches ``context.selection``."""
        ...


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    """Transient outcome consumed before a candidate may enter task JSONL."""

    passed: bool
    message: str = ""
    metrics: Mapping[str, int | float | bool | str] = field(
        default_factory=dict[str, int | float | bool | str]
    )


@runtime_checkable
class TaskQualityGate(Protocol):
    """Apply Adapter-owned quality semantics without prescribing a Gold representation."""

    async def check(
        self,
        task: TaskRecord,
        payload: BaseModel,
        context: GeneratorContext,
    ) -> QualityGateResult:
        """Return an expected rejection; raise when the candidate could not be checked."""
        ...


@runtime_checkable
class AdapterLifecycle(Protocol):
    """Optional Adapter-owned durable lifecycle behind the shared Generate workflow."""

    async def run_started(
        self,
        environment: EnvironmentRecord,
        context: GeneratorContext,
    ) -> None: ...

    async def candidate_checked(
        self,
        task: TaskRecord,
        result: QualityGateResult,
        context: GeneratorContext,
    ) -> None: ...

    async def candidate_failed(
        self,
        task: TaskRecord,
        error: BaseException,
        context: GeneratorContext,
    ) -> None: ...

    async def run_finished(
        self,
        context: GeneratorContext,
        error: BaseException | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AdapterDefinition:
    """Runtime manifest for one registered data-producing Adapter."""

    type: str
    payload_model: type[BaseModel]
    generator: AdapterGenerator
    quality_gate: TaskQualityGate
    quality_gate_concurrency: int = 1
    maximum_tasks: int | None = None
    display_name: str | None = None
    progress_phases: tuple[AdapterPhase, ...] = ()
    lifecycle: AdapterLifecycle | None = None

    def __post_init__(self) -> None:
        if not self.type or self.type != self.type.strip():
            raise ValueError("adapter type must be non-empty without surrounding whitespace")
        if self.maximum_tasks is not None and self.maximum_tasks < 1:
            raise ValueError("adapter maximum tasks must be positive")
        if self.quality_gate_concurrency < 1:
            raise ValueError("adapter quality gate concurrency must be positive")
        if self.display_name is not None and (
            not self.display_name or self.display_name != self.display_name.strip()
        ):
            raise ValueError(
                "adapter display name must be non-empty without surrounding whitespace"
            )
        phase_ids = [phase.id for phase in self.progress_phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("adapter progress phase IDs must be unique")

    def validate_task(self, task: TaskRecord) -> BaseModel:
        payload = self.payload_model.model_validate(task.adapter)
        if getattr(payload, "type", None) != self.type:
            raise ValueError(f"adapter payload type does not match definition {self.type!r}")
        return payload
