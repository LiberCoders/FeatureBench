"""Small, stable models shared by the core and every adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model for the single current schema; unknown fields are always errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunnerLimits(StrictModel):
    """Complete host-side limits for executing one external Runner artifact."""

    collect_timeout_seconds: float = Field(default=120, gt=0)
    baseline_total_timeout_seconds: float = Field(default=3600, gt=0)


class RuntimeResources(StrictModel):
    """Container resources required to reproduce the environment at runtime."""

    network: str = Field(default="none", min_length=1)
    cpus: float | None = Field(default=None, gt=0)
    memory: str | None = None
    gpus: str | None = None

    @field_validator("network")
    @classmethod
    def normalized_network(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("runtime network must not have surrounding whitespace")
        return value

    @field_validator("memory", "gpus")
    @classmethod
    def normalized_optional_resource(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("optional runtime resources must be null or non-empty")
        return normalized


class RuntimeSpec(StrictModel):
    """Self-contained host controls carried outside the clean Docker image."""

    workdir: Literal["/testbed"] = "/testbed"
    activation: str = Field(min_length=20)
    runner: str = Field(min_length=100)
    limits: RunnerLimits = RunnerLimits()
    resources: RuntimeResources = RuntimeResources()

    @field_validator("activation", "runner")
    @classmethod
    def bash_artifact(cls, value: str) -> str:
        if not value.startswith(("#!/bin/bash", "#!/usr/bin/env bash")):
            raise ValueError("runtime scripts must be Bash-compatible and start with a shebang")
        return value


class TaskRecord(StrictModel):
    """The complete universal task envelope.

    Core identity and runtime fields stay at the top level. Adapter-specific task
    inputs and reconstruction/verification data live under ``adapter``.
    """

    task_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    runtime: RuntimeSpec
    adapter: dict[str, Any]

    @field_validator("task_id", "environment_id", "image")
    @classmethod
    def no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("must not have surrounding whitespace")
        return value

    @field_validator("adapter")
    @classmethod
    def adapter_has_type(cls, value: dict[str, Any]) -> dict[str, Any]:
        adapter_type = value.get("type")
        if not isinstance(adapter_type, str) or not adapter_type.strip():
            raise ValueError("adapter.type must be a non-empty string")
        if adapter_type != adapter_type.strip():
            raise ValueError("adapter.type must not have surrounding whitespace")
        return value
