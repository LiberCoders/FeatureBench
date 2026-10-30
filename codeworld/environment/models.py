"""Environment Build records; runtime controls remain outside the image."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator

from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.models import RuntimeSpec, StrictModel


class EnvironmentRecord(StrictModel):
    environment_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    language: str = Field(min_length=1)
    gpu_requirement: Literal["none", "optional", "required"]
    image: str = Field(min_length=1)
    dockerfile: str = Field(min_length=40)
    runtime: RuntimeSpec
    verified_test_points: tuple[CollectedTestPoint, ...] = Field(min_length=1)

    @field_validator("verified_test_points")
    @classmethod
    def canonical_verified_tests(
        cls, values: tuple[CollectedTestPoint, ...]
    ) -> tuple[CollectedTestPoint, ...]:
        identifiers = [item.id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verified test-point IDs must be unique")
        canonical = tuple(sorted(values, key=lambda item: (item.id, item.path)))
        if values != canonical:
            raise ValueError("verified test points must be sorted by ID and path")
        return values


class EnvironmentBuildFailure(StrictModel):
    run_id: str | None = None
    task_id: str | None = None
    github_id: int | None = None
    repository: str = Field(min_length=1)
    ref: str | None = None
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    disposition: Literal["failed", "rejected", "interrupted"] = "failed"
    category: str | None = None
    environment_id: str | None = None
    failed_phase: str | None = None
    attempts: int = Field(default=0, ge=0)
    diagnostics_path: str | None = None
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
