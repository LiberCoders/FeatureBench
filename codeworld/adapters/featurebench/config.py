"""FeatureBench-owned generation settings."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from codeworld.adapters.base import AdapterSettings
from codeworld.config import (
    AgentExecutionSettings,
    AgentFactorySettings,
    AgentTrajectorySettings,
)
from codeworld.models import StrictModel


class FeatureBenchAgentRunSettings(StrictModel):
    """Model and wall-clock limit owned by one FeatureBench Agent role."""

    model: str = Field(default="openai:gpt-5.6-terra", min_length=1)
    run_timeout_seconds: float = Field(default=3600, gt=0)


class FeatureBenchAgentSettings(AgentExecutionSettings):
    """Shared execution policy plus independently selectable Adapter Agent roles."""

    top: FeatureBenchAgentRunSettings = FeatureBenchAgentRunSettings()
    repair: FeatureBenchAgentRunSettings = FeatureBenchAgentRunSettings()
    issue_author: FeatureBenchAgentRunSettings = FeatureBenchAgentRunSettings()
    trajectory: AgentTrajectorySettings = AgentTrajectorySettings(
        enabled=True,
        retention="failures",
    )

    @field_validator("trajectory", mode="before")
    @classmethod
    def default_to_failed_seed_retention(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        configured = dict(cast(dict[str, object], value))
        configured.setdefault("retention", "failures")
        return configured

    def factory_settings(self, role: FeatureBenchAgentRunSettings) -> AgentFactorySettings:
        execution = self.model_dump(exclude={"top", "repair", "issue_author"})
        return AgentFactorySettings.model_validate({**execution, "model": role.model})


class FeatureBenchRepairSettings(StrictModel):
    """Select the post-Mask recovery engine and bound deterministic search work."""

    strategy: Literal["rollback", "agent"] = "rollback"
    optimization_target: Literal["first_accepted", "interval_center"] = "first_accepted"
    maximum_validations: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=3600, gt=0)


class FeatureBenchAuditSettings(StrictModel):
    """Retention for Adapter-owned candidate evidence, independent of Agent trajectories."""

    retention: Literal["failures", "all"] = "failures"


class FeatureBenchSettings(AdapterSettings):
    environment_concurrency: int = Field(default=8, ge=1, le=32)
    gold_patch_line_minimum: int = Field(default=200, ge=1)
    gold_patch_line_preferred_minimum: int = Field(default=600, ge=1)
    gold_patch_line_maximum: int = Field(default=1000, ge=1)
    minimum_f2p_test_points: int = Field(default=1, ge=1)
    maximum_f2p_test_points: int | None = Field(default=None, ge=1)
    maximum_f2p_ratio: float = Field(default=0.7, gt=0, lt=1)
    maximum_seed_pass_rate: float = Field(default=0.9, ge=0, lt=1)
    minimum_remaining_p2p_test_points: int = Field(default=10, ge=1)
    maximum_p2p_test_points: int | None = Field(default=3000, ge=1)
    maximum_selected_p2p_test_files: int | None = Field(default=64, ge=1)
    maximum_tasks_per_environment: int | None = Field(default=None, ge=1)
    seed_concurrency: int = Field(default=16, ge=1, le=64)
    seed_ordering: Literal["easy_first", "path"] = "easy_first"

    top_max_depth: int = Field(default=5, ge=1, le=20)

    top_batch_size: int = Field(default=20, ge=1, le=50)
    top_batch_concurrency: int = Field(default=1, ge=1, le=32)
    agent_concurrency: int = Field(default=4, ge=1, le=32)
    repair: FeatureBenchRepairSettings = FeatureBenchRepairSettings()
    agent: FeatureBenchAgentSettings = FeatureBenchAgentSettings()

    test_command: tuple[str, ...] = ("pytest", "-q")
    compatibility_timeout_seconds: float = Field(default=120, gt=0)
    full_test_timeout_seconds: float = Field(default=600, gt=0)
    validation_timeout_seconds: float = Field(default=3600, gt=0)
    trace_timeout_seconds: float = Field(default=600, gt=0)
    trace_concurrency: int = Field(default=24, ge=1, le=64)
    optimizer_timeout_seconds: float = Field(default=60, gt=0)
    optimizer_workers: int = Field(default=1, ge=1, le=32)
    random_seed: int = 0

    maximum_trace_report_bytes: int = Field(default=200_000_000, ge=1_000_000)
    index_timeout_seconds: float = Field(default=300, gt=0)
    maximum_index_files: int = Field(default=10_000, ge=1)
    maximum_index_file_bytes: int = Field(default=2_000_000, ge=1)
    maximum_index_total_bytes: int = Field(default=100_000_000, ge=1)

    library_name: str | None = None
    black_links: tuple[str, ...] = ()

    audit: FeatureBenchAuditSettings = FeatureBenchAuditSettings()
    critical_magic_methods: tuple[str, ...] = (
        "__new__",
        "__init__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
    )

    @model_validator(mode="after")
    def valid_featurebench_ranges(self) -> FeatureBenchSettings:
        if (
            self.maximum_f2p_test_points is not None
            and self.minimum_f2p_test_points > self.maximum_f2p_test_points
        ):
            raise ValueError("minimum_f2p_test_points must not exceed maximum_f2p_test_points")
        if (
            self.maximum_p2p_test_points is not None
            and self.minimum_remaining_p2p_test_points > self.maximum_p2p_test_points
        ):
            raise ValueError(
                "minimum_remaining_p2p_test_points must not exceed maximum_p2p_test_points"
            )
        if self.gold_patch_line_minimum > self.gold_patch_line_preferred_minimum:
            raise ValueError(
                "gold_patch_line_minimum must not exceed gold_patch_line_preferred_minimum"
            )
        if self.gold_patch_line_preferred_minimum > self.gold_patch_line_maximum:
            raise ValueError(
                "gold_patch_line_preferred_minimum must not exceed gold_patch_line_maximum"
            )
        return self

    def gold_patch_minimum_for_tier(self, tier: str) -> int:
        if tier == "preferred":
            return self.gold_patch_line_preferred_minimum
        if tier == "fallback":
            return self.gold_patch_line_minimum
        raise ValueError(f"unknown Gold Patch tier: {tier}")

    def gold_patch_maximum_for_tier(self, tier: str) -> int:
        if tier == "preferred":
            return self.gold_patch_line_maximum
        if tier == "fallback":
            return self.gold_patch_line_preferred_minimum - 1
        raise ValueError(f"unknown Gold Patch tier: {tier}")
