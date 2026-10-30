"""Repository records shared across data-factory stages."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field, field_validator

from codeworld.models import StrictModel


class RepositoryCandidate(StrictModel):
    discovery_run_id: str = Field(min_length=1)
    github_id: int = Field(gt=0)
    node_id: str = ""
    full_name: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    name: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    clone_url: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    api_url: str = Field(min_length=1)
    default_branch: str = Field(min_length=1)
    description: str | None = None
    stars: int = Field(ge=0)
    forks: int = Field(ge=0)
    open_issues: int = Field(ge=0)
    size_kb: int = Field(ge=0)
    language: str
    license: str | None = None
    topics: tuple[str, ...] = ()
    visibility: str = "public"
    archived: bool = False
    fork: bool = False
    private: bool = False
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("created_at", "updated_at", "pushed_at", "discovered_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
