"""Strict, appendable GitHub discovery records and lossless search partitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import Field, field_validator, model_validator

from codeworld.config import DiscoverySettings
from codeworld.models import StrictModel


def _github_time(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _keyword(value: str) -> str:
    escaped = value.strip().replace('"', '\\"')
    return f'"{escaped}"' if any(char.isspace() for char in escaped) else escaped


class SearchPartition(StrictModel):
    language: str | None = None
    license: str | None = None
    created_start: datetime
    created_end: datetime
    depth: int = Field(default=0, ge=0)
    expected_count: int = Field(default=0, ge=0)

    @field_validator("created_start", "created_end")
    @classmethod
    def normalized_bound(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).replace(microsecond=0)

    @model_validator(mode="after")
    def ordered_bounds(self) -> SearchPartition:
        if self.created_start > self.created_end:
            raise ValueError("partition start must not be later than its end")
        return self

    @property
    def span_seconds(self) -> int:
        return int((self.created_end - self.created_start).total_seconds())

    def bisect(self) -> tuple[SearchPartition, SearchPartition]:
        if self.span_seconds <= 0:
            raise ValueError("a one-second GitHub partition cannot be bisected further")
        midpoint = self.created_start + (self.created_end - self.created_start) / 2
        midpoint = midpoint.replace(microsecond=0)
        return (
            self.model_copy(
                update={"created_end": midpoint, "depth": self.depth + 1, "expected_count": 0}
            ),
            self.model_copy(
                update={
                    "created_start": midpoint + timedelta(seconds=1),
                    "depth": self.depth + 1,
                    "expected_count": 0,
                }
            ),
        )

    def query(self, settings: DiscoverySettings) -> str:
        terms: list[str] = []
        if settings.public_only:
            terms.append("is:public")
        if settings.exclude_forks:
            terms.append("fork:false")
        if settings.exclude_archived:
            terms.append("archived:false")
        terms.append(
            f"created:{_github_time(self.created_start)}..{_github_time(self.created_end)}"
        )
        if self.language:
            terms.append(f"language:{self.language}")
        if self.license:
            terms.append(f"license:{self.license}")
        if settings.pushed_after and settings.pushed_before:
            terms.append(
                f"pushed:{_github_time(settings.pushed_after)}..{_github_time(settings.pushed_before)}"
            )
        elif settings.pushed_after:
            terms.append(f"pushed:>={_github_time(settings.pushed_after)}")
        elif settings.pushed_before:
            terms.append(f"pushed:<={_github_time(settings.pushed_before)}")
        if settings.maximum_stars is not None:
            terms.append(f"stars:{settings.minimum_stars}..{settings.maximum_stars}")
        else:
            terms.append(f"stars:>={settings.minimum_stars}")
        terms.extend(_keyword(keyword) for keyword in settings.keywords)
        return " ".join(terms)
