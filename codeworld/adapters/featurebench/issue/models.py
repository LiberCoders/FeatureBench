"""Typed completion and submit results for FeatureBench Issue authoring."""

from __future__ import annotations

from pydantic import Field

from codeworld.models import StrictModel


class IssueSubmitResult(StrictModel):
    """Small submit-tool result; detailed corrections remain in the workspace."""

    accepted: bool
    attempt_id: str = Field(min_length=1)
    feedback_dir: str = Field(min_length=1)


class IssueAuthorCompletion(StrictModel):
    """The Agent's final response only references its accepted workspace file."""

    path: str = Field(min_length=1)
