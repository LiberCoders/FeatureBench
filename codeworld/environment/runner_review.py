"""Independent typed LLM review for generated Runner integrity."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import Field, model_validator

from codeworld.agents.factory import AgentFactory
from codeworld.agents.feedback import (
    AgentFeedback,
    FeedbackIssue,
)
from codeworld.config import AgentSettings
from codeworld.environment.feedback import FEEDBACK_TEMPLATES, FeedbackCode
from codeworld.environment.prompt_assets import load_prompt_asset
from codeworld.models import StrictModel
from codeworld.runtime.container import ContainerSession


class RunnerIntegrityReview(StrictModel):
    cheating_detected: bool
    reason: str | None

    @model_validator(mode="after")
    def verdict_matches_reason(self) -> RunnerIntegrityReview:
        if self.cheating_detected and (self.reason is None or not self.reason.strip()):
            raise ValueError("reason is required when cheating_detected is true")
        if not self.cheating_detected and self.reason is not None:
            raise ValueError("reason must be null when cheating_detected is false")
        return self


class RunnerReviewContext(StrictModel):
    test_frameworks: tuple[str, ...]
    runner_contract: dict[str, Any]
    applicable_runner_guidance: tuple[str, ...]
    runner_source: str = Field(min_length=100)


async def review_runner_integrity(
    *,
    agents: AgentFactory[AgentSettings],
    session: ContainerSession,
    context: RunnerReviewContext,
    progress: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
    metadata: dict[str, str] | None = None,
) -> RunnerIntegrityReview:
    """Ask a tool-free reviewer for a source-grounded integrity verdict."""

    prompt = (
        load_prompt_asset("runner-review.user")
        + "\n\n"
        + json.dumps(
            context.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )
    return await agents.run_structured(
        session=session,
        prompt=prompt,
        instructions=load_prompt_asset("runner-review.instructions"),
        output_type=RunnerIntegrityReview,
        container_tools=False,
        progress=progress,
        run_timeout_seconds=agents.settings.runner_review_run_timeout_seconds,
        agent_name="environment_runner_review",
        metadata=metadata,
    )


def runner_review_feedback(
    review: RunnerIntegrityReview,
    *,
    applicable_runner_guidance: tuple[str, ...] = (),
) -> AgentFeedback:
    """Return the review reason with the selected ecosystem repair contract."""
    if not review.cheating_detected or review.reason is None:
        raise ValueError("an accepted Runner review has no feedback")
    message, correction = FEEDBACK_TEMPLATES.resolve(FeedbackCode.ARTIFACT_RUNNER_INTEGRITY)
    return AgentFeedback(
        issues=(
            FeedbackIssue(
                code=str(FeedbackCode.ARTIFACT_RUNNER_INTEGRITY),
                path=("runner",),
                message=message,
                correction=correction,
                expected="Runner does not fake, bypass, or misrepresent test execution",
                actual=review.reason,
                context={
                    "applicable_runner_guidance": json.dumps(
                        applicable_runner_guidance,
                        ensure_ascii=False,
                    )
                },
            ),
        )
    )
