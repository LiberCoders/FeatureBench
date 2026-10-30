from __future__ import annotations

import pytest

from codeworld.environment.feedback import FeedbackCode
from codeworld.environment.prompt_assets import load_prompt_asset
from codeworld.environment.runner_review import (
    RunnerIntegrityReview,
    runner_review_feedback,
)


def test_runner_review_verdict_and_reason_are_consistent() -> None:
    RunnerIntegrityReview(cheating_detected=False, reason=None)
    RunnerIntegrityReview(
        cheating_detected=True,
        reason="run returns requested IDs as passed without executing pytest",
    )
    with pytest.raises(ValueError, match="reason is required"):
        RunnerIntegrityReview(cheating_detected=True, reason=None)
    with pytest.raises(ValueError, match="reason must be null"):
        RunnerIntegrityReview(cheating_detected=False, reason="not cheating")
    with pytest.raises(ValueError, match="reason"):
        RunnerIntegrityReview.model_validate({"cheating_detected": False})


def test_runner_review_feedback_preserves_model_reason() -> None:
    review = RunnerIntegrityReview(
        cheating_detected=True,
        reason="the branch labels every collected ID as passed without a test report",
    )
    guidance = ("Use framework-native machine-readable hooks.",)
    feedback = runner_review_feedback(review, applicable_runner_guidance=guidance)
    issue = feedback.issues[0]
    assert issue.code == FeedbackCode.ARTIFACT_RUNNER_INTEGRITY
    assert issue.path == ("runner",)
    assert issue.actual == review.reason
    assert issue.evidence is None
    assert "framework-native machine-readable hooks" in issue.context["applicable_runner_guidance"]


def test_runner_review_requires_conservative_error_for_missing_real_outcomes() -> None:
    instructions = load_prompt_asset("runner-review.instructions")
    assert "exactly one result for every requested ID" in instructions
    assert "missing real outcome conservatively as" in instructions
    assert "never turn missing evidence into `passed`, `failed`, or `skipped`" in instructions
    assert "ordinary correctness\ndefect" in instructions
    assert "fabricates a non-error outcome" in instructions
    assert "applicable_runner_guidance" in instructions
    assert "synthesized `error` outcome" in instructions
