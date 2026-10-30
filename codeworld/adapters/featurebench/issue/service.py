"""Host-owned validation for Agent-authored FeatureBench Issue files."""

from __future__ import annotations

from pydantic import ValidationError

from codeworld.adapters.featurebench.issue.models import IssueSubmitResult
from codeworld.adapters.featurebench.issue.validation import issue_validation_issues
from codeworld.adapters.featurebench.issue.workspace import IssueAuthorWorkspace
from codeworld.adapters.featurebench.models import IssueDraft, MaskPlan
from codeworld.agents.feedback import FeedbackIssue


def _schema_issues(error: ValidationError) -> tuple[FeedbackIssue, ...]:
    issues: list[FeedbackIssue] = []
    for entry in error.errors(include_url=False, include_input=False):
        raw_context = entry.get("ctx") or {}
        issues.append(
            FeedbackIssue(
                code=f"schema.{entry['type']}",
                path=tuple(entry.get("loc", ())),
                message=str(entry["msg"]),
                correction=(
                    "Correct this field in the submitted Issue file, preserve all valid content, "
                    "and submit the same path again."
                ),
                context={str(key): str(value) for key, value in raw_context.items()},
            )
        )
    return tuple(issues)


class IssueAuthorService:
    """Parse, validate, and accept exactly one Agent-authored Issue file."""

    def __init__(self, *, workspace: IssueAuthorWorkspace, plan: MaskPlan) -> None:
        self.workspace = workspace
        self.plan = plan
        self.attempt_index = 0
        self.accepted_issue: IssueDraft | None = None
        self.accepted_path: str | None = None
        self.accepted_attempt_id: str | None = None

    async def submit_file(self, submitted_path: str) -> IssueSubmitResult:
        """Validate one path without echoing its potentially large contents into context."""
        if self.accepted_issue is not None:
            return IssueSubmitResult(
                accepted=True,
                attempt_id=self.accepted_attempt_id or "attempt-0000",
                feedback_dir=self.workspace.feedback_path(
                    self.accepted_attempt_id or "attempt-0000"
                ),
            )
        attempt_id = f"attempt-{self.attempt_index:04d}"
        self.attempt_index += 1
        issue: IssueDraft | None = None
        issues: tuple[FeedbackIssue, ...]
        try:
            path = self.workspace.outbox_host_path(submitted_path)
            if not path.is_file():
                raise ValueError(f"submitted Issue file does not exist: {submitted_path}")
            if path.stat().st_size > 10_000_000:
                raise ValueError("submitted Issue file exceeds 10000000 bytes")
            issue = IssueDraft.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            issues = _schema_issues(exc)
        except (OSError, UnicodeError, ValueError) as exc:
            issues = (
                FeedbackIssue(
                    code="featurebench.issue.file_invalid",
                    message=str(exc) or type(exc).__name__,
                    correction=(
                        "Write one UTF-8 JSON file under outbox/ following protocol/README.md, "
                        "then submit its relative path."
                    ),
                ),
            )
        else:
            issues = issue_validation_issues(issue, self.plan)
        accepted = issue is not None and not issues
        feedback_dir = self.workspace.write_feedback(
            attempt_id=attempt_id,
            accepted=accepted,
            issues=issues,
        )
        if accepted:
            self.accepted_issue = issue
            self.accepted_path = submitted_path
            self.accepted_attempt_id = attempt_id
        return IssueSubmitResult(
            accepted=accepted,
            attempt_id=attempt_id,
            feedback_dir=feedback_dir,
        )
