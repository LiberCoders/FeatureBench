"""Deterministic contracts for FeatureBench Issue Author output."""

from __future__ import annotations

import json

from codeworld.adapters.featurebench.models import IssueDraft, MaskPlan
from codeworld.agents.feedback import FeedbackIssue

_FORBIDDEN_CONSTRUCTION_PHRASES = (
    "benchmark construction",
    "featurebench",
    "gold patch",
    "hidden test",
    "mask plan",
    "object classification",
)


def issue_validation_issues(
    draft: IssueDraft,
    plan: MaskPlan,
) -> tuple[FeedbackIssue, ...]:
    """Return precise semantic violations in one submitted Issue file."""
    expected = {item.object_id for item in plan.top_objects}
    actual = [item.object_id for item in draft.interface_contracts]
    issues: list[FeedbackIssue] = []
    if len(actual) != len(set(actual)):
        issues.append(
            FeedbackIssue(
                code="featurebench.issue.duplicate_interface",
                path=("interface_contracts",),
                message="The file contains duplicate interface object IDs.",
                correction="Keep exactly one interface contract for every supplied object_id.",
            )
        )
    if set(actual) != expected:
        issues.append(
            FeedbackIssue(
                code="featurebench.issue.interface_mismatch",
                path=("interface_contracts",),
                message="The interface IDs do not match the final interface manifest.",
                correction=(
                    "Use every object_id from input/top-interfaces.jsonl exactly once and do not "
                    "invent IDs. Preserve valid contracts while adding or removing only mismatches."
                ),
                expected=json.dumps(sorted(expected)),
                actual=json.dumps(sorted(set(actual))),
            )
        )
    generated_parts = [("task_statement", draft.task_statement, False)] + [
        (f"interface_contracts[{index}].docstring", item.docstring, True)
        for index, item in enumerate(draft.interface_contracts)
    ]
    for path, content, is_docstring in generated_parts:
        if not content.strip():
            issues.append(
                FeedbackIssue(
                    code="featurebench.issue.empty_text",
                    path=tuple(path.replace("]", "").replace("[", ".").split(".")),
                    message="Required user-facing text is empty or whitespace-only.",
                    correction="Write the shortest accurate repository-native description needed here.",
                )
            )
        lowered = content.lower()
        leaked = tuple(phrase for phrase in _FORBIDDEN_CONSTRUCTION_PHRASES if phrase in lowered)
        if leaked:
            issues.append(
                FeedbackIssue(
                    code="featurebench.issue.construction_leak",
                    path=tuple(path.replace("]", "").replace("[", ".").split(".")),
                    message=f"User-facing text exposes construction terms: {leaked!r}.",
                    correction="Describe only repository-native feature and interface behavior.",
                )
            )
        if is_docstring and "```" in content:
            issues.append(
                FeedbackIssue(
                    code="featurebench.issue.invalid_doc_markup",
                    path=tuple(path.replace("]", "").replace("[", ".").split(".")),
                    message="Docstring text contains a Markdown fenced-code delimiter.",
                    correction=(
                        "Describe code inline in this docstring without triple-backtick fences; "
                        "fenced Markdown remains available in task_statement."
                    ),
                )
            )
    return tuple(issues)
