"""Pytest Runner selection validators."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.artifact_validation import (
    ArtifactValidationContext,
    ArtifactValidationRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_POST_COLLECTION_ITEM_FILTER = re.compile(
    r"def\s+pytest_collection_modifyitems\b[\s\S]*?items\s*\[\s*:\s*\]\s*=",
)
_PRE_COLLECTION_SELECTOR = re.compile(
    r"pytest_ignore_collect\b|(?:[rubfRUBF]*['\"]@['\"]\s*\+)|(?:f['\"]@\{)",
)
_SELECTION_VALUE = (
    r"(?:requested(?:_test)?_ids|requested|test_ids|"
    r"request\s*\[\s*['\"]test_ids['\"]\s*\]|"
    r"request\.get\(\s*['\"]test_ids['\"]\s*\))"
)
_SERIALIZED_SELECTION = rf"(?:json\s*\.\s*)?dumps\s*\(\s*{_SELECTION_VALUE}"
_ENV_SELECTION_PAYLOAD = re.compile(
    rf"(?:os\s*\.\s*environ|env(?:ironment)?)\s*\[[^\]\n]+\]\s*=\s*"
    rf"(?:{_SERIALIZED_SELECTION}|{_SELECTION_VALUE})",
    re.IGNORECASE,
)
_ARGV_SELECTION_PAYLOAD = re.compile(
    rf"\b(?:cmd|command|argv|args)\s*(?:"
    rf"\.\s*(?:extend|append)\s*\(\s*(?:{_SERIALIZED_SELECTION}|{_SELECTION_VALUE})|"
    rf"\+=\s*(?:list\s*\(\s*)?{_SELECTION_VALUE})",
    re.IGNORECASE,
)


def _pytest_selection_transport(context: ArtifactValidationContext) -> AgentFeedback | None:
    runner = context.runner
    mechanism: str | None = None
    if _ENV_SELECTION_PAYLOAD.search(runner):
        mechanism = "an environment-variable value"
    elif _ARGV_SELECTION_PAYLOAD.search(runner):
        mechanism = "the pytest command argv"
    if mechanism is None:
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "keep arbitrary requested pytest IDs in the Runner request or another "
                    "selection file; pass only bounded file paths through argv and environment "
                    "variables, then have the pytest plugin read the IDs from that file"
                ),
                actual=f"the Runner expands arbitrary requested pytest IDs into {mechanism}",
            ),
        )
    )


def _pytest_selection_contract(context: ArtifactValidationContext) -> AgentFeedback | None:
    runner = context.runner
    if not _POST_COLLECTION_ITEM_FILTER.search(runner):
        return None
    if _PRE_COLLECTION_SELECTOR.search(runner):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "scope pytest to the requested node IDs before unrelated test modules are "
                    "imported, using @ARGFILE only with pytest>=8.2 or a bounded "
                    "pytest_ignore_collect filter compatible with earlier or unknown versions"
                ),
                actual=(
                    "the Runner starts an unscoped pytest collection and filters items only in "
                    "pytest_collection_modifyitems"
                ),
            ),
        )
    )


def register(registry: ArtifactValidationRegistry) -> None:
    registry.register(
        "pytest.selection-transport",
        _pytest_selection_transport,
        ecosystem_modules=("pytest",),
    )
    registry.register(
        "pytest.selection-contract",
        _pytest_selection_contract,
        ecosystem_modules=("pytest",),
    )
