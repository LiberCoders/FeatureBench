"""Go test failure diagnostics."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.diagnostics import (
    FailureDiagnosticContext,
    FailureDiagnosticRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_COLLECT_EXECUTED_GO_TESTS = re.compile(
    r"runner collect failed:[\s\S]*(?:--- pass:|--- fail:)", re.IGNORECASE
)


def _collect_executed_tests(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _COLLECT_EXECUTED_GO_TESTS.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected="collect uses `go test -list` or equivalent listing without executing bodies",
                actual="the collect action executed Go tests and inherited their aggregate failure",
                evidence=detail[-5000:],
            ),
        )
    )


def register(registry: FailureDiagnosticRegistry) -> None:
    registry.register(
        "go.collect-executed-tests",
        _collect_executed_tests,
        priority=430,
        languages=("go",),
        test_frameworks=("go test", "gotest"),
    )
