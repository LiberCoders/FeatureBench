"""Ruby and Bundler failure diagnostics."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.diagnostics import (
    FailureDiagnosticContext,
    FailureDiagnosticRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_MISSING_REQUIRED_FILE = re.compile(
    r"LoadError:[\\nr\s]*cannot load such file -- (?P<require>[^\s\\]+)",
    re.IGNORECASE,
)
_DEPLOYMENT_WITHOUT_LOCK = re.compile(
    r"deployment setting requires a lockfile|"
    r"deployment mode.*(?:Gemfile\.lock|lockfile)",
    re.IGNORECASE,
)


def _deployment_without_lock(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _DEPLOYMENT_WITHOUT_LOCK.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_RUBY_DEPLOYMENT_LOCK,
                path=("dockerfile",),
                expected=(
                    "a repository-supported Bundler install that does not require an absent "
                    "committed lockfile"
                ),
                actual="Bundler deployment mode was enabled without a committed lockfile",
                evidence=detail[-5000:],
            ),
        )
    )


def _missing_test_dependency(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    match = _MISSING_REQUIRED_FILE.search(detail)
    if match is None:
        return None
    required_file = match.group("require")
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_RUBY_TEST_DEPENDENCY,
                path=("dockerfile",),
                expected=(
                    "every Bundler group and gem required by the accepted test suite is "
                    "installed in the image"
                ),
                actual=f"Ruby could not require {required_file}",
                evidence=detail[-5000:],
            ),
        )
    )


def register(registry: FailureDiagnosticRegistry) -> None:
    registry.register(
        "ruby.deployment-without-lock",
        _deployment_without_lock,
        priority=400,
        languages=("ruby",),
        build_systems=("bundler",),
        package_managers=("bundler",),
    )
    registry.register(
        "ruby.missing-test-dependency",
        _missing_test_dependency,
        priority=390,
        languages=("ruby",),
        package_managers=("bundler", "rubygems"),
        test_frameworks=("rspec", "minitest", "cucumber"),
    )
