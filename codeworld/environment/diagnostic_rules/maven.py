"""Maven reactor failure diagnostics."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.diagnostics import (
    FailureDiagnosticContext,
    FailureDiagnosticRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_LOCAL_REACTOR_DEPENDENCY_FAILURE = re.compile(
    r"Could not resolve dependencies for project [^\n]+:\s*Could not find artifact "
    r"(?P<artifact>[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:"
    r"[A-Za-z0-9_.+-]*SNAPSHOT)",
    re.IGNORECASE,
)

_DYNAMIC_TEST_PROVIDER_FAILURE = re.compile(
    r"Failed to execute goal org\.apache\.maven\.plugins:"
    r"maven-(?:surefire|failsafe)-plugin:[^\s:]+:[^\s:]+.*?"
    r"Failed to collect dependencies at "
    r"(?P<provider>org\.apache\.maven\.(?:surefire|failsafe):[^:\s]+:[^\s]+)",
    re.IGNORECASE | re.DOTALL,
)


def _local_reactor(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    match = _LOCAL_REACTOR_DEPENDENCY_FAILURE.search(detail)
    if match is None:
        return None
    artifact = match.group("artifact")
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_LOCAL_REACTOR_DEPENDENCY,
                path=("dockerfile",),
                expected=(
                    "the required reactor snapshot is installed in the image's local Maven "
                    "repository before downstream dependency resolution"
                ),
                actual=f"Maven could not find local reactor artifact {artifact}",
                evidence=detail[-5000:],
            ),
        )
    )


def _dynamic_test_provider(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    match = _DYNAMIC_TEST_PROVIDER_FAILURE.search(detail)
    if match is None:
        return None
    provider = match.group("provider")
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_MAVEN_TEST_PROVIDER,
                path=("dockerfile",),
                expected=(
                    "the dynamically selected Maven test provider and its transitive dependencies "
                    "are present before network-disabled validation"
                ),
                actual=f"Maven could not resolve dynamically selected test provider {provider}",
                evidence=detail[-5000:],
            ),
        )
    )


def register(registry: FailureDiagnosticRegistry) -> None:
    registry.register(
        "java.maven-dynamic-test-provider",
        _dynamic_test_provider,
        priority=380,
        languages=("java", "kotlin", "scala"),
        build_systems=("maven", "apache maven", "maven wrapper", "mvnw"),
        package_managers=("maven",),
        test_frameworks=("junit", "junit4", "junit5", "testng"),
    )
    registry.register(
        "java.maven-local-reactor",
        _local_reactor,
        priority=370,
        languages=("java", "kotlin", "scala"),
        build_systems=("maven", "apache maven", "maven wrapper", "mvnw"),
        package_managers=("maven",),
    )
