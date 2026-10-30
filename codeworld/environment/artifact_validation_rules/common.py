"""Language-neutral artifact validation rules."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.artifact_validation import (
    ArtifactValidationContext,
    ArtifactValidationRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_SYSTEM_PACKAGE_MUTATION = re.compile(
    r"\b(?:apt-get\s+(?:update|install)|apk\s+add|dnf\s+install)\b",
    re.IGNORECASE,
)


def _runtime_system_package_mutation(
    context: ArtifactValidationContext,
) -> AgentFeedback | None:
    for field_name in ("activation", "runner"):
        source = getattr(context, field_name).replace("\\\r\n", " ").replace("\\\n", " ")
        if match := _SYSTEM_PACKAGE_MUTATION.search(source):
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNTIME_MUTATION,
                        path=(field_name,),
                        actual=match.group(0),
                    ),
                )
            )
    return None


def register(registry: ArtifactValidationRegistry) -> None:
    registry.register(
        "common.runtime-system-package-mutation",
        _runtime_system_package_mutation,
        ecosystem_modules=("*",),
    )
