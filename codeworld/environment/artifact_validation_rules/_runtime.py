"""Shared registration helper for ecosystem-owned runtime mutation rules."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.artifact_validation import (
    ArtifactValidationContext,
    ArtifactValidationRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue


def register_runtime_mutation_rule(
    registry: ArtifactValidationRegistry,
    *,
    name: str,
    ecosystem_module: str,
    pattern: str,
) -> None:
    compiled = re.compile(pattern, re.IGNORECASE)

    def validate(context: ArtifactValidationContext) -> AgentFeedback | None:
        for field_name in ("activation", "runner"):
            source = getattr(context, field_name).replace("\\\r\n", " ").replace("\\\n", " ")
            if match := compiled.search(source):
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

    registry.register(
        name,
        validate,
        ecosystem_modules=(ecosystem_module,),
    )
