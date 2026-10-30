"""Go runtime artifact rules."""

from __future__ import annotations

from codeworld.environment.artifact_validation import ArtifactValidationRegistry
from codeworld.environment.artifact_validation_rules._runtime import (
    register_runtime_mutation_rule,
)


def register(registry: ArtifactValidationRegistry) -> None:
    register_runtime_mutation_rule(
        registry,
        name="go.runtime-dependency-mutation",
        ecosystem_module="go",
        pattern=r"\bgo\s+mod\s+download\b",
    )
