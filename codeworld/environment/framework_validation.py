"""Registrable validation for framework-specific runtime evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from codeworld.agents.feedback import AgentFeedback, AgentFeedbackError
from codeworld.environment.labels import normalize_ecosystem_label
from codeworld.environment.runner_contract import CollectedTestPoint


@dataclass(frozen=True, slots=True)
class FrameworkCollectionValidationContext:
    """The canonical collection returned by one environment Runner."""

    collected: tuple[CollectedTestPoint, ...]


FrameworkCollectionValidator = Callable[
    [FrameworkCollectionValidationContext], AgentFeedback | None
]


@dataclass(frozen=True, slots=True)
class _RegisteredCollectionValidator:
    name: str
    frameworks: frozenset[str]
    validate: FrameworkCollectionValidator


class FrameworkValidationRegistry:
    """Select runtime validators by analyzed test framework without core branches."""

    def __init__(self) -> None:
        self._collection_validators: dict[str, _RegisteredCollectionValidator] = {}

    def register_collection(
        self,
        name: str,
        validate: FrameworkCollectionValidator,
        *,
        frameworks: tuple[str, ...],
    ) -> None:
        normalized = frozenset(normalize_ecosystem_label(item) for item in frameworks)
        if not name or name in self._collection_validators:
            raise ValueError(f"duplicate or empty framework collection validator: {name!r}")
        if not normalized:
            raise ValueError(f"framework collection validator {name!r} has no frameworks")
        self._collection_validators[name] = _RegisteredCollectionValidator(
            name=name,
            frameworks=normalized,
            validate=validate,
        )

    def validate_collection(
        self,
        context: FrameworkCollectionValidationContext,
        *,
        frameworks: tuple[str, ...],
    ) -> AgentFeedback | None:
        selected = frozenset(normalize_ecosystem_label(item) for item in frameworks)
        for validator in self._collection_validators.values():
            if validator.frameworks & selected and (feedback := validator.validate(context)):
                return feedback
        return None


def _default_registry() -> FrameworkValidationRegistry:
    from codeworld.registry_loading import register_package_adapters

    return register_package_adapters(
        "codeworld.environment.framework_validation_rules",
        FrameworkValidationRegistry(),
    )


DEFAULT_FRAMEWORK_VALIDATORS = _default_registry()


def validate_framework_collection(
    collected: tuple[CollectedTestPoint, ...],
    *,
    frameworks: tuple[str, ...],
) -> None:
    """Raise structured Agent feedback when framework evidence violates its contract."""
    feedback = DEFAULT_FRAMEWORK_VALIDATORS.validate_collection(
        FrameworkCollectionValidationContext(collected=collected),
        frameworks=frameworks,
    )
    if feedback is not None:
        raise AgentFeedbackError(feedback)
