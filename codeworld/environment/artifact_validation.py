"""Registrable ecosystem-specific validation for authored environment artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from codeworld.agents.feedback import AgentFeedback


@dataclass(frozen=True, slots=True)
class ArtifactValidationContext:
    """Artifact source available to one deterministic ecosystem validator."""

    dockerfile: str
    activation: str
    runner: str


ArtifactValidator = Callable[[ArtifactValidationContext], AgentFeedback | None]


@dataclass(frozen=True, slots=True)
class ArtifactRepairValidationContext:
    """One proposed repair plus the constraints it must preserve."""

    current: ArtifactValidationContext
    candidate: ArtifactValidationContext
    constraints: AgentFeedback


ArtifactRepairValidator = Callable[[ArtifactRepairValidationContext], AgentFeedback | None]


@dataclass(frozen=True, slots=True)
class _RegisteredArtifactValidator:
    name: str
    ecosystem_modules: frozenset[str]
    validate: ArtifactValidator


@dataclass(frozen=True, slots=True)
class _RegisteredArtifactRepairValidator:
    name: str
    ecosystem_modules: frozenset[str]
    validate: ArtifactRepairValidator


class ArtifactValidationRegistry:
    """Select deterministic artifact rules without ecosystem branches in orchestration."""

    def __init__(self) -> None:
        self._validators: dict[str, _RegisteredArtifactValidator] = {}
        self._repair_validators: dict[str, _RegisteredArtifactRepairValidator] = {}

    def register(
        self,
        name: str,
        validate: ArtifactValidator,
        *,
        ecosystem_modules: tuple[str, ...],
    ) -> None:
        normalized_modules = frozenset(item.casefold() for item in ecosystem_modules)
        if not name or name in self._validators:
            raise ValueError(f"duplicate or empty artifact validator: {name!r}")
        if not normalized_modules:
            raise ValueError(f"artifact validator {name!r} has no ecosystem modules")
        self._validators[name] = _RegisteredArtifactValidator(
            name=name,
            ecosystem_modules=normalized_modules,
            validate=validate,
        )

    def register_repair(
        self,
        name: str,
        validate: ArtifactRepairValidator,
        *,
        ecosystem_modules: tuple[str, ...],
    ) -> None:
        normalized_modules = frozenset(item.casefold() for item in ecosystem_modules)
        if not name or name in self._repair_validators:
            raise ValueError(f"duplicate or empty artifact repair validator: {name!r}")
        if not normalized_modules:
            raise ValueError(f"artifact repair validator {name!r} has no ecosystem modules")
        self._repair_validators[name] = _RegisteredArtifactRepairValidator(
            name=name,
            ecosystem_modules=normalized_modules,
            validate=validate,
        )

    @staticmethod
    def _applies(registered: frozenset[str], selected: frozenset[str]) -> bool:
        return "*" in registered or bool(registered & selected)

    def validate(
        self,
        context: ArtifactValidationContext,
        *,
        ecosystem_modules: tuple[str, ...],
    ) -> AgentFeedback | None:
        selected = frozenset(item.casefold() for item in ecosystem_modules)
        for validator in self._validators.values():
            if self._applies(validator.ecosystem_modules, selected) and (
                feedback := validator.validate(context)
            ):
                return feedback
        return None

    def validate_repair(
        self,
        context: ArtifactRepairValidationContext,
        *,
        ecosystem_modules: tuple[str, ...],
    ) -> AgentFeedback | None:
        selected = frozenset(item.casefold() for item in ecosystem_modules)
        for validator in self._repair_validators.values():
            if self._applies(validator.ecosystem_modules, selected) and (
                feedback := validator.validate(context)
            ):
                return feedback
        return None


def _default_registry() -> ArtifactValidationRegistry:
    from codeworld.registry_loading import register_package_adapters

    return register_package_adapters(
        "codeworld.environment.artifact_validation_rules",
        ArtifactValidationRegistry(),
    )


DEFAULT_ARTIFACT_VALIDATORS = _default_registry()
