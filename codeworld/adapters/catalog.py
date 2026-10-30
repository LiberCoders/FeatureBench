"""Automatic discovery and configuration of task adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from codeworld.adapters.base import AdapterDefinition, AdapterSettings
from codeworld.errors import ContractError
from codeworld.registry_loading import register_package_adapters

AdapterDefinitionFactory = Callable[[AdapterSettings], AdapterDefinition]
AdapterInvocationFactory = Callable[[AdapterSettings], "AdapterInvocation"]


@dataclass(frozen=True, slots=True)
class AdapterInvocation:
    """Build isolated Environment adapters that may share invocation-owned services."""

    type: str
    definition_factory: Callable[[], AdapterDefinition]

    def build_definition(self) -> AdapterDefinition:
        definition = self.definition_factory()
        if definition.type != self.type:
            raise ContractError(
                f"adapter definition type {definition.type!r} does not match "
                f"invocation {self.type!r}"
            )
        return definition


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    """One Adapter package's configuration and composition contract."""

    type: str
    settings_model: type[AdapterSettings]
    definition_factory: AdapterDefinitionFactory
    invocation_factory: AdapterInvocationFactory | None = None

    def __post_init__(self) -> None:
        if not self.type or self.type != self.type.strip():
            raise ValueError("adapter type must be non-empty without surrounding whitespace")
        # Keep the plugin boundary defensive even when a dynamically loaded package
        # does not run a static type checker.
        if AdapterSettings not in self.settings_model.__mro__:
            raise TypeError(
                f"adapter settings model for {self.type!r} must inherit AdapterSettings"
            )

    def validate_settings(self, value: object | None = None) -> AdapterSettings:
        validated = self.settings_model.model_validate({} if value is None else value)
        if value is not None and (
            not isinstance(value, Mapping) or "environment_concurrency" not in value
        ):
            raise ValueError(
                f"adapter {self.type!r} configuration must explicitly declare "
                "environment_concurrency"
            )
        return validated

    def build(self, value: object | None = None) -> AdapterDefinition:
        return self.build_validated(self.validate_settings(value))

    def build_validated(self, settings: AdapterSettings) -> AdapterDefinition:
        """Build a fresh Adapter instance from settings validated once by the caller."""
        return self.begin_validated(settings).build_definition()

    def begin_validated(self, settings: AdapterSettings) -> AdapterInvocation:
        """Create one run scope whose definitions may share Adapter-owned services."""
        if not isinstance(settings, self.settings_model):
            raise TypeError(
                f"Adapter {self.type!r} requires {self.settings_model.__name__} settings"
            )
        invocation = (
            self.invocation_factory(settings)
            if self.invocation_factory is not None
            else AdapterInvocation(
                type=self.type,
                definition_factory=lambda: self.definition_factory(settings),
            )
        )
        if invocation.type != self.type:
            raise ContractError(
                f"adapter invocation type {invocation.type!r} does not match "
                f"registration {self.type!r}"
            )
        return invocation


class AdapterCatalog:
    """Discovered Adapter registrations, independent of run configuration."""

    def __init__(self) -> None:
        self._registrations: dict[str, AdapterRegistration] = {}

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def register(self, registration: AdapterRegistration) -> None:
        if registration.type in self._registrations:
            raise ContractError(f"adapter already registered: {registration.type}")
        self._registrations[registration.type] = registration

    def get(self, adapter_type: str) -> AdapterRegistration:
        try:
            return self._registrations[adapter_type]
        except KeyError as exc:
            raise ContractError(f"adapter is not registered: {adapter_type}") from exc

    def validate_settings(
        self, configurations: Mapping[str, object]
    ) -> dict[str, dict[str, object]]:
        """Validate configured Adapter sections without materializing absent defaults."""
        unknown = sorted(set(configurations) - set(self._registrations))
        if unknown:
            raise ContractError(f"adapter configuration is not registered: {', '.join(unknown)}")
        return {
            adapter_type: self.get(adapter_type).validate_settings(value).model_dump(mode="python")
            for adapter_type, value in configurations.items()
        }


def default_adapter_catalog() -> AdapterCatalog:
    """Discover every public Adapter package through its local ``register`` hook."""
    return register_package_adapters(
        "codeworld.adapters",
        AdapterCatalog(),
        packages_only=True,
    )
