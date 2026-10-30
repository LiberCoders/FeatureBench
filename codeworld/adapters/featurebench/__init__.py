"""Built-in FeatureBench Level 1 adapter definition and components."""

from __future__ import annotations

from codeworld.adapters.base import AdapterDefinition, AdapterSettings
from codeworld.adapters.catalog import AdapterCatalog, AdapterInvocation, AdapterRegistration
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.definition import featurebench_definition
from codeworld.adapters.featurebench.generator import FeatureBenchGenerator
from codeworld.adapters.featurebench.models import FeatureBenchPayload
from codeworld.adapters.featurebench.resources import FeatureBenchRunResources


def _definition_from_settings(settings: AdapterSettings) -> AdapterDefinition:
    return featurebench_definition(FeatureBenchSettings.model_validate(settings))


def _invocation_from_settings(settings: AdapterSettings) -> AdapterInvocation:
    resolved = FeatureBenchSettings.model_validate(settings)
    resources = FeatureBenchRunResources.from_settings(resolved)
    return AdapterInvocation(
        type="featurebench",
        definition_factory=lambda: featurebench_definition(
            resolved,
            resources=resources,
        ),
    )


def register(catalog: AdapterCatalog) -> None:
    """Declare FeatureBench configuration and component composition."""
    catalog.register(
        AdapterRegistration(
            type="featurebench",
            settings_model=FeatureBenchSettings,
            definition_factory=_definition_from_settings,
            invocation_factory=_invocation_from_settings,
        )
    )


__all__ = [
    "FeatureBenchGenerator",
    "FeatureBenchPayload",
    "FeatureBenchSettings",
    "featurebench_definition",
    "register",
]
