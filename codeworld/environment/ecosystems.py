"""Validated ecosystem prompt assets selected through a small runtime registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from codeworld.environment.labels import normalize_ecosystem_label
from codeworld.models import StrictModel

_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "prompts" / "ecosystems"


def _labels(values: tuple[str, ...]) -> frozenset[str]:
    return frozenset(normalize_ecosystem_label(value) for value in values if value.strip())


class EcosystemMatch(StrictModel):
    """Analysis labels that activate one prompt asset."""

    languages: tuple[str, ...] = ()
    build_systems: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    test_frameworks: tuple[str, ...] = ()

    @field_validator("languages", "build_systems", "package_managers", "test_frameworks")
    @classmethod
    def normalized_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(normalize_ecosystem_label(value) for value in values))
        if any(not value for value in normalized):
            raise ValueError("ecosystem match labels may not be empty")
        return normalized

    @model_validator(mode="after")
    def has_selector(self) -> EcosystemMatch:
        if not any(
            (
                self.languages,
                self.build_systems,
                self.package_managers,
                self.test_frameworks,
            )
        ):
            raise ValueError("an ecosystem prompt asset must define at least one match label")
        return self


class EcosystemPromptModule(StrictModel):
    """One YAML-backed language, build-tool, or test-framework guide."""

    module: str = Field(min_length=1)
    match: EcosystemMatch
    dockerfile_guidance: tuple[str, ...] = ()
    activation_guidance: tuple[str, ...] = ()
    runner_guidance: tuple[str, ...] = ()

    @field_validator("module")
    @classmethod
    def normalized_module(cls, value: str) -> str:
        if value != value.strip().casefold():
            raise ValueError("ecosystem module must be lowercase without surrounding whitespace")
        return value

    @field_validator("dockerfile_guidance", "activation_guidance", "runner_guidance")
    @classmethod
    def nonempty_guidance(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("ecosystem guidance entries may not be empty")
        return normalized

    @model_validator(mode="after")
    def has_guidance(self) -> EcosystemPromptModule:
        if not (self.dockerfile_guidance or self.activation_guidance or self.runner_guidance):
            raise ValueError("an ecosystem prompt asset must contain guidance")
        return self

    def matches(
        self,
        *,
        languages: frozenset[str],
        build_systems: frozenset[str],
        package_managers: frozenset[str],
        test_frameworks: frozenset[str],
    ) -> bool:
        return bool(
            frozenset(self.match.languages) & languages
            or frozenset(self.match.build_systems) & build_systems
            or frozenset(self.match.package_managers) & package_managers
            or frozenset(self.match.test_frameworks) & test_frameworks
        )

    def as_prompt_value(self) -> EcosystemPromptGuidance:
        return EcosystemPromptGuidance(
            module=self.module,
            dockerfile_guidance=self.dockerfile_guidance,
            activation_guidance=self.activation_guidance,
            runner_guidance=self.runner_guidance,
        )


class EcosystemPromptGuidance(StrictModel):
    """Selected authoring knowledge exposed to the Artifact Agent."""

    module: str = Field(min_length=1)
    dockerfile_guidance: tuple[str, ...] = ()
    activation_guidance: tuple[str, ...] = ()
    runner_guidance: tuple[str, ...] = ()


class EcosystemPromptRegistry:
    """Ordered registry loaded from assets; it contains no ecosystem prose."""

    def __init__(self) -> None:
        self._modules: dict[str, EcosystemPromptModule] = {}

    def register(self, module: EcosystemPromptModule) -> None:
        if module.module in self._modules:
            raise ValueError(f"duplicate ecosystem prompt module: {module.module!r}")
        self._modules[module.module] = module

    def select(
        self,
        *,
        languages: tuple[str, ...] = (),
        build_systems: tuple[str, ...] = (),
        package_managers: tuple[str, ...] = (),
        test_frameworks: tuple[str, ...] = (),
    ) -> tuple[EcosystemPromptModule, ...]:
        normalized = {
            "languages": _labels(languages),
            "build_systems": _labels(build_systems),
            "package_managers": _labels(package_managers),
            "test_frameworks": _labels(test_frameworks),
        }
        return tuple(module for module in self._modules.values() if module.matches(**normalized))

    def resolve(self, module_ids: tuple[str, ...]) -> tuple[EcosystemPromptModule, ...]:
        requested = set(module_ids)
        unknown = requested - self._modules.keys()
        if unknown:
            raise ValueError("unknown ecosystem prompt modules: " + ", ".join(sorted(unknown)))
        return tuple(module for name, module in self._modules.items() if name in requested)

    @classmethod
    def from_directory(cls, directory: Path) -> EcosystemPromptRegistry:
        registry = cls()
        paths = tuple(sorted(directory.glob("*.yaml")))
        if not paths:
            raise ValueError(f"no ecosystem prompt YAML assets found in {directory}")
        for path in paths:
            try:
                document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
                module = EcosystemPromptModule.model_validate(document)
            except (OSError, yaml.YAMLError, ValueError) as exc:
                raise ValueError(f"invalid ecosystem prompt asset {path}: {exc}") from exc
            registry.register(module)
        return registry


DEFAULT_ECOSYSTEM_PROMPTS = EcosystemPromptRegistry.from_directory(_ASSET_DIRECTORY)


def ecosystem_prompt_guidance(
    *,
    module_ids: tuple[str, ...] = (),
    languages: tuple[str, ...] = (),
    build_systems: tuple[str, ...] = (),
    package_managers: tuple[str, ...] = (),
    test_frameworks: tuple[str, ...] = (),
) -> tuple[EcosystemPromptGuidance, ...]:
    """Return only prompt assets relevant to the typed repository analysis."""
    modules = (
        DEFAULT_ECOSYSTEM_PROMPTS.resolve(module_ids)
        if module_ids
        else DEFAULT_ECOSYSTEM_PROMPTS.select(
            languages=languages,
            build_systems=build_systems,
            package_managers=package_managers,
            test_frameworks=test_frameworks,
        )
    )
    return tuple(module.as_prompt_value() for module in modules)
