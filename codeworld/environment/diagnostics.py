"""Priority-ordered, registrable ecosystem failure diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.labels import normalize_ecosystem_label


@dataclass(frozen=True, slots=True)
class FailureDiagnosticContext:
    """Typed repository facts available when a real build or validation fails."""

    primary_language: str | None = None
    build_systems: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ()
    test_frameworks: tuple[str, ...] = ()
    runner_source: str | None = None


FailureDiagnostic = Callable[[str, FailureDiagnosticContext], AgentFeedback | None]


@dataclass(frozen=True, slots=True)
class _RegisteredDiagnostic:
    name: str
    priority: int
    diagnose: FailureDiagnostic
    languages: frozenset[str]
    build_systems: frozenset[str]
    package_managers: frozenset[str]
    test_frameworks: frozenset[str]

    def applies_to(self, context: FailureDiagnosticContext) -> bool:
        configured = (
            self.languages or self.build_systems or self.package_managers or self.test_frameworks
        )
        if not configured:
            return True
        observed: frozenset[str] = (
            frozenset({normalize_ecosystem_label(context.primary_language)})
            if context.primary_language
            else frozenset()
        )
        observed_build = frozenset(
            normalize_ecosystem_label(item) for item in context.build_systems
        )
        observed_packages = frozenset(
            normalize_ecosystem_label(item) for item in context.package_managers
        )
        observed_tests = frozenset(
            normalize_ecosystem_label(item) for item in context.test_frameworks
        )
        if not (observed or observed_build or observed_packages or observed_tests):
            # Standalone diagnostic callers may not have analysis context. Strong error
            # signatures remain useful in that case.
            return True
        return bool(
            self.languages & observed
            or self.build_systems & observed_build
            or self.package_managers & observed_packages
            or self.test_frameworks & observed_tests
        )


class FailureDiagnosticRegistry:
    """Registry whose rules own ecosystem knowledge while core owns ordering."""

    def __init__(self) -> None:
        self._rules: dict[str, _RegisteredDiagnostic] = {}

    def register(
        self,
        name: str,
        diagnose: FailureDiagnostic,
        *,
        priority: int = 0,
        languages: tuple[str, ...] = (),
        build_systems: tuple[str, ...] = (),
        package_managers: tuple[str, ...] = (),
        test_frameworks: tuple[str, ...] = (),
    ) -> None:
        if not name or name in self._rules:
            raise ValueError(f"duplicate or empty failure diagnostic: {name!r}")
        self._rules[name] = _RegisteredDiagnostic(
            name,
            priority,
            diagnose,
            frozenset(normalize_ecosystem_label(item) for item in languages),
            frozenset(normalize_ecosystem_label(item) for item in build_systems),
            frozenset(normalize_ecosystem_label(item) for item in package_managers),
            frozenset(normalize_ecosystem_label(item) for item in test_frameworks),
        )

    def diagnose(
        self,
        detail: str,
        context: FailureDiagnosticContext | None = None,
    ) -> AgentFeedback | None:
        resolved_context = context or FailureDiagnosticContext()
        ordered = sorted(
            self._rules.values(),
            key=lambda rule: (-rule.priority, rule.name),
        )
        for rule in ordered:
            if not rule.applies_to(resolved_context):
                continue
            if feedback := rule.diagnose(detail, resolved_context):
                return feedback
        return None


def _default_registry() -> FailureDiagnosticRegistry:
    from codeworld.registry_loading import register_package_adapters

    return register_package_adapters(
        "codeworld.environment.diagnostic_rules",
        FailureDiagnosticRegistry(),
    )


DEFAULT_FAILURE_DIAGNOSTICS = _default_registry()
