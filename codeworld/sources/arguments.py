"""Small adapters for ecosystems that consume source URLs through build arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from codeworld.sources.base import SourceAdapter

SourceValueTransform = Callable[[str], str]


def _empty_argument_mapping() -> Mapping[str, tuple[str, ...]]:
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class ArgumentSourceAdapter(SourceAdapter):
    """A declarative adapter for tools with a native build-time environment variable."""

    name: str
    services: frozenset[str]
    arguments: Mapping[str, tuple[str, ...]]
    hints: Mapping[str, tuple[str, ...]]
    transforms: Mapping[str, SourceValueTransform]
    download_timeout_arguments: Mapping[str, tuple[str, ...]] = field(
        default_factory=_empty_argument_mapping
    )
    download_retry_arguments: Mapping[str, tuple[str, ...]] = field(
        default_factory=_empty_argument_mapping
    )
    priority: int = 150

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        variables = self.arguments.get(service, ())
        transform = self.transforms.get(service)
        value = transform(url) if transform is not None else url
        return dict.fromkeys(variables, value)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return self.hints.get(service, ())

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        del connect_timeout_seconds
        values = {
            name: str(timeout_seconds * 1000)
            for name in self.download_timeout_arguments.get(service, ())
        }
        values.update(
            {name: str(retries) for name in self.download_retry_arguments.get(service, ())}
        )
        return values
