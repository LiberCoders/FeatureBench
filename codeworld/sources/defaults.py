"""Automatic construction of the built-in source adapter registry."""

from codeworld.registry_loading import register_package_adapters
from codeworld.sources.registry import SourceAdapterRegistry


def default_source_adapters() -> SourceAdapterRegistry:
    return register_package_adapters(
        "codeworld.sources.adapter_registrations",
        SourceAdapterRegistry(),
    )
