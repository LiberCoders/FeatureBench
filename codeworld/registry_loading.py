"""Automatic discovery for built-in registry adapters."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from typing import cast


def register_package_adapters[RegistryT](
    package_name: str,
    registry: RegistryT,
    *,
    packages_only: bool = False,
) -> RegistryT:
    """Import every public module in a package and invoke its standard register hook."""
    package = import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        raise ValueError(f"registry adapter package has no module search path: {package_name}")

    modules = sorted(iter_modules(package_paths), key=lambda item: item.name)
    for module_info in modules:
        if module_info.name.startswith("_") or (packages_only and not module_info.ispkg):
            continue
        qualified_name = f"{package_name}.{module_info.name}"
        module = import_module(qualified_name)
        hook = getattr(module, "register", None)
        if not callable(hook):
            raise ValueError(f"registry adapter {qualified_name} must define register(registry)")
        cast(Callable[[RegistryT], None], hook)(registry)
    return registry
