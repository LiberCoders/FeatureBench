from codeworld.sources.cargo import CargoSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(CargoSourceAdapter())
