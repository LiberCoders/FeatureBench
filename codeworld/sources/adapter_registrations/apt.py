from codeworld.sources.apt import AptSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(AptSourceAdapter())
