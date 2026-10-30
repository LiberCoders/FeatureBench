from codeworld.sources.playwright import PlaywrightSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(PlaywrightSourceAdapter())
