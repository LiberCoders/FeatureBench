from codeworld.sources.registry import SourceAdapterRegistry
from codeworld.sources.rustup import RustupSourceAdapter


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(RustupSourceAdapter())
