from codeworld.sources.registry import SourceAdapterRegistry
from codeworld.sources.rubygems import RubyGemsSourceAdapter


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(RubyGemsSourceAdapter())
