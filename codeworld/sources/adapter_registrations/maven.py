from codeworld.sources.gradle import GradleSourceAdapter
from codeworld.sources.maven import MavenSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(MavenSourceAdapter())
    registry.register(GradleSourceAdapter())
