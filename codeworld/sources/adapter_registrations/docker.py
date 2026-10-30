from codeworld.sources.docker import DockerSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(DockerSourceAdapter())
