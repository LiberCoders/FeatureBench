from codeworld.sources.huggingface import HuggingFaceSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(HuggingFaceSourceAdapter())
