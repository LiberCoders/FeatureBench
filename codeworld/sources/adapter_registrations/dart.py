from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="dart",
            services=frozenset({"pub", "flutter"}),
            arguments={
                "pub": ("PUB_HOSTED_URL",),
                "flutter": ("FLUTTER_STORAGE_BASE_URL",),
            },
            hints={
                "pub": ("dart pub", "flutter pub", "pub.dev"),
                "flutter": ("flutter",),
            },
            transforms={},
        )
    )
