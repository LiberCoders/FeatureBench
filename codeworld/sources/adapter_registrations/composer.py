from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="composer",
            services=frozenset({"composer", "composer_binary"}),
            arguments={
                "composer": ("COMPOSER_REPO_PACKAGIST",),
                "composer_binary": ("COMPOSER_BINARY_URL",),
            },
            hints={
                "composer": ("composer ", "packagist"),
                "composer_binary": ("composer.phar", "getcomposer.org"),
            },
            transforms={},
        )
    )
