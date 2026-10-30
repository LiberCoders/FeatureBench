from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="hex",
            services=frozenset({"hex"}),
            arguments={"hex": ("HEX_MIRROR",)},
            hints={"hex": ("mix deps", "hex.pm")},
            transforms={},
        )
    )
