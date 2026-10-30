from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="cran",
            services=frozenset({"cran"}),
            arguments={"cran": ("CRAN_MIRROR",)},
            hints={"cran": ("install.packages", "cran")},
            transforms={},
        )
    )
