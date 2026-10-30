from urllib.parse import urlsplit

from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def _proxy_chain(url: str) -> str:
    return f"{url},direct"


def _host(url: str) -> str:
    return urlsplit(url).netloc


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="go",
            services=frozenset({"go_proxy", "go_sumdb"}),
            arguments={"go_proxy": ("GOPROXY",), "go_sumdb": ("GOSUMDB",)},
            hints={
                "go_proxy": ("go mod", "go get", "goproxy"),
                "go_sumdb": ("gosumdb", "sum.golang"),
            },
            transforms={"go_proxy": _proxy_chain, "go_sumdb": _host},
        )
    )
