from codeworld.sources.arguments import ArgumentSourceAdapter
from codeworld.sources.registry import SourceAdapterRegistry


def register(registry: SourceAdapterRegistry) -> None:
    registry.register(
        ArgumentSourceAdapter(
            name="node",
            services=frozenset({"npm", "node_dist"}),
            arguments={
                "npm": ("NPM_CONFIG_REGISTRY", "COREPACK_NPM_REGISTRY"),
                "node_dist": ("NODEJS_ORG_MIRROR",),
            },
            hints={
                "npm": ("npm ", "pnpm ", "yarn ", "corepack"),
                "node_dist": ("nodejs.org", "node distribution"),
            },
            transforms={},
            download_timeout_arguments={"npm": ("NPM_CONFIG_FETCH_TIMEOUT",)},
            download_retry_arguments={"npm": ("NPM_CONFIG_FETCH_RETRIES",)},
        )
    )
