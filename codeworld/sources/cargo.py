"""Temporary Cargo registry configuration adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceBuildFragments,
    harness_run,
)

_COMMAND = re.compile(r"(?:^|[\s;&|/])(?:cargo|maturin|rustup)(?:\s|$)", re.IGNORECASE)

_SETUP = harness_run("""set -eu; \
    cargo_home=/root/.cargo; \
    config="$cargo_home/config.toml"; \
    legacy="$cargo_home/config"; \
    backup=/var/tmp/.featurebench-cargo-config.toml; \
    legacy_backup=/var/tmp/.featurebench-cargo-config; \
    generated=/var/tmp/.featurebench-cargo-config-generated; \
    mkdir -p "$cargo_home"; \
    if [ ! -f "$backup" ] && [ ! -f "$legacy_backup" ] && [ ! -f "$generated" ]; then \
      if [ -f "$config" ]; then cp -a "$config" "$backup"; fi; \
      if [ -f "$legacy" ]; then cp -a "$legacy" "$legacy_backup"; rm -f "$legacy"; fi; \
      if [ ! -f "$backup" ]; then : > "$generated"; fi; \
    fi; \
    printf '%s\\n' \
      '[source.crates-io]' \
      'replace-with = "featurebench-build-mirror"' \
      '[source.featurebench-build-mirror]' > "$config"; \
    printf 'registry = "%s"\\n' "$CARGO_REGISTRIES_CRATES_IO_INDEX" >> "$config"
""")

_CLEANUP = harness_run("""set -eu; \
    cargo_home=/root/.cargo; \
    config="$cargo_home/config.toml"; \
    legacy="$cargo_home/config"; \
    backup=/var/tmp/.featurebench-cargo-config.toml; \
    legacy_backup=/var/tmp/.featurebench-cargo-config; \
    generated=/var/tmp/.featurebench-cargo-config-generated; \
    if [ -f "$backup" ]; then mv "$backup" "$config"; \
    elif [ -f "$generated" ]; then rm -f "$config"; fi; \
    if [ -f "$legacy_backup" ]; then mv "$legacy_backup" "$legacy"; fi; \
    rm -f "$generated"; \
    rmdir "$cargo_home" 2>/dev/null || true
""")


def _sparse_index(url: str) -> str:
    return f"sparse+{url.rstrip('/')}/"


@dataclass(frozen=True, slots=True)
class CargoSourceAdapter(SourceAdapter):
    name: str = "cargo"
    priority: int = 200
    services: frozenset[str] = frozenset({"cargo"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        if service != "cargo":
            return {}
        return {"CARGO_REGISTRIES_CRATES_IO_INDEX": _sparse_index(url)}

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        del connect_timeout_seconds
        if service != "cargo":
            return {}
        return {
            "CARGO_HTTP_TIMEOUT": str(timeout_seconds),
            "CARGO_NET_RETRY": str(retries),
        }

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "cargo" not in context.services or _COMMAND.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return ("cargo ", "crates.io", "maturin") if service == "cargo" else ()
