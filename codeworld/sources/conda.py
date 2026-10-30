"""Temporary conda-forge channel configuration and download controls."""

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

_COMMAND = re.compile(
    r"(?:^|[\s;&|/])(?:conda|mamba|micromamba)(?:\s|$)",
    re.IGNORECASE,
)
_CONFIG = "/var/tmp/featurebench-condarc.yaml"
_BACKUP = "/var/tmp/.featurebench-condarc-backup"
_GENERATED = "/var/tmp/.featurebench-condarc-generated"

_SETUP = harness_run(f"""set -eu; \
    config="$CONDARC"; \
    backup={_BACKUP}; \
    generated={_GENERATED}; \
    if [ ! -f "$backup" ] && [ ! -f "$generated" ]; then \
      if [ -f "$config" ]; then cp -a "$config" "$backup"; else : > "$generated"; fi; \
    fi; \
    printf '%s\\n' \
      'custom_channels:' \
      "  conda-forge: $FEATUREBENCH_CONDA_CHANNEL_ROOT" > "$config"
""")

_CLEANUP = harness_run(f"""set -eu; \
    config="$CONDARC"; \
    backup={_BACKUP}; \
    generated={_GENERATED}; \
    if [ -f "$backup" ]; then mv "$backup" "$config"; \
    elif [ -f "$generated" ]; then rm -f "$config"; fi; \
    rm -f "$generated"
""")


def _channel_root(url: str) -> str:
    root, separator, channel = url.rstrip("/").rpartition("/")
    if not separator or channel.casefold() != "conda-forge":
        raise ValueError("conda source endpoint must end with /conda-forge")
    return root


@dataclass(frozen=True, slots=True)
class CondaSourceAdapter(SourceAdapter):
    """Route named conda-forge channels without persisting build-only config."""

    name: str = "conda"
    priority: int = 150
    services: frozenset[str] = frozenset({"conda"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        if service != "conda":
            return {}
        return {
            "CONDA_CHANNEL_URL": url,
            "FEATUREBENCH_CONDA_CHANNEL_ROOT": _channel_root(url),
            "CONDARC": _CONFIG,
        }

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        if service != "conda":
            return {}
        return {
            "CONDA_REMOTE_CONNECT_TIMEOUT_SECS": str(connect_timeout_seconds),
            "CONDA_REMOTE_READ_TIMEOUT_SECS": str(timeout_seconds),
            "CONDA_REMOTE_MAX_RETRIES": str(retries),
        }

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "conda" not in context.services or _COMMAND.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "conda":
            return ()
        return ("conda ", "mamba ", "micromamba ", "conda-forge")
