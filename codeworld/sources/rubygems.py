"""Temporary RubyGems and Bundler source configuration adapter."""

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

_COMMAND = re.compile(r"(?:^|[\s;&|])(?:gem|bundle|bundler)(?:\s|$)", re.IGNORECASE)

_SETUP = harness_run("""set -eu; \
    gemrc=/root/.gemrc; \
    gemrc_backup=/var/tmp/.featurebench-rubygems-gemrc; \
    gemrc_generated=/var/tmp/.featurebench-rubygems-gemrc-generated; \
    bundle_config=/root/.bundle/config; \
    bundle_backup=/var/tmp/.featurebench-bundler-config; \
    bundle_generated=/var/tmp/.featurebench-bundler-config-generated; \
    if [ ! -f "$gemrc_backup" ] && [ ! -f "$gemrc_generated" ]; then \
      if [ -f "$gemrc" ]; then cp -a "$gemrc" "$gemrc_backup"; else : > "$gemrc_generated"; fi; \
    fi; \
    if [ ! -f "$bundle_backup" ] && [ ! -f "$bundle_generated" ]; then \
      if [ -f "$bundle_config" ]; then cp -a "$bundle_config" "$bundle_backup"; \
      else : > "$bundle_generated"; fi; \
    fi; \
    mkdir -p /root/.bundle; \
    printf '%s\\n' \
      '---' \
      ':sources:' \
      "- ${RUBYGEMS_MIRROR}" \
      ':update_sources: true' > "$gemrc"; \
    printf '%s\\n' \
      '---' \
      "BUNDLE_MIRROR__HTTPS://RUBYGEMS__ORG/: \\\"${RUBYGEMS_MIRROR}\\\"" \
      > "$bundle_config"
""")

_CLEANUP = harness_run("""set -eu; \
    gemrc=/root/.gemrc; \
    gemrc_backup=/var/tmp/.featurebench-rubygems-gemrc; \
    gemrc_generated=/var/tmp/.featurebench-rubygems-gemrc-generated; \
    bundle_config=/root/.bundle/config; \
    bundle_backup=/var/tmp/.featurebench-bundler-config; \
    bundle_generated=/var/tmp/.featurebench-bundler-config-generated; \
    if [ -f "$gemrc_backup" ]; then mv "$gemrc_backup" "$gemrc"; \
    elif [ -f "$gemrc_generated" ]; then rm -f "$gemrc"; fi; \
    if [ -f "$bundle_backup" ]; then mv "$bundle_backup" "$bundle_config"; \
    elif [ -f "$bundle_generated" ]; then rm -f "$bundle_config"; fi; \
    rm -f "$gemrc_generated" "$bundle_generated"; \
    rmdir /root/.bundle 2>/dev/null || true
""")


@dataclass(frozen=True, slots=True)
class RubyGemsSourceAdapter(SourceAdapter):
    name: str = "rubygems"
    priority: int = 200
    services: frozenset[str] = frozenset({"rubygems"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        return {"RUBYGEMS_MIRROR": url} if service == "rubygems" else {}

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "rubygems" not in context.services or _COMMAND.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return ("bundle ", "bundler", "gem install", "rubygems") if service == "rubygems" else ()
