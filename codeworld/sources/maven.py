"""Temporary Maven settings adapter."""

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

_COMMAND = re.compile(r"(?:^|[\s;&|])(?:\./)?mvnw?(?:\s|$)", re.IGNORECASE)

_SETUP = harness_run("""set -eu; \
    settings=/root/.m2/settings.xml; \
    backup=/var/tmp/.featurebench-maven-settings.xml; \
    generated=/var/tmp/.featurebench-maven-settings-generated; \
    mkdir -p /root/.m2; \
    if [ ! -f "$backup" ] && [ ! -f "$generated" ]; then \
      if [ -f "$settings" ]; then cp -a "$settings" "$backup"; else : > "$generated"; fi; \
    fi; \
    printf '%s\\n' \
      '<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">' \
      '  <mirrors>' \
      '    <mirror>' \
      '      <id>featurebench-build-mirror</id>' \
      "      <url>${MAVEN_MIRROR}</url>" \
      '      <mirrorOf>*</mirrorOf>' \
      '    </mirror>' \
      '  </mirrors>' \
      '</settings>' > "$settings"
""")

_CLEANUP = harness_run("""set -eu; \
    settings=/root/.m2/settings.xml; \
    backup=/var/tmp/.featurebench-maven-settings.xml; \
    generated=/var/tmp/.featurebench-maven-settings-generated; \
    repository=/root/.m2/repository; \
    if [ -d "$repository" ]; then \
      find "$repository" -type f -name '_remote.repositories' -exec sh -c '\
        for path do \
          if grep -q "featurebench-build-mirror" "$path"; then rm -f "$path"; fi; \
        done\
      ' sh {} +; \
      find "$repository" -type f -name '*.lastUpdated' -delete; \
      find "$repository" -type f -name 'maven-metadata-featurebench-build-mirror.xml*' -exec sh -c '\
        for path do \
          directory=${path%/*}; \
          filename=${path##*/}; \
          suffix=${filename#*maven-metadata-featurebench-build-mirror.xml}; \
          mv -f "$path" "$directory/maven-metadata-central.xml$suffix"; \
        done\
      ' sh {} +; \
    fi; \
    if [ -f "$backup" ]; then mv "$backup" "$settings"; \
    elif [ -f "$generated" ]; then rm -f "$settings"; fi; \
    rm -f "$generated"
""")


@dataclass(frozen=True, slots=True)
class MavenSourceAdapter(SourceAdapter):
    name: str = "maven"
    priority: int = 200
    services: frozenset[str] = frozenset({"maven"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        return {"MAVEN_MIRROR": url} if service == "maven" else {}

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "maven" not in context.services or _COMMAND.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return ("mvn ", "maven") if service == "maven" else ()
