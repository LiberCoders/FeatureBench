"""Temporary Gradle repository adapter backed by the Maven source service."""

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

_COMMAND = re.compile(r"(?:^|[\s;&|])(?:\./)?gradlew?(?:\s|$)", re.IGNORECASE)

_SETUP = harness_run("""set -eu; \
    init=/root/.gradle/init.d/featurebench-mirror.gradle; \
    backup=/var/tmp/.featurebench-gradle-mirror.gradle; \
    generated=/var/tmp/.featurebench-gradle-mirror-generated; \
    mkdir -p /root/.gradle/init.d; \
    if [ ! -f "$backup" ] && [ ! -f "$generated" ]; then \
      if [ -f "$init" ]; then cp -a "$init" "$backup"; else : > "$generated"; fi; \
    fi; \
    printf '%s\\n' \
      'def featurebenchMirror = System.getenv("GRADLE_MIRROR")' \
      'allprojects {' \
      '  buildscript {' \
      '    repositories {' \
      '      maven { url featurebenchMirror }' \
      '    }' \
      '  }' \
      '  repositories {' \
      '    maven { url featurebenchMirror }' \
      '  }' \
      '}' > "$init"
""")

_CLEANUP = harness_run("""set -eu; \
    init=/root/.gradle/init.d/featurebench-mirror.gradle; \
    backup=/var/tmp/.featurebench-gradle-mirror.gradle; \
    generated=/var/tmp/.featurebench-gradle-mirror-generated; \
    if [ -f "$backup" ]; then mv "$backup" "$init"; \
    elif [ -f "$generated" ]; then rm -f "$init"; fi; \
    rm -f "$generated"
""")


@dataclass(frozen=True, slots=True)
class GradleSourceAdapter(SourceAdapter):
    """Prepend the selected Maven repository to Gradle dependency resolution."""

    name: str = "gradle"
    priority: int = 201
    services: frozenset[str] = frozenset({"maven"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        return {"GRADLE_MIRROR": url} if service == "maven" else {}

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "maven" not in context.services or _COMMAND.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return ("gradle ", "gradlew", "GRADLE_MIRROR") if service == "maven" else ()
