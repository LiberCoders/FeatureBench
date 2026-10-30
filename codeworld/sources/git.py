"""GitHub API and Git transport source application."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceBuildFragments,
    SourceFailureKind,
    SourceMatchConfidence,
    harness_run,
    observed_failure_output,
)

_GITHUB_HOST = re.compile(
    r"https://(?:github\.com|raw\.githubusercontent\.com|"
    r"gist\.github\.com|gist\.githubusercontent\.com)(?:/[^\s'\"\\]*)?",
    re.IGNORECASE,
)
_GITHUB_URL = re.compile(
    r"(?<!\$\{GITHUB_PROXY_PREFIX\})(?<!/)"
    r"https://(?:github\.com|raw\.githubusercontent\.com|"
    r"gist\.github\.com|gist\.githubusercontent\.com)(?:/[^\s'\"\\]*)?",
    re.IGNORECASE,
)
_GITHUB_PROXY_REFERENCE = "${GITHUB_PROXY_PREFIX}"
_GITHUB_SCP_URL = re.compile(
    r"(?<![A-Za-z0-9_@./-])git@github\.com:"
    r"(?P<path>[A-Za-z0-9_.-]+/[^\s'\"\\?#]+)",
    re.IGNORECASE,
)
_GITHUB_SCHEME_URL = re.compile(
    r"(?P<vcs>git\+)?(?:ssh|git)://(?:git@)?github\.com/"
    r"(?P<path>[A-Za-z0-9_.-]+/[^\s'\"\\?#]+)",
    re.IGNORECASE,
)
_GIT_NETWORK_COMMAND = re.compile(
    r"(?:^|[\s;&|/])(?:git|git_transport)\s+"
    r"(?:-[A-Za-z]\s+[^\s]+\s+|-c\s+[^\s]+\s+|--[^\s]+\s+)*"
    r"(?:clone|fetch|submodule|lfs)(?:\s|$)|"
    r"(?:git\+[^\s'\"]*github\.com|git://github\.com|"
    r"ssh://git@github\.com|git@github\.com:)",
    re.IGNORECASE,
)
_GIT_REMOTE_MISS = re.compile(
    r"\bnot our ref\b|server does not allow request for unadvertised object|"
    r"couldn't find remote ref|did not contain [0-9a-f]{7,64}",
    re.IGNORECASE,
)
_PUBLIC_GITHUB_TRANSPORTS = (
    "git@github.com:",
    "ssh://git@github.com/",
    "git://github.com/",
)


def _proxy_public_url(match: re.Match[str]) -> str:
    """Route public literal GitHub assets without proxying credential-bearing URLs."""

    url = match.group(0)
    if "?" in url or "#" in url:
        return url
    return f"{_GITHUB_PROXY_REFERENCE}{url}"


def _normalize_scp_url(match: re.Match[str]) -> str:
    return f"https://github.com/{match.group('path')}"


def _normalize_scheme_url(match: re.Match[str]) -> str:
    prefix = "git+" if match.group("vcs") else ""
    return f"{prefix}https://github.com/{match.group('path')}"


def github_transport_config_arguments(
    target_prefix: str,
    *,
    redirect_https: bool,
) -> tuple[str, ...]:
    """Return Git command-scoped rewrites for public GitHub transport forms."""

    originals = (
        *(("https://github.com/",) if redirect_https else ()),
        *_PUBLIC_GITHUB_TRANSPORTS,
    )
    return tuple(
        item
        for original in originals
        for item in ("-c", f"url.{target_prefix}.insteadOf={original}")
    )


_SETUP = harness_run("""set -eu; \
    config=/root/.gitconfig; \
    backup=/var/tmp/.featurebench-gitconfig; \
    generated=/var/tmp/.featurebench-gitconfig-generated; \
    if command -v git >/dev/null 2>&1; then \
      if [ ! -f "$backup" ] && [ ! -f "$generated" ]; then \
        if [ -f "$config" ]; then cp -a "$config" "$backup"; else : > "$generated"; fi; \
      fi; \
      target="${GITHUB_PROXY_PREFIX}https://github.com/"; \
      if [ -n "${GITHUB_PROXY_PREFIX}" ]; then \
        git config --global --add "url.${target}.insteadOf" "https://github.com/"; \
      fi; \
      git config --global --add "url.${target}.insteadOf" "git@github.com:"; \
      git config --global --add "url.${target}.insteadOf" "ssh://git@github.com/"; \
      git config --global --add "url.${target}.insteadOf" "git://github.com/"; \
    fi
""")

_CLEANUP = harness_run("""set -eu; \
    config=/root/.gitconfig; \
    backup=/var/tmp/.featurebench-gitconfig; \
    generated=/var/tmp/.featurebench-gitconfig-generated; \
    if [ -f "$backup" ]; then mv "$backup" "$config"; \
    elif [ -f "$generated" ]; then rm -f "$config"; fi; \
    rm -f "$generated"
""")


@dataclass(frozen=True, slots=True)
class GitSourceAdapter(SourceAdapter):
    name: str = "git"
    priority: int = 40
    services: frozenset[str] = frozenset({"github_api", "github_git"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        if service != "github_git":
            return {}
        return {"GITHUB_PROXY_PREFIX": "" if canonical else f"{url.rstrip('/')}/"}

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "github_git":
            return ()
        return (
            "GITHUB_PROXY_PREFIX",
            "git clone",
            "git fetch",
            "git submodule",
            "githubusercontent.com",
            "gist.github.com",
        )

    def match_build_step(
        self,
        service: str,
        *,
        instruction: str,
        output: str,
        endpoint_url: str,
    ) -> SourceMatchConfidence | None:
        if service != "github_git":
            return None
        observed = observed_failure_output(output)
        if endpoint_url.casefold() in observed.casefold():
            return SourceMatchConfidence.ENDPOINT
        detail = f"{instruction}\n{observed}"
        if _GIT_NETWORK_COMMAND.search(detail) is None:
            return None
        if (
            "github.com" not in detail.casefold()
            and "githubusercontent.com" not in detail.casefold()
        ):
            return None
        return SourceMatchConfidence.OPERATION

    def classify_build_failure(
        self,
        service: str,
        *,
        detail: str,
        confidence: SourceMatchConfidence,
    ) -> SourceFailureKind | None:
        if service == "github_git" and _GIT_REMOTE_MISS.search(detail) is not None:
            return SourceFailureKind.NOT_FOUND
        return SourceAdapter.classify_build_failure(
            self,
            service,
            detail=detail,
            confidence=confidence,
        )

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "github_git" not in context.services or _GITHUB_HOST.search(source) is None:
            return None
        return SourceBuildFragments(after_from=_SETUP, before_final_user=_CLEANUP)

    def transform(self, source: str, context: SourceBuildContext) -> str:
        if "github_git" not in context.services:
            return source
        normalized = _GITHUB_SCP_URL.sub(_normalize_scp_url, source)
        normalized = _GITHUB_SCHEME_URL.sub(_normalize_scheme_url, normalized)
        return _GITHUB_URL.sub(_proxy_public_url, normalized)

    def resolve_url(
        self,
        service: str,
        *,
        endpoint_url: str,
        canonical: bool,
        canonical_url: str,
    ) -> str | None:
        if service not in self.services:
            return None
        if canonical:
            return canonical_url
        if service == "github_git":
            return f"{endpoint_url.rstrip('/')}/{canonical_url}"
        upstream = urlsplit(canonical_url)
        replacement = urlsplit(endpoint_url)
        return urlunsplit(
            (
                replacement.scheme,
                replacement.netloc,
                f"{replacement.path.rstrip('/')}/{upstream.path.lstrip('/')}",
                upstream.query,
                "",
            )
        )
