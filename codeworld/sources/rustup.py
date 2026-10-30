"""Rustup installer and toolchain distribution source adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import SourceAdapter, SourceBuildContext

_INSTALLER_URL = re.compile(
    r"https://sh\.rustup\.rs/?(?=[\s'\"|;&)]|$)",
    re.IGNORECASE,
)
_INSTALLER_REFERENCE = "${RUSTUP_INIT_URL}"


@dataclass(frozen=True, slots=True)
class RustupSourceAdapter(SourceAdapter):
    """Route rustup self-updates and toolchains through one coherent endpoint."""

    name: str = "rustup"
    priority: int = 150
    services: frozenset[str] = frozenset({"rustup"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        if service != "rustup":
            return {}
        dist_server = url.rstrip("/")
        return {
            "RUSTUP_DIST_SERVER": dist_server,
            "RUSTUP_INIT_URL": (
                "https://sh.rustup.rs" if canonical else f"{dist_server}/rustup-init.sh"
            ),
            "RUSTUP_UPDATE_ROOT": f"{dist_server}/rustup",
        }

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        del connect_timeout_seconds
        if service != "rustup":
            return {}
        return {
            "RUSTUP_DOWNLOAD_TIMEOUT": str(timeout_seconds),
            "RUSTUP_MAX_RETRIES": str(retries),
        }

    def transform(self, source: str, context: SourceBuildContext) -> str:
        if "rustup" not in context.services:
            return source
        return _INSTALLER_URL.sub(_INSTALLER_REFERENCE, source)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "rustup":
            return ()
        return (
            "rustup",
            "RUSTUP_DIST_SERVER",
            "RUSTUP_INIT_URL",
            "RUSTUP_UPDATE_ROOT",
            "sh.rustup.rs",
            "static.rust-lang.org",
            "downloading installer",
            "syncing channel updates",
        )
