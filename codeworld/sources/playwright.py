"""Playwright browser-binary download source adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import SourceAdapter


@dataclass(frozen=True, slots=True)
class PlaywrightSourceAdapter(SourceAdapter):
    """Route Playwright-managed browser artifacts through one build-only host."""

    name: str = "playwright"
    priority: int = 150
    services: frozenset[str] = frozenset({"playwright"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        if service != "playwright":
            return {}
        return {"PLAYWRIGHT_DOWNLOAD_HOST": url.rstrip("/")}

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        del connect_timeout_seconds, retries
        if service != "playwright":
            return {}
        return {"PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT": str(timeout_seconds * 1000)}

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "playwright":
            return ()
        return (
            "playwright install",
            "PLAYWRIGHT_DOWNLOAD_HOST",
            "cdn.playwright.dev",
            "playwright.download.prss.microsoft.com",
            "chrome-for-testing-public",
            "Downloading Chrome for Testing",
            "Downloading Chromium",
            "Downloading Firefox",
            "Downloading Webkit",
        )
