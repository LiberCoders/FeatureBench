"""System-level Debian/Ubuntu APT source adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceBuildFragments,
    harness_run,
)

_ARGUMENTS: Mapping[str, str] = {
    "debian_apt": "DEBIAN_APT_MIRROR",
    "debian_security": "DEBIAN_SECURITY_APT_MIRROR",
    "ubuntu_apt": "UBUNTU_APT_MIRROR",
    "ubuntu_ports": "UBUNTU_PORTS_APT_MIRROR",
}

_SETUP = harness_run("""set -eu; \
    if command -v apt-get >/dev/null 2>&1 && [ -d /etc/apt ]; then \
      find /etc/apt -type f \\( -name '*.list' -o -name '*.sources' \\) -exec sed -i \
        -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_APT_MIRROR}|g" \
        -e "s|https://deb.debian.org/debian-security|${DEBIAN_SECURITY_APT_MIRROR}|g" \
        -e "s|http://security.debian.org/debian-security|${DEBIAN_SECURITY_APT_MIRROR}|g" \
        -e "s|https://security.debian.org/debian-security|${DEBIAN_SECURITY_APT_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian|${DEBIAN_APT_MIRROR}|g" \
        -e "s|https://deb.debian.org/debian|${DEBIAN_APT_MIRROR}|g" \
        -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|https://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|https://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|http://ports.ubuntu.com/ubuntu-ports|${UBUNTU_PORTS_APT_MIRROR}|g" \
        -e "s|https://ports.ubuntu.com/ubuntu-ports|${UBUNTU_PORTS_APT_MIRROR}|g" {} +; \
    fi
""")


@dataclass(frozen=True, slots=True)
class AptSourceAdapter(SourceAdapter):
    name: str = "apt"
    priority: int = 100
    services: frozenset[str] = frozenset(_ARGUMENTS)

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        del canonical
        variable = _ARGUMENTS.get(service)
        return {variable: url} if variable is not None else {}

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "apt-get" not in source or not (self.services & context.services):
            return None
        return SourceBuildFragments(after_from=_SETUP)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        return {
            "debian_apt": ("apt-get", "debian"),
            "debian_security": ("debian-security",),
            "ubuntu_apt": ("apt-get", "ubuntu"),
            "ubuntu_ports": ("ubuntu-ports",),
        }.get(service, ())
