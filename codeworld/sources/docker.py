"""OCI registry source adapter with canonical-reference restoration."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from codeworld.sources.base import SourceAdapter, SourceImageRequest

_REGISTRIES = {
    "docker.io": "docker_io",
    "index.docker.io": "docker_io",
    "ghcr.io": "ghcr_io",
    "mcr.microsoft.com": "mcr_microsoft",
    "gcr.io": "gcr_io",
    "quay.io": "quay_io",
    "registry.k8s.io": "k8s_registry",
}


@dataclass(frozen=True, slots=True)
class DockerSourceAdapter(SourceAdapter):
    name: str = "docker"
    priority: int = 50
    services: frozenset[str] = frozenset(_REGISTRIES.values())

    def image_request(self, reference: str) -> SourceImageRequest | None:
        value = reference.strip()
        if not value or value == "scratch" or value.startswith(("$", "sha256:")):
            return None
        first, separator, remainder = value.partition("/")
        explicit_registry = bool(
            separator and ("." in first or ":" in first or first == "localhost")
        )
        if not explicit_registry:
            return SourceImageRequest(
                adapter=self.name,
                service="docker_io",
                remainder=value if separator else f"library/{value}",
            )
        service = _REGISTRIES.get(first.lower())
        if service is None:
            return None
        return SourceImageRequest(adapter=self.name, service=service, remainder=remainder)

    def image_reference(self, request: SourceImageRequest, *, endpoint_url: str) -> str:
        endpoint = urlsplit(endpoint_url)
        prefix = f"{endpoint.netloc}{endpoint.path}".strip("/")
        return f"{prefix}/{request.remainder.lstrip('/')}"
