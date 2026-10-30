"""Deterministic registration and composition of independent source adapters."""

from __future__ import annotations

import re

from codeworld.errors import (
    DOCKER_HARNESS_FRAGMENT_BEGIN,
    DOCKER_HARNESS_FRAGMENT_END,
)
from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceImageRequest,
)

_DOCKERFILE_FROM = re.compile(r"^\s*FROM\s+.+?\s*$", re.IGNORECASE)
_DOCKERFILE_USER = re.compile(r"^\s*USER\s+", re.IGNORECASE)
_SAFE_ARGUMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _harness_fragment(content: str) -> str:
    if not content:
        return ""
    body = content if content.endswith("\n") else f"{content}\n"
    return f"{DOCKER_HARNESS_FRAGMENT_BEGIN}\n{body}{DOCKER_HARNESS_FRAGMENT_END}\n"


def _insert_fragment(source: str, offset: int, fragment: str) -> str:
    """Insert one Dockerfile fragment without merging adjacent instructions."""

    if not fragment:
        return source
    prefix = source[:offset]
    separator = "" if not prefix or prefix.endswith(("\n", "\r")) else "\n"
    return prefix + separator + fragment + source[offset:]


def _last_user_offset(source: str) -> int:
    offset = len(source)
    cursor = 0
    for line in source.splitlines(keepends=True):
        if _DOCKERFILE_USER.match(line):
            offset = cursor
        cursor += len(line)
    return offset


class SourceAdapterRegistry:
    """Own registrations so the source manager never branches on ecosystem names."""

    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}
        self._services: dict[str, list[SourceAdapter]] = {}

    @property
    def adapters(self) -> tuple[SourceAdapter, ...]:
        return tuple(sorted(self._adapters.values(), key=lambda item: (item.priority, item.name)))

    def register(self, adapter: SourceAdapter) -> None:
        if not adapter.name or adapter.name in self._adapters:
            raise ValueError(f"duplicate or empty source adapter: {adapter.name!r}")
        self._adapters[adapter.name] = adapter
        for service in adapter.services:
            self._services.setdefault(service, []).append(adapter)

    def for_service(self, service: str) -> tuple[SourceAdapter, ...]:
        return tuple(
            sorted(
                self._services.get(service, ()),
                key=lambda item: (item.priority, item.name),
            )
        )

    def image_request(self, reference: str) -> SourceImageRequest | None:
        for adapter in self.adapters:
            if (request := adapter.image_request(reference)) is not None:
                if request.adapter != adapter.name or request.service not in adapter.services:
                    raise ValueError(f"source adapter {adapter.name!r} returned an invalid request")
                return request
        return None

    def image_reference(self, request: SourceImageRequest, *, endpoint_url: str) -> str:
        adapter = self._adapters.get(request.adapter)
        if adapter is None:
            raise ValueError(f"unknown source adapter: {request.adapter}")
        return adapter.image_reference(request, endpoint_url=endpoint_url)

    def resolve_url(
        self,
        service: str,
        *,
        endpoint_url: str,
        canonical: bool,
        canonical_url: str,
    ) -> str:
        resolved = tuple(
            result
            for adapter in self.for_service(service)
            if (
                result := adapter.resolve_url(
                    service,
                    endpoint_url=endpoint_url,
                    canonical=canonical,
                    canonical_url=canonical_url,
                )
            )
            is not None
        )
        if len(resolved) != 1:
            raise ValueError(
                f"source service {service!r} requires exactly one URL-capable adapter; "
                f"found {len(resolved)}"
            )
        return resolved[0]

    def render(self, source: str, context: SourceBuildContext) -> str:
        names = tuple(sorted(set(context.build_argument_names)))
        if any(_SAFE_ARGUMENT.fullmatch(name) is None for name in names):
            raise ValueError("Docker build argument names must be shell-safe identifiers")
        declarations = "".join(f"ARG {name}\n" for name in names)
        lines = source.splitlines(keepends=True)
        stage_starts = [
            index
            for index, line in enumerate(lines)
            if _DOCKERFILE_FROM.match(line.rstrip("\r\n")) is not None
        ]
        if not stage_starts:
            return source
        rendered = list(lines[: stage_starts[0]])
        boundaries = (*stage_starts, len(lines))
        for position, start in enumerate(stage_starts):
            end = boundaries[position + 1]
            stage = "".join(lines[start:end])
            for adapter in self.adapters:
                stage = adapter.transform(stage, context)
            fragments = tuple(
                fragment
                for adapter in self.adapters
                if (fragment := adapter.fragments(stage, context)) is not None
            )
            setup = _harness_fragment(
                declarations + "".join(fragment.after_from for fragment in fragments)
            )
            first = stage.splitlines(keepends=True)[0]
            materialized = _insert_fragment(stage, len(first), setup)
            cleanup = _harness_fragment(
                "".join(fragment.before_final_user for fragment in reversed(fragments))
            )
            if cleanup:
                cleanup_at = _last_user_offset(materialized)
                materialized = _insert_fragment(materialized, cleanup_at, cleanup)
            rendered.append(materialized)
        return "".join(rendered)
