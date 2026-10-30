"""Contracts shared by the source manager and independently registered adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from codeworld.errors import DOCKER_HARNESS_STEP_MARKER

_SOURCE_AVAILABILITY_FAILURE = re.compile(
    r"could not resolve host|temporary failure in name resolution|timed? out|"
    r"(?:connect|read)timeouterror|timeout(?:error| was reached| exceeded)|"
    r"connection (?:failed|reset|refused)|"
    r"operation too slow|less than [^\n]*bytes/sec transferred|"
    r"network is unreachable|(?:tls )?handshake|"
    r"certificate (?:verification failed|is not trusted|issuer is unknown)|"
    r"http/2[^\n]*protocol_error|"
    r"unexpected (?:eof|end of file)|not in gzip format|invalid compressed data|"
    r"requested url returned error: (?:404|429|5\d\d)|"
    r"remote end closed|http[^\n ]* (?:404|429|5\d\d)",
    re.IGNORECASE,
)
_SOURCE_TRANSFER_FAILURE = re.compile(
    r"\bdownload failed\b|\bfailed to (?:fetch|download)\b",
    re.IGNORECASE,
)
_BUILDKIT_STEP_INSTRUCTION = re.compile(r"^#\S+\s+\[[^\]]+\]\s+RUN(?:\s|$)", re.IGNORECASE)
_BUILDKIT_PROCESS_ERROR = re.compile(r'^#\S+\s+ERROR:\s+process\s+["\']', re.IGNORECASE)
_BUILDKIT_OUTPUT_PREFIX = re.compile(r"^#\S+\s+(?:\d+(?:\.\d+)?\s+)?")
_HARNESS_RUN_INSTRUCTION = re.compile(r"^\s*RUN(?:\s|$)", re.IGNORECASE)
_HARNESS_RUN_OPTION = re.compile(r"--[A-Za-z][A-Za-z0-9-]*(?:=[^\s\x00]+)?")


def observed_failure_output(detail: str) -> str:
    """Exclude rendered commands whose literals are not observed failures."""

    return "\n".join(
        line
        for line in detail.splitlines()
        if _HARNESS_RUN_INSTRUCTION.match(line) is None
        and _BUILDKIT_STEP_INSTRUCTION.match(line) is None
        and _BUILDKIT_PROCESS_ERROR.match(line) is None
    )


def causal_failure_output(detail: str) -> str:
    """Return the deepest explicit failure cause, or all observed output.

    Build tools commonly wrap a precise semantic error in a generic transfer
    message.  Rotating a source based on the wrapper is incorrect when the
    deepest ``Caused by:`` block says that a manifest, toolchain, or build is
    invalid.  BuildKit prefixes are removed only for locating that block; the
    original observed output remains the fallback for tools without cause
    chains.
    """

    observed = observed_failure_output(detail)
    lines = [_BUILDKIT_OUTPUT_PREFIX.sub("", line) for line in observed.splitlines()]
    cause_indexes = [
        index for index, line in enumerate(lines) if line.strip().casefold() == "caused by:"
    ]
    if not cause_indexes:
        return observed

    marker_index = cause_indexes[-1]
    marker_indent = len(lines[marker_index]) - len(lines[marker_index].lstrip())
    cause: list[str] = []
    for line in lines[marker_index + 1 :]:
        if not line.strip():
            if cause:
                cause.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= marker_indent:
            break
        cause.append(line)
    rendered = "\n".join(cause).strip()
    return rendered or observed


def harness_run(command: str, *, options: tuple[str, ...] = ()) -> str:
    """Render one adapter-owned Dockerfile RUN instruction with explicit provenance."""

    body = command.strip()
    if not body or "\x00" in body:
        raise ValueError("Harness RUN command must be non-empty and contain no NUL bytes")
    unescaped_lines = tuple(
        line
        for line in body.splitlines(keepends=True)
        if line.endswith(("\n", "\r")) and not line.rstrip("\r\n").endswith("\\")
    )
    if unescaped_lines:
        raise ValueError("Harness RUN command newlines must use Docker line continuations")
    invalid = tuple(option for option in options if _HARNESS_RUN_OPTION.fullmatch(option) is None)
    if invalid:
        raise ValueError(f"Invalid Dockerfile RUN options: {invalid!r}")
    rendered_options = "".join(f" {option}" for option in options)
    return f"RUN{rendered_options} {DOCKER_HARNESS_STEP_MARKER}; {body}\n"


class SourceFailureKind(StrEnum):
    """Adapter-owned source failure semantics consumed by common retry policy."""

    AVAILABILITY = "availability"
    NOT_FOUND = "not_found"


class SourceMatchConfidence(IntEnum):
    """Strength of the evidence that one failed step belongs to a source service."""

    OPERATION = 1
    ENDPOINT = 2


@dataclass(frozen=True, slots=True)
class SourceBuildContext:
    """The source snapshot selected for one isolated Docker build attempt."""

    services: frozenset[str]
    build_argument_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceBuildFragments:
    """Temporary Dockerfile material applied and removed by the common renderer."""

    after_from: str
    before_final_user: str = ""

    def __post_init__(self) -> None:
        for fragment in (self.after_from, self.before_final_user):
            for line in fragment.splitlines():
                if (
                    _HARNESS_RUN_INSTRUCTION.match(line) is not None
                    and DOCKER_HARNESS_STEP_MARKER not in line
                ):
                    raise ValueError(
                        "Source adapter RUN instructions must be rendered with harness_run()"
                    )


@dataclass(frozen=True, slots=True)
class SourceImageRequest:
    """An adapter-owned mapping from a canonical image to one logical service."""

    adapter: str
    service: str
    remainder: str


class SourceAdapter:
    """Extensible application boundary for one source-consuming ecosystem."""

    name: str = ""
    priority: int = 100
    services: frozenset[str] = frozenset()

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        """Return build-only arguments required to apply this service."""

        del service, url, canonical
        return {}

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        """Translate the common download policy to adapter-owned native variables."""

        del service, connect_timeout_seconds, timeout_seconds, retries
        return {}

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        """Return temporary Dockerfile setup/cleanup, if the source needs it."""

        del source, context
        return None

    def transform(self, source: str, context: SourceBuildContext) -> str:
        """Apply one adapter-owned source rewrite before common composition."""

        del context
        return source

    def failure_hints(self, service: str) -> tuple[str, ...]:
        """Return output tokens that attribute a failed build step to this service."""

        del service
        return ()

    def match_build_step(
        self,
        service: str,
        *,
        instruction: str,
        output: str,
        endpoint_url: str,
    ) -> SourceMatchConfidence | None:
        """Return positive evidence that the active BuildKit step consumes this service.

        Endpoint evidence is stronger than an operation marker. Rendered command lines
        are deliberately excluded from endpoint matching because they describe what was
        configured, not what actually failed.
        """

        observed = observed_failure_output(output)
        if endpoint_url.casefold() in observed.casefold():
            return SourceMatchConfidence.ENDPOINT
        active_instruction = instruction.casefold()
        if any(marker.casefold() in active_instruction for marker in self.failure_hints(service)):
            return SourceMatchConfidence.OPERATION
        return None

    def classify_build_failure(
        self,
        service: str,
        *,
        detail: str,
        confidence: SourceMatchConfidence,
    ) -> SourceFailureKind | None:
        """Classify a failure already attributed to this adapter's service."""

        del service
        cause = causal_failure_output(detail)
        if _SOURCE_AVAILABILITY_FAILURE.search(cause) is not None:
            return SourceFailureKind.AVAILABILITY
        if (
            confidence is SourceMatchConfidence.ENDPOINT
            and _SOURCE_TRANSFER_FAILURE.search(cause) is not None
        ):
            return SourceFailureKind.AVAILABILITY
        return None

    def resolve_url(
        self,
        service: str,
        *,
        endpoint_url: str,
        canonical: bool,
        canonical_url: str,
    ) -> str | None:
        """Apply an adapter-owned HTTP URL transformation, if supported."""

        del service, endpoint_url, canonical, canonical_url
        return None

    def image_request(self, reference: str) -> SourceImageRequest | None:
        """Map a canonical OCI reference to a service owned by this adapter."""

        del reference
        return None

    def image_reference(self, request: SourceImageRequest, *, endpoint_url: str) -> str:
        """Materialize a candidate image reference for a selected endpoint."""

        del request, endpoint_url
        raise ValueError(f"source adapter {self.name!r} does not handle OCI images")
