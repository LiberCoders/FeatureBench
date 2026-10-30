"""Repository target parsing shared by configuration and input adapters."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit


def normalize_repository_target(value: str) -> str:
    """Normalize one ``owner/repository`` identity or GitHub HTTPS URL."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("repository target must not be empty")

    if candidate.startswith(("https://", "http://")):
        parsed = urlsplit(candidate)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
            raise ValueError("expected owner/repo or an https://github.com/owner/repo URL")
        if parsed.query or parsed.fragment:
            raise ValueError("GitHub repository URL must not contain a query or fragment")
        candidate = parsed.path

    candidate = candidate.strip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    parts = candidate.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(character.isspace() for character in candidate)
        or any(character in candidate for character in "?#\\")
    ):
        raise ValueError("expected owner/repo or an https://github.com/owner/repo URL")
    return "/".join(parts)


def read_repository_targets(path: str | Path) -> tuple[str, ...]:
    """Read an ordered, deduplicated repository list from a UTF-8 text file."""
    source = Path(path)
    result: list[str] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                normalized = normalize_repository_target(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid repository target at {source}:{line_number}: {exc}"
                ) from exc
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                result.append(normalized)
    if not result:
        raise ValueError(f"repository target file contains no repositories: {source}")
    return tuple(result)
