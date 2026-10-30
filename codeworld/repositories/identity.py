"""Stable aliases for repository-addressed work across stage output formats."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_repository_url(value: str) -> str:
    """Normalize equivalent Git repository spellings without resolving the network."""

    candidate = value.strip()
    if "://" not in candidate and candidate.count("/") == 1:
        candidate = f"https://github.com/{candidate}"
    parsed = urlsplit(candidate)
    if not parsed.scheme or not parsed.netloc:
        return candidate.removesuffix("/").removesuffix(".git").casefold()
    path = parsed.path.rstrip("/")
    if path.casefold().endswith(".git"):
        path = path[:-4]
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), path.casefold(), "", ""))


def repository_aliases(
    repository: str,
    *,
    github_id: int | None = None,
    full_name: str | None = None,
) -> tuple[str, ...]:
    """Return every stable identity available before repository materialization."""

    aliases = [f"repository:{normalize_repository_url(repository)}"]
    if github_id is not None:
        aliases.append(f"github:{github_id}")
    if full_name:
        aliases.append(f"name:{full_name.casefold()}")
    return tuple(aliases)
