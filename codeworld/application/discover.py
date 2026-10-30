"""Repository Discovery application workflow."""

from __future__ import annotations

import os
from pathlib import Path

from codeworld.config import FactorySettings
from codeworld.discovery import DiscoveryRun, GitHubDiscovery
from codeworld.errors import ConfigurationError
from codeworld.events import EventSink
from codeworld.repositories.jsonl import RepositoryJsonlAppender
from codeworld.repositories.targets import read_repository_targets
from codeworld.selection import WorkSelection
from codeworld.sources import SourceManager
from codeworld.state import StateStore


def _select_targets_file(cli_file: Path | None, configured_file: Path | None) -> Path | None:
    if (
        cli_file is not None
        and configured_file is not None
        and cli_file.resolve() != configured_file.resolve()
    ):
        raise ConfigurationError(
            "discover received different TXT inputs from the CLI and YAML repositories_file"
        )
    return cli_file or configured_file


async def run_discover(
    *,
    settings: FactorySettings,
    output: Path,
    events: EventSink,
    state: StateStore,
    run_id: str,
    resume: bool,
    selection: WorkSelection,
    targets_file: Path | None = None,
) -> str:
    """Discover repositories and checkpoint them to one durable JSONL file."""
    target_repositories: tuple[str, ...] | None = None
    selected_file = _select_targets_file(targets_file, settings.discovery.repositories_file)
    if selected_file is not None:
        if settings.discovery.target_repositories:
            raise ConfigurationError(
                "discover TXT input cannot be combined with discovery.target_repositories"
            )
        target_repositories = read_repository_targets(selected_file)
    sources = SourceManager(settings.sources, run_id=run_id)
    token_value = os.getenv(settings.discovery.github_token_env, "")
    async with GitHubDiscovery(
        settings.discovery,
        tokens=(value for value in token_value.split(",") if value),
        events=events,
        run_id=run_id,
        sources=sources,
    ) as discovery:
        with RepositoryJsonlAppender(
            output,
            fsync_interval=settings.jsonl_fsync_interval,
        ) as writer:
            discovered, appended = await DiscoveryRun(
                discovery=discovery,
                writer=writer,
                state=state,
                run_id=run_id,
                output=output,
                resume=resume,
                selection=selection,
                target_repositories=target_repositories,
                events=events,
            ).run()
    return (
        f"checkpointed {discovered} repositories and appended {appended} records "
        f"this invocation to {output}"
    )
