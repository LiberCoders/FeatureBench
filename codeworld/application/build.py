"""Environment Build application workflows."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from codeworld.agents.factory import AgentFactory
from codeworld.application.resources import cleanup_owned_resources, finish_owned_cleanup
from codeworld.config import FactorySettings
from codeworld.console.models import load_run_invocation
from codeworld.environment.jsonl import iter_environments, write_environment_snapshot
from codeworld.environment.production import EnvironmentProducer
from codeworld.events import EventSink
from codeworld.models import StrictModel
from codeworld.repositories.identity import normalize_repository_url
from codeworld.repositories.jsonl import iter_repositories
from codeworld.run_layout import RunLayout
from codeworld.runtime.container import DockerRuntime
from codeworld.selection import WorkSelection
from codeworld.sources import SourceManager
from codeworld.state import RunLock, StateStore

_OWNER_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class BuildRemovalSummary(StrictModel):
    """One explicit removal from a durable Build output."""

    repositories: int
    environments: int
    images: int
    stale_containers: int
    stale_images: int
    archive: Path


def parse_build_removal_targets(
    project_root: Path,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    """Parse owner/repository operands or one TXT file without accepting other formats."""

    if not values:
        raise ValueError("build remove requires at least one owner/repository or TXT file")
    if len(values) == 1 and values[0].lower().endswith(".txt"):
        source = Path(values[0])
        if not source.is_absolute():
            source = project_root / source
        if not source.is_file():
            raise ValueError(f"repository TXT file does not exist: {source}")
        candidates = tuple(
            line
            for raw_line in source.read_text(encoding="utf-8").splitlines()
            if (line := raw_line.strip()) and not line.startswith("#")
        )
    else:
        if any(value.lower().endswith(".txt") for value in values):
            raise ValueError("a repository TXT file cannot be combined with repository operands")
        candidates = values

    targets: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        target = value.strip()
        if not _OWNER_REPOSITORY.fullmatch(target):
            raise ValueError(f"invalid repository target {value!r}; expected owner/repository")
        key = target.casefold()
        if key not in seen:
            seen.add(key)
            targets.append(target)
    if not targets:
        raise ValueError("repository target list is empty")
    return tuple(targets)


async def remove_build_environments(
    *,
    project_root: Path,
    run_directory: Path,
    repositories: tuple[str, ...],
) -> BuildRemovalSummary:
    """Atomically remove selected Build outputs, then reclaim their owned Docker resources."""

    invocation = load_run_invocation(project_root, run_directory)
    if invocation.command != "build":
        raise ValueError(
            f"build remove requires a Build run, got {invocation.command!r}: "
            f"{invocation.output_directory}"
        )
    if invocation.run_id is None:
        raise ValueError(f"Build run is missing its run ID: {invocation.output_directory}")
    settings = FactorySettings.from_files(invocation.config_files)
    settings.load_host_environment()
    settings.initialize_workspace()
    runtime = DockerRuntime(
        settings.docker,
        run_id=invocation.run_id,
        resource_scope=DockerRuntime.scope_for_workspace(settings.workspace),
    )
    requested = {normalize_repository_url(repository): repository for repository in repositories}
    output = invocation.environments_path

    layout = RunLayout(invocation.output_directory)
    with RunLock(layout.lock):
        records = tuple(iter_environments(output)) if output.is_file() else ()
        removed = tuple(
            record for record in records if normalize_repository_url(record.repository) in requested
        )
        matched = {normalize_repository_url(record.repository) for record in removed}
        missing = tuple(requested[key] for key in requested.keys() - matched)
        if missing:
            rendered = ", ".join(sorted(missing, key=str.casefold))
            raise ValueError(f"repositories have no Environment in this Build run: {rendered}")
        remaining = tuple(record for record in records if record not in removed)

        await runtime.check_available()
        archive = layout.removals / f"{uuid4().hex}.jsonl"
        write_environment_snapshot(archive, removed)
        write_environment_snapshot(output, remaining)

        protected_images = frozenset(record.image for record in remaining)
        stale_containers, stale_images = await runtime.cleanup_run_resources(
            protected_images=protected_images
        )
        removed_images = frozenset(record.image for record in removed) - protected_images
        for image in removed_images:
            await runtime.remove_image(image, force=True)
        uncleared: list[str] = []
        for image in removed_images:
            if await runtime.image_exists(image):
                uncleared.append(image)
        if uncleared:
            rendered = ", ".join(sorted(uncleared))
            raise ValueError(
                f"removed {len(removed)} Environment record(s), but Docker resources remain "
                f"in use: {rendered}; removed records are archived at {archive}"
            )

    return BuildRemovalSummary(
        repositories=len(repositories),
        environments=len(removed),
        images=len(removed_images),
        stale_containers=len(stale_containers),
        stale_images=len(frozenset(stale_images) - removed_images),
        archive=archive,
    )


def _durable_images(output: Path) -> frozenset[str] | None:
    if not output.is_file():
        return frozenset()
    try:
        return frozenset(record.image for record in iter_environments(output))
    except Exception:
        # A malformed durability boundary must make image cleanup more conservative.
        return None


def count_build_targets(
    *,
    repository: str | None,
    repositories: Path | None,
    limit: int | None,
) -> int:
    """Count distinct repository inputs for dashboard initialization."""
    if repository is not None:
        return 1
    if repositories is None:
        raise ValueError("build requires one repository or a repository JSONL")
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    count = 0
    for candidate in iter_repositories(repositories):
        name_key = candidate.full_name.casefold()
        if candidate.github_id in seen_ids or name_key in seen_names:
            continue
        seen_ids.add(candidate.github_id)
        seen_names.add(name_key)
        count += 1
        if limit is not None and count >= limit:
            break
    return count


async def run_build(
    *,
    settings: FactorySettings,
    repository: str | None,
    repositories: Path | None,
    ref: str | None,
    limit: int | None,
    selection: WorkSelection,
    output: Path,
    failures: Path,
    events: EventSink,
    state: StateStore,
    run_id: str,
) -> str:
    """Build one repository or consume a Discovery JSONL batch."""
    sources = SourceManager(settings.sources, run_id=run_id)
    runtime = DockerRuntime(
        settings.docker,
        sources=sources,
        run_id=run_id,
        resource_scope=DockerRuntime.scope_for_workspace(settings.workspace),
    )
    cleanup_images = settings.environment.cleanup_intermediate_images
    await cleanup_owned_resources(
        runtime,
        protected_images=_durable_images(output) if cleanup_images else None,
    )
    try:
        producer = EnvironmentProducer(
            settings=settings,
            runtime=runtime,
            agents=AgentFactory(settings.agent),
            events=events,
            state=state,
            run_id=run_id,
            selection=selection,
        )
        if repositories is not None:
            summary = await producer.produce(
                repositories=repositories,
                output=output,
                failures=failures,
                limit=limit,
            )
            return (
                f"submitted {summary.submitted}, built {summary.built}, "
                f"skipped {summary.skipped}, rejected {summary.rejected}, failed {summary.failed}; "
                f"environments: {output}; failures: {failures}"
            )
        if repository is None:
            raise ValueError("build requires one repository or a repository JSONL")
        outcome = await producer.produce_one(
            repository=repository,
            ref=ref,
            output=output,
            failures=failures,
        )
        if outcome is None or outcome.skipped:
            return "repository already has a durable terminal output for this selection"
        return f"appended environment to {output}"
    finally:
        await finish_owned_cleanup(
            runtime,
            protected_images=_durable_images(output) if cleanup_images else None,
        )
