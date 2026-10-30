"""FeatureBench data factory command entry point."""

from __future__ import annotations

import argparse
import asyncio
import re
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from codeworld.adapters.catalog import AdapterCatalog, default_adapter_catalog
from codeworld.application import (
    count_build_targets,
    parse_build_removal_targets,
    remove_build_environments,
    run_build,
    run_discover,
    run_generate,
)
from codeworld.config import FactorySettings
from codeworld.console.models import (
    CliInvocation,
    default_config_paths,
    default_output_directory,
    load_run_invocation,
    validate_config_paths,
    write_run_invocation_manifest,
    write_run_manifest,
)
from codeworld.display import AdapterDashboard, BuildDashboard, DiscoveryDashboard
from codeworld.errors import ConfigurationError, ContractError
from codeworld.events import (
    BestEffortEventSink,
    CompositeEventSink,
    Event,
    EventSink,
    JsonlEventSink,
)
from codeworld.provenance import write_run_provenance
from codeworld.run_layout import RunLayout
from codeworld.selection import default_work_selection_registry
from codeworld.state import RunLock, StateStore

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class InterruptController[ResultT]:
    """Turn repeated SIGINT requests into one cancellation and protected cleanup time."""

    def __init__(self, notify: Callable[[str], None]) -> None:
        self.notify = notify
        self.requests = 0
        self.task: asyncio.Task[ResultT] | None = None

    def request_shutdown(self) -> None:
        self.requests += 1
        if self.requests == 1:
            self.notify("Stopping safely; cleaning data-factory resources…")
            if self.task is not None and not self.task.done():
                self.task.cancel()
            return
        self.notify("Cleanup already in progress; additional Ctrl+C ignored.")

    async def run(self, operation: Awaitable[ResultT]) -> ResultT:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("interrupt controller requires an asyncio task")
        self.task = task
        if self.requests:
            task.cancel()
        try:
            return await operation
        finally:
            self.task = None


def _run_interruptible[ResultT](operation: Awaitable[ResultT]) -> ResultT:
    console = Console(stderr=True)
    controller: InterruptController[ResultT] = InterruptController(
        lambda message: console.print(f"[yellow]{message}[/]")
    )
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        previous_handler = signal.getsignal(signal.SIGINT)
        installed = False

        def handle_sigint(_signum: int, _frame: object) -> None:
            loop.call_soon_threadsafe(controller.request_shutdown)

        try:
            signal.signal(signal.SIGINT, handle_sigint)
            installed = True
        except ValueError:
            # signal.signal is restricted to the main interpreter thread.
            pass
        try:
            return runner.run(controller.run(operation))
        finally:
            if installed:
                signal.signal(signal.SIGINT, previous_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fb",
        description="Discover repositories, build environments, and generate adapter tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  fb discover
  fb discover repositories.txt
  fb build owner/project
  fb build outputs/discover-YYYYMMDD-HHMMSS/repositories.jsonl --limit 10
  fb build remove outputs/build-YYYYMMDD-HHMMSS owner/project
  fb generate outputs/build-YYYYMMDD-HHMMSS/environments.jsonl --adapter featurebench
  fb configure outputs/build-YYYYMMDD-HHMMSS
  fb --resume outputs/build-YYYYMMDD-HHMMSS""",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="incrementally run current configuration against an existing output directory",
    )
    parser.add_argument(
        "--select",
        dest="work_selection",
        metavar="POLICY",
        help=(
            "select resumable work by policy or '+'-joined states "
            "(unseen, failed, rejected, interrupted; default: unseen)"
        ),
    )
    commands = parser.add_subparsers(dest="command")

    discover = commands.add_parser("discover", help="discover candidate GitHub repositories")
    discover.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="optional TXT file containing one owner/repo or GitHub URL per line",
    )
    discover.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        help="semantic YAML file; repeat to compose run, source, and discovery settings",
    )
    discover.add_argument(
        "-o",
        "--output",
        type=Path,
        help="run output directory (default: outputs/discover-TIMESTAMP)",
    )

    build = commands.add_parser(
        "build", help="build one repository or every repository in a discovery JSONL"
    )
    build.add_argument(
        "source",
        help=(
            "GitHub repository (URL or owner/repo), repositories.jsonl, or 'remove' "
            "followed by a Build run and repository targets"
        ),
    )
    build.add_argument(
        "operands",
        nargs="*",
        help=argparse.SUPPRESS,
    )
    build.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        help="semantic YAML file; repeat to compose run, source, and build settings",
    )
    build.add_argument("--ref", help="branch, tag, or commit for a single repository")
    build.add_argument(
        "-o",
        "--output",
        type=Path,
        help="run output directory (default: outputs/build-TIMESTAMP)",
    )
    build.add_argument("--limit", type=int, help="maximum records to consume from JSONL")

    generate = commands.add_parser(
        "generate", help="generate adapter tasks from a startup snapshot of environments"
    )
    generate.add_argument("environment", type=Path, help="environment JSONL to snapshot")
    generate.add_argument(
        "-c",
        "--config",
        type=Path,
        action="append",
        help="semantic YAML file; repeat to compose host, source, and adapter settings",
    )
    generate.add_argument(
        "--environment-id",
        help="process only one environment from the startup snapshot",
    )
    generate.add_argument(
        "--limit",
        type=int,
        help="maximum environments selected from the startup snapshot",
    )
    generate.add_argument(
        "--adapter",
        required=True,
        help="registered task adapter type",
    )
    generate.add_argument(
        "-o",
        "--output",
        type=Path,
        help="run output directory (default: outputs/generate-TIMESTAMP)",
    )

    configure = commands.add_parser(
        "configure",
        help="edit an inactive run's YAML paths and execution inputs",
    )
    configure.add_argument(
        "run_directory",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="output directory containing run.yaml and .featurebench/state.sqlite3",
    )

    return parser


def resolve_workspace_root(path: Path | None = None) -> Path:
    """Return the directory that owns run outputs and host-local configuration."""
    root = (path or Path.cwd()).resolve()
    if not root.is_dir():
        raise ConfigurationError(f"FeatureBench workspace is not a directory: {root}")
    return root


def invocation_from_arguments(arguments: argparse.Namespace, project_root: Path) -> CliInvocation:
    if arguments.resume is not None:
        if arguments.command is not None:
            raise ValueError("--resume cannot be combined with another command")
        requested_selection = arguments.work_selection or "unseen"
        selection = default_work_selection_registry().resolve(requested_selection)
        return replace(
            load_run_invocation(project_root, arguments.resume),
            selection=selection.name,
        )
    if arguments.work_selection is not None:
        raise ValueError("--select requires --resume")
    if arguments.command not in ("discover", "build", "generate"):
        raise ValueError("choose discover, build, generate, or configure")
    command = arguments.command
    adapter: str | None = None
    if command == "generate":
        adapter = str(arguments.adapter).strip()
        if not adapter:
            raise ValueError("adapter must be non-empty")
    config_files = validate_config_paths(
        tuple(arguments.config or default_config_paths(project_root, command, adapter=adapter)),
        project_root,
    )
    output = arguments.output or default_output_directory(project_root, command)
    if not output.is_absolute():
        output = project_root / output
    if command == "discover":
        source: str | None = None
        if arguments.source is not None:
            source_path = Path(arguments.source)
            if not source_path.is_absolute():
                source_path = project_root / source_path
            if source_path.suffix.lower() != ".txt" or not source_path.is_file():
                raise ValueError(f"repository source is not a TXT file: {source_path}")
            source = str(source_path.resolve())
        return CliInvocation(
            command="discover",
            project_root=project_root,
            config_files=config_files,
            output_directory=output,
            repository_source=source,
        )
    if command == "generate":
        environment = Path(arguments.environment)
        if not environment.is_absolute():
            environment = project_root / environment
        if not environment.is_file():
            raise ValueError(f"environment JSONL does not exist: {environment}")
        if arguments.limit is not None and arguments.limit < 1:
            raise ValueError("--limit must be a positive integer")
        if arguments.environment_id is not None and arguments.limit is not None:
            raise ValueError("--limit cannot be combined with --environment-id")
        assert adapter is not None
        return CliInvocation(
            command="generate",
            project_root=project_root,
            config_files=config_files,
            output_directory=output,
            environment_source=str(environment),
            environment_id=arguments.environment_id,
            adapter=adapter,
            limit=arguments.limit,
        )
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("--limit must be a positive integer")
    if arguments.operands:
        raise ValueError(
            "build accepts one repository source; use 'fb build remove RUN repo...' "
            "only for Environment removal"
        )
    source = str(arguments.source).strip()
    if source.lower().endswith(".jsonl"):
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.is_file():
            raise ValueError(f"repository JSONL does not exist: {source_path}")
        if arguments.ref:
            raise ValueError("--ref only applies to one repository, not repository JSONL")
        source = str(source_path)
    return CliInvocation(
        command="build",
        project_root=project_root,
        config_files=config_files,
        output_directory=output,
        repository_source=source,
        ref=arguments.ref,
        limit=arguments.limit,
    )


def execution_arguments(invocation: CliInvocation) -> argparse.Namespace:
    source = invocation.repository_source
    repositories: Path | None = None
    repository: str | None = None
    if invocation.command == "build":
        if source is None:
            raise ValueError("build requires a repository source")
        if source.lower().endswith(".jsonl"):
            repositories = Path(source)
        else:
            repository = source
            if re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
                repository = f"https://github.com/{repository}.git"
    environment = (
        Path(invocation.environment_source)
        if invocation.command == "generate" and invocation.environment_source is not None
        else None
    )
    discovery_targets = (
        Path(source) if invocation.command == "discover" and source is not None else None
    )
    return argparse.Namespace(
        command=invocation.command,
        config=list(invocation.config_files),
        output_directory=invocation.output_directory,
        output={
            "discover": invocation.repositories_path,
            "build": invocation.environments_path,
            "generate": invocation.tasks_path,
        }[invocation.command],
        failures=invocation.failures_path,
        repository=repository,
        repositories=repositories,
        discovery_targets=discovery_targets,
        environment=environment,
        environment_id=invocation.environment_id,
        adapter=invocation.adapter,
        ref=invocation.ref,
        limit=invocation.limit,
        work_selection=invocation.selection,
        resume=invocation.resume,
        run_id=invocation.run_id,
        invocation=invocation,
    )


async def _run_application(
    arguments: argparse.Namespace,
    settings: FactorySettings,
    catalog: AdapterCatalog | None,
    events: EventSink,
    state: StateStore,
    run_id: str,
) -> str:
    command = str(arguments.command)
    selection = default_work_selection_registry().resolve(str(arguments.work_selection))
    await events.emit(
        Event(
            run_id=run_id,
            kind="work_selection",
            stage=command,
            message=f"selected durable output policy {selection.name}",
            data={"selection": selection.name},
        )
    )
    if command == "discover":
        return await run_discover(
            settings=settings,
            output=arguments.output,
            events=events,
            state=state,
            run_id=run_id,
            resume=bool(arguments.resume),
            selection=selection,
            targets_file=arguments.discovery_targets,
        )
    if command == "build":
        return await run_build(
            settings=settings,
            repository=arguments.repository,
            repositories=arguments.repositories,
            ref=arguments.ref,
            limit=arguments.limit,
            selection=selection,
            output=arguments.output,
            failures=arguments.failures,
            events=events,
            state=state,
            run_id=run_id,
        )
    if command == "generate":
        if catalog is None:
            raise AssertionError("generate application requires an Adapter catalog")
        return await run_generate(
            settings=settings,
            catalog=catalog,
            environment_path=arguments.environment,
            environment_id=arguments.environment_id,
            limit=arguments.limit,
            adapter_type=arguments.adapter,
            output=arguments.output,
            failures=arguments.failures,
            events=events,
            state=state,
            run_id=run_id,
            selection=selection,
        )
    raise AssertionError(f"unhandled command: {command}")


async def _async_main(arguments: argparse.Namespace) -> int:
    config_files = tuple(arguments.config)
    catalog = default_adapter_catalog() if arguments.command == "generate" else None
    settings = FactorySettings.from_files(config_files, adapter_catalog=catalog)
    settings.load_host_environment()
    settings.initialize_workspace()
    resume = bool(getattr(arguments, "resume", False))
    requested_run_id = getattr(arguments, "run_id", None)
    if resume and not requested_run_id:
        raise ValueError("resumed run manifest is missing its run ID")
    run_id = requested_run_id or uuid4().hex
    if not _RUN_ID.fullmatch(run_id) or run_id in (".", ".."):
        raise ValueError(
            "run ID must be 1-128 safe filename characters beginning with a letter or digit"
        )
    output_directory = Path(arguments.output_directory)
    layout = RunLayout(output_directory)
    invocation: CliInvocation = arguments.invocation
    with RunLock(layout.lock):
        if not resume:
            write_run_manifest(invocation, run_id)
        invocation_manifest = write_run_invocation_manifest(invocation, run_id)
        invocation_directory = invocation_manifest.parent
        await asyncio.to_thread(
            write_run_provenance,
            invocation_directory,
            run_id=run_id,
            settings=settings,
            project_root=Path.cwd(),
        )
        state = StateStore(layout.state)
        try:
            if resume:
                state.resume_run(
                    run_id,
                    str(arguments.command),
                    settings.model_dump(mode="json"),
                )
            else:
                state.create_run(
                    run_id,
                    str(arguments.command),
                    settings.model_dump(mode="json"),
                )
            if arguments.command == "discover":
                dashboard = DiscoveryDashboard(
                    output_directory=output_directory,
                    search_mode=(
                        arguments.discovery_targets is None
                        and settings.discovery.repositories_file is None
                        and not bool(settings.discovery.target_repositories)
                    ),
                    refresh_per_second=settings.refresh_per_second,
                )
            elif arguments.command == "build":
                dashboard = BuildDashboard(
                    output_directory=output_directory,
                    repository_target=count_build_targets(
                        repository=arguments.repository,
                        repositories=arguments.repositories,
                        limit=arguments.limit,
                    ),
                    repository_name=arguments.repository,
                    model=settings.agent.model,
                    refresh_per_second=settings.refresh_per_second,
                )
            else:
                dashboard = AdapterDashboard(
                    adapter=str(arguments.adapter),
                    output_directory=output_directory,
                    refresh_per_second=settings.refresh_per_second,
                )
            try:
                with dashboard:
                    async with BestEffortEventSink(dashboard) as display_events:
                        events = CompositeEventSink(
                            (
                                state,
                                JsonlEventSink(layout.events),
                                display_events,
                            )
                        )
                        await events.emit(
                            Event(
                                run_id=run_id,
                                kind="run_invocation_started",
                                stage=str(arguments.command),
                                message=f"started immutable invocation {invocation_directory.name}",
                                data={
                                    "invocation": invocation_directory.name,
                                    "manifest_path": str(
                                        invocation_manifest.relative_to(output_directory)
                                    ),
                                    "provenance_path": str(
                                        (invocation_directory / "provenance.json").relative_to(
                                            output_directory
                                        )
                                    ),
                                    "selection": str(arguments.work_selection),
                                    "resume": resume,
                                },
                            )
                        )
                        result = await _run_application(
                            arguments,
                            settings,
                            catalog,
                            events,
                            state,
                            run_id,
                        )
                state.finish_run(run_id)
                Console().print(f"[bold green]Done:[/] {result}")
                return 0
            except Exception as exc:
                state.finish_run(run_id, error=str(exc))
                Console(stderr=True).print_exception(show_locals=False)
                return 1
            except BaseException as exc:
                state.interrupt_run(run_id, str(exc) or type(exc).__name__)
                raise
        finally:
            state.close()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    if not values:
        parser.print_help()
        return 0
    try:
        parsed = parser.parse_args(values)
        project_root = resolve_workspace_root()
        if parsed.command == "configure":
            if parsed.resume is not None:
                raise ValueError("configure cannot be combined with --resume")
            if parsed.work_selection is not None:
                raise ValueError("configure cannot be combined with --select")
            from codeworld.console.config_tui import launch_configuration_editor

            changed = launch_configuration_editor(project_root, Path(parsed.run_directory))
            Console().print(f"[bold green]Saved:[/] {changed} run input change(s)")
            return 0
        if parsed.command == "build" and parsed.source == "remove":
            if parsed.resume is not None:
                raise ValueError("build remove cannot be combined with --resume")
            if parsed.work_selection is not None:
                raise ValueError("build remove cannot be combined with --select")
            if parsed.config or parsed.ref or parsed.output or parsed.limit:
                raise ValueError(
                    "build remove uses the existing Run configuration and cannot be combined "
                    "with --config, --ref, --output, or --limit"
                )
            if len(parsed.operands) < 2:
                raise ValueError(
                    "usage: fb build remove RUN_DIRECTORY owner/repository... | repositories.txt"
                )
            run_directory = Path(parsed.operands[0])
            repositories = parse_build_removal_targets(
                project_root,
                tuple(str(value) for value in parsed.operands[1:]),
            )
            summary = _run_interruptible(
                remove_build_environments(
                    project_root=project_root,
                    run_directory=run_directory,
                    repositories=repositories,
                )
            )
            Console().print(
                "[bold green]Removed:[/] "
                f"{summary.environments} Environment(s) for {summary.repositories} "
                f"repository target(s); deleted {summary.images} formal image(s), "
                f"{summary.stale_images} stale image(s), and "
                f"{summary.stale_containers} stale container(s).\n"
                f"Archived records: {summary.archive}\n"
                f"Resume with: fb --resume {run_directory} --select non-success"
            )
            return 0
        invocation = invocation_from_arguments(parsed, project_root)
        return _run_interruptible(_async_main(execution_arguments(invocation)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        Console(stderr=True).print(
            "[yellow]Interrupted. Resume with: fb --resume <run-output-directory>[/]"
        )
        return 130
    except (ConfigurationError, ContractError, ValueError) as exc:
        Console(stderr=True).print(f"[bold red]Error:[/] {exc}")
        return 2
    except Exception:
        Console(stderr=True).print_exception(show_locals=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
