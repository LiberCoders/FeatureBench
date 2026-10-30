"""Async Docker CLI runtime with no shell interpolation and bounded output."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

from pydantic import Field

from codeworld.config import DockerSettings
from codeworld.errors import (
    DOCKER_HARNESS_FRAGMENT_BEGIN,
    DOCKER_HARNESS_FRAGMENT_END,
    DOCKER_HARNESS_STEP_MARKER,
    CommandTimeoutError,
    ContractError,
    DockerBuildError,
    DockerBuildOrigin,
    DockerBuildStep,
    InfrastructureError,
    SourceMaterializationError,
)
from codeworld.models import RuntimeSpec, StrictModel
from codeworld.sources import SourceAttemptError, SourceManager


class CommandResult(StrictModel):
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = Field(ge=0)

    @property
    def combined_output(self) -> str:
        return self.stdout + self.stderr


_DOCKERFILE_FROM = re.compile(r"^\s*FROM\s+(?P<body>.+?)\s*$", re.IGNORECASE)
_DOCKERFILE_SYNTAX = re.compile(r"^\s*#\s*syntax\s*=\s*(?P<image>[^\s]+)", re.IGNORECASE)
_BUILDKIT_STEP = re.compile(
    r"(?:^|\n)#(?P<step_id>\d+)\s+\[(?P<stage>[^\]]+)\]\s+(?P<instruction>[^\n]+)",
    re.IGNORECASE,
)
_BUILDKIT_ERROR = re.compile(r"(?:^|\n)#(?P<step_id>\d+)\s+ERROR(?::|\s|$)", re.IGNORECASE)
_BUILDKIT_PROGRESS = re.compile(
    r"^#(?P<step_id>\d+)\s+\[(?P<stage>[^\]]+)\]\s+(?P<instruction>.+)$",
    re.IGNORECASE,
)
_DOCKERFILE_PARSE_LINE = re.compile(
    r"(?:Dockerfile|[^\s:]*\.Dockerfile):(?P<location>\d+)|"
    r"dockerfile parse error on line (?P<message>\d+)",
    re.IGNORECASE,
)


def _source_owned_parse_failure(failure: DockerBuildError, dockerfile: str) -> bool:
    """Return whether a Docker parser failure points inside a Source-owned fragment."""

    detail = failure.process_output_tail
    if "dockerfile parse error" not in detail.casefold():
        return False
    matches = tuple(_DOCKERFILE_PARSE_LINE.finditer(detail))
    if not matches:
        return False
    match = matches[-1]
    line_number = int(match.group("location") or match.group("message"))
    source_owned = False
    for index, line in enumerate(dockerfile.splitlines(), start=1):
        marker = line.strip()
        if index == line_number:
            return source_owned or marker in (
                DOCKER_HARNESS_FRAGMENT_BEGIN,
                DOCKER_HARNESS_FRAGMENT_END,
            )
        if marker == DOCKER_HARNESS_FRAGMENT_BEGIN:
            source_owned = True
        elif marker == DOCKER_HARNESS_FRAGMENT_END:
            source_owned = False
    return False


_RESOURCE_ID = re.compile(r"^(?:sha256:)?[0-9a-f]{12,64}$")
_MANAGED_LABEL = "io.featurebench.managed"
_SCOPE_LABEL = "io.featurebench.scope"
_RUN_LABEL = "io.featurebench.run-id"
_KIND_LABEL = "io.featurebench.resource-kind"


@dataclass(frozen=True, slots=True)
class DockerBuildProgress:
    """One normalized BuildKit step observed at the Docker adapter boundary."""

    step_id: str
    stage: str
    instruction: str


DockerBuildProgressCallback = Callable[[DockerBuildProgress], Awaitable[None]]


def _dockerfile_images(source: str) -> tuple[str, ...]:
    images: list[str] = []
    stage_aliases: set[str] = set()
    for line in source.splitlines():
        syntax = _DOCKERFILE_SYNTAX.match(line)
        if syntax is not None:
            images.append(syntax.group("image"))
            continue
        match = _DOCKERFILE_FROM.match(line)
        if match is None:
            continue
        tokens = match.group("body").split()
        index = 0
        while index < len(tokens) and tokens[index].startswith("--"):
            index += 1
        if index >= len(tokens):
            continue
        image = tokens[index]
        if image not in stage_aliases and not image.startswith("$") and image != "scratch":
            images.append(image)
        for token_index, token in enumerate(tokens[:-1]):
            if token.lower() == "as":
                stage_aliases.add(tokens[token_index + 1])
                break
    return tuple(dict.fromkeys(images))


def _build_step_origin(stage: str, instruction: str) -> DockerBuildOrigin:
    """Derive ownership from explicit rendered provenance, never failure text."""
    normalized_stage = stage.casefold()
    if normalized_stage.startswith("auth"):
        return DockerBuildOrigin.PLATFORM
    if normalized_stage.startswith("featurebench_managed_base"):
        return DockerBuildOrigin.HARNESS
    if DOCKER_HARNESS_STEP_MARKER in instruction:
        return DockerBuildOrigin.HARNESS
    if normalized_stage.startswith("internal"):
        return DockerBuildOrigin.UNKNOWN
    return DockerBuildOrigin.ARTIFACT


def _step_output(output: str, step_id: str) -> str:
    return "\n".join(
        line for line in output.splitlines() if re.match(rf"^#{re.escape(step_id)}(?:\s|$)", line)
    )


def _failed_buildkit_step(
    output: str,
    *,
    observed: Sequence[DockerBuildProgress] = (),
) -> DockerBuildStep | None:
    """Parse the decisive BuildKit step once at the Docker process boundary."""
    steps = tuple(_BUILDKIT_STEP.finditer(output))
    failed_ids = tuple(match.group("step_id") for match in _BUILDKIT_ERROR.finditer(output))
    observed_by_id = {item.step_id: item for item in observed}
    selected = None
    if failed_ids:
        selected = next(
            (match for match in reversed(steps) if match.group("step_id") == failed_ids[-1]),
            None,
        )
        if selected is None and (fallback := observed_by_id.get(failed_ids[-1])) is not None:
            return DockerBuildStep(
                step_id=fallback.step_id,
                stage=fallback.stage,
                instruction=fallback.instruction,
                output=_step_output(output, fallback.step_id),
                origin=_build_step_origin(fallback.stage, fallback.instruction),
            )
    if selected is None:
        if steps:
            selected = steps[-1]
        elif observed:
            fallback = observed[-1]
            return DockerBuildStep(
                step_id=fallback.step_id,
                stage=fallback.stage,
                instruction=fallback.instruction,
                output=_step_output(output, fallback.step_id),
                origin=_build_step_origin(fallback.stage, fallback.instruction),
            )
        else:
            return None
    step_id = selected.group("step_id")
    step_output = _step_output(output, step_id)
    stage = selected.group("stage")
    instruction = selected.group("instruction")
    return DockerBuildStep(
        step_id=step_id,
        stage=stage,
        instruction=instruction,
        output=step_output,
        origin=_build_step_origin(stage, instruction),
    )


def _docker_build_error(
    *,
    argv: tuple[str, ...],
    stdout: str,
    stderr: str,
    exit_code: int | None = None,
    duration_seconds: float | None = None,
    timeout_seconds: float | None = None,
    observed_progress: Sequence[DockerBuildProgress] = (),
) -> DockerBuildError:
    output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    return DockerBuildError(
        argv=argv,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration_seconds,
        timeout_seconds=timeout_seconds,
        failed_step=_failed_buildkit_step(output, observed=observed_progress),
    )


def _bind_mount(host_path: str | Path, container_path: str) -> tuple[Path, PurePosixPath]:
    source = Path(host_path).resolve()
    target = PurePosixPath(container_path)
    if not source.exists():
        raise ValueError(f"bind-mount source does not exist: {source}")
    if not target.is_absolute() or ".." in target.parts or str(target) == "/":
        raise ValueError("bind-mount target must be an absolute non-root container path")
    if "," in str(source) or "," in str(target):
        raise ValueError("bind-mount paths may not contain commas")
    return source, target


class DockerRuntime:
    def __init__(
        self,
        settings: DockerSettings | None = None,
        *,
        sources: SourceManager | None = None,
        run_id: str | None = None,
        resource_scope: str | None = None,
    ) -> None:
        if (run_id is None) != (resource_scope is None):
            raise ValueError("run_id and resource_scope must be configured together")
        if run_id is not None and (not run_id or not resource_scope):
            raise ValueError("Docker resource ownership values must not be empty")
        self.settings = settings or DockerSettings()
        self.sources = sources
        self.run_id = run_id
        self.resource_scope = resource_scope
        self._prepare_lock = asyncio.Lock()
        self._prepared_images: dict[tuple[str, bool], str] = {}

    @staticmethod
    def scope_for_workspace(workspace: str | Path) -> str:
        """Return a non-reversible owner-and-workspace identity for Docker labels."""
        owner = str(os.getuid()) if hasattr(os, "getuid") else "unknown"
        identity = f"{owner}\0{Path(workspace).resolve()}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def _resource_labels(self, kind: str) -> dict[str, str]:
        if self.run_id is None or self.resource_scope is None:
            return {}
        return {
            _MANAGED_LABEL: "true",
            _SCOPE_LABEL: self.resource_scope,
            _RUN_LABEL: self.run_id,
            _KIND_LABEL: kind,
        }

    def _label_arguments(self, kind: str) -> tuple[str, ...]:
        return tuple(
            item
            for key, value in sorted(self._resource_labels(kind).items())
            for item in ("--label", f"{key}={value}")
        )

    def _label_filters(self, kind: str) -> tuple[str, ...]:
        return tuple(
            item
            for key, value in sorted(self._resource_labels(kind).items())
            for item in ("--filter", f"label={key}={value}")
        )

    async def _invoke(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        input_bytes: bytes | None = None,
        max_output_chars: int | None = None,
        check: bool = True,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> CommandResult:
        argv = (self.settings.binary, *arguments)
        loop = asyncio.get_running_loop()
        started = loop.time()
        max_chars = max_output_chars or self.settings.max_output_chars
        max_bytes = max(4096, max_chars + 4096)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE
                if input_bytes is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise InfrastructureError(f"cannot start Docker CLI: {exc}") from exc
        deadline = timeout_seconds or self.settings.command_timeout_seconds

        stdout_tail = bytearray()
        stderr_tail = bytearray()

        async def drain_tail(stream: asyncio.StreamReader, tail: bytearray) -> None:
            pending = bytearray()
            while True:
                chunk = await stream.read(65_536)
                if not chunk:
                    if pending and on_output is not None:
                        # Progress projection must never stop draining a subprocess pipe.
                        with suppress(Exception):
                            await on_output(pending.decode("utf-8", errors="replace").rstrip("\r"))
                    return
                tail.extend(chunk)
                if len(tail) > max_bytes:
                    del tail[: len(tail) - max_bytes]
                if on_output is None:
                    continue
                pending.extend(chunk)
                while (newline := pending.find(b"\n")) >= 0:
                    raw_line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    # The command result remains authoritative even if a best-effort
                    # progress consumer cannot accept an update.
                    with suppress(Exception):
                        await on_output(raw_line.decode("utf-8", errors="replace").rstrip("\r"))

        async def communicate() -> tuple[bytes, bytes]:
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_task = asyncio.create_task(drain_tail(process.stdout, stdout_tail))
            stderr_task = asyncio.create_task(drain_tail(process.stderr, stderr_tail))
            try:
                if input_bytes is not None:
                    assert process.stdin is not None
                    try:
                        process.stdin.write(input_bytes)
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        process.stdin.close()
                        with suppress(BrokenPipeError, ConnectionResetError):
                            await process.stdin.wait_closed()
                await process.wait()
                await stdout_task
                await stderr_task
                return bytes(stdout_tail), bytes(stderr_tail)
            finally:
                for task in (stdout_task, stderr_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        communication = asyncio.create_task(communicate())
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), deadline)
        except TimeoutError as exc:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            stdout, stderr = await communication
            decoded_stdout = stdout.decode("utf-8", errors="replace")[-max_chars:]
            decoded_stderr = stderr.decode("utf-8", errors="replace")[-max_chars:]
            output = (decoded_stderr.strip() or decoded_stdout.strip())[-max_chars:]
            raise CommandTimeoutError(
                f"command timed out after {deadline}s: {' '.join(argv)}"
                + (f"\n{output}" if output else ""),
                argv=argv,
                stdout=decoded_stdout,
                stderr=decoded_stderr,
                timeout_seconds=deadline,
            ) from exc
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            await asyncio.gather(communication, return_exceptions=True)
            raise
        elapsed = loop.time() - started
        result = CommandResult(
            argv=argv,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace")[-max_chars:],
            stderr=stderr.decode("utf-8", errors="replace")[-max_chars:],
            duration_seconds=elapsed,
        )
        if check and result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise InfrastructureError(
                f"Docker command failed ({result.exit_code}): {' '.join(argv)}\n{detail}"
            )
        return result

    async def check_available(self) -> None:
        await self.invoke(("version", "--format", "{{.Server.Version}}"), timeout_seconds=20)

    async def resolve_image(self, image: str) -> str:
        result = await self.invoke(
            (
                "image",
                "inspect",
                image,
                "--format",
                "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
            )
        )
        reference = result.stdout.strip()
        if "@sha256:" not in reference and not reference.startswith("sha256:"):
            raise ContractError(f"image has no immutable digest: {image}")
        return reference

    async def prepare_image(self, image: str, *, pull: bool) -> str:
        key = (image, pull)
        async with self._prepare_lock:
            prepared = self._prepared_images.get(key)
            if prepared is not None:
                return prepared
            if pull:

                async def pull_source(candidate: str) -> None:
                    try:
                        result = await self.invoke(
                            ("pull", candidate),
                            timeout_seconds=self.settings.pull_timeout_seconds,
                            check=False,
                        )
                    except CommandTimeoutError as exc:
                        raise SourceAttemptError(str(exc)) from exc
                    if result.exit_code != 0:
                        raise SourceAttemptError(result.stderr.strip() or result.stdout.strip())

                async def restore_tag(candidate: str, canonical: str) -> None:
                    await self.invoke(("tag", candidate, canonical), timeout_seconds=60)

                handled = (
                    await self.sources.pull_image(
                        image,
                        pull=pull_source,
                        tag=restore_tag,
                    )
                    if self.sources is not None
                    else False
                )
                if not handled:
                    await self.invoke(
                        ("pull", image),
                        timeout_seconds=self.settings.pull_timeout_seconds,
                    )
            prepared = await self.resolve_image(image)
            self._prepared_images[key] = prepared
            return prepared

    async def build(
        self,
        context: str | Path,
        *,
        tag: str,
        dockerfile: str | Path = "Dockerfile",
        build_args: dict[str, str] | None = None,
        pull: bool = True,
        progress: DockerBuildProgressCallback | None = None,
    ) -> str:
        context_path = Path(context).resolve()  # noqa: ASYNC240 - validate before subprocess work
        dockerfile_path = Path(dockerfile)
        if not dockerfile_path.is_absolute():
            dockerfile_path = context_path / dockerfile_path
        dockerfile_path = dockerfile_path.resolve()
        if not dockerfile_path.is_relative_to(context_path):
            raise ValueError("Dockerfile must be inside the build context")
        effective_dockerfile = dockerfile_path
        temporary_dockerfile: Path | None = None
        explicit_build_args = dict(build_args or {})
        sources = self.sources
        using_sources = sources is not None and sources.settings.enabled
        original: str | None = None
        if using_sources:
            assert sources is not None
            original = dockerfile_path.read_text(encoding="utf-8")
            if pull:
                for base_image in _dockerfile_images(original):
                    await self.prepare_image(base_image, pull=True)
            with tempfile.NamedTemporaryFile(
                prefix="featurebench-sources-",
                suffix=".Dockerfile",
                delete=False,
            ) as temporary:
                temporary_dockerfile = Path(temporary.name)
            effective_dockerfile = temporary_dockerfile
        try:
            return await self._build_attempts(
                context_path=context_path,
                tag=tag,
                effective_dockerfile=effective_dockerfile,
                explicit_build_args=explicit_build_args,
                pull=pull,
                sources=sources,
                using_sources=using_sources,
                original=original,
                progress=progress,
            )
        finally:
            if temporary_dockerfile is not None:
                temporary_dockerfile.unlink(missing_ok=True)

    async def _build_attempts(
        self,
        *,
        context_path: Path,
        tag: str,
        effective_dockerfile: Path,
        explicit_build_args: dict[str, str],
        pull: bool,
        sources: SourceManager | None,
        using_sources: bool,
        original: str | None,
        progress: DockerBuildProgressCallback | None,
    ) -> str:
        attempted: dict[str, set[str]] = {}
        recovery_services: set[str] = set()
        last_progress: DockerBuildProgress | None = None
        observed_progress: dict[str, DockerBuildProgress] = {}

        async def observe_build_output(line: str) -> None:
            nonlocal last_progress
            if (match := _BUILDKIT_PROGRESS.match(line)) is None:
                return
            observed = DockerBuildProgress(
                step_id=match.group("step_id"),
                stage=match.group("stage").strip(),
                instruction=match.group("instruction").strip(),
            )
            if observed == last_progress:
                return
            last_progress = observed
            observed_progress[observed.step_id] = observed
            if progress is not None:
                await progress(observed)

        while True:
            last_progress = None
            observed_progress.clear()
            plan = (
                await sources.build_plan(excluded=attempted) if using_sources and sources else None
            )
            effective_build_args = dict(plan.arguments) if plan is not None else {}
            effective_build_args.update(explicit_build_args)
            managed_choices = plan.managed_choices(explicit_build_args) if plan is not None else {}
            rewritten: str | None = None
            if original is not None:
                assert sources is not None and plan is not None
                rewritten = sources.materialize_build_sources(
                    original,
                    plan,
                    build_arguments=effective_build_args,
                )
                await asyncio.to_thread(
                    effective_dockerfile.write_text,
                    rewritten,
                    encoding="utf-8",
                )
            arguments = [
                "build",
                "--progress",
                "plain",
                "--tag",
                tag,
                "--file",
                str(effective_dockerfile),
                "--network",
                self.settings.build_network,
                "--ulimit",
                "nofile=65535:65535",
            ]
            arguments.extend(self._label_arguments("candidate-image"))
            if pull and not using_sources:
                arguments.append("--pull")
            for key, value in sorted(effective_build_args.items()):
                arguments.extend(("--build-arg", f"{key}={value}"))
            arguments.append(str(context_path))
            started_at = datetime.now(UTC)
            started_monotonic = time.monotonic()

            try:
                result = await self.invoke(
                    arguments,
                    timeout_seconds=self.settings.build_timeout_seconds,
                    check=False,
                    on_output=observe_build_output,
                )
            except CommandTimeoutError as exc:
                failure = _docker_build_error(
                    argv=exc.argv,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    timeout_seconds=exc.timeout_seconds,
                    observed_progress=tuple(observed_progress.values()),
                )
            else:
                if result.exit_code == 0:
                    if sources is not None:
                        for service in recovery_services:
                            choice = managed_choices.get(service)
                            if choice is not None:
                                await sources.record_success(
                                    choice,
                                    started_at=started_at,
                                    started_monotonic=started_monotonic,
                                )
                    break
                failure = _docker_build_error(
                    argv=result.argv,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=result.duration_seconds,
                    observed_progress=tuple(observed_progress.values()),
                )
            if rewritten is not None and _source_owned_parse_failure(failure, rewritten):
                raise SourceMaterializationError(
                    "Source materialization produced an invalid Dockerfile\n"
                    f"{failure.decisive_output[-5000:]}"
                ) from failure
            if failure.failed_step is None:
                # Unknown build failures remain eligible for artifact repair. Stop only
                # when an independent probe positively proves the daemon is unavailable.
                await self.check_available()
            failed_service = (
                sources.build_failure_service(failure, managed_choices)
                if sources is not None
                else None
            )
            if failed_service is None:
                raise failure
            choice = managed_choices[failed_service]
            used = attempted.setdefault(failed_service, set())
            used.add(choice.endpoint.name)
            recovery_services.add(failed_service)
            assert sources is not None
            await sources.record_failure(
                choice,
                SourceAttemptError(str(failure)),
                started_at=started_at,
                started_monotonic=started_monotonic,
            )
            if not await sources.attempt_choices(
                failed_service,
                excluded=frozenset(used),
            ):
                raise InfrastructureError(
                    f"all configured sources failed for {failed_service}\n"
                    f"{failure.decisive_output[-5000:]}"
                )
        return await self.resolve_image(tag)

    async def remove_image(self, image: str, *, force: bool = False) -> bool:
        arguments = ("image", "rm", "--force", image) if force else ("image", "rm", image)
        result = await self.invoke(arguments, timeout_seconds=60, check=False)
        return result.exit_code == 0

    async def image_exists(self, image: str) -> bool:
        """Return whether one exact local Docker image reference still resolves."""

        result = await self.invoke(
            ("image", "inspect", image, "--format", "{{.Id}}"),
            timeout_seconds=30,
            check=False,
        )
        return result.exit_code == 0

    async def _has_resource_labels(
        self,
        resource_type: str,
        resource_id: str,
        *,
        kind: str,
    ) -> bool:
        result = await self.invoke(
            (
                resource_type,
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                resource_id,
            ),
            timeout_seconds=30,
            check=False,
        )
        if result.exit_code != 0:
            return False
        try:
            labels = cast(object, json.loads(result.stdout.strip()))
        except (TypeError, json.JSONDecodeError):
            return False
        expected = self._resource_labels(kind)
        return (
            bool(expected)
            and isinstance(labels, dict)
            and all(
                cast(dict[object, object], labels).get(key) == value
                for key, value in expected.items()
            )
        )

    async def _listed_resources(self, resource_type: str, *, kind: str) -> tuple[str, ...]:
        labels = self._resource_labels(kind)
        if not labels:
            return ()
        result = await self.invoke(
            (
                resource_type,
                "ls",
                "--quiet",
                "--no-trunc",
                *self._label_filters(kind),
            ),
            timeout_seconds=30,
            check=False,
        )
        if result.exit_code != 0:
            return ()
        return tuple(
            dict.fromkeys(
                resource_id
                for line in result.stdout.splitlines()
                if (resource_id := line.strip()) and _RESOURCE_ID.fullmatch(resource_id)
            )
        )

    async def cleanup_run_resources(
        self,
        *,
        protected_images: Collection[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Remove only transient resources carrying this runtime's exact ownership labels."""
        removed_containers: list[str] = []
        for container_id in await self._listed_resources("container", kind="transient-container"):
            if not await self._has_resource_labels(
                "container", container_id, kind="transient-container"
            ):
                continue
            result = await self.invoke(
                ("container", "rm", "--force", container_id),
                timeout_seconds=self.settings.command_timeout_seconds,
                check=False,
            )
            if result.exit_code == 0:
                removed_containers.append(container_id)

        removed_images: list[str] = []
        if protected_images is None:
            return tuple(removed_containers), ()
        protected = {reference.strip() for reference in protected_images if reference.strip()}
        for image_id in await self._listed_resources("image", kind="candidate-image"):
            if image_id in protected:
                continue
            if not await self._has_resource_labels("image", image_id, kind="candidate-image"):
                continue
            result = await self.invoke(
                ("image", "rm", image_id),
                timeout_seconds=60,
                check=False,
            )
            if result.exit_code == 0:
                removed_images.append(image_id)
        return tuple(removed_containers), tuple(removed_images)

    @asynccontextmanager
    async def session(
        self,
        image: str,
        *,
        runtime_spec: RuntimeSpec | None = None,
        read_only: bool = False,
        gpus: str | None = None,
        bind_mounts: Sequence[tuple[str | Path, str, bool]] = (),
    ) -> AsyncGenerator[ContainerSession, None]:
        name = f"featurebench-{uuid4().hex[:16]}"
        resources = runtime_spec.resources if runtime_spec is not None else None
        network = resources.network if resources is not None else self.settings.network
        cpus = resources.cpus if resources is not None else self.settings.cpus
        memory = resources.memory if resources is not None else self.settings.memory
        effective_gpus = gpus if gpus is not None else (resources.gpus if resources else None)
        arguments = ["create", "--name", name, "--init", "--network", network]
        arguments.extend(self._label_arguments("transient-container"))
        if cpus is not None:
            arguments.extend(("--cpus", str(cpus)))
        if memory is not None:
            arguments.extend(("--memory", memory))
        if effective_gpus is not None:
            arguments.extend(("--gpus", effective_gpus))
        for host_path, container_path, mount_read_only in bind_mounts:
            source, target = _bind_mount(host_path, container_path)
            mount = f"type=bind,source={source},target={target}"
            if mount_read_only:
                mount += ",readonly"
            arguments.extend(("--mount", mount))
        if read_only:
            arguments.append("--read-only")
        arguments.extend(("--entrypoint", "sleep", image, "infinity"))
        result = await self.invoke(arguments)
        container_id = result.stdout.strip()
        session = ContainerSession(self, container_id, runtime_spec=runtime_spec)
        try:
            await self.invoke(("start", container_id))
            yield session
        finally:
            # A first SIGINT cancels the body; a repeated SIGINT may arrive while
            # its async-context cleanup is removing the container.  Keep that
            # second cancellation from cancelling the Docker removal itself, then
            # propagate cancellation only after the bounded cleanup has finished.
            removal = asyncio.create_task(
                self.invoke(
                    ("rm", "--force", container_id),
                    timeout_seconds=self.settings.command_timeout_seconds,
                    check=False,
                )
            )
            try:
                await asyncio.shield(removal)
            except asyncio.CancelledError:
                with suppress(asyncio.CancelledError):
                    await removal
                raise

    async def invoke(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        input_bytes: bytes | None = None,
        max_output_chars: int | None = None,
        check: bool = True,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> CommandResult:
        """Run a Docker CLI operation without shell parsing."""
        return await self._invoke(
            arguments,
            timeout_seconds=timeout_seconds,
            input_bytes=input_bytes,
            max_output_chars=max_output_chars,
            check=check,
            on_output=on_output,
        )


class ContainerSession:
    def __init__(
        self,
        runtime: DockerRuntime,
        container_id: str,
        *,
        runtime_spec: RuntimeSpec | None = None,
    ) -> None:
        self.runtime = runtime
        self.container_id = container_id
        self.runtime_spec = runtime_spec

    def _workdir(self, workdir: str | None) -> str:
        if workdir is not None:
            return workdir
        if self.runtime_spec is None:
            return "/"
        return self.runtime_spec.workdir

    async def exec(
        self,
        command: Sequence[str],
        *,
        activated: bool = False,
        workdir: str | None = None,
        timeout_seconds: float | None = None,
        env: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
        max_output_chars: int | None = None,
        check: bool = False,
    ) -> CommandResult:
        if not command:
            raise ValueError("container command must not be empty")
        actual_command: tuple[str, ...]
        if activated:
            if self.runtime_spec is None:
                raise ContractError("cannot activate an environment without a runtime artifact")
            if input_bytes is not None:
                raise ValueError("activated commands cannot also consume standard input")
            actual_command = (
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'set -euo pipefail; source /dev/stdin; cd -- /testbed; exec "$@"',
                "featurebench-activate",
                *command,
            )
            input_bytes = self.runtime_spec.activation.encode("utf-8")
        else:
            actual_command = tuple(command)
        arguments = ["exec"]
        if input_bytes is not None:
            arguments.append("--interactive")
        arguments.extend(("--workdir", self._workdir(workdir)))
        for key, value in sorted((env or {}).items()):
            arguments.extend(("--env", f"{key}={value}"))
        arguments.append(self.container_id)
        arguments.extend(actual_command)
        return await self.runtime.invoke(
            arguments,
            timeout_seconds=timeout_seconds,
            input_bytes=input_bytes,
            max_output_chars=max_output_chars,
            check=check,
        )

    async def run_script(
        self,
        script: str,
        arguments: Sequence[str] = (),
        *,
        activated: bool = False,
        timeout_seconds: float | None = None,
        max_output_chars: int | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Stream a host artifact to Bash without copying it into the image or container."""
        if self.runtime_spec is None:
            raise ContractError("cannot run an environment artifact without a runtime spec")
        prelude = "set -euo pipefail\n"
        if activated:
            prelude += self.runtime_spec.activation + "\ncd -- /testbed\n"
        content = (prelude + script).encode("utf-8")
        return await self.exec(
            ("bash", "--noprofile", "--norc", "-s", "--", *arguments),
            workdir=self.runtime_spec.workdir,
            timeout_seconds=timeout_seconds,
            input_bytes=content,
            max_output_chars=max_output_chars,
            check=check,
        )

    def _resolve_workspace_path(self, path: str) -> str:
        if self.runtime_spec is None:
            raise ContractError("container runtime spec has not been provided")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("path must be relative to the image workspace")
        return str(PurePosixPath(self.runtime_spec.workdir) / relative)

    async def read_absolute_file(self, path: str, *, max_bytes: int = 1_000_000) -> bytes:
        absolute = PurePosixPath(path)
        if not absolute.is_absolute() or ".." in absolute.parts:
            raise ValueError("absolute container file path is required")
        result = await self.exec(
            ("head", "-c", str(max_bytes + 1), str(absolute)),
            workdir="/",
            max_output_chars=max_bytes + 1,
            check=False,
        )
        if result.exit_code != 0:
            raise ContractError(f"cannot read container file {path}: {result.combined_output}")
        content = result.stdout.encode("utf-8", errors="replace")
        if len(content) > max_bytes:
            raise ValueError(f"file exceeds {max_bytes} bytes: {path}")
        return content

    async def write_file(self, path: str, content: bytes) -> None:
        absolute = self._resolve_workspace_path(path)
        await self.write_absolute_file(absolute, content)

    async def write_absolute_file(self, path: str, content: bytes) -> None:
        """Write one absolute container path without exposing the host filesystem."""
        absolute_path = PurePosixPath(path)
        if not absolute_path.is_absolute() or ".." in absolute_path.parts:
            raise ValueError("absolute container file path is required")
        absolute = str(absolute_path)
        parent = str(PurePosixPath(absolute).parent)
        await self.exec(("mkdir", "-p", parent), workdir="/", check=True)
        await self.exec(("tee", absolute), workdir="/", input_bytes=content, check=True)

    async def git_diff(self, *arguments: str, max_bytes: int = 20_000_000) -> str:
        """Read one complete bounded patch without the command-output tail policy."""
        if max_bytes < 1:
            raise ValueError("git diff max_bytes must be positive")
        output_path = f"/tmp/featurebench-git-diff-{uuid4().hex}.patch"
        try:
            await self.exec(
                (
                    "git",
                    "diff",
                    "--no-ext-diff",
                    f"--output={output_path}",
                    *arguments,
                ),
                check=True,
            )
            size_result = await self.exec(
                ("stat", "-c", "%s", output_path),
                workdir="/",
                check=True,
            )
            try:
                size = int(size_result.stdout.strip())
            except ValueError as exc:
                raise ContractError(
                    f"cannot determine generated git diff size: {size_result.stdout!r}"
                ) from exc
            if size > max_bytes:
                raise ContractError(
                    f"generated git diff is {size} bytes and exceeds {max_bytes} bytes"
                )
            return (await self.read_absolute_file(output_path, max_bytes=max_bytes)).decode(
                "utf-8", errors="replace"
            )
        finally:
            await self.exec(("rm", "-f", output_path), workdir="/", check=False)

    async def clean_workspace(self) -> None:
        await self.exec(("git", "reset", "--hard", "HEAD"), check=True)
        await self.exec(("git", "clean", "-ffd"), check=True)

    async def seal_workspace(self) -> None:
        """Discard clean-image Git history and commit only the prepared public task state."""
        if self.runtime_spec is None:
            raise ContractError("container runtime spec has not been provided")
        git_directory = str(PurePosixPath(self.runtime_spec.workdir) / ".git")
        await self.exec(("rm", "-rf", git_directory), workdir="/", check=True)
        await self.exec(("git", "init"), check=True)
        await self.exec(("git", "config", "user.email", "codeworld@localhost"), check=True)
        await self.exec(("git", "config", "user.name", "FeatureBench"), check=True)
        await self.exec(("git", "add", "-A"), check=True)
        await self.exec(("git", "commit", "-m", "FeatureBench task baseline"), check=True)
