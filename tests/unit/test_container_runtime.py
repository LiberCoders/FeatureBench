from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import cast

import pytest

from codeworld.config import (
    DockerSettings,
    SourceEndpointSettings,
    SourceMetricsSettings,
    SourceServiceSettings,
    SourceSettings,
)
from codeworld.errors import (
    DOCKER_HARNESS_FRAGMENT_BEGIN,
    DOCKER_HARNESS_FRAGMENT_END,
    DOCKER_HARNESS_STEP_MARKER,
    CommandTimeoutError,
    DockerBuildError,
    DockerBuildOrigin,
    SourceMaterializationError,
)
from codeworld.models import RuntimeResources, RuntimeSpec
from codeworld.runtime.container import (
    CommandResult,
    ContainerSession,
    DockerBuildProgress,
    DockerRuntime,
    _failed_buildkit_step,  # pyright: ignore[reportPrivateUsage]
    _source_owned_parse_failure,  # pyright: ignore[reportPrivateUsage]
)
from codeworld.sources import SourceAttemptMetric, SourceManager


class FakeRuntime:
    def __init__(self) -> None:
        self.arguments: tuple[str, ...] = ()
        self.input_bytes: bytes | None = None

    async def invoke(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: float | None = None,
        input_bytes: bytes | None = None,
        max_output_chars: int | None = None,
        check: bool = True,
    ) -> CommandResult:
        del timeout_seconds, max_output_chars, check
        self.arguments = tuple(arguments)
        self.input_bytes = input_bytes
        return CommandResult(
            argv=("docker", *arguments),
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0,
        )


class RecordingDockerRuntime(DockerRuntime):
    def __init__(
        self,
        *,
        run_id: str | None = None,
        resource_scope: str | None = None,
    ) -> None:
        super().__init__(run_id=run_id, resource_scope=resource_scope)
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float | None] = []

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
        del input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        self.calls.append(call)
        self.timeouts.append(timeout_seconds)
        return CommandResult(
            argv=("docker", *call),
            exit_code=0,
            stdout="container-1\n" if call[0] == "create" else "",
            stderr="",
            duration_seconds=0,
        )


class CancellationRecordingDockerRuntime(RecordingDockerRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.removal_started = asyncio.Event()
        self.release_removal = asyncio.Event()
        self.removal_completed = False

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
        if tuple(arguments)[:2] == ("rm", "--force"):
            self.removal_started.set()
            await self.release_removal.wait()
            self.removal_completed = True
        return await super().invoke(
            arguments,
            timeout_seconds=timeout_seconds,
            input_bytes=input_bytes,
            max_output_chars=max_output_chars,
            check=check,
            on_output=on_output,
        )


class SourceBuildRuntime(DockerRuntime):
    def __init__(self, sources: SourceManager) -> None:
        super().__init__(sources=sources)
        self.calls: list[tuple[str, ...]] = []

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
        del timeout_seconds, input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        self.calls.append(call)
        failed = call[0] == "build" and any("mirrors.tuna" in item for item in call)
        return CommandResult(
            argv=("docker", *call),
            exit_code=1 if failed else 0,
            stdout="sha256:digest\n" if call[:2] == ("image", "inspect") else "",
            stderr=(
                (
                    "#8 [stage-1 3/6] RUN python -m pip install demo\n"
                    "#8 pip install: Could not resolve host mirrors.tuna.tsinghua.edu.cn\n"
                    "#8 ERROR: process exited with code 1"
                )
                if failed
                else ""
            ),
            duration_seconds=0,
        )


class SourceParseFailureRuntime(DockerRuntime):
    def __init__(self, sources: SourceManager) -> None:
        super().__init__(sources=sources)

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
        del timeout_seconds, input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        assert call[0] == "build"
        dockerfile = await asyncio.to_thread(
            Path(call[call.index("--file") + 1]).read_text,
            encoding="utf-8",
        )
        line_number = next(
            index
            for index, line in enumerate(dockerfile.splitlines(), start=1)
            if line == "ARG CONDA_CHANNEL_URL"
        )
        return CommandResult(
            argv=("docker", *call),
            exit_code=1,
            stdout="",
            stderr=(
                "#1 [internal] load build definition from Dockerfile\n"
                f"Dockerfile:{line_number}\n"
                f"ERROR: dockerfile parse error on line {line_number}: invalid ARG"
            ),
            duration_seconds=0,
        )


class ManagedResourceDockerRuntime(DockerRuntime):
    owned_container = "a" * 64
    foreign_container = "b" * 64
    protected_image = "sha256:" + "c" * 64
    orphan_image = "sha256:" + "d" * 64
    foreign_image = "sha256:" + "e" * 64

    def __init__(self) -> None:
        super().__init__(run_id="run-1", resource_scope="scope-1")
        self.calls: list[tuple[str, ...]] = []

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
        del timeout_seconds, input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        self.calls.append(call)
        stdout = ""
        if call[:2] == ("container", "ls"):
            stdout = f"{self.owned_container}\n{self.foreign_container}\n"
        elif call[:2] == ("image", "ls"):
            stdout = f"{self.protected_image}\n{self.orphan_image}\n{self.foreign_image}\n"
        elif len(call) >= 5 and call[1] == "inspect":
            resource_type = call[0]
            resource_id = call[-1]
            kind = "transient-container" if resource_type == "container" else "candidate-image"
            labels = self._resource_labels(kind)
            if resource_id in (self.foreign_container, self.foreign_image):
                labels = {**labels, "io.featurebench.run-id": "somebody-elses-run"}
            stdout = json.dumps(labels)
        return CommandResult(
            argv=("docker", *call),
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0,
        )


class ManagedBuildDockerRuntime(DockerRuntime):
    def __init__(self) -> None:
        super().__init__(run_id="run-1", resource_scope="scope-1")
        self.calls: list[tuple[str, ...]] = []

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
        del timeout_seconds, input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        self.calls.append(call)
        stdout = "sha256:" + "f" * 64 if call[:2] == ("image", "inspect") else ""
        return CommandResult(
            argv=("docker", *call),
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0,
        )


class PullTimeoutRuntime(DockerRuntime):
    def __init__(self, sources: SourceManager) -> None:
        super().__init__(sources=sources)
        self.calls: list[tuple[str, ...]] = []

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
        del timeout_seconds, input_bytes, max_output_chars, check, on_output
        call = tuple(arguments)
        self.calls.append(call)
        if call[0] == "pull" and call[1].startswith("docker.example/"):
            raise CommandTimeoutError(
                f"command timed out while pulling {call[1]}",
                argv=("docker", *call),
                timeout_seconds=1,
            )
        return CommandResult(
            argv=("docker", *call),
            exit_code=0,
            stdout=("python@sha256:" + "a" * 64 + "\n" if call[:2] == ("image", "inspect") else ""),
            stderr="",
            duration_seconds=0,
        )


class SourceBuildTimeoutRuntime(SourceBuildRuntime):
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
        call = tuple(arguments)
        if call[0] == "build" and any("mirrors.tuna" in item for item in call):
            self.calls.append(call)
            stderr = "#8 [stage-1 3/6] RUN python -m pip install --no-cache-dir -e .\n"
            raise CommandTimeoutError(
                f"command timed out after {timeout_seconds}s: docker build\n{stderr}",
                argv=("docker", *call),
                stderr=stderr,
                timeout_seconds=timeout_seconds,
            )
        return await super().invoke(
            arguments,
            timeout_seconds=timeout_seconds,
            input_bytes=input_bytes,
            max_output_chars=max_output_chars,
            check=check,
            on_output=on_output,
        )


class CondaSourceBuildRuntime(SourceBuildRuntime):
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
        call = tuple(arguments)
        if call[0] == "build" and any("mirrors.tuna" in item for item in call):
            self.calls.append(call)
            return CommandResult(
                argv=("docker", *call),
                exit_code=1,
                stdout="",
                stderr=(
                    "#11 [featurebench_managed_base 6/9] RUN conda install "
                    '--channel "$CONDA_CHANNEL_URL" python=3.13\n'
                    "#11 CondaHTTPError: HTTP 000 CONNECTION FAILED for url "
                    "<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/"
                    "linux-64/repodata.json>\n"
                    "#11 ERROR: process exited with code 1"
                ),
                duration_seconds=1,
            )
        return await super().invoke(
            arguments,
            timeout_seconds=timeout_seconds,
            input_bytes=input_bytes,
            max_output_chars=max_output_chars,
            check=check,
            on_output=on_output,
        )


def runtime_spec() -> RuntimeSpec:
    return RuntimeSpec(
        activation="#!/usr/bin/env bash\nexport TOOLCHAIN=ready\ncd /testbed\n",
        runner=(
            "#!/usr/bin/env bash\n"
            "# Parse one Runner V1 request and write the corresponding structured response.\n"
            "printf '%s\\n' \"$@\" >/dev/null\n"
            "echo runner-placeholder-with-sufficient-contract-source-length >/dev/null\n"
        ),
    )


def resolved(path: Path) -> Path:
    return path.resolve()


def source_settings(tmp_path: Path) -> SourceSettings:
    return SourceSettings(
        enabled=True,
        state_path=tmp_path / "source-state.json",
        metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
        services={
            "pypi": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(
                        name="tuna",
                        url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                    ),
                    SourceEndpointSettings(
                        name="official",
                        url="https://pypi.org/simple",
                        canonical=True,
                    ),
                ),
            )
        },
    )


def test_buildkit_failure_parser_returns_only_the_explicit_failed_step() -> None:
    step = _failed_buildkit_step(
        "#5 [featurebench_managed_base 2/4] RUN apt-get update\n"
        "#5 DONE 1.2s\n"
        "#17 [stage-1 4/8] RUN python -m pip install 'browserforge>=1.2.5'\n"
        "#17 ERROR: No matching distribution found for browserforge>=1.2.5\n"
        "#18 [stage-1 5/8] RUN echo unrelated\n"
        "#17 ERROR: process exited with code 1"
    )

    assert step is not None
    assert step.step_id == "17"
    assert step.stage == "stage-1 4/8"
    assert step.origin is DockerBuildOrigin.ARTIFACT
    assert "browserforge" in step.output
    assert "apt-get" not in step.output
    assert "unrelated" not in step.output


def test_docker_build_error_keeps_step_provenance_and_unprefixed_process_error() -> None:
    output = (
        "#1 [internal] load build definition from Dockerfile\n"
        "#1 transferring dockerfile: 16.44kB done\n"
        "#1 DONE 0.1s\n"
        "Dockerfile:225\n"
        "ERROR: dockerfile parse error on line 225: unknown instruction: PYTHON3\n"
    )
    step = _failed_buildkit_step(output)
    assert step is not None
    assert "unknown instruction" not in step.detail

    error = DockerBuildError(exit_code=1, stderr=output, failed_step=step)

    assert "BuildKit failed step" in error.decisive_output
    assert "load build definition" in error.decisive_output
    assert "Docker process output tail" in error.decisive_output
    assert "unknown instruction: PYTHON3" in error.decisive_output
    assert "unknown instruction: PYTHON3" in str(error)


def test_dockerfile_parse_failure_is_owned_only_inside_source_fragment() -> None:
    dockerfile = (
        "FROM scratch\n"
        f"{DOCKER_HARNESS_FRAGMENT_BEGIN}\n"
        f"RUN {DOCKER_HARNESS_STEP_MARKER}; printf '%s\n"
        "' value\n"
        f"{DOCKER_HARNESS_FRAGMENT_END}\n"
        "BROKEN artifact instruction\n"
    )
    source_line = dockerfile.splitlines().index("' value") + 1
    artifact_line = dockerfile.splitlines().index("BROKEN artifact instruction") + 1

    source_failure = DockerBuildError(
        exit_code=1,
        stderr=(
            f"Dockerfile:{source_line}\n"
            f"ERROR: dockerfile parse error on line {source_line}: unknown instruction: '"
        ),
    )
    artifact_failure = DockerBuildError(
        exit_code=1,
        stderr=(
            f"Dockerfile:{artifact_line}\n"
            f"ERROR: dockerfile parse error on line {artifact_line}: unknown instruction: BROKEN"
        ),
    )

    assert _source_owned_parse_failure(source_failure, dockerfile)
    assert not _source_owned_parse_failure(artifact_failure, dockerfile)


@pytest.mark.parametrize(
    ("stage", "instruction", "origin"),
    [
        ("featurebench_managed_base 2/4", "RUN apt-get update", DockerBuildOrigin.HARNESS),
        (
            "stage-1 2/8",
            f"RUN {DOCKER_HARNESS_STEP_MARKER}; git clone example /testbed",
            DockerBuildOrigin.HARNESS,
        ),
        ("auth", "sharing credentials for registry", DockerBuildOrigin.PLATFORM),
        ("internal", "load build definition from Dockerfile", DockerBuildOrigin.UNKNOWN),
        ("stage-1 4/8", "RUN make test", DockerBuildOrigin.ARTIFACT),
    ],
)
def test_buildkit_step_origin_comes_from_rendered_provenance(
    stage: str,
    instruction: str,
    origin: DockerBuildOrigin,
) -> None:
    step = _failed_buildkit_step(f"#8 [{stage}] {instruction}\n#8 ERROR: failed")

    assert step is not None
    assert step.origin is origin


@pytest.mark.asyncio
async def test_docker_build_retains_live_step_when_bounded_log_loses_its_header(
    tmp_path: Path,
) -> None:
    class TruncatedFailureRuntime(DockerRuntime):
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
            del timeout_seconds, input_bytes, max_output_chars, check
            call = tuple(arguments)
            if call[0] == "build":
                assert on_output is not None
                await on_output("#21 [stage-1 8/13] RUN python -m pip install -e .")
                await on_output("#22 [stage-1 9/13] RUN echo unrelated parallel step")
                return CommandResult(
                    argv=("docker", *call),
                    exit_code=1,
                    stdout="",
                    stderr="#21 error: can't find Rust compiler\n#21 ERROR: exit code 1",
                    duration_seconds=1,
                )
            raise AssertionError(f"unexpected Docker call: {call}")

    (tmp_path / "Dockerfile").write_text("FROM scratch\nRUN true\n", encoding="utf-8")

    with pytest.raises(DockerBuildError) as caught:
        await TruncatedFailureRuntime().build(tmp_path, tag="example:failed", pull=False)

    step = caught.value.failed_step
    assert step is not None
    assert step.step_id == "21"
    assert step.instruction == "RUN python -m pip install -e ."
    assert step.origin is DockerBuildOrigin.ARTIFACT
    assert "Rust compiler" in step.output


def test_docker_build_error_does_not_render_raw_build_arguments() -> None:
    error = DockerBuildError(
        argv=("docker", "build", "--build-arg", "PRIVATE_VALUE=secret"),
        exit_code=1,
        stderr="failed without BuildKit step output",
    )

    assert "failed without BuildKit step output" in str(error)
    assert "PRIVATE_VALUE" not in str(error)
    assert "secret" not in str(error)


@pytest.mark.asyncio
async def test_activated_command_streams_artifact_without_copying_it() -> None:
    fake = FakeRuntime()
    session = ContainerSession(
        cast(DockerRuntime, fake), "container-1", runtime_spec=runtime_spec()
    )
    await session.exec(("git", "status", "--short"), activated=True)
    assert fake.arguments[:4] == ("exec", "--interactive", "--workdir", "/testbed")
    assert fake.arguments[-3:] == ("git", "status", "--short")
    assert fake.input_bytes == runtime_spec().activation.encode()
    assert all("/opt/featurebench" not in argument for argument in fake.arguments)


@pytest.mark.asyncio
async def test_runner_artifact_is_streamed_as_a_script() -> None:
    fake = FakeRuntime()
    spec = runtime_spec()
    session = ContainerSession(cast(DockerRuntime, fake), "container-1", runtime_spec=spec)
    await session.run_script(
        spec.runner,
        ("--request", "/tmp/in", "--response", "/tmp/out"),
        activated=True,
    )
    assert fake.arguments[-9:] == (
        "bash",
        "--noprofile",
        "--norc",
        "-s",
        "--",
        "--request",
        "/tmp/in",
        "--response",
        "/tmp/out",
    )
    streamed = (fake.input_bytes or b"").decode()
    assert spec.activation in streamed
    assert spec.runner in streamed


@pytest.mark.asyncio
async def test_session_scopes_gpu_and_read_only_bind_to_the_requested_container(
    tmp_path: Path,
) -> None:
    runtime = RecordingDockerRuntime()
    mount_source = resolved(tmp_path)
    async with runtime.session(
        "planner@sha256:digest",
        read_only=True,
        gpus="device=0",
        bind_mounts=((tmp_path, "/testbed", True),),
    ):
        pass
    create = runtime.calls[0]
    assert create[:2] == ("create", "--name")
    assert create[create.index("--gpus") : create.index("--gpus") + 2] == (
        "--gpus",
        "device=0",
    )
    mount = create[create.index("--mount") + 1]
    assert f"source={mount_source}" in mount
    assert "target=/testbed" in mount
    assert mount.endswith(",readonly")
    assert "--read-only" in create
    assert runtime.calls[-1][:3] == ("rm", "--force", "container-1")
    assert runtime.timeouts[-1] == runtime.settings.command_timeout_seconds


@pytest.mark.asyncio
async def test_session_applies_self_contained_runtime_resources_by_default() -> None:
    runtime = RecordingDockerRuntime()
    spec = runtime_spec().model_copy(
        update={
            "resources": RuntimeResources(
                network="host",
                cpus=2.5,
                memory="3g",
                gpus="device=1",
            )
        }
    )
    async with runtime.session("task@sha256:digest", runtime_spec=spec):
        pass
    create = runtime.calls[0]
    assert create[create.index("--network") : create.index("--network") + 2] == (
        "--network",
        "host",
    )
    assert create[create.index("--cpus") : create.index("--cpus") + 2] == (
        "--cpus",
        "2.5",
    )
    assert create[create.index("--memory") : create.index("--memory") + 2] == (
        "--memory",
        "3g",
    )
    assert create[create.index("--gpus") : create.index("--gpus") + 2] == (
        "--gpus",
        "device=1",
    )


@pytest.mark.asyncio
async def test_session_finishes_container_removal_during_repeated_cancellation() -> None:
    runtime = CancellationRecordingDockerRuntime()
    body_started = asyncio.Event()

    async def use_session() -> None:
        async with runtime.session("task@sha256:digest"):
            body_started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use_session())
    await body_started.wait()
    task.cancel()
    await runtime.removal_started.wait()
    task.cancel()
    runtime.release_removal.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.removal_completed is True
    assert runtime.calls[-1][:3] == ("rm", "--force", "container-1")


@pytest.mark.asyncio
async def test_managed_sessions_receive_exact_run_ownership_labels() -> None:
    runtime = RecordingDockerRuntime(run_id="run-1", resource_scope="scope-1")

    async with runtime.session("task@sha256:digest"):
        pass

    create = runtime.calls[0]
    labels = {
        create[index + 1] for index, argument in enumerate(create[:-1]) if argument == "--label"
    }
    assert labels == {
        "io.featurebench.managed=true",
        "io.featurebench.resource-kind=transient-container",
        "io.featurebench.run-id=run-1",
        "io.featurebench.scope=scope-1",
    }


@pytest.mark.asyncio
async def test_cleanup_removes_only_exactly_owned_unprotected_resources() -> None:
    runtime = ManagedResourceDockerRuntime()

    removed = await runtime.cleanup_run_resources(protected_images=(runtime.protected_image,))

    assert removed == ((runtime.owned_container,), (runtime.orphan_image,))
    removals = [
        call
        for call in runtime.calls
        if call[:3] in (("container", "rm", "--force"), ("image", "rm", runtime.orphan_image))
    ]
    assert ("container", "rm", "--force", runtime.owned_container) in removals
    assert ("image", "rm", runtime.orphan_image) in removals
    assert all(runtime.foreign_container not in call for call in removals)
    assert all(runtime.foreign_image not in call for call in removals)
    assert all(runtime.protected_image not in call for call in removals)

    list_calls = [call for call in runtime.calls if len(call) > 1 and call[1] == "ls"]
    assert list_calls
    for call in list_calls:
        filters = {call[index + 1] for index, item in enumerate(call[:-1]) if item == "--filter"}
        assert "label=io.featurebench.managed=true" in filters
        assert "label=io.featurebench.run-id=run-1" in filters
        assert "label=io.featurebench.scope=scope-1" in filters


@pytest.mark.asyncio
async def test_managed_builds_receive_candidate_image_ownership_labels(tmp_path: Path) -> None:
    runtime = ManagedBuildDockerRuntime()
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    await runtime.build(tmp_path, tag="codeworld/test:1", pull=False)

    build = next(call for call in runtime.calls if call[0] == "build")
    labels = {
        build[index + 1] for index, argument in enumerate(build[:-1]) if argument == "--label"
    }
    assert labels == {
        "io.featurebench.managed=true",
        "io.featurebench.resource-kind=candidate-image",
        "io.featurebench.run-id=run-1",
        "io.featurebench.scope=scope-1",
    }


@pytest.mark.asyncio
async def test_docker_build_rotates_failed_package_source_without_agent_repair(
    tmp_path: Path,
) -> None:
    manager = SourceManager(source_settings(tmp_path), run_id="build")
    runtime = SourceBuildRuntime(manager)
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nRUN pip install demo\n",
        encoding="utf-8",
    )
    assert await runtime.build(tmp_path, tag="example:test", pull=False) == "sha256:digest"
    builds = [call for call in runtime.calls if call[0] == "build"]
    assert len(builds) == 2
    assert all(
        call[call.index("--ulimit") : call.index("--ulimit") + 2]
        == ("--ulimit", "nofile=65535:65535")
        for call in builds
    )
    wrapper = Path(builds[0][builds[0].index("--file") + 1])
    assert wrapper.parent != tmp_path
    assert not wrapper.exists()  # noqa: ASYNC240 - verify synchronous cleanup after await
    assert any("mirrors.tuna" in item for item in builds[0])
    assert any("pypi.org" in item for item in builds[1])
    metrics = [
        SourceAttemptMetric.model_validate_json(line)
        for line in (tmp_path / "source-metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(metric.endpoint, metric.status) for metric in metrics] == [
        ("tuna", "failed"),
        ("official", "succeeded"),
    ]


@pytest.mark.asyncio
async def test_docker_build_never_reselects_an_attempted_source(
    tmp_path: Path,
) -> None:
    settings = source_settings(tmp_path)
    settings.state_path.write_text(
        json.dumps(
            {
                "endpoints": {
                    "pypi/official": {
                        "service": "pypi",
                        "endpoint": "official",
                        "url": "https://pypi.org/simple",
                        "failures": 4,
                        "consecutive_failures": 4,
                        "cooldown_until": time.time() + 3600,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = SourceBuildRuntime(SourceManager(settings, run_id="build-exclusion"))
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nRUN pip install demo\n",
        encoding="utf-8",
    )

    assert await runtime.build(tmp_path, tag="example:exclusion", pull=False) == "sha256:digest"
    builds = [call for call in runtime.calls if call[0] == "build"]
    assert len(builds) == 2
    assert any("mirrors.tuna" in item for item in builds[0])
    assert any("pypi.org" in item for item in builds[1])


@pytest.mark.asyncio
async def test_docker_pull_timeout_rotates_to_the_next_image_source(tmp_path: Path) -> None:
    settings = source_settings(tmp_path).model_copy(
        update={
            "services": {
                "docker_io": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://docker.example/cache",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://docker.io",
                            canonical=True,
                        ),
                    )
                )
            }
        }
    )
    runtime = PullTimeoutRuntime(SourceManager(settings, run_id="pull-timeout"))

    resolved = await runtime.prepare_image("python:3.12", pull=True)

    assert resolved == "python@sha256:" + "a" * 64
    pulls = [call[1] for call in runtime.calls if call[0] == "pull"]
    assert pulls == [
        "docker.example/cache/library/python:3.12",
        "docker.io/library/python:3.12",
    ]


@pytest.mark.asyncio
async def test_global_docker_build_timeout_does_not_guess_a_source(
    tmp_path: Path,
) -> None:
    manager = SourceManager(source_settings(tmp_path), run_id="build-timeout")
    runtime = SourceBuildTimeoutRuntime(manager)
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nRUN python -m pip install demo\n",
        encoding="utf-8",
    )

    with pytest.raises(DockerBuildError, match="timed out after 3600s"):
        await runtime.build(tmp_path, tag="example:timeout", pull=False)
    builds = [call for call in runtime.calls if call[0] == "build"]
    assert len(builds) == 1
    assert any("mirrors.tuna" in item for item in builds[0])


@pytest.mark.asyncio
async def test_docker_build_rotates_conda_channel_after_connection_failed(
    tmp_path: Path,
) -> None:
    settings = SourceSettings(
        enabled=True,
        state_path=tmp_path / "source-state.json",
        metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
        services={
            "conda": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(
                        name="tuna",
                        url=("https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"),
                    ),
                    SourceEndpointSettings(
                        name="official",
                        url="https://conda.anaconda.org/conda-forge",
                        canonical=True,
                    ),
                )
            )
        },
    )
    runtime = CondaSourceBuildRuntime(SourceManager(settings, run_id="conda-build"))
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\n"
        "ARG CONDA_CHANNEL_URL\n"
        'RUN conda install --channel "$CONDA_CHANNEL_URL" python=3.13\n',
        encoding="utf-8",
    )

    assert await runtime.build(tmp_path, tag="example:conda", pull=False) == "sha256:digest"
    builds = [call for call in runtime.calls if call[0] == "build"]
    assert len(builds) == 2
    assert any("mirrors.tuna" in item for item in builds[0])
    assert any("conda.anaconda.org" in item for item in builds[1])


@pytest.mark.asyncio
async def test_source_owned_dockerfile_parse_failure_is_not_agent_repairable(
    tmp_path: Path,
) -> None:
    settings = SourceSettings(
        enabled=True,
        state_path=tmp_path / "source-state.json",
        metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
        services={
            "conda": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(
                        name="official",
                        url="https://conda.anaconda.org/conda-forge",
                        canonical=True,
                    ),
                )
            )
        },
    )
    runtime = SourceParseFailureRuntime(SourceManager(settings, run_id="source-parse"))
    (tmp_path / "Dockerfile").write_text(
        "FROM scratch\nRUN conda install -n base -y python\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceMaterializationError, match="Source materialization"):
        await runtime.build(tmp_path, tag="example:source-parse", pull=False)


@pytest.mark.asyncio
async def test_command_timeout_retains_bounded_process_output() -> None:
    runtime = DockerRuntime(
        DockerSettings(
            binary="bash",
            command_timeout_seconds=0.1,
            max_output_chars=1000,
        )
    )

    with pytest.raises(CommandTimeoutError) as caught:
        await runtime.invoke(
            (
                "-c",
                "printf '#7 [stage-1 2/4] RUN make all\\n' >&2; exec sleep 5",
            )
        )

    assert "RUN make all" in caught.value.stderr
    assert "RUN make all" in str(caught.value)


@pytest.mark.asyncio
async def test_docker_build_projects_normalized_buildkit_steps(tmp_path: Path) -> None:
    docker = tmp_path / "fake-docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = build ]; then\n'
        "  printf '#1 [internal] load build definition from Dockerfile\\n' >&2\n"
        "  printf '#7 [featurebench_managed_base 3/5] RUN uv sync --frozen\\n' >&2\n"
        "  printf '#7 DONE 1.2s\\n' >&2\n"
        "  exit 0\n"
        "fi\n"
        f"printf 'sha256:{'a' * 64}\\n'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runtime = DockerRuntime(DockerSettings(binary=str(docker)))
    updates: list[DockerBuildProgress] = []

    async def record(update: DockerBuildProgress) -> None:
        updates.append(update)

    image = await runtime.build(
        tmp_path,
        tag="example:progress",
        pull=False,
        progress=record,
    )

    assert image == "sha256:" + "a" * 64
    assert updates == [
        DockerBuildProgress(
            step_id="1",
            stage="internal",
            instruction="load build definition from Dockerfile",
        ),
        DockerBuildProgress(
            step_id="7",
            stage="featurebench_managed_base 3/5",
            instruction="RUN uv sync --frozen",
        ),
    ]
