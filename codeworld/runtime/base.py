"""Container backend contracts consumed by adapters and environment runners."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable

from codeworld.models import RuntimeSpec


@runtime_checkable
class CommandResultProtocol(Protocol):
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def combined_output(self) -> str: ...


@runtime_checkable
class ContainerSessionProtocol(Protocol):
    runtime_spec: RuntimeSpec | None

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
    ) -> CommandResultProtocol: ...

    async def run_script(
        self,
        script: str,
        arguments: Sequence[str] = (),
        *,
        activated: bool = False,
        timeout_seconds: float | None = None,
        max_output_chars: int | None = None,
        check: bool = False,
    ) -> CommandResultProtocol: ...

    async def read_absolute_file(self, path: str, *, max_bytes: int = 1_000_000) -> bytes: ...

    async def write_file(self, path: str, content: bytes) -> None: ...

    async def write_absolute_file(self, path: str, content: bytes) -> None: ...

    async def git_diff(self, *arguments: str, max_bytes: int = 20_000_000) -> str: ...

    async def clean_workspace(self) -> None: ...

    async def seal_workspace(self) -> None: ...


@runtime_checkable
class ContainerRuntimeProtocol(Protocol):
    def session(
        self,
        image: str,
        *,
        runtime_spec: RuntimeSpec | None = None,
        read_only: bool = False,
        gpus: str | None = None,
        bind_mounts: Sequence[tuple[str | Path, str, bool]] = (),
    ) -> AbstractAsyncContextManager[ContainerSessionProtocol]: ...
