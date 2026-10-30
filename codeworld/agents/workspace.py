"""Pydantic AI workspace capability backed by one real data-factory container."""

from __future__ import annotations

import hashlib
import posixpath
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from codeworld.errors import CommandTimeoutError, ContractError
from codeworld.runtime.base import ContainerSessionProtocol


def _short_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _absolute_path(session: ContainerSessionProtocol, path: str) -> str:
    """Resolve a model-supplied path inside the container, not on the host."""
    if not path or "\x00" in path:
        raise ModelRetry("path must be a non-empty container path")
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        normalized = posixpath.normpath(str(candidate))
    else:
        if session.runtime_spec is None:
            raise ModelRetry("the container has no runtime workspace")
        normalized = posixpath.normpath(
            str(PurePosixPath(session.runtime_spec.workdir) / candidate)
        )
    if not normalized.startswith("/"):
        raise ModelRetry("path could not be resolved inside the container")
    return normalized


@dataclass
class ContainerWorkspace(AbstractCapability[object]):
    """Classic shell and filesystem tools executed in one activated task container.

    Docker is the isolation boundary. Relative paths start at the runtime workdir;
    absolute paths address other mounted container paths such as an adapter workspace.
    Adapter-specific tools are deliberately supplied as separate capabilities.
    """

    session: ContainerSessionProtocol = field(repr=False)
    writable: bool = False
    default_timeout_seconds: float = 120
    max_output_chars: int = 50_000
    max_read_bytes: int = 5_000_000
    max_file_results: int = 1_000

    def get_instructions(self) -> str:
        mode = "read, write, edit, and run" if self.writable else "read and run"
        return (
            f"Container workspace tools can {mode} inside the isolated task container. "
            "Every run_command executes after the environment activation artifact has been "
            "sourced and starts in /testbed. Relative file paths resolve from the runtime "
            "workdir; absolute paths can address adapter workspaces mounted in the container."
        )

    def get_toolset(self) -> FunctionToolset[object]:
        toolset: FunctionToolset[object] = FunctionToolset(id="container_workspace")

        async def run_command(command: str, timeout_seconds: float | None = None) -> str:
            """Run a shell command in the activated repository environment.

            Args:
                command: Complete Bash command. Pipes, redirection, globs, and chaining work.
                timeout_seconds: Optional deadline no greater than the configured tool ceiling.
            """
            if not command.strip():
                raise ModelRetry("command must not be empty")
            deadline = min(
                timeout_seconds or self.default_timeout_seconds,
                self.default_timeout_seconds,
            )
            try:
                result = await self.session.run_script(
                    command,
                    activated=True,
                    timeout_seconds=deadline,
                    max_output_chars=self.max_output_chars,
                    check=False,
                )
            except CommandTimeoutError as exc:
                raise ModelRetry(str(exc)) from exc
            sections: list[str] = []
            if result.stdout:
                sections.append(f"[stdout]\n{result.stdout}")
            if result.stderr:
                sections.append(f"[stderr]\n{result.stderr}")
            body = "\n".join(sections) if sections else "(no output)"
            if result.exit_code:
                body += f"\n[exit code: {result.exit_code}]"
            return body[-self.max_output_chars :]

        async def read_file(path: str, offset: int = 0, limit: int = 2_000) -> str:
            """Read a UTF-8 file with line numbers and continuation metadata.

            Args:
                path: Absolute container path or path relative to the runtime workdir.
                offset: Zero-based first line.
                limit: Maximum lines returned, capped at 2000.
            """
            absolute = _absolute_path(self.session, path)
            try:
                raw = await self.session.read_absolute_file(absolute, max_bytes=self.max_read_bytes)
            except (ContractError, ValueError) as exc:
                raise ModelRetry(str(exc)) from exc
            if b"\x00" in raw:
                return f"[binary file: {len(raw)} bytes]"
            content = raw.decode("utf-8", errors="replace")
            lines = content.splitlines()
            start = max(0, offset)
            count = min(max(1, limit), 2_000)
            selected = lines[start : start + count]
            body = "\n".join(
                f"{number:>6}\t{line}" for number, line in enumerate(selected, start=start + 1)
            )
            remaining = len(lines) - start - len(selected)
            continuation = (
                f"\n... ({remaining} more lines; use offset={start + len(selected)})"
                if remaining > 0
                else ""
            )
            return (
                f"[{absolute} | {len(lines)} lines | hash:{_short_hash(content)}]\n"
                f"{body}{continuation}"
            )

        async def list_directory(path: str = ".") -> str:
            """List one container directory.

            Args:
                path: Absolute container path or path relative to the runtime workdir.
            """
            absolute = _absolute_path(self.session, path)
            result = await self.session.exec(
                (
                    "find",
                    absolute,
                    "-mindepth",
                    "1",
                    "-maxdepth",
                    "1",
                    "-printf",
                    "%y %p %s bytes\\n",
                ),
                workdir="/",
                check=False,
                max_output_chars=self.max_output_chars,
            )
            if result.exit_code != 0:
                raise ModelRetry(result.combined_output)
            lines = result.stdout.splitlines()
            shown = lines[: self.max_file_results]
            suffix = (
                f"\n... ({len(lines) - len(shown)} more entries)" if len(lines) > len(shown) else ""
            )
            return "\n".join(shown) + suffix

        async def search_files(query: str, path: str = ".") -> str:
            """Search literal text recursively inside container files.

            Args:
                query: Literal text to search for.
                path: Absolute container path or path relative to the runtime workdir.
            """
            if not query:
                raise ModelRetry("query must not be empty")
            absolute = _absolute_path(self.session, path)
            result = await self.session.exec(
                ("grep", "-R", "-n", "-F", "--", query, absolute),
                workdir="/",
                check=False,
                max_output_chars=self.max_output_chars,
            )
            if result.exit_code not in (0, 1):
                raise ModelRetry(result.combined_output)
            return result.combined_output[-self.max_output_chars :]

        async def find_files(pattern: str, path: str = ".") -> str:
            """Find files by shell-style filename pattern.

            Args:
                pattern: Filename pattern such as '*.py'.
                path: Absolute container path or path relative to the runtime workdir.
            """
            if not pattern:
                raise ModelRetry("pattern must not be empty")
            absolute = _absolute_path(self.session, path)
            result = await self.session.exec(
                ("find", absolute, "-type", "f", "-name", pattern),
                workdir="/",
                check=False,
                max_output_chars=self.max_output_chars,
            )
            if result.exit_code != 0:
                raise ModelRetry(result.combined_output)
            lines = result.stdout.splitlines()
            shown = lines[: self.max_file_results]
            suffix = (
                f"\n... ({len(lines) - len(shown)} more files)" if len(lines) > len(shown) else ""
            )
            return "\n".join(shown) + suffix

        async def file_info(path: str) -> str:
            """Return type, permissions, size, and timestamps for one container path."""
            absolute = _absolute_path(self.session, path)
            result = await self.session.exec(("stat", "--", absolute), workdir="/", check=False)
            if result.exit_code != 0:
                raise ModelRetry(result.combined_output)
            return result.combined_output[-self.max_output_chars :]

        # Shell commands may mutate the repository and test runners commonly share
        # caches or generated state.  Use Pydantic AI's native barrier so a model
        # response containing several calls cannot execute them concurrently.
        toolset.add_function(run_command, sequential=True)
        toolset.add_function(read_file)
        toolset.add_function(list_directory)
        toolset.add_function(search_files)
        toolset.add_function(find_files)
        toolset.add_function(file_info)

        if self.writable:

            async def write_file(path: str, content: str, expected_hash: str | None = None) -> str:
                """Create or replace a UTF-8 file inside the container.

                Args:
                    path: Absolute container path or path relative to the runtime workdir.
                    content: Complete new file content.
                    expected_hash: Optional hash from read_file for conflict detection.
                """
                absolute = _absolute_path(self.session, path)
                if expected_hash is not None:
                    try:
                        old = (
                            await self.session.read_absolute_file(
                                absolute, max_bytes=self.max_read_bytes
                            )
                        ).decode("utf-8", errors="replace")
                    except (ContractError, ValueError) as exc:
                        raise ModelRetry(str(exc)) from exc
                    if _short_hash(old) != expected_hash:
                        raise ModelRetry(
                            "file changed since it was read; read it again before writing"
                        )
                try:
                    await self.session.write_absolute_file(absolute, content.encode("utf-8"))
                except (ContractError, ValueError) as exc:
                    raise ModelRetry(str(exc)) from exc
                return (
                    f"wrote {len(content.encode('utf-8'))} bytes to {absolute}; "
                    f"new hash:{_short_hash(content)}"
                )

            async def edit_file(
                path: str,
                old_text: str,
                new_text: str,
                expected_hash: str | None = None,
            ) -> str:
                """Replace exactly one text occurrence inside a container file.

                Args:
                    path: Absolute container path or path relative to the runtime workdir.
                    old_text: Exact unique text to replace.
                    new_text: Replacement text.
                    expected_hash: Optional hash from read_file for conflict detection.
                """
                absolute = _absolute_path(self.session, path)
                try:
                    content = (
                        await self.session.read_absolute_file(
                            absolute, max_bytes=self.max_read_bytes
                        )
                    ).decode("utf-8", errors="replace")
                except (ContractError, ValueError) as exc:
                    raise ModelRetry(str(exc)) from exc
                if expected_hash is not None and _short_hash(content) != expected_hash:
                    raise ModelRetry("file changed since it was read; read it again before editing")
                if content.count(old_text) != 1:
                    raise ModelRetry("old_text must occur exactly once")
                updated = content.replace(old_text, new_text, 1)
                await self.session.write_absolute_file(absolute, updated.encode("utf-8"))
                return f"edited {absolute}; new hash:{_short_hash(updated)}"

            async def create_directory(path: str) -> str:
                """Create a directory and missing parents inside the container."""
                absolute = _absolute_path(self.session, path)
                result = await self.session.exec(
                    ("mkdir", "-p", "--", absolute), workdir="/", check=False
                )
                if result.exit_code != 0:
                    raise ModelRetry(result.combined_output)
                return f"created {absolute}"

            toolset.add_function(write_file, sequential=True)
            toolset.add_function(edit_file, sequential=True)
            toolset.add_function(create_directory, sequential=True)

        return toolset
