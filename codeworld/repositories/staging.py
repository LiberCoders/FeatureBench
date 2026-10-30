"""Run-local repository staging shared by repository consumers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import socket
import tempfile
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from codeworld.config import RepositorySettings
from codeworld.errors import (
    CommandTimeoutError,
    ConfigurationError,
    InfrastructureError,
    RepositoryMaterializationRejected,
)
from codeworld.sources import SourceAttemptError, SourceChoice, SourceManager
from codeworld.sources.git import github_transport_config_arguments

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_UNAVAILABLE_REMOTE_OBJECT = re.compile(
    r"\bnot our ref\b|"
    r"server does not allow request for unadvertised object|"
    r"couldn't find remote ref|"
    r"did not contain [0-9a-f]{7,64}\. direct fetching of that commit failed",
    re.IGNORECASE,
)


class _GitObjectUnavailableError(InfrastructureError):
    """A Git server definitively does not expose the requested ref or object."""


class RepositoryStager:
    """Clone only while one Environment Build job is active, then remove it."""

    def __init__(
        self,
        root: str | Path,
        settings: RepositorySettings | None = None,
        *,
        sources: SourceManager | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.settings = settings or RepositorySettings()
        self.sources = sources
        self.root.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_checkouts()

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _cleanup_stale_checkouts(self) -> None:
        hostname = socket.gethostname()
        now = time.time()
        for candidate in self.root.glob("stage-*"):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            lease_path = candidate / ".featurebench-stage-lease.json"
            owner_host: str | None = None
            owner_pid: int | None = None
            try:
                raw = json.loads(lease_path.read_text(encoding="utf-8"))
                owner_host = str(raw["hostname"])
                owner_pid = int(raw["pid"])
            except (OSError, ValueError, TypeError, KeyError):
                pass
            try:
                age = now - candidate.stat().st_mtime
            except OSError:
                continue
            same_host_dead = (
                owner_host == hostname
                and owner_pid is not None
                and not self._pid_is_alive(owner_pid)
            )
            unknown_owner_expired = (
                owner_host is None or owner_host != hostname
            ) and age >= self.settings.stale_checkout_seconds
            if same_host_dead or unknown_owner_expired:
                shutil.rmtree(candidate, ignore_errors=True)

    def _git_environment(self) -> dict[str, str]:
        passthrough = (
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "TMPDIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "GIT_SSL_CAINFO",
        )
        environment = {key: os.environ[key] for key in passthrough if key in os.environ}
        environment.setdefault("PATH", os.defpath)
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_ASKPASS": "/bin/false",
                "SSH_ASKPASS": "/bin/false",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            }
        )
        return environment

    def _git_directory(self, cwd: Path | None) -> Path | None:
        """Resolve a job-owned checkout's Git directory without invoking Git."""

        if cwd is None:
            return None
        marker = cwd / ".git"
        if marker.is_dir():
            candidate = marker
        elif marker.is_file():
            try:
                declaration = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            prefix = "gitdir:"
            if not declaration.lower().startswith(prefix):
                return None
            value = declaration[len(prefix) :].strip()
            if not value:
                return None
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = marker.parent / candidate
        else:
            return None

        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def _reset_failed_git_attempt(self, cwd: Path | None) -> None:
        """Discard process-owned Git state before another source retries the command.

        One staging checkout belongs to one repository job and runs Git commands serially.
        Once ``_git_once`` has reaped a failed process, every remaining Git lock is orphaned.
        Valid objects are deliberately retained so another source can reuse completed work.
        """

        git_directory = self._git_directory(cwd)
        if git_directory is None:
            return
        transient_paths = (
            *git_directory.rglob("*.lock"),
            *git_directory.rglob("tmp_pack_*"),
        )
        for path in transient_paths:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise InfrastructureError(
                    f"failed to remove orphaned Git state {path}: {exc}"
                ) from exc

    @staticmethod
    def _resolve_indexed_submodule_paths(cwd: Path, output: str) -> tuple[Path, ...]:
        root = cwd.resolve()
        paths: list[Path] = []
        for record in output.split("\0"):
            metadata, separator, value = record.partition("\t")
            if not separator or not value or metadata.partition(" ")[0] != "160000":
                continue
            path = (root / value).resolve(strict=False)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise InfrastructureError(
                    f"repository indexes an unsafe submodule path: {value!r}"
                ) from exc
            if path == root:
                raise InfrastructureError("repository indexes its root as a submodule path")
            paths.append(path)
        return tuple(dict.fromkeys(paths))

    async def _indexed_submodule_paths(self, cwd: Path) -> tuple[Path, ...]:
        _, output, _ = await self._git_once(
            ("ls-files", "--stage", "-z"),
            cwd=cwd,
            accepted_codes=(0,),
        )
        return await asyncio.to_thread(self._resolve_indexed_submodule_paths, cwd, output)

    def _remove_failed_submodule_state(
        self,
        paths: tuple[Path, ...],
        git_directory: Path | None,
    ) -> None:
        module_state = (git_directory / "modules",) if git_directory is not None else ()
        targets = (*paths, *module_state)
        for path in targets:
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path)
            except OSError as exc:
                raise InfrastructureError(
                    f"failed to reset partial submodule state {path}: {exc}"
                ) from exc

    async def _reset_failed_submodule_update(self, cwd: Path) -> None:
        """Return every declared submodule to an uninitialized, retryable state."""

        paths = await self._indexed_submodule_paths(cwd)
        git_directory = self._git_directory(cwd)
        deinit = ("submodule", "deinit", "--force", "--all")
        try:
            await self._git_once(deinit, cwd=cwd, accepted_codes=(0,))
        except InfrastructureError:
            # A half-cloned nested repository can make the first deinit fail. Remove
            # its process-owned state, then retry the canonical Git cleanup once.
            await asyncio.to_thread(
                self._remove_failed_submodule_state,
                paths,
                git_directory,
            )
            await asyncio.to_thread(self._reset_failed_git_attempt, cwd)
            await self._git_once(deinit, cwd=cwd, accepted_codes=(0,))
        await asyncio.to_thread(
            self._remove_failed_submodule_state,
            paths,
            git_directory,
        )

    @staticmethod
    def _is_submodule_update(arguments: Sequence[str]) -> bool:
        return any(
            tuple(arguments[index : index + 2]) == ("submodule", "update")
            for index in range(len(arguments) - 1)
        )

    async def _recover_failed_git_attempt(
        self,
        cwd: Path | None,
        arguments: Sequence[str],
    ) -> None:
        await asyncio.to_thread(self._reset_failed_git_attempt, cwd)
        if cwd is not None and self._is_submodule_update(arguments):
            await self._reset_failed_submodule_update(cwd)

    async def _git(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path | None = None,
        network: bool = False,
        accepted_codes: tuple[int, ...] = (0,),
    ) -> tuple[int, str, str]:
        sources = self.sources
        network_arguments = (
            (
                "-c",
                f"http.lowSpeedLimit={self.settings.low_speed_bytes_per_second}",
                "-c",
                f"http.lowSpeedTime={self.settings.low_speed_timeout_seconds}",
                *arguments,
            )
            if network
            else tuple(arguments)
        )
        if network and sources is not None and sources.service("github_git"):

            async def via_source(choice: SourceChoice) -> tuple[int, str, str]:
                canonical_prefix = "https://github.com/"
                transport_prefix = sources.resolve_url(choice, canonical_prefix)
                actual = (
                    *github_transport_config_arguments(
                        transport_prefix,
                        redirect_https=not choice.endpoint.canonical,
                    ),
                    *network_arguments,
                )
                try:
                    return await self._git_once(
                        actual,
                        cwd=cwd,
                        accepted_codes=accepted_codes,
                    )
                except _GitObjectUnavailableError as exc:
                    if choice.endpoint.canonical:
                        raise RepositoryMaterializationRejected(str(exc)) from exc
                    await self._recover_failed_git_attempt(cwd, arguments)
                    raise SourceAttemptError(str(exc)) from exc
                except InfrastructureError as exc:
                    await self._recover_failed_git_attempt(cwd, arguments)
                    raise SourceAttemptError(str(exc)) from exc

            return await sources.execute(
                "github_git",
                via_source,
            )

        attempts = self.settings.network_attempts if network else 1
        last_detail = ""
        for attempt in range(1, attempts + 1):
            try:
                return await self._git_once(
                    network_arguments,
                    cwd=cwd,
                    accepted_codes=accepted_codes,
                )
            except _GitObjectUnavailableError as exc:
                raise RepositoryMaterializationRejected(str(exc)) from exc
            except InfrastructureError as exc:
                last_detail = str(exc)
                await self._recover_failed_git_attempt(cwd, arguments)
                if attempt >= attempts:
                    raise
            if attempt < attempts:
                await asyncio.sleep(self.settings.retry_base_seconds * (2 ** (attempt - 1)))
        raise InfrastructureError(f"Git command failed: {last_detail}")

    async def _git_once(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path | None,
        accepted_codes: tuple[int, ...],
    ) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.git_binary,
                *arguments,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._git_environment(),
                start_new_session=True,
            )
        except OSError as exc:
            raise InfrastructureError(f"cannot start Git: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), self.settings.command_timeout_seconds
            )
        except TimeoutError as exc:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise CommandTimeoutError(
                f"Git command {arguments[-1]!r} timed out after "
                f"{self.settings.command_timeout_seconds}s"
            ) from exc
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise
        output = stdout.decode("utf-8", errors="replace")
        error = stderr.decode("utf-8", errors="replace")
        if process.returncode in accepted_codes:
            return process.returncode or 0, output.strip(), error.strip()
        detail = error.strip() or output.strip()
        if _UNAVAILABLE_REMOTE_OBJECT.search(detail) is not None:
            raise _GitObjectUnavailableError(
                f"Git remote does not expose the requested object: {detail[-5000:]}"
            )
        raise InfrastructureError(f"Git command failed ({process.returncode}): {detail[-5000:]}")

    async def check_available(self) -> None:
        """Validate run-wide host capabilities before repository jobs are scheduled."""
        try:
            await self._git(("--version",), cwd=self.root)
        except InfrastructureError as exc:
            raise ConfigurationError(
                f"repository staging requires Git executable {self.settings.git_binary!r}"
            ) from exc
        if not self.settings.fetch_lfs:
            return
        try:
            await self._git(("lfs", "version"), cwd=self.root)
        except InfrastructureError as exc:
            raise ConfigurationError(
                "repository.fetch_lfs=true requires git-lfs on the host; install git-lfs "
                "or explicitly set repository.fetch_lfs=false"
            ) from exc

    @staticmethod
    def _uses_lfs(destination: Path) -> bool:
        for attributes in destination.rglob(".gitattributes"):
            try:
                if "filter=lfs" in attributes.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                continue
        return False

    async def _hydrate_checkout(self, destination: Path) -> None:
        await self._git(("reset", "--hard", "HEAD"), cwd=destination)
        await self._git(("clean", "-ffdx"), cwd=destination)
        if self.settings.update_submodules and (destination / ".gitmodules").is_file():
            await self._git(("submodule", "sync", "--recursive"), cwd=destination)
            await self._git(
                (
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--force",
                    "--jobs",
                    str(self.settings.submodule_jobs),
                ),
                cwd=destination,
                network=True,
            )
        if self.settings.fetch_lfs and await asyncio.to_thread(self._uses_lfs, destination):
            await self._git(("lfs", "install", "--local"), cwd=destination)
            await self._git(("lfs", "pull"), cwd=destination, network=True)
            # git-lfs also hydrates pointer-shaped files that are not governed by
            # filter=lfs attributes. Re-checking out HEAD preserves real LFS files
            # through the installed filter while restoring malformed pointers.
            await self._git(("reset", "--hard", "HEAD"), cwd=destination)
        _, status, _ = await self._git(
            ("status", "--porcelain", "--untracked-files=all"), cwd=destination
        )
        if status:
            raise InfrastructureError(
                "staged repository is not clean after submodule/LFS hydration"
            )

    async def _configure_partial_clone(self, destination: Path) -> None:
        """Mark one initialized repository as a partial clone before filtered fetches."""
        await self._git(("config", "core.repositoryFormatVersion", "1"), cwd=destination)
        await self._git(("config", "extensions.partialClone", "origin"), cwd=destination)
        await self._git(("config", "remote.origin.promisor", "true"), cwd=destination)
        await self._git(
            ("config", "remote.origin.partialCloneFilter", "blob:none"),
            cwd=destination,
        )

    @asynccontextmanager
    async def stage(
        self, url: str, ref: str | None = None
    ) -> AsyncGenerator[tuple[Path, str], None]:
        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("repository URL must not be empty")
        parsed = urlsplit(normalized_url)
        if parsed.scheme and (parsed.username is not None or parsed.password is not None):
            raise ValueError("repository URLs may not contain credentials")
        staging_root = Path(tempfile.mkdtemp(prefix="stage-", dir=self.root))
        (staging_root / ".featurebench-stage-lease.json").write_text(
            json.dumps(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_at": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        destination = staging_root / "repository"
        try:
            fetch_ref = ref.strip() if ref is not None and ref.strip() else "HEAD"
            if fetch_ref.startswith("-"):
                raise ValueError("repository ref may not start with '-'")
            await self._git(("init", "--quiet", "--", str(destination)))
            await self._git(("remote", "add", "origin", normalized_url), cwd=destination)
            if self.settings.partial_clone:
                await self._configure_partial_clone(destination)
            fetch = ["fetch", "--force", "--no-tags", "--depth=1"]
            if self.settings.partial_clone:
                fetch.append("--filter=blob:none")
            fetch.extend(("origin", fetch_ref))
            try:
                await self._git(tuple(fetch), cwd=destination, network=True)
            except InfrastructureError:
                fallback = ["fetch", "--force", "--no-tags"]
                if self.settings.partial_clone:
                    fallback.append("--filter=blob:none")
                fallback.extend(("origin", fetch_ref))
                await self._git(tuple(fallback), cwd=destination, network=True)
            _, commit, _ = await self._git(("rev-parse", "FETCH_HEAD^{commit}"), cwd=destination)
            if not _COMMIT.fullmatch(commit):
                raise InfrastructureError(f"Git resolved a non-commit object: {commit}")
            await self._git(
                ("checkout", "--force", "--detach", commit), cwd=destination, network=True
            )
            await self._hydrate_checkout(destination)
            yield destination, commit
        finally:
            cleanup = asyncio.create_task(asyncio.to_thread(shutil.rmtree, staging_root, True))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                with suppress(asyncio.CancelledError):
                    await cleanup
                raise
