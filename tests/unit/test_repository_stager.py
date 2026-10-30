from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from codeworld.config import (
    RepositorySettings,
    SourceEndpointSettings,
    SourceMetricsSettings,
    SourceServiceSettings,
    SourceSettings,
)
from codeworld.errors import (
    CommandTimeoutError,
    ConfigurationError,
    InfrastructureError,
    RepositoryMaterializationRejected,
)
from codeworld.repositories.staging import RepositoryStager
from codeworld.sources import SourceManager


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_clone", [False, True])
async def test_repository_stager_pins_commit_and_removes_worker_checkout(
    tmp_path: Path, partial_clone: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    git(source, "add", "pyproject.toml")
    git(source, "commit", "-m", "fixture")
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    git(source, "add", "README.md")
    git(source, "commit", "-m", "second fixture")
    expected = git(source, "rev-parse", "HEAD")

    staging = tmp_path / "staging"
    stager = RepositoryStager(
        staging,
        RepositorySettings(partial_clone=partial_clone, update_submodules=False, fetch_lfs=False),
    )
    async with stager.stage(str(source), "HEAD") as (checkout, commit):
        assert commit == expected
        assert checkout.name == "repository"
        assert (checkout / ".git").exists()
        assert git(checkout, "status", "--porcelain") == ""
        assert git(checkout, "rev-list", "--count", "HEAD") == "1"
        if partial_clone:
            assert git(checkout, "config", "--get", "core.repositoryFormatVersion") == "1"
            assert git(checkout, "config", "--get", "extensions.partialClone") == "origin"
            assert git(checkout, "config", "--get", "remote.origin.promisor") == "true"
            assert (
                git(checkout, "config", "--get", "remote.origin.partialCloneFilter") == "blob:none"
            )
    assert not list(staging.glob("stage-*"))


@pytest.mark.asyncio
async def test_repository_stager_kills_git_process_group_when_cancelled(tmp_path: Path) -> None:
    pid_path = tmp_path / "git.pid"
    git_wrapper = tmp_path / "git-wrapper"
    git_wrapper.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$$" > "$1"\nsleep 60\n',
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)

    class CancellationTestStager(RepositoryStager):
        async def wait_in_git(self, pid_path: Path, cwd: Path) -> tuple[int, str, str]:
            return await self._git_once((str(pid_path),), cwd=cwd, accepted_codes=(0,))

    stager = CancellationTestStager(
        tmp_path / "staging",
        RepositorySettings(git_binary=str(git_wrapper), command_timeout_seconds=120),
    )

    task = asyncio.create_task(stager.wait_in_git(pid_path, tmp_path))
    for _ in range(100):
        if pid_path.is_file():
            break
        await asyncio.sleep(0.01)
    assert pid_path.is_file()
    pid = int(pid_path.read_text(encoding="utf-8"))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_repository_stager_resets_transient_git_state_before_source_failover(
    tmp_path: Path,
) -> None:
    source_settings = SourceSettings(
        enabled=True,
        state_path=tmp_path / "source-state.json",
        metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
        services={
            "github_git": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(name="mirror", url="https://mirror.example"),
                    SourceEndpointSettings(
                        name="official",
                        url="https://github.com",
                        canonical=True,
                    ),
                )
            )
        },
    )
    sources = SourceManager(source_settings, run_id="source-recovery")

    class SourceRecoveryStager(RepositoryStager):
        attempts = 0

        async def _git_once(
            self,
            arguments: Sequence[str],
            *,
            cwd: Path | None,
            accepted_codes: tuple[int, ...],
        ) -> tuple[int, str, str]:
            del arguments, accepted_codes
            assert cwd is not None
            self.attempts += 1
            git_directory = cwd / ".git"
            if self.attempts == 1:
                (git_directory / "refs" / "heads").mkdir(parents=True)
                (git_directory / "objects" / "pack").mkdir(parents=True)
                (git_directory / "index.lock").write_text("orphan", encoding="utf-8")
                (git_directory / "refs" / "heads" / "main.lock").write_text(
                    "orphan",
                    encoding="utf-8",
                )
                (git_directory / "objects" / "pack" / "tmp_pack_fixture").write_text(
                    "partial",
                    encoding="utf-8",
                )
                raise CommandTimeoutError("first source timed out")

            assert not (git_directory / "index.lock").exists()
            assert not (git_directory / "refs" / "heads" / "main.lock").exists()
            assert not (git_directory / "objects" / "pack" / "tmp_pack_fixture").exists()
            assert (git_directory / "config").read_text(encoding="utf-8") == "preserved"
            assert (cwd / "repository.lock").read_text(encoding="utf-8") == "preserved"
            return 0, "recovered", ""

        async def network_checkout(self, cwd: Path) -> tuple[int, str, str]:
            return await self._git(
                ("checkout", "--force", "--detach", "a" * 40),
                cwd=cwd,
                network=True,
            )

    staging = tmp_path / "staging"
    checkout = staging / "stage-fixture" / "repository"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "config").write_text("preserved", encoding="utf-8")
    (checkout / "repository.lock").write_text("preserved", encoding="utf-8")
    stager = SourceRecoveryStager(
        staging,
        RepositorySettings(network_attempts=1),
        sources=sources,
    )

    result = await stager.network_checkout(checkout)

    assert result == (0, "recovered", "")
    assert stager.attempts == 2


@pytest.mark.asyncio
async def test_repository_stager_normalizes_public_github_submodule_transports(
    tmp_path: Path,
) -> None:
    source_settings = SourceSettings(
        enabled=True,
        state_path=tmp_path / "source-state.json",
        metrics=SourceMetricsSettings(enabled=False),
        services={
            "github_git": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(
                        name="official",
                        url="https://github.com",
                        canonical=True,
                    ),
                ),
                maximum_attempts=1,
            )
        },
    )
    captured: list[tuple[str, ...]] = []

    class CaptureStager(RepositoryStager):
        async def _git_once(
            self,
            arguments: Sequence[str],
            *,
            cwd: Path | None,
            accepted_codes: tuple[int, ...],
        ) -> tuple[int, str, str]:
            del cwd, accepted_codes
            captured.append(tuple(arguments))
            return 0, "ok", ""

        async def update_submodules(self, cwd: Path) -> None:
            await self._git(
                ("submodule", "update", "--init", "--recursive"),
                cwd=cwd,
                network=True,
            )

    stager = CaptureStager(
        tmp_path / "staging",
        RepositorySettings(fetch_lfs=False),
        sources=SourceManager(source_settings),
    )

    await stager.update_submodules(tmp_path)

    arguments = captured[0]
    for transport in (
        "git@github.com:",
        "ssh://git@github.com/",
        "git://github.com/",
    ):
        assert f"url.https://github.com/.insteadOf={transport}" in arguments


@pytest.mark.asyncio
async def test_repository_stager_rotates_a_stale_mirror_before_rejecting(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "git-calls"
    git_wrapper = tmp_path / "git-wrapper"
    git_wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{calls}'\n"
        'case "$*" in\n'
        "  *mirror.example*)\n"
        "    printf '%s\\n' 'fatal: remote error: upload-pack: not our ref " + "a" * 40 + "' >&2\n"
        "    exit 128\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)
    sources = SourceManager(
        SourceSettings(
            enabled=True,
            state_path=tmp_path / "source-state.json",
            metrics=SourceMetricsSettings(path=tmp_path / "source-metrics.jsonl"),
            services={
                "github_git": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://mirror.example/",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://github.com/",
                            canonical=True,
                        ),
                    )
                )
            },
        ),
        run_id="missing-object",
    )
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(
            git_binary=str(git_wrapper),
            network_attempts=1,
            update_submodules=False,
            fetch_lfs=False,
        ),
        sources=sources,
    )

    result = await stager._git(  # pyright: ignore[reportPrivateUsage]
        ("fetch", "origin", "a" * 40),
        network=True,
    )

    assert result == (0, "", "")
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.asyncio
async def test_repository_stager_rejects_a_definitively_missing_remote_object(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "git-calls"
    git_wrapper = tmp_path / "git-wrapper"
    git_wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{calls}'\n"
        "printf '%s\\n' 'fatal: remote error: upload-pack: not our ref " + "b" * 40 + "' >&2\n"
        "exit 128\n",
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(
            git_binary=str(git_wrapper),
            network_attempts=3,
            update_submodules=False,
            fetch_lfs=False,
        ),
    )

    with pytest.raises(RepositoryMaterializationRejected) as caught:
        await stager._git(  # pyright: ignore[reportPrivateUsage]
            ("fetch", "origin", "b" * 40),
            network=True,
        )

    assert caught.value.category == "repository_object_unavailable"
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_repository_stager_resets_partial_submodules_before_source_failover(
    tmp_path: Path,
) -> None:
    module = tmp_path / "module"
    module.mkdir()
    git(module, "init")
    git(module, "config", "user.email", "test@example.com")
    git(module, "config", "user.name", "Test")
    (module / "fixture.txt").write_text("module fixture\n", encoding="utf-8")
    git(module, "add", "fixture.txt")
    git(module, "commit", "-m", "module fixture")

    staging = tmp_path / "staging"
    checkout = staging / "stage-fixture" / "repository"
    checkout.mkdir(parents=True)
    git(checkout, "init")
    git(checkout, "config", "user.email", "test@example.com")
    git(checkout, "config", "user.name", "Test")
    git(
        checkout,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(module),
        "vendor/module",
    )
    git(checkout, "commit", "-m", "submodule fixture")
    ordinary = checkout / "ordinary"
    ordinary.mkdir()
    (ordinary / "preserved.txt").write_text("preserved\n", encoding="utf-8")
    with (checkout / ".gitmodules").open("a", encoding="utf-8") as stream:
        stream.write(
            f'[submodule "declaration-without-gitlink"]\n\tpath = ordinary\n\turl = {module}\n'
        )
    assert (checkout / "vendor" / "module" / "fixture.txt").is_file()
    assert (checkout / ".git" / "modules" / "vendor" / "module").is_dir()

    class SubmoduleRecoveryStager(RepositoryStager):
        async def recover_submodule_update(self, cwd: Path) -> None:
            await self._recover_failed_git_attempt(
                cwd,
                ("submodule", "update", "--init", "--recursive", "--force"),
            )

    stager = SubmoduleRecoveryStager(
        staging,
        RepositorySettings(update_submodules=True, fetch_lfs=False),
    )

    await stager.recover_submodule_update(checkout)

    assert not (checkout / "vendor" / "module").exists()
    assert not (checkout / ".git" / "modules").exists()
    assert (ordinary / "preserved.txt").read_text(encoding="utf-8") == "preserved\n"
    git(
        checkout,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--force",
    )
    assert (checkout / "vendor" / "module" / "fixture.txt").is_file()


@pytest.mark.asyncio
async def test_repository_stager_preflight_checks_git_and_configured_lfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(fetch_lfs=True),
    )
    calls: list[tuple[str, ...]] = []

    async def record(arguments: tuple[str, ...], **_: object) -> tuple[int, str, str]:
        calls.append(arguments)
        return 0, "available", ""

    monkeypatch.setattr(stager, "_git", record)

    await stager.check_available()

    assert calls == [("--version",), ("lfs", "version")]


@pytest.mark.asyncio
async def test_repository_stager_preflight_fails_before_jobs_when_lfs_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(fetch_lfs=True),
    )

    async def missing_lfs(arguments: tuple[str, ...], **_: object) -> tuple[int, str, str]:
        if arguments == ("lfs", "version"):
            raise InfrastructureError("git: 'lfs' is not a git command")
        return 0, "git version fixture", ""

    monkeypatch.setattr(stager, "_git", missing_lfs)

    with pytest.raises(ConfigurationError, match=r"repository\.fetch_lfs=true requires git-lfs"):
        await stager.check_available()


@pytest.mark.asyncio
async def test_repository_stager_preflight_skips_lfs_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(fetch_lfs=False),
    )
    calls: list[tuple[str, ...]] = []

    async def record(arguments: tuple[str, ...], **_: object) -> tuple[int, str, str]:
        calls.append(arguments)
        return 0, "available", ""

    monkeypatch.setattr(stager, "_git", record)

    await stager.check_available()

    assert calls == [("--version",)]


@pytest.mark.asyncio
async def test_repository_stager_restores_head_after_lfs_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()
    (destination / ".gitattributes").write_text("*.png filter=lfs\n", encoding="utf-8")
    stager = RepositoryStager(
        tmp_path / "staging",
        RepositorySettings(update_submodules=False, fetch_lfs=True),
    )
    calls: list[tuple[str, ...]] = []

    async def record(arguments: tuple[str, ...], **_: object) -> tuple[int, str, str]:
        calls.append(arguments)
        return 0, "", ""

    monkeypatch.setattr(stager, "_git", record)

    await stager._hydrate_checkout(destination)  # pyright: ignore[reportPrivateUsage]

    assert calls == [
        ("reset", "--hard", "HEAD"),
        ("clean", "-ffdx"),
        ("lfs", "install", "--local"),
        ("lfs", "pull"),
        ("reset", "--hard", "HEAD"),
        ("status", "--porcelain", "--untracked-files=all"),
    ]
