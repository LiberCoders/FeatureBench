from __future__ import annotations

from pathlib import Path

import pytest

from codeworld.application import build as build_application
from codeworld.application.build import (
    parse_build_removal_targets,
    remove_build_environments,
)
from codeworld.console.models import CliInvocation, write_run_manifest
from codeworld.environment.jsonl import iter_environments, write_environment_snapshot
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.models import RuntimeSpec
from codeworld.run_layout import RunLayout


def _environment(owner: str, repository: str, marker: str) -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id=f"environment-{marker}",
        repository=f"https://github.com/{owner}/{repository}.git",
        commit=marker * 40,
        language="python",
        gpu_requirement="none",
        image=f"sha256:{marker * 64}",
        dockerfile=(
            "FROM python:3.13\n"
            f"RUN git clone https://github.com/{owner}/{repository}.git /testbed\n"
        ),
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Runner V1 placeholder with enough source to satisfy the strict model.\n"
                "request=''\nresponse=''\necho \"$request$response\" >/dev/null\n"
            ),
        ),
        verified_test_points=(CollectedTestPoint(id="test_one", path="tests/test_one.py"),),
    )


def _build_run(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "build.yaml"
    config.write_text(f"workspace: {tmp_path / 'workspace'}\n", encoding="utf-8")
    run_directory = project / "outputs" / "build-test"
    invocation = CliInvocation(
        command="build",
        project_root=project,
        config_files=(config,),
        output_directory=run_directory,
        repository_source="owner/source",
    )
    write_run_manifest(invocation, "build-run")
    return project, run_directory


class RemovalRuntime:
    instance: RemovalRuntime | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.protected_images: frozenset[str] | None = None
        self.forced_removals: list[str] = []
        RemovalRuntime.instance = self

    @classmethod
    def current(cls) -> RemovalRuntime:
        assert cls.instance is not None
        return cls.instance

    @staticmethod
    def scope_for_workspace(_workspace: Path) -> str:
        return "scope"

    async def check_available(self) -> None:
        return None

    async def cleanup_run_resources(
        self, *, protected_images: frozenset[str] | None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        self.protected_images = protected_images
        return (("stale-container",), ("stale-image",))

    async def remove_image(self, image: str, *, force: bool = False) -> bool:
        assert force
        self.forced_removals.append(image)
        return True

    async def image_exists(self, image: str) -> bool:
        return image not in self.forced_removals


def test_build_remove_targets_accept_repositories_or_one_txt(tmp_path: Path) -> None:
    assert parse_build_removal_targets(
        tmp_path,
        ("Owner/One", "owner/one", "Another/Two"),
    ) == ("Owner/One", "Another/Two")

    source = tmp_path / "repositories.txt"
    source.write_text("# selected repositories\nOwner/One\n\nAnother/Two\n", encoding="utf-8")
    assert parse_build_removal_targets(tmp_path, ("repositories.txt",)) == (
        "Owner/One",
        "Another/Two",
    )

    with pytest.raises(ValueError, match="expected owner/repository"):
        parse_build_removal_targets(tmp_path, ("https://github.com/owner/repo",))
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_build_removal_targets(tmp_path, ("repositories.txt", "owner/repo"))


@pytest.mark.asyncio
async def test_build_remove_atomically_rewrites_outputs_and_cleans_owned_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run_directory = _build_run(tmp_path)
    removed = _environment("Owner", "One", "a")
    retained = _environment("Another", "Two", "b")
    write_environment_snapshot(run_directory / "environments.jsonl", (removed, retained))
    monkeypatch.setattr(build_application, "DockerRuntime", RemovalRuntime)

    summary = await remove_build_environments(
        project_root=project,
        run_directory=run_directory,
        repositories=("owner/one",),
    )

    assert tuple(iter_environments(run_directory / "environments.jsonl")) == (retained,)
    assert tuple(iter_environments(summary.archive)) == (removed,)
    assert summary.environments == 1
    assert summary.images == 1
    assert summary.stale_containers == 1
    assert summary.stale_images == 1
    runtime = RemovalRuntime.current()
    assert runtime.protected_images == frozenset((retained.image,))
    assert runtime.forced_removals == [removed.image]


@pytest.mark.asyncio
async def test_build_remove_validates_every_repository_before_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run_directory = _build_run(tmp_path)
    original = _environment("Owner", "One", "a")
    output = run_directory / "environments.jsonl"
    write_environment_snapshot(output, (original,))
    before = output.read_bytes()
    monkeypatch.setattr(build_application, "DockerRuntime", RemovalRuntime)

    with pytest.raises(ValueError, match="no Environment"):
        await remove_build_environments(
            project_root=project,
            run_directory=run_directory,
            repositories=("owner/one", "missing/repository"),
        )

    assert output.read_bytes() == before
    assert not RunLayout(run_directory).removals.exists()
