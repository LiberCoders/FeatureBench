from __future__ import annotations

import asyncio
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from codeworld.cli import (
    InterruptController,
    build_parser,
    execution_arguments,
    invocation_from_arguments,
    main,
    resolve_workspace_root,
)
from codeworld.console import config_tui
from codeworld.console.models import (
    RunManifest,
    default_config_paths,
    load_run_invocation,
    write_run_invocation_manifest,
    write_run_manifest,
)
from codeworld.errors import ConfigurationError, ContractError
from codeworld.run_layout import RunLayout


def test_project_exposes_only_featurebench_cli_names() -> None:
    project_root = Path(__file__).resolve().parents[2]
    document = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert document["project"]["scripts"] == {
        "fb": "featurebench.cli:main",
        "featurebench": "featurebench.cli:main",
    }
    assert build_parser().prog == "fb"


def test_bare_cli_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Discover repositories, build environments" in output
    assert "fb discover" in output
    assert "fb --resume" in output


@pytest.mark.asyncio
async def test_repeated_interrupt_requests_do_not_cancel_cleanup_twice() -> None:
    messages: list[str] = []
    operation_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def operation() -> int:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
        return 0

    controller: InterruptController[int] = InterruptController(messages.append)
    task = asyncio.create_task(controller.run(operation()))
    await operation_started.wait()
    controller.request_shutdown()
    await cleanup_started.wait()
    controller.request_shutdown()

    assert not task.done()
    assert messages == [
        "Stopping safely; cleaning data-factory resources…",
        "Cleanup already in progress; additional Ctrl+C ignored.",
    ]

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_configure_command_launches_the_run_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _project(tmp_path)
    observed: dict[str, Path] = {}

    def fake_launch(project_root: Path, run_directory: Path) -> int:
        observed["project_root"] = project_root
        observed["run_directory"] = run_directory
        return 2

    monkeypatch.chdir(root)
    monkeypatch.setattr(config_tui, "launch_configuration_editor", fake_launch)

    assert main(["configure", "outputs/build-test"]) == 0
    assert observed == {
        "project_root": root.resolve(),
        "run_directory": Path("outputs/build-test"),
    }
    assert "Saved: 2 run input change(s)" in capsys.readouterr().out


def _project(tmp_path: Path) -> Path:
    (tmp_path / "configs" / "default").mkdir(parents=True)
    (tmp_path / "configs" / "default" / "adapters").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'featurebench'\n", encoding="utf-8")
    for name in ("host", "sources", "discover", "build"):
        (tmp_path / "configs" / "default" / f"{name}.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "configs" / "default" / "adapters" / "featurebench.yaml").write_text(
        "featurebench:\n  environment_concurrency: 2\n", encoding="utf-8"
    )
    return tmp_path


def test_workspace_root_is_the_exact_requested_directory(tmp_path: Path) -> None:
    root = _project(tmp_path)
    assert resolve_workspace_root(root) == root.resolve()
    child = root / "child"
    child.mkdir()
    assert resolve_workspace_root(child) == child.resolve()


def test_workspace_root_must_be_a_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ConfigurationError, match="not a directory"):
        resolve_workspace_root(missing)


def test_default_config_paths_only_load_stage_settings(tmp_path: Path) -> None:
    root = _project(tmp_path)

    assert default_config_paths(root, "discover") == (
        root / "configs" / "default" / "discover.yaml",
    )
    assert default_config_paths(root, "build") == (root / "configs" / "default" / "build.yaml",)
    assert default_config_paths(root, "generate", adapter="featurebench") == (
        root / "configs" / "default" / "adapters" / "featurebench.yaml",
    )
    assert default_config_paths(root, "generate", adapter="custom") == (
        root / "configs" / "default" / "adapters" / "custom.yaml",
    )


def test_default_config_paths_fall_back_to_packaged_profiles(tmp_path: Path) -> None:
    expected = Path(__file__).resolve().parents[2] / "configs" / "default"

    assert default_config_paths(tmp_path, "discover") == (expected / "discover.yaml",)
    assert default_config_paths(tmp_path, "build") == (expected / "build.yaml",)
    assert default_config_paths(tmp_path, "generate", adapter="featurebench") == (
        expected / "adapters" / "featurebench.yaml",
    )


def test_build_unifies_one_repository_and_repository_jsonl(tmp_path: Path) -> None:
    root = _project(tmp_path)
    single = invocation_from_arguments(build_parser().parse_args(["build", "owner/repo"]), root)
    single_args = execution_arguments(single)
    assert single_args.repository == "https://github.com/owner/repo.git"
    assert single_args.repositories is None
    assert single.output_directory.parent == root / "outputs"

    repositories = root / "repositories.jsonl"
    repositories.write_text("\n", encoding="utf-8")
    batch = invocation_from_arguments(
        build_parser().parse_args(["build", str(repositories), "--limit", "10"]), root
    )
    batch_args = execution_arguments(batch)
    assert batch_args.repository is None
    assert batch_args.repositories == repositories
    assert batch_args.limit == 10


def test_build_parser_reserves_remove_operands_without_weakening_normal_builds(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    removal = build_parser().parse_args(
        ["build", "remove", "outputs/build-test", "owner/one", "another/two"]
    )
    assert removal.source == "remove"
    assert removal.operands == ["outputs/build-test", "owner/one", "another/two"]

    ordinary = build_parser().parse_args(["build", "owner/one", "unexpected/second"])
    with pytest.raises(ValueError, match="accepts one repository source"):
        invocation_from_arguments(ordinary, root)


def test_discover_accepts_a_durable_txt_repository_source(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source = root / "repositories.txt"
    source.write_text("owner/repo\n", encoding="utf-8")
    invocation = invocation_from_arguments(
        build_parser().parse_args(
            ["discover", "repositories.txt", "--output", "outputs/discover-test"]
        ),
        root,
    )

    assert invocation.repository_source == str(source.resolve())
    assert execution_arguments(invocation).discovery_targets == source.resolve()

    manifest = write_run_manifest(invocation, "discover-run")
    resumed = load_run_invocation(root, manifest.parent)
    assert resumed.repository_source == str(source.resolve())


def test_generate_snapshots_environments_and_writes_task_jsonl(tmp_path: Path) -> None:
    root = _project(tmp_path)
    environments = root / "environments.jsonl"
    environments.write_text("{}\n", encoding="utf-8")
    invocation = invocation_from_arguments(
        build_parser().parse_args(
            [
                "generate",
                str(environments),
                "--environment-id",
                "environment-1",
                "--adapter",
                "featurebench",
            ]
        ),
        root,
    )
    arguments = execution_arguments(invocation)

    assert invocation.command == "generate"
    assert invocation.adapter == "featurebench"
    assert invocation.config_files[-1] == (
        root / "configs" / "default" / "adapters" / "featurebench.yaml"
    )
    assert arguments.environment == environments
    assert arguments.environment_id == "environment-1"
    assert arguments.limit is None
    assert arguments.output == invocation.output_directory / "tasks.jsonl"


def test_generate_accepts_a_positive_environment_limit(tmp_path: Path) -> None:
    root = _project(tmp_path)
    environments = root / "environments.jsonl"
    environments.write_text("{}\n", encoding="utf-8")
    invocation = invocation_from_arguments(
        build_parser().parse_args(
            [
                "generate",
                str(environments),
                "--limit",
                "10",
                "--adapter",
                "featurebench",
            ]
        ),
        root,
    )

    assert invocation.environment_id is None
    assert invocation.limit == 10
    assert execution_arguments(invocation).limit == 10


def test_generate_rejects_limit_with_one_environment_id(tmp_path: Path) -> None:
    root = _project(tmp_path)
    environments = root / "environments.jsonl"
    environments.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        invocation_from_arguments(
            build_parser().parse_args(
                [
                    "generate",
                    str(environments),
                    "--limit",
                    "1",
                    "--environment-id",
                    "environment-1",
                    "--adapter",
                    "featurebench",
                ]
            ),
            root,
        )


def test_generate_requires_an_explicit_adapter(tmp_path: Path) -> None:
    root = _project(tmp_path)
    environments = root / "environments.jsonl"
    environments.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate", str(environments)])


def test_run_directory_is_the_complete_resume_handle(tmp_path: Path) -> None:
    root = _project(tmp_path)
    invocation = invocation_from_arguments(
        build_parser().parse_args(["build", "owner/repo", "--output", "outputs/build-test"]),
        root,
    )
    manifest = write_run_manifest(invocation, "run-123")
    assert manifest == root / "outputs" / "build-test" / "run.yaml"

    resumed = load_run_invocation(root, manifest.parent)
    assert resumed.resume is True
    assert resumed.run_id == "run-123"
    assert resumed.command == "build"
    assert resumed.repository_source == "owner/repo"
    assert resumed.environments_path == manifest.parent / "environments.jsonl"

    with pytest.raises(ValueError, match="not empty"):
        write_run_manifest(invocation, "another-run")


def test_run_layout_keeps_only_stage_contracts_visible(tmp_path: Path) -> None:
    root = _project(tmp_path)
    invocation = invocation_from_arguments(
        build_parser().parse_args(["build", "owner/repo", "--output", "outputs/build-test"]),
        root,
    )
    layout = RunLayout(invocation.output_directory)
    layout.lock.parent.mkdir(parents=True)
    layout.lock.touch()

    write_run_manifest(invocation, "run-123")
    write_run_invocation_manifest(invocation, "run-123")

    assert {path.name for path in layout.root.iterdir()} == {".featurebench", "run.yaml"}
    assert layout.state.parent == layout.internal
    assert (layout.invocations / "0001" / "run.yaml").is_file()


def test_actual_run_invocations_append_immutable_manifests(tmp_path: Path) -> None:
    root = _project(tmp_path)
    invocation = invocation_from_arguments(
        build_parser().parse_args(["build", "owner/repo", "--output", "outputs/build-test"]),
        root,
    )
    write_run_manifest(invocation, "run-123")

    first = write_run_invocation_manifest(invocation, "run-123")
    first_bytes = first.read_bytes()
    second = write_run_invocation_manifest(
        replace(invocation, resume=True, selection="non-success", run_id="run-123"),
        "run-123",
    )

    assert first.parent.name == "0001"
    assert second.parent.name == "0002"
    assert first.read_bytes() == first_bytes
    assert "version" not in yaml.safe_load(first_bytes)
    assert yaml.safe_load(first_bytes)["selection"] == "unseen"
    assert yaml.safe_load(second.read_bytes())["selection"] == "non-success"


def test_run_manifest_rejects_a_manifest_without_selection() -> None:
    with pytest.raises(ValueError, match="selection"):
        RunManifest.model_validate(
            {
                "run_id": "run-123",
                "command": "build",
                "config_files": ["configs/default/build.yaml"],
                "output_directory": "outputs/build-test",
                "repository_source": "owner/repo",
            }
        )


def test_resume_accepts_and_canonicalizes_a_one_shot_work_selection(tmp_path: Path) -> None:
    root = _project(tmp_path)
    invocation = invocation_from_arguments(
        build_parser().parse_args(["build", "owner/repo", "--output", "outputs/build-test"]),
        root,
    )
    write_run_manifest(invocation, "run-123")

    resumed = invocation_from_arguments(
        build_parser().parse_args(
            [
                "--resume",
                "outputs/build-test",
                "--select",
                "rejected+unseen+interrupted",
            ]
        ),
        root,
    )

    assert resumed.selection == "unseen+rejected+interrupted"
    assert resumed.resume is True
    invocation_manifest = write_run_invocation_manifest(resumed, "run-123")
    assert yaml.safe_load(invocation_manifest.read_text())["selection"] == (
        "unseen+rejected+interrupted"
    )
    with pytest.raises(ContractError, match="did you mean 'rejected'"):
        invocation_from_arguments(
            build_parser().parse_args(
                ["--resume", "outputs/build-test", "--select", "unseen+rejcted"]
            ),
            root,
        )
