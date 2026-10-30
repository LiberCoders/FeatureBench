from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeworld.adapters.catalog import default_adapter_catalog
from codeworld.cli import build_parser
from codeworld.config import FactorySettings
from codeworld.console.configuration import RunConfigurationSession
from codeworld.console.models import (
    CliInvocation,
    load_run_invocation,
    write_run_manifest,
)
from codeworld.errors import ContractError
from codeworld.run_layout import RunLayout
from codeworld.state import RunLock, StateStore


def _project(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='codeworld'\n", encoding="utf-8")
    return tmp_path


def _unfinished_run(
    project: Path,
    *,
    command: str = "build",
    settings: FactorySettings | None = None,
) -> tuple[Path, Path, FactorySettings]:
    config = project / "configs" / f"{command}.yaml"
    config.write_text("concurrency: 4\n", encoding="utf-8")
    resolved = settings or FactorySettings.from_files((config,))
    run_directory = project / "outputs" / f"{command}-test"
    invocation = CliInvocation(
        command=command,  # pyright: ignore[reportArgumentType]
        project_root=project,
        config_files=(config,),
        output_directory=run_directory,
        repository_source="owner/repo" if command == "build" else None,
        environment_source="environments.jsonl" if command == "generate" else None,
        adapter="featurebench" if command == "generate" else None,
    )
    write_run_manifest(invocation, "run-123")
    state = StateStore(RunLayout(run_directory).state)
    state.create_run("run-123", command, resolved.model_dump(mode="json"))
    state.interrupt_run("run-123", "stopped")
    state.close()
    return run_directory, config, resolved


def test_configure_command_accepts_one_run_directory() -> None:
    parsed = build_parser().parse_args(["configure", "outputs/build-test"])

    assert parsed.command == "configure"
    assert parsed.run_directory == Path("outputs/build-test")


def test_resume_reloads_current_manifest_yaml_and_refreshes_cached_state(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    run_directory, source_config, _ = _unfinished_run(project)
    source_config.write_text("concurrency: 9\n", encoding="utf-8")

    resumed = load_run_invocation(project, run_directory)
    revised = FactorySettings.from_files(resumed.config_files)
    assert revised.concurrency == 9

    state = StateStore(RunLayout(run_directory).state)
    state.resume_run("run-123", "build", revised.model_dump(mode="json"))
    _, stored = state.load_run_config("run-123", "build")
    state.close()
    assert stored == revised.model_dump(mode="json")


def test_run_editor_switches_the_manifest_to_another_validated_yaml(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    run_directory, source_config, _ = _unfinished_run(project)
    replacement_config = project / "configs" / "revised.yaml"
    replacement_config.write_text("concurrency: 8\n", encoding="utf-8")

    with RunConfigurationSession(project, run_directory) as session:
        assert [entry.name for entry in session.group_entries("execution")] == [
            "config_files",
            "source",
            "ref",
            "limit",
        ]
        result = session.edit(
            ("@execution", "config_files"),
            "[configs/revised.yaml]",
        )
        assert result.old_value == ["configs/build.yaml"]
        assert result.new_value == ["configs/revised.yaml"]
        assert session.pending_change_count == 1
        assert not (run_directory / "config.yaml").exists()
        saved = session.save()
        assert saved.count == 1
        assert session.pending_change_count == 0

    assert source_config.read_text(encoding="utf-8") == "concurrency: 4\n"
    assert not (run_directory / "config.yaml").exists()
    resumed = load_run_invocation(project, run_directory)
    assert resumed.config_files == (replacement_config,)
    revised = FactorySettings.from_files(resumed.config_files)
    assert revised.concurrency == 8

    layout = RunLayout(run_directory)
    state = StateStore(layout.state)
    state.resume_run("run-123", "build", revised.model_dump(mode="json"))
    _, stored = state.load_run_config("run-123", "build")
    state.close()
    assert stored == revised.model_dump(mode="json")

    history = [
        json.loads(line) for line in layout.config_changes.read_text(encoding="utf-8").splitlines()
    ]
    assert history[-1]["changes"] == [
        {
            "path": "config_files",
            "old_value": ["configs/build.yaml"],
            "new_value": ["configs/revised.yaml"],
        }
    ]


def test_run_editor_rejects_invalid_values_without_changing_the_run(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, original = _unfinished_run(project)
    original_manifest = (run_directory / "run.yaml").read_bytes()

    with (
        RunConfigurationSession(project, run_directory) as session,
        pytest.raises(ValueError, match="at least one YAML"),
    ):
        session.edit(("@execution", "config_files"), "[]")

    assert (run_directory / "run.yaml").read_bytes() == original_manifest
    assert not (run_directory / "config.yaml").exists()
    state = StateStore(RunLayout(run_directory).state)
    _, stored = state.load_run_config("run-123", "build")
    state.close()
    assert stored == original.model_dump(mode="json")


def test_run_editor_discards_an_unconfirmed_valid_draft(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, original = _unfinished_run(project)
    original_manifest = (run_directory / "run.yaml").read_bytes()

    with RunConfigurationSession(project, run_directory) as session:
        session.edit(("@execution", "limit"), "8")
        assert session.pending_change_count == 1

    assert (run_directory / "run.yaml").read_bytes() == original_manifest
    assert not (run_directory / "config.yaml").exists()
    layout = RunLayout(run_directory)
    assert not layout.config_changes.exists()
    state = StateStore(layout.state)
    _, stored = state.load_run_config("run-123", "build")
    state.close()
    assert stored == original.model_dump(mode="json")


def test_run_editor_tracks_only_distinct_net_draft_changes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, _ = _unfinished_run(project)

    with RunConfigurationSession(project, run_directory) as session:
        session.edit(("@execution", "limit"), "8")
        session.edit(("@execution", "limit"), "16")
        assert session.pending_changes[0].old_value is None
        assert session.pending_changes[0].new_value == 16
        assert session.pending_change_count == 1

        session.edit(("@execution", "limit"), "null")
        assert session.pending_change_count == 0
        assert session.save().count == 0

    assert not (run_directory / "config.yaml").exists()
    assert not RunLayout(run_directory).config_changes.exists()


def test_run_editor_searches_only_run_inputs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, _ = _unfinished_run(project)

    with RunConfigurationSession(project, run_directory) as session:
        config_results = session.search("config build")
        assert any(result.entry.path == ("@execution", "config_files") for result in config_results)
        limit_results = session.search("limit")
        assert any(result.entry.path == ("@execution", "limit") for result in limit_results)
        assert not session.search("docker timeout")


def test_run_editor_hides_adapter_yaml_fields_and_can_switch_adapter_config(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    adapter_config = project / "configs" / "generate.yaml"
    adapter_document = (
        "featurebench:\n  environment_concurrency: 2\n  gold_patch_line_minimum: 200\n"
    )
    adapter_config.write_text(adapter_document, encoding="utf-8")
    (project / "environments.jsonl").write_text("\n", encoding="utf-8")
    catalog = default_adapter_catalog()
    settings = FactorySettings.from_files((adapter_config,), adapter_catalog=catalog)
    run_directory, _, _ = _unfinished_run(
        project,
        command="generate",
        settings=settings,
    )
    active_config = project / "configs" / "generate.yaml"
    active_config.write_text(adapter_document, encoding="utf-8")
    revised_config = project / "configs" / "generate-revised.yaml"
    revised_config.write_text(
        "featurebench:\n  environment_concurrency: 2\n  gold_patch_line_minimum: 300\n",
        encoding="utf-8",
    )

    with RunConfigurationSession(
        project,
        run_directory,
        adapter_catalog=catalog,
    ) as session:
        assert [entry.name for entry in session.group_entries("execution")] == [
            "config_files",
            "environment",
            "environment_id",
            "limit",
        ]
        with pytest.raises(KeyError):
            session.entry(("featurebench", "gold_patch_line_minimum"))
        result = session.edit(
            ("@execution", "config_files"),
            "[configs/generate-revised.yaml]",
        )
        assert result.new_value == ["configs/generate-revised.yaml"]
        session.save()

    resumed = load_run_invocation(project, run_directory)
    revised = FactorySettings.from_files(resumed.config_files, adapter_catalog=catalog)
    assert revised.adapters["featurebench"]["gold_patch_line_minimum"] == 300


def test_run_editor_updates_and_reopens_completed_runs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, _ = _unfinished_run(project)
    state = StateStore(RunLayout(run_directory).state)
    state.finish_run("run-123")
    state.close()

    with RunConfigurationSession(project, run_directory) as session:
        assert session.status == "completed"
        result = session.edit(("@execution", "limit"), "1000")
        assert result.new_value == 1000
        session.save()

    resumed = load_run_invocation(project, run_directory)
    assert resumed.limit == 1000
    state = StateStore(RunLayout(run_directory).state)
    state.resume_run("run-123", "build", FactorySettings().model_dump(mode="json"))
    state.close()


def test_run_editor_persists_build_execution_inputs_with_full_validation(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    run_directory, _, _ = _unfinished_run(project)
    repositories = project / "repositories.jsonl"
    repositories.write_text("\n", encoding="utf-8")

    with RunConfigurationSession(project, run_directory) as session:
        session.edit(("@execution", "source"), str(repositories))
        session.edit(("@execution", "limit"), "1000")
        with pytest.raises(ValueError, match="ref only applies"):
            session.edit(("@execution", "ref"), "main")
        session.save()

    resumed = load_run_invocation(project, run_directory)
    assert resumed.repository_source == str(repositories)
    assert resumed.limit == 1000
    assert resumed.ref is None


def test_run_editor_persists_discover_and_generate_execution_inputs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    targets = project / "targets.txt"
    targets.write_text("owner/repo\n", encoding="utf-8")
    discover_run, _, _ = _unfinished_run(project, command="discover")
    with RunConfigurationSession(project, discover_run) as session:
        session.edit(("@execution", "source"), str(targets))
        session.save()
    assert load_run_invocation(project, discover_run).repository_source == str(targets)

    environments = project / "environments.jsonl"
    environments.write_text("\n", encoding="utf-8")
    generate_run, _, _ = _unfinished_run(project, command="generate")
    with RunConfigurationSession(
        project,
        generate_run,
        adapter_catalog=default_adapter_catalog(),
    ) as session:
        session.edit(("@execution", "environment"), str(environments))
        session.edit(("@execution", "environment_id"), "environment-2")
        session.save()
    resumed = load_run_invocation(project, generate_run)
    assert resumed.environment_source == str(environments)
    assert resumed.environment_id == "environment-2"

    with RunConfigurationSession(
        project,
        generate_run,
        adapter_catalog=default_adapter_catalog(),
    ) as session:
        session.edit(("@execution", "environment_id"), "null")
        session.edit(("@execution", "limit"), "25")
        session.save()
    resumed = load_run_invocation(project, generate_run)
    assert resumed.environment_id is None
    assert resumed.limit == 25


def test_run_editor_rejects_a_concurrently_active_run(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_directory, _, _ = _unfinished_run(project)

    with (
        RunLock(RunLayout(run_directory).lock),
        pytest.raises(ContractError, match="already active"),
        RunConfigurationSession(project, run_directory),
    ):
        pass
