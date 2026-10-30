from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from codeworld.display import (
    AdapterDashboard,
    BuildDashboard,
    DiscoveryDashboard,
)
from codeworld.events import Event
from codeworld.progress import AdapterProgressSnapshot, AdapterWorkItemProgress


@pytest.mark.asyncio
async def test_discovery_dashboard_shows_only_user_facing_planning_progress(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=100, color_system=None)
    dashboard = DiscoveryDashboard(
        output_directory=tmp_path,
        search_mode=True,
        console=console,
    )
    timestamp = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="discovery_progress",
            stage="repository-discovery",
            timestamp=timestamp - timedelta(seconds=35),
            data={
                "phase": "planning",
                "progress": 0.0,
                "current_language": "Python",
                "search_group_index": 1,
                "search_group_total": 20,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="discovery_progress",
            stage="repository-discovery",
            timestamp=timestamp,
            data={
                "phase": "planning",
                "progress": 0.35,
                "current_language": "Python",
                "search_group_index": 1,
                "search_group_total": 20,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="partition_probed",
            stage="repository-discovery",
            message="2016-01-01..2016-03-01: 602",
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Step 1 of 2 · Planning GitHub search" in rendered
    assert "35% · ETA ≈" in rendered
    assert "Python · 1/20 filters" in rendered
    assert str(tmp_path) in rendered
    assert "partition_probed" not in rendered
    assert "2016-01-01" not in rendered
    assert "Repository jobs" not in rendered
    assert "Recent events" not in rendered


@pytest.mark.asyncio
async def test_discovery_dashboard_switches_to_fetching_and_rate_limit_countdown(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=100, color_system=None)
    dashboard = DiscoveryDashboard(
        output_directory=tmp_path,
        search_mode=True,
        console=console,
    )
    timestamp = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="discovery_progress",
            stage="repository-discovery",
            timestamp=timestamp,
            data={
                "phase": "fetching",
                "progress": 0.4,
                "repositories": 40,
                "repository_target": 100,
                "completed_jobs": 8,
                "total_jobs": 20,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="rate_limited",
            stage="repository-discovery",
            timestamp=timestamp,
            data={"cooldown_seconds": 60},
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Step 2 of 2 · Fetching repository metadata" in rendered
    assert "40% · ETA estimating" in rendered
    assert "40 / 100 target" in rendered
    assert "GitHub rate limit · retrying in" in rendered


@pytest.mark.asyncio
async def test_build_dashboard_summarizes_repository_outcomes_without_raw_events(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=120, color_system=None)
    dashboard = BuildDashboard(
        output_directory=tmp_path,
        repository_target=2,
        model="openai:h:gpt-5.6-luna",
        console=console,
    )
    timestamp = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="materialize_started",
            stage="environment-build",
            task_id="example/one",
            timestamp=timestamp - timedelta(seconds=40),
            message="materializing repository",
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="environment_completed",
            stage="environment-build",
            task_id="example/one",
            timestamp=timestamp - timedelta(seconds=30),
            message="validated environment is durable in output JSONL",
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="environment_rejected",
            stage="environment-build",
            task_id="example/two",
            timestamp=timestamp,
            message="repository is outside the configured acceptance policy",
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="analysis_agent_agent_run_metrics",
            stage="environment-build",
            task_id="example/one",
            timestamp=timestamp - timedelta(seconds=20),
            data={
                "run_id": "analysis-1",
                "model": "openai:h:gpt-5.6-luna",
                "input_tokens": 100_000,
                "output_tokens": 2_000,
                "cache_read_tokens": 75_000,
                "cache_write_tokens": 1_000,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="artifact_agent_agent_run_metrics",
            stage="environment-build",
            task_id="example/two",
            timestamp=timestamp - timedelta(seconds=10),
            data={
                "run_id": "artifact-1",
                "model": "openai:h:gpt-5.6-luna",
                "input_tokens": 23_456,
                "output_tokens": 3_678,
                "cache_read_tokens": 5_000,
                "cache_write_tokens": 200,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="environment_production_completed",
            stage="environment-build",
            timestamp=timestamp,
            data={"submitted": 2, "built": 1, "skipped": 0, "rejected": 1, "failed": 0},
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Build complete" in rendered
    assert "100% · 2 / 2 repositories" in rendered
    results_line = next(line for line in rendered.splitlines() if "Results" in line)
    assert "Built 1" in results_line
    assert "Skipped 0" in results_line
    assert "Rejected 1" in results_line
    assert "Failed 0" in results_line
    status_line = next(line for line in rendered.splitlines() if "Status" in line)
    assert "Complete" in status_line
    assert "Elapsed" in status_line
    assert not any(line.lstrip().startswith("Elapsed") for line in rendered.splitlines())
    usage_line = next(line for line in rendered.splitlines() if "Usage" in line)
    assert "Model openai:h:gpt-5.6-luna" in usage_line
    assert "Input 123,456" in usage_line
    assert "Output 5,678" in usage_line
    assert "Cache read 80,000" in usage_line
    assert "Cache write 1,200" in usage_line
    assert str(tmp_path) in rendered
    assert "environment_completed" not in rendered
    assert "Repository jobs" not in rendered
    assert "Recent events" not in rendered


@pytest.mark.asyncio
async def test_build_dashboard_active_jobs_adapt_to_terminal_height(tmp_path: Path) -> None:
    async def render_active_jobs(height: int) -> str:
        console = Console(record=True, width=120, height=height, color_system=None)
        dashboard = BuildDashboard(
            output_directory=tmp_path,
            repository_target=8,
            console=console,
        )
        timestamp = datetime.now(UTC)
        for index in range(8):
            await dashboard.emit(
                Event(
                    run_id="run",
                    kind="analysis_agent_started",
                    stage="environment-build",
                    task_id=f"example/repository-{index}",
                    timestamp=timestamp + timedelta(seconds=index),
                )
            )
        console.print(dashboard.live.get_renderable())
        return console.export_text()

    short = await render_active_jobs(16)
    tall = await render_active_jobs(40)
    short_count = sum(f"example/repository-{index}" in short for index in range(8))
    tall_count = sum(f"example/repository-{index}" in tall for index in range(8))

    assert 1 <= short_count < tall_count
    assert tall_count == 8


@pytest.mark.asyncio
async def test_build_dashboard_keeps_eta_on_progress_row(tmp_path: Path) -> None:
    console = Console(record=True, width=120, color_system=None)
    dashboard = BuildDashboard(
        output_directory=tmp_path,
        repository_target=4,
        console=console,
    )
    timestamp = datetime.now(UTC)
    for index, offset in enumerate((-60, 0), start=1):
        await dashboard.emit(
            Event(
                run_id="run",
                kind="environment_completed",
                stage="environment-build",
                task_id=f"example/{index}",
                timestamp=timestamp + timedelta(seconds=offset),
            )
        )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "50% · 2 / 4 repositories · ETA ≈ 2m 00s" in rendered
    assert not any(line.lstrip().startswith("≈") for line in rendered.splitlines())


@pytest.mark.asyncio
async def test_build_dashboard_shows_current_build_step_and_validation_phase(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=180, color_system=None)
    dashboard = BuildDashboard(
        output_directory=tmp_path,
        repository_target=2,
        console=console,
    )
    timestamp = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="build_progress",
            stage="environment-build",
            task_id="example/building",
            timestamp=timestamp - timedelta(seconds=20),
            data={
                "attempt": 2,
                "max_attempts": 10,
                "step_id": "7",
                "step_stage": "featurebench_managed_base 3/5",
                "step_instruction": "RUN uv sync --frozen",
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="validation_phase_started",
            stage="environment-build",
            task_id="example/verifying",
            timestamp=timestamp - timedelta(seconds=10),
            data={
                "attempt": 1,
                "phase": "quickcheck",
                "phase_label": "Running quickcheck",
            },
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "example/building · Building environment · attempt 2/10" in rendered
    assert "#7 [featurebench_managed_base 3/5] RUN uv sync --frozen" in rendered
    assert (
        "example/verifying · Verifying environment · attempt 1 · current 10s · Running quickcheck"
    ) in rendered
    assert "current " in rendered


@pytest.mark.asyncio
async def test_build_dashboard_keeps_active_rows_single_line_and_stable(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=100, color_system=None)
    dashboard = BuildDashboard(
        output_directory=tmp_path,
        repository_target=2,
        console=console,
    )
    timestamp = datetime.now(UTC)
    long_instruction = "RUN set -eux; " + " && ".join(
        f"install-component-{index}" for index in range(20)
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="build_progress",
            stage="environment-build",
            task_id="example/first",
            timestamp=timestamp,
            data={
                "attempt": 2,
                "max_attempts": 10,
                "step_id": "18",
                "step_stage": "stage-1 5/12",
                "step_instruction": long_instruction,
            },
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="artifact_agent_started",
            stage="environment-build",
            task_id="example/second",
            timestamp=timestamp + timedelta(seconds=1),
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()
    active_lines = [
        line
        for line in rendered.splitlines()
        if "example/first" in line or "example/second" in line
    ]

    assert len(active_lines) == 2
    assert "example/first" in active_lines[0]
    assert "example/second" in active_lines[1]
    assert active_lines[0].index("attempt 2/10") < active_lines[0].index("current")
    assert active_lines[0].index("current") < active_lines[0].index("#18")
    assert "#18 [stage-1" in active_lines[0]
    assert "…" in active_lines[0]


@pytest.mark.asyncio
async def test_build_dashboard_uses_semantic_color_roles(tmp_path: Path) -> None:
    dashboard = BuildDashboard(output_directory=tmp_path, repository_target=1)
    now = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="build_progress",
            stage="environment-build",
            task_id="example/project",
            timestamp=now,
            data={
                "attempt": 1,
                "max_attempts": 10,
                "step_id": "1",
                "step_instruction": "RUN uv sync",
            },
        )
    )

    results = dashboard._results_text()  # pyright: ignore[reportPrivateUsage]
    activity = dashboard._activity_text(  # pyright: ignore[reportPrivateUsage]
        dashboard.jobs["example/project"], now
    )
    usage = dashboard._usage_text()  # pyright: ignore[reportPrivateUsage]

    assert {str(span.style) for span in results.spans} >= {"green", "yellow", "red"}
    assert {str(span.style) for span in activity.spans} >= {
        "bold cyan",
        "yellow",
        "bright_black",
    }
    assert {str(span.style) for span in usage.spans} >= {
        "magenta",
        "cyan",
        "bright_blue",
    }


def test_build_dashboard_status_and_elapsed_share_status_style(tmp_path: Path) -> None:
    dashboard = BuildDashboard(output_directory=tmp_path, repository_target=1)

    rendered = dashboard._status_elapsed_text(datetime.now(UTC))  # pyright: ignore[reportPrivateUsage]

    assert rendered.style == "cyan"
    assert all(span.style != "dim" for span in rendered.spans)


@pytest.mark.asyncio
async def test_adapter_dashboard_renders_only_normalized_user_progress(tmp_path: Path) -> None:
    console = Console(record=True, width=120, color_system=None)
    dashboard = AdapterDashboard(
        adapter="featurebench",
        output_directory=tmp_path,
        console=console,
    )
    timestamp = datetime.now(UTC)
    for completed, event_time in (
        (1, timestamp - timedelta(seconds=30)),
        (6, timestamp),
    ):
        snapshot = AdapterProgressSnapshot(
            adapter="featurebench",
            adapter_name="FeatureBench",
            status="working",
            phase_id="analyze_tests",
            phase_label="Tracing",
            completed=completed,
            total=10,
            unit="test files",
            current="tests/test_api.py",
            tasks_created=2,
            candidates_rejected=1,
            candidates_failed=3,
        )
        await dashboard.emit(
            Event(
                run_id="run",
                kind="adapter_progress",
                stage="featurebench",
                timestamp=event_time,
                data=snapshot.model_dump(mode="json"),
            )
        )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="dynamic_trace",
            stage="featurebench",
            message="internal trace detail",
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Generate · FeatureBench" in rendered
    assert "Tracing" in rendered
    assert "2/5" not in rendered
    assert "Step 2 of 5" not in rendered
    assert "60% · 6 / 10 test files · ETA ≈" in rendered
    assert "Tasks" in rendered and "Created 2" in rendered
    assert "Candidates rejected 1" in rendered
    assert "failed 3" in rendered
    assert "tests/test_api.py" in rendered
    assert str(tmp_path) in rendered
    assert "dynamic_trace" not in rendered
    assert "internal trace detail" not in rendered


@pytest.mark.asyncio
async def test_adapter_dashboard_completes_without_a_declared_progress_total(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=100, color_system=None)
    dashboard = AdapterDashboard(
        adapter="example",
        output_directory=tmp_path,
        console=console,
    )
    snapshot = AdapterProgressSnapshot(
        adapter="example",
        adapter_name="Example",
        status="completed",
        tasks_created=3,
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="adapter_progress",
            stage="example",
            data=snapshot.model_dump(mode="json"),
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Generate complete" in rendered
    assert "100% · Complete" in rendered
    assert "Tasks" in rendered and "Created 3" in rendered
    assert "None" not in rendered


@pytest.mark.asyncio
async def test_adapter_dashboard_aggregates_parallel_environment_progress(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=110, color_system=None)
    dashboard = AdapterDashboard(
        adapter="featurebench",
        output_directory=tmp_path,
        console=console,
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_snapshot_captured",
            stage="featurebench",
            data={"selected_environments": 2},
        )
    )
    for environment_id in ("environment-1", "environment-2"):
        await dashboard.emit(
            Event(
                run_id="run",
                kind="generation_environment_started",
                stage="featurebench",
                data={"environment_id": environment_id},
            )
        )
        snapshot = AdapterProgressSnapshot(
            adapter="featurebench",
            adapter_name="FeatureBench",
            environment_id=environment_id,
            status="working",
            phase_id="analyze_tests",
            phase_label="Tracing",
            tasks_created=1,
        )
        await dashboard.emit(
            Event(
                run_id="run",
                kind="adapter_progress",
                stage="featurebench",
                data=snapshot.model_dump(mode="json"),
            )
        )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_environment_completed",
            stage="featurebench",
            data={"environment_id": "environment-1"},
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Generating tasks" in rendered
    assert "50% · 1 / 2 environments" in rendered
    assert "Tasks" in rendered and "Created 2" in rendered
    assert "Results" in rendered and "Completed 1" in rendered
    assert "Active" in rendered and "environment-2" in rendered
    assert "Generate complete" not in rendered


@pytest.mark.asyncio
async def test_adapter_dashboard_keeps_created_count_invocation_scoped(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=110, color_system=None)
    dashboard = AdapterDashboard(
        adapter="featurebench",
        output_directory=tmp_path,
        console=console,
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_snapshot_captured",
            stage="featurebench",
            data={"selected_environments": 1},
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="completed",
            stage="featurebench",
            data={
                "completed": 1,
                "failed": 0,
                "rejected": 0,
                "skipped": 0,
                "tasks_created": 1,
                "tasks_total": 7,
            },
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()

    assert "Generate complete" in rendered
    assert "Created 1" in rendered
    assert "Created 7" not in rendered


@pytest.mark.asyncio
async def test_adapter_dashboard_is_generic_for_custom_adapters(tmp_path: Path) -> None:
    console = Console(record=True, width=112, color_system=None)
    dashboard = AdapterDashboard(
        adapter="custom-adapter",
        output_directory=tmp_path,
        console=console,
    )
    timestamp = datetime.now(UTC)
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_snapshot_captured",
            stage="custom-adapter",
            data={"selected_environments": 1},
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_environment_started",
            stage="custom-adapter",
            timestamp=timestamp,
            data={
                "environment_id": "environment-1",
                "repository": "https://git.example/acme/project.git",
                "attempt": 2,
            },
        )
    )
    snapshot = AdapterProgressSnapshot(
        adapter="custom-adapter",
        adapter_name="Custom Adapter",
        environment_id="environment-1",
        status="working",
        phase_id="inspect_sources",
        phase_label="Inspecting sources",
        completed=2,
        total=9,
        unit="files",
        current="src/" + "deeply-nested/" * 20 + "module.py",
        work_items=(
            AdapterWorkItemProgress(
                item_id="src/module.py",
                activity="Resolving imports",
                current="src/" + "deeply-nested/" * 20 + "module.py",
            ),
        ),
        tasks_created=4,
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="adapter_progress",
            stage="custom-adapter",
            timestamp=timestamp + timedelta(seconds=1),
            data=snapshot.model_dump(mode="json"),
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="custom_planner_agent_run_started",
            stage="custom-adapter",
            data={"run_id": "agent-1", "model": "openai:h:gpt-5.6-luna"},
        )
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="custom_planner_agent_run_metrics",
            stage="custom-adapter",
            data={
                "run_id": "agent-1",
                "model": "openai:h:gpt-5.6-luna",
                "input_tokens": 12_345,
                "output_tokens": 678,
                "cache_read_tokens": 9_000,
                "cache_write_tokens": 12,
            },
        )
    )

    console.print(dashboard.live.get_renderable())
    rendered = console.export_text()
    activity_line = next(line for line in rendered.splitlines() if "acme/project" in line)
    usage_line = next(line for line in rendered.splitlines() if "Usage" in line)

    assert "Generate · Custom Adapter" in rendered
    assert "FeatureBench" not in rendered
    assert (
        "acme/project · Inspecting sources · 2/9 files · 1 active · Resolving imports"
        in activity_line
    )
    assert "1/3" not in activity_line
    assert "attempt" not in activity_line
    assert "current" in activity_line
    assert "…" in activity_line
    assert "openai:h:gpt-5.6-luna" in usage_line
    assert "Input 12,345" in usage_line
    assert "Output 678" in usage_line
    assert "Cache read 9,000" in usage_line


@pytest.mark.asyncio
async def test_adapter_dashboard_times_the_current_activity(tmp_path: Path) -> None:
    dashboard = AdapterDashboard(adapter="example", output_directory=tmp_path)
    timestamp = datetime.now(UTC)
    baseline = AdapterProgressSnapshot(
        adapter="example",
        adapter_name="Example",
        environment_id="environment-1",
        status="working",
        phase_id="create_tasks",
        phase_label="Generating",
        work_items=(AdapterWorkItemProgress(item_id="seed", activity="Baseline"),),
    )
    validating = baseline.model_copy(
        update={
            "work_items": (AdapterWorkItemProgress(item_id="seed", activity="Validating mask"),)
        }
    )

    for snapshot, event_time in (
        (baseline, timestamp),
        (validating, timestamp + timedelta(seconds=12)),
        (validating.model_copy(update={"completed": 1}), timestamp + timedelta(seconds=20)),
    ):
        await dashboard.emit(
            Event(
                run_id="run",
                kind="adapter_progress",
                stage="example",
                timestamp=event_time,
                data=snapshot.model_dump(mode="json"),
            )
        )

    assert dashboard.environment_activity_started_at["environment-1"] == timestamp
    assert dashboard.environment_work_item_started_at[("environment-1", "seed")] == (
        timestamp + timedelta(seconds=12)
    )
    assert dashboard.snapshot.work_items[0].activity == "Validating mask"


@pytest.mark.asyncio
async def test_adapter_dashboard_times_concurrent_work_items_independently(
    tmp_path: Path,
) -> None:
    dashboard = AdapterDashboard(adapter="example", output_directory=tmp_path)
    timestamp = datetime.now(UTC)
    baseline = AdapterProgressSnapshot(
        adapter="example",
        adapter_name="Example",
        environment_id="environment-1",
        status="working",
        phase_id="create_tasks",
        phase_label="Generating",
        work_items=(
            AdapterWorkItemProgress(item_id="seed-a", activity="Baseline"),
            AdapterWorkItemProgress(item_id="seed-b", activity="Targeting"),
        ),
    )
    seed_b_advanced = baseline.model_copy(
        update={
            "work_items": (
                baseline.work_items[0],
                AdapterWorkItemProgress(item_id="seed-b", activity="Planning"),
            )
        }
    )

    for snapshot, event_time in (
        (baseline, timestamp),
        (seed_b_advanced, timestamp + timedelta(seconds=12)),
    ):
        await dashboard.emit(
            Event(
                run_id="run",
                kind="adapter_progress",
                stage="example",
                timestamp=event_time,
                data=snapshot.model_dump(mode="json"),
            )
        )

    assert dashboard.environment_work_item_started_at[("environment-1", "seed-a")] == timestamp
    assert dashboard.environment_work_item_started_at[("environment-1", "seed-b")] == (
        timestamp + timedelta(seconds=12)
    )
    activity = dashboard._activity_text(  # pyright: ignore[reportPrivateUsage]
        "environment-1", timestamp + timedelta(seconds=20)
    )
    assert "Baseline · current 20s · seed-a" in activity.plain


@pytest.mark.asyncio
async def test_adapter_dashboard_summarizes_concurrent_work_items(tmp_path: Path) -> None:
    console = Console(record=True, width=180, color_system=None)
    dashboard = AdapterDashboard(
        adapter="featurebench",
        output_directory=tmp_path,
        console=console,
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="generation_environment_started",
            stage="generate",
            data={
                "environment_id": "environment-1",
                "repository": "https://github.com/acme/project.git",
            },
        )
    )
    snapshot = AdapterProgressSnapshot(
        adapter="featurebench",
        adapter_name="FeatureBench",
        environment_id="environment-1",
        status="working",
        phase_id="create_tasks",
        phase_label="Generating",
        total=8,
        work_items=(
            AdapterWorkItemProgress(
                item_id="tests/test_a.py",
                activity="Targeting",
            ),
            AdapterWorkItemProgress(
                item_id="tests/test_b.py",
                activity="Validating mask",
            ),
        ),
    )
    await dashboard.emit(
        Event(
            run_id="run",
            kind="adapter_progress",
            stage="featurebench",
            data=snapshot.model_dump(mode="json"),
        )
    )

    console.print(dashboard.live.get_renderable())
    activity_line = next(
        line for line in console.export_text().splitlines() if "acme/project" in line
    )
    assert (
        "Generating · 0/8 items · 2 active · Targeting · current 0s · tests/test_a.py"
        in activity_line
    )


@pytest.mark.asyncio
async def test_adapter_dashboard_active_rows_adapt_to_terminal_height(tmp_path: Path) -> None:
    async def render_active_environments(height: int) -> str:
        console = Console(record=True, width=120, height=height, color_system=None)
        dashboard = AdapterDashboard(
            adapter="example",
            output_directory=tmp_path,
            console=console,
        )
        await dashboard.emit(
            Event(
                run_id="run",
                kind="generation_snapshot_captured",
                stage="example",
                data={"selected_environments": 8},
            )
        )
        timestamp = datetime.now(UTC)
        for index in range(8):
            await dashboard.emit(
                Event(
                    run_id="run",
                    kind="generation_environment_started",
                    stage="example",
                    timestamp=timestamp + timedelta(seconds=index),
                    data={
                        "environment_id": f"environment-{index}",
                        "repository": f"example/repository-{index}",
                    },
                )
            )
        console.print(dashboard.live.get_renderable())
        return console.export_text()

    short = await render_active_environments(16)
    tall = await render_active_environments(40)
    short_count = sum(f"example/repository-{index}" in short for index in range(8))
    tall_count = sum(f"example/repository-{index}" in tall for index in range(8))

    assert 1 <= short_count < tall_count
    assert tall_count == 8


def test_adapter_dashboard_uses_semantic_color_roles(tmp_path: Path) -> None:
    dashboard = AdapterDashboard(adapter="example", output_directory=tmp_path)

    tasks = dashboard._tasks_text()  # pyright: ignore[reportPrivateUsage]
    results = dashboard._results_text()  # pyright: ignore[reportPrivateUsage]
    usage = dashboard.usage.render()

    assert {str(span.style) for span in tasks.spans} >= {
        "green",
        "bold green",
        "bold yellow",
        "bold red",
    }
    assert {str(span.style) for span in results.spans} >= {
        "green",
        "red",
        "yellow",
        "bright_blue",
    }
    assert {str(span.style) for span in usage.spans} >= {
        "magenta",
        "cyan",
        "bright_blue",
    }
