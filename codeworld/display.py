"""Rich terminal dashboard driven by the same structured event stream as SQLite."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from codeworld.events import Event
from codeworld.progress import AdapterProgressSnapshot


def _duration(seconds: float) -> str:
    total = max(0, math.ceil(seconds))
    if total < 60:
        return f"{total}s"
    minutes, remaining_seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def _progress_grid() -> Table:
    """Keep the progress summary, including ETA, on one terminal row."""
    progress = Table.grid(expand=True)
    progress.add_column(ratio=1)
    progress.add_column(justify="right", no_wrap=True)
    return progress


def _single_line() -> Text:
    """Create terminal-width-aware text that never changes a dashboard row's height."""
    return Text(overflow="ellipsis", no_wrap=True)


def _progress_bar(completed: float) -> ProgressBar:
    return ProgressBar(
        total=1.0,
        completed=completed,
        style="grey23",
        complete_style="cyan",
        finished_style="green",
    )


def _rendered_height(console: Console, renderable: RenderableType) -> int:
    options = console.options.update(height=None)
    return len(console.render_lines(renderable, options=options, pad=False))


def _fit_active_rows(
    console: Console,
    active_count: int,
    render: Callable[[int], Panel],
) -> Panel:
    """Render as many stable one-line Active rows as the terminal can show."""
    panel = render(active_count)
    if active_count == 0 or _rendered_height(console, panel) <= console.size.height:
        return panel

    # Preserve at least one active job even in a terminal too short for fixed rows.
    best = 1
    low = 2
    high = active_count - 1
    while low <= high:
        candidate = (low + high) // 2
        candidate_panel = render(candidate)
        if _rendered_height(console, candidate_panel) <= console.size.height:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    return render(best)


def _clip(value: object, *, limit: int = 140) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _repository_label(value: str) -> str:
    normalized = value.strip().rstrip("/").removesuffix(".git")
    parts = [part for part in normalized.split("/") if part]
    return "/".join(parts[-2:]) if len(parts) >= 2 else normalized


@dataclass(slots=True)
class _AgentUsage:
    models: set[str] = field(default_factory=lambda: set[str]())
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    metered_runs: set[str] = field(default_factory=lambda: set[str]())

    def observe(self, event: Event) -> bool:
        if not (
            event.kind.endswith("_agent_run_started") or event.kind.endswith("_agent_run_metrics")
        ):
            return False
        changed = False
        raw_model = event.data.get("model")
        if isinstance(raw_model, str) and raw_model and raw_model not in self.models:
            self.models.add(raw_model)
            changed = True
        if not event.kind.endswith("_agent_run_metrics"):
            return changed
        raw_run_id = event.data.get("run_id")
        metric_id = str(raw_run_id) if raw_run_id else event.event_id
        if metric_id in self.metered_runs:
            return changed
        self.metered_runs.add(metric_id)
        self.input_tokens += int(event.data.get("input_tokens", 0))
        self.output_tokens += int(event.data.get("output_tokens", 0))
        self.cache_read_tokens += int(event.data.get("cache_read_tokens", 0))
        self.cache_write_tokens += int(event.data.get("cache_write_tokens", 0))
        return True

    def render(self) -> Text:
        model = ", ".join(sorted(self.models)) or "—"
        value = _single_line()
        usage = (
            ("Model", model, "magenta"),
            ("Input", f"{self.input_tokens:,}", "cyan"),
            ("Output", f"{self.output_tokens:,}", "cyan"),
            ("Cache read", f"{self.cache_read_tokens:,}", "bright_blue"),
            ("Cache write", f"{self.cache_write_tokens:,}", "bright_blue"),
        )
        for index, (label, metric, style) in enumerate(usage):
            if index:
                value.append(" · ", style="dim")
            value.append(f"{label} ", style="dim")
            value.append(metric, style=style)
        return value


class DiscoveryDashboard:
    """A deliberately small, phase-aware view of repository discovery."""

    def __init__(
        self,
        *,
        output_directory: str | Path,
        search_mode: bool,
        console: Console | None = None,
        refresh_per_second: float = 8,
    ) -> None:
        self.console = console or Console()
        self.output_directory = Path(output_directory).resolve()
        self.search_mode = search_mode
        self.started_at = datetime.now(UTC)
        self.phase = "planning" if search_mode else "fetching"
        self.progress = 0.0
        self.repositories = 0
        self.repository_target: int | None = None
        self.completed_jobs = 0
        self.total_jobs = 0
        self.current_language: str | None = None
        self.current_license: str | None = None
        self.search_group_index = 0
        self.search_group_total = 0
        self.rate_limited_until: datetime | None = None
        self.samples: deque[tuple[datetime, float]] = deque(maxlen=24)
        self.stopped = False
        self.live = Live(
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
            get_renderable=self._render,
        )

    def __enter__(self) -> DiscoveryDashboard:
        self.live.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        if exception_type is not None and self.phase != "completed":
            self.stopped = True
            self.live.refresh()
        self.live.stop()

    async def emit(self, event: Event) -> None:
        if event.kind == "rate_limited":
            cooldown = float(event.data.get("cooldown_seconds", 0))
            limited_until = event.timestamp + timedelta(seconds=max(cooldown, 0))
            if self.rate_limited_until is None or limited_until > self.rate_limited_until:
                self.rate_limited_until = limited_until
        elif event.kind == "discovery_progress":
            phase = str(event.data.get("phase", self.phase))
            if phase != self.phase:
                self.phase = phase
                self.samples.clear()
            self.progress = min(max(float(event.data.get("progress", 0)), 0.0), 1.0)
            self.repositories = int(event.data.get("repositories", self.repositories))
            raw_target = event.data.get("repository_target", self.repository_target)
            self.repository_target = int(raw_target) if raw_target is not None else None
            self.completed_jobs = int(event.data.get("completed_jobs", self.completed_jobs))
            self.total_jobs = int(event.data.get("total_jobs", self.total_jobs))
            raw_language = event.data.get("current_language")
            raw_license = event.data.get("current_license")
            self.current_language = str(raw_language) if raw_language else None
            self.current_license = str(raw_license) if raw_license else None
            self.search_group_index = int(
                event.data.get("search_group_index", self.search_group_index)
            )
            self.search_group_total = int(
                event.data.get("search_group_total", self.search_group_total)
            )
            self.samples.append((event.timestamp, self.progress))
            self.rate_limited_until = None
        if self.live.is_started:
            self.live.refresh()

    def _eta(self) -> float | None:
        if self.progress <= 0 or self.progress >= 1 or len(self.samples) < 2:
            return None
        last_time, last_progress = self.samples[-1]
        first_time, first_progress = self.samples[0]
        elapsed = (last_time - first_time).total_seconds()
        completed = last_progress - first_progress
        if elapsed < 1 or completed <= 0:
            return None
        return (1 - self.progress) / (completed / elapsed)

    def _step_title(self) -> str:
        if self.phase == "completed":
            return "Discover complete"
        if not self.search_mode:
            return "Fetching selected repositories"
        if self.phase == "planning":
            return "Step 1 of 2 · Planning GitHub search"
        return "Step 2 of 2 · Fetching repository metadata"

    def _status(self, now: datetime) -> Text:
        if self.stopped:
            return Text("Stopped safely · ready to resume", style="yellow")
        if self.phase == "completed":
            return Text("Complete", style="bold green")
        if self.rate_limited_until is not None and self.rate_limited_until > now:
            remaining = (self.rate_limited_until - now).total_seconds()
            return Text(
                f"GitHub rate limit · retrying in {_duration(remaining)}",
                style="yellow",
            )
        return Text("Working", style="cyan")

    def _render(self) -> RenderableType:
        now = datetime.now(UTC)
        heading = Text(self._step_title(), style="bold")

        progress = _progress_grid()
        eta = self._eta()
        if self.phase == "completed":
            progress_label = "100% · Complete"
        elif eta is None:
            progress_label = f"{self.progress:>3.0%} · ETA estimating…"
        else:
            progress_label = f"{self.progress:>3.0%} · ETA ≈ {_duration(eta)}"
        progress.add_row(
            ProgressBar(total=1.0, completed=self.progress),
            progress_label,
        )

        details = Table.grid(padding=(0, 2))
        details.add_column(width=13, style="dim")
        details.add_column()
        if self.phase == "planning":
            current_filter = self.current_language or "All languages"
            if self.current_license:
                current_filter = f"{current_filter} / {self.current_license}"
            if self.search_group_total:
                current_filter += f" · {self.search_group_index}/{self.search_group_total} filters"
            details.add_row("Scanning", current_filter)
        else:
            repository_count = str(self.repositories)
            if self.repository_target is not None:
                repository_count += f" / {self.repository_target} target"
            details.add_row("Repositories", repository_count)
        details.add_row("Status", self._status(now))
        details.add_row(
            "Elapsed",
            _duration(math.floor((now - self.started_at).total_seconds())),
        )
        details.add_row("Output", str(self.output_directory))

        content: list[RenderableType] = [heading, Text(""), progress, Text(""), details]
        if self.phase != "completed":
            content.extend(
                (
                    Text(""),
                    Text(
                        "Ctrl+C stop safely · Resume from Discover → Resume",
                        style="dim",
                    ),
                )
            )
        return Panel(
            Group(*content),
            title="Discover",
            border_style="green" if self.phase == "completed" else "cyan",
        )


@dataclass(frozen=True, slots=True)
class _BuildJobView:
    repository: str
    operation: str
    attempt: int | None
    max_attempts: int | None
    detail: str | None
    activity_started_at: datetime
    updated_at: datetime


class BuildDashboard:
    """User-facing progress for one or many Environment Build jobs."""

    def __init__(
        self,
        *,
        output_directory: str | Path,
        repository_target: int,
        repository_name: str | None = None,
        model: str | None = None,
        console: Console | None = None,
        refresh_per_second: float = 8,
    ) -> None:
        self.console = console or Console()
        self.output_directory = Path(output_directory).resolve()
        self.repository_target = repository_target
        self.repository_name = repository_name
        self.usage = _AgentUsage(models={model} if model else set())
        self.started_at = datetime.now(UTC)
        self.phase = "Preparing"
        self.status = "preparing"
        self.built = 0
        self.skipped = 0
        self.rejected = 0
        self.failed = 0
        self.jobs: dict[str, _BuildJobView] = {}
        self._finished_jobs: set[str] = set()
        self.samples: deque[tuple[datetime, float]] = deque(maxlen=24)
        self.live = Live(
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
            get_renderable=self._render,
        )

    def __enter__(self) -> BuildDashboard:
        self.live.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        if exception_type is not None and self.status != "completed":
            self.status = (
                "stopped"
                if issubclass(exception_type, (KeyboardInterrupt, asyncio.CancelledError))
                else "failed"
            )
            self.live.refresh()
        self.live.stop()

    async def emit(self, event: Event) -> None:
        operation = self._operation(event)
        if operation is not None:
            self.phase = operation
            if self.status not in ("completed", "failed", "stopped"):
                self.status = "working"
        if "retry" in event.kind and self.status not in ("completed", "failed", "stopped"):
            self.status = "retrying"

        self.usage.observe(event)

        if event.task_id is not None:
            terminal = {
                "environment_completed": "built",
                "environment_skipped": "skipped",
                "environment_failed": "failed",
                "environment_timed_out": "failed",
                "environment_rejected": "rejected",
            }
            outcome = terminal.get(event.kind)
            if outcome is None:
                previous = self.jobs.get(event.task_id)
                next_operation = operation or (
                    previous.operation if previous is not None else self.phase
                )
                detail = self._operation_detail(event)
                if detail is None and previous is not None and next_operation == previous.operation:
                    detail = previous.detail
                raw_attempt = event.data.get("attempt")
                attempt = raw_attempt if isinstance(raw_attempt, int) else None
                raw_max_attempts = event.data.get("max_attempts")
                max_attempts = raw_max_attempts if isinstance(raw_max_attempts, int) else None
                if previous is not None:
                    if attempt is None and next_operation == previous.operation:
                        attempt = previous.attempt
                    if max_attempts is None and (
                        next_operation == previous.operation or attempt is not None
                    ):
                        max_attempts = previous.max_attempts
                same_activity = (
                    previous is not None
                    and next_operation == previous.operation
                    and attempt == previous.attempt
                    and detail == previous.detail
                )
                activity_started_at = event.timestamp
                if same_activity:
                    assert previous is not None
                    activity_started_at = previous.activity_started_at
                self.jobs[event.task_id] = _BuildJobView(
                    repository=event.task_id,
                    operation=next_operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    detail=detail,
                    activity_started_at=activity_started_at,
                    updated_at=event.timestamp,
                )
            else:
                self.jobs.pop(event.task_id, None)
            if outcome is not None and event.task_id not in self._finished_jobs:
                self._finished_jobs.add(event.task_id)
                setattr(self, outcome, getattr(self, outcome) + 1)
                self._sample(event.timestamp)

        if event.kind == "environment_production_completed":
            self.built = int(event.data.get("built", self.built))
            self.skipped = int(event.data.get("skipped", self.skipped))
            self.rejected = int(event.data.get("rejected", self.rejected))
            self.failed = int(event.data.get("failed", self.failed))
            self.jobs.clear()
            self.status = "completed"
            self.phase = "Complete"
            self._sample(event.timestamp)
        elif (
            event.kind == "completed"
            and event.stage == "environment-build"
            and event.task_id is None
        ):
            self.built = max(1, self.built)
            self.status = "completed"
            self.phase = "Complete"
            self._sample(event.timestamp)

        if self.live.is_started:
            self.live.refresh()

    @property
    def completed_jobs(self) -> int:
        return self.built + self.skipped + self.rejected + self.failed

    def _sample(self, timestamp: datetime) -> None:
        if self.repository_target <= 0:
            return
        progress = min(self.completed_jobs / self.repository_target, 1.0)
        self.samples.append((timestamp, progress))

    def _eta(self, progress: float) -> float | None:
        if progress <= 0 or progress >= 1 or len(self.samples) < 2:
            return None
        first_time, first_progress = self.samples[0]
        last_time, last_progress = self.samples[-1]
        elapsed = (last_time - first_time).total_seconds()
        completed = last_progress - first_progress
        if elapsed < 1 or completed <= 0:
            return None
        return (1 - progress) / (completed / elapsed)

    @staticmethod
    def _operation(event: Event) -> str | None:
        kind = event.kind
        if event.stage == "repository-materialization" or kind.startswith("materialize_"):
            return "Fetching repository"
        if kind in ("planner_started", "analysis_completed") or kind.startswith("analysis_agent_"):
            return "Analyzing repository"
        if kind == "artifacts_authored" or kind.startswith("artifact_agent_"):
            return "Authoring environment"
        if kind.startswith("runner_review_"):
            return "Reviewing Runner"
        if kind in ("build_started", "build_progress"):
            return "Building environment"
        if kind.startswith("validation_"):
            return "Verifying environment"
        if kind.startswith("intermediate_image_cleanup"):
            return "Cleaning up"
        return None

    @staticmethod
    def _operation_detail(event: Event) -> str | None:
        if event.kind in ("build_started", "build_progress"):
            step_id = event.data.get("step_id")
            stage = event.data.get("step_stage")
            instruction = event.data.get("step_instruction")
            if step_id is not None and instruction:
                step = f"#{step_id}"
                if stage:
                    step += f" [{_clip(stage, limit=40)}]"
                # Preserve the detailed BuildKit instruction in state; the Text renderable
                # clips it against the actual terminal width instead of a fixed UI width.
                step += f" {_clip(instruction, limit=2000)}"
                return step
            return None
        if event.kind == "validation_phase_started":
            label = event.data.get("phase_label")
            return _clip(label) if label else None
        return None

    def _status_text(self) -> Text:
        labels = {
            "preparing": ("Preparing", "cyan"),
            "working": ("Working", "cyan"),
            "retrying": ("Retrying", "yellow"),
            "completed": ("Complete", "bold green"),
            "failed": ("Failed", "bold red"),
            "stopped": ("Stopped safely", "yellow"),
        }
        label, style = labels[self.status]
        return Text(label, style=style)

    def _status_elapsed_text(self, now: datetime) -> Text:
        value = self._status_text()
        value.append(" · Elapsed ")
        value.append(_duration(math.floor((now - self.started_at).total_seconds())))
        return value

    def _results_text(self) -> Text:
        value = _single_line()
        for index, (label, count, style) in enumerate(
            (
                ("Built", self.built, "green"),
                ("Skipped", self.skipped, "bright_blue"),
                ("Rejected", self.rejected, "yellow"),
                ("Failed", self.failed, "red"),
            )
        ):
            if index:
                value.append(" · ", style="dim")
            value.append(f"{label} ", style=style)
            value.append(str(count), style=f"bold {style}")
        return value

    @staticmethod
    def _activity_text(job: _BuildJobView, now: datetime) -> Text:
        operation_styles = {
            "Fetching repository": "bright_blue",
            "Analyzing repository": "magenta",
            "Authoring environment": "bright_magenta",
            "Reviewing Runner": "bright_blue",
            "Building environment": "yellow",
            "Verifying environment": "green",
            "Cleaning up": "cyan",
        }
        value = _single_line()
        value.append(job.repository, style="bold cyan")
        value.append(" · ", style="dim")
        value.append(job.operation, style=operation_styles.get(job.operation, "cyan"))
        if job.attempt is not None:
            value.append(" · attempt ", style="dim")
            attempt = str(job.attempt)
            if job.max_attempts is not None:
                attempt += f"/{job.max_attempts}"
            value.append(attempt, style="bright_blue")
        value.append(" · current ", style="dim")
        value.append(
            _duration(math.floor((now - job.activity_started_at).total_seconds())),
            style="cyan",
        )
        if job.detail:
            value.append(" · ", style="dim")
            value.append(job.detail, style="bright_black")
        return value

    def _usage_text(self) -> Text:
        return self.usage.render()

    def _build_panel(
        self,
        *,
        heading: Text,
        progress_grid: Table,
        active: Sequence[_BuildJobView],
        now: datetime,
    ) -> Panel:
        details = Table.grid(padding=(0, 2), expand=True)
        details.add_column(width=13, style="dim", no_wrap=True)
        details.add_column(ratio=1, overflow="fold")
        details.add_row("Results", self._results_text())
        if active:
            for index, job in enumerate(active):
                details.add_row(
                    "Active" if index == 0 else "",
                    self._activity_text(job, now),
                )
        elif self.repository_name and self.status not in ("completed", "failed", "stopped"):
            repository = _single_line()
            repository.append(self.repository_name, style="bold cyan")
            details.add_row("Repository", repository)
        status = self._status_elapsed_text(now)
        status.overflow = "ellipsis"
        status.no_wrap = True
        details.add_row("Status", status)
        details.add_row("Usage", self._usage_text())
        output = _single_line()
        output.append(str(self.output_directory), style="bright_blue")
        details.add_row("Output", output)

        content: list[RenderableType] = [heading, Text(""), progress_grid, Text(""), details]
        if self.status not in ("completed", "failed", "stopped"):
            content.extend((Text(""), Text("Ctrl+C stop safely", style="dim yellow")))
        border_style = {
            "completed": "green",
            "failed": "red",
            "stopped": "yellow",
            "retrying": "yellow",
        }.get(self.status, "cyan")
        return Panel(
            Group(*content),
            title="Build",
            border_style=border_style,
        )

    def _render(self) -> RenderableType:
        now = datetime.now(UTC)
        heading = Text(
            "Build complete" if self.status == "completed" else self.phase,
            style=(
                "bold green"
                if self.status == "completed"
                else "bold red"
                if self.status == "failed"
                else "bold yellow"
                if self.status in ("retrying", "stopped")
                else "bold cyan"
            ),
        )
        progress_grid = _progress_grid()
        if self.repository_target == 0:
            progress = 1.0
        else:
            progress = min(self.completed_jobs / self.repository_target, 1.0)
        if self.status == "completed":
            progress = 1.0
        eta = self._eta(progress)
        label = _single_line()
        label.append(f"{progress:>3.0%}", style="bold cyan")
        label.append(" · ", style="dim")
        label.append(
            f"{self.completed_jobs} / {self.repository_target}",
            style="bright_blue",
        )
        label.append(" repositories", style="dim")
        if eta is not None:
            label.append(" · ETA ", style="dim")
            label.append(f"≈ {_duration(eta)}", style="yellow")
        elif progress < 1:
            label.append(" · ETA ", style="dim")
            label.append("estimating…", style="yellow")
        progress_grid.add_row(
            _progress_bar(progress),
            label,
        )

        # Dict insertion order is the first-seen activity order. Progress heartbeats update
        # a value without moving its key, so Active rows remain visually stable.
        active = list(self.jobs.values())
        return _fit_active_rows(
            self.console,
            len(active),
            lambda count: self._build_panel(
                heading=heading,
                progress_grid=progress_grid,
                active=active[:count],
                now=now,
            ),
        )


class AdapterDashboard:
    """Adapter-neutral, environment-aware progress for task generation."""

    def __init__(
        self,
        *,
        adapter: str,
        output_directory: str | Path,
        console: Console | None = None,
        refresh_per_second: float = 8,
    ) -> None:
        self.console = console or Console()
        self.output_directory = Path(output_directory).resolve()
        self.started_at = datetime.now(UTC)
        self.snapshot = AdapterProgressSnapshot(
            adapter=adapter,
            adapter_name=adapter.replace("-", " ").title(),
        )
        self.environment_total: int | None = None
        self.environment_completed = 0
        self.environment_failed = 0
        self.environment_rejected = 0
        self.environment_skipped = 0
        self.summary_tasks_created: int | None = None
        self.active_environments: dict[str, datetime] = {}
        self.environment_repositories: dict[str, str] = {}
        self.environment_activity_started_at: dict[str, datetime] = {}
        self.environment_work_item_started_at: dict[tuple[str, str], datetime] = {}
        self.environment_snapshots: dict[str, AdapterProgressSnapshot] = {}
        self.usage = _AgentUsage()
        self.batch_finished = False
        self.samples: deque[tuple[datetime, float]] = deque(maxlen=24)
        self.live = Live(
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
            get_renderable=self._render,
        )

    def __enter__(self) -> AdapterDashboard:
        self.live.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        if exception_type is not None and self.snapshot.status != "completed":
            status = (
                "stopped"
                if issubclass(exception_type, (KeyboardInterrupt, asyncio.CancelledError))
                else "failed"
            )
            self.snapshot = self.snapshot.model_copy(update={"status": status})
            self.live.refresh()
        self.live.stop()

    async def emit(self, event: Event) -> None:
        usage_changed = self.usage.observe(event)
        if "retry" in event.kind and self.snapshot.status not in (
            "completed",
            "failed",
            "stopped",
        ):
            self.snapshot = self.snapshot.model_copy(update={"status": "retrying"})
            usage_changed = True

        if event.kind == "generation_snapshot_captured":
            total = event.data.get("selected_environments")
            if isinstance(total, int) and total >= 0:
                self.environment_total = total
                self.samples.clear()
                self._sample(event.timestamp)
            self._refresh()
            return
        if event.kind == "generation_environment_started":
            environment_id = event.data.get("environment_id")
            if isinstance(environment_id, str):
                self.active_environments.setdefault(environment_id, event.timestamp)
                self.environment_activity_started_at.setdefault(environment_id, event.timestamp)
                repository = event.data.get("repository")
                if isinstance(repository, str) and repository:
                    self.environment_repositories[environment_id] = repository
            self._refresh()
            return
        terminal = {
            "generation_environment_completed": "completed",
            "generation_environment_failed": "failed",
            "generation_environment_rejected": "rejected",
            "generation_environment_interrupted": "failed",
            "generation_environment_skipped": "skipped",
        }.get(event.kind)
        if terminal is not None:
            environment_id = event.data.get("environment_id")
            if isinstance(environment_id, str):
                self.active_environments.pop(environment_id, None)
                for key in tuple(self.environment_work_item_started_at):
                    if key[0] == environment_id:
                        self.environment_work_item_started_at.pop(key)
                repository = event.data.get("repository")
                if isinstance(repository, str) and repository:
                    self.environment_repositories[environment_id] = repository
            if terminal == "completed":
                self.environment_completed += 1
            elif terminal == "failed":
                self.environment_failed += 1
            elif terminal == "rejected":
                self.environment_rejected += 1
            else:
                self.environment_skipped += 1
            self._sample(event.timestamp)
            self._refresh()
            return
        if event.kind == "completed" and self.environment_total is not None:
            for attribute, key in (
                ("environment_completed", "completed"),
                ("environment_failed", "failed"),
                ("environment_rejected", "rejected"),
                ("environment_skipped", "skipped"),
            ):
                value = event.data.get(key)
                if isinstance(value, int):
                    setattr(self, attribute, value)
            tasks_created = event.data.get("tasks_created")
            if isinstance(tasks_created, int):
                self.summary_tasks_created = tasks_created
            self.active_environments.clear()
            self.environment_work_item_started_at.clear()
            self.batch_finished = True
            self.snapshot = self.snapshot.model_copy(update={"status": "completed"})
            self._sample(event.timestamp)
            self._refresh()
            return
        if event.kind != "adapter_progress":
            if usage_changed:
                self._refresh()
            return
        snapshot = AdapterProgressSnapshot.model_validate(event.data)
        if snapshot.environment_id is not None:
            environment_id = snapshot.environment_id
            previous = self.environment_snapshots.get(environment_id)
            previous_work_items = (
                {item.item_id: item for item in previous.work_items} if previous is not None else {}
            )
            current_work_item_ids = {item.item_id for item in snapshot.work_items}
            for item in snapshot.work_items:
                prior = previous_work_items.get(item.item_id)
                key = (environment_id, item.item_id)
                if (
                    prior is None
                    or prior.activity != item.activity
                    or prior.current != item.current
                ):
                    self.environment_work_item_started_at[key] = event.timestamp
            for item_id in previous_work_items.keys() - current_work_item_ids:
                self.environment_work_item_started_at.pop((environment_id, item_id), None)
            self.environment_snapshots[environment_id] = snapshot
            if previous is None or previous.phase_id != snapshot.phase_id:
                self.environment_activity_started_at[environment_id] = event.timestamp
            if snapshot.status not in ("completed", "failed", "stopped"):
                self.active_environments.setdefault(environment_id, event.timestamp)
            if not self.batch_finished and snapshot.status in ("completed", "failed", "stopped"):
                snapshot = snapshot.model_copy(update={"status": "working"})
        if self.environment_total is None and snapshot.phase_id != self.snapshot.phase_id:
            self.samples.clear()
        self.snapshot = snapshot
        self._sample(event.timestamp)
        self._refresh()

    def _refresh(self) -> None:
        if self.live.is_started:
            self.live.refresh()

    def _sample(self, timestamp: datetime) -> None:
        progress = self._progress()
        if progress is None:
            return
        if not self.samples or self.samples[-1][1] != progress:
            self.samples.append((timestamp, progress))

    def _progress(self) -> float | None:
        if self.environment_total is not None:
            if self.environment_total == 0:
                return 1.0
            terminal = (
                self.environment_completed
                + self.environment_failed
                + self.environment_rejected
                + self.environment_skipped
            )
            return min(terminal / self.environment_total, 1.0)
        if self.snapshot.status == "completed":
            return 1.0
        if self.snapshot.total is None:
            return None
        if self.snapshot.total == 0:
            return 1.0
        return min(self.snapshot.completed / self.snapshot.total, 1.0)

    def _eta(self, progress: float | None) -> float | None:
        if progress is None or progress <= 0 or progress >= 1 or len(self.samples) < 2:
            return None
        first_time, first_progress = self.samples[0]
        last_time, last_progress = self.samples[-1]
        elapsed = (last_time - first_time).total_seconds()
        completed = last_progress - first_progress
        if elapsed < 1 or completed <= 0:
            return None
        return (1 - progress) / (completed / elapsed)

    def _heading(self) -> str:
        if self.batch_finished or (
            self.environment_total is None and self.snapshot.status == "completed"
        ):
            return "Generate complete"
        if self.environment_total is not None:
            return "Generating tasks" if self.active_environments else "Preparing environments"
        return self.snapshot.phase_label

    def _status(self) -> str:
        if self.batch_finished:
            return "completed"
        active_statuses = {
            self.environment_snapshots[environment_id].status
            for environment_id in self.active_environments
            if environment_id in self.environment_snapshots
        }
        if "retrying" in active_statuses or self.snapshot.status == "retrying":
            return "retrying"
        if "working" in active_statuses:
            return "working"
        if active_statuses and active_statuses <= {"waiting"}:
            return "waiting"
        return self.snapshot.status

    def _status_text(self) -> Text:
        labels = {
            "preparing": ("Preparing", "cyan"),
            "working": ("Working", "cyan"),
            "waiting": ("Waiting", "yellow"),
            "retrying": ("Retrying", "yellow"),
            "completed": ("Complete", "bold green"),
            "failed": ("Failed", "bold red"),
            "stopped": ("Stopped safely", "yellow"),
        }
        label, style = labels[self._status()]
        return Text(label, style=style)

    def _status_elapsed_text(self, now: datetime) -> Text:
        value = self._status_text()
        value.append(" · Elapsed ")
        value.append(_duration(math.floor((now - self.started_at).total_seconds())))
        value.overflow = "ellipsis"
        value.no_wrap = True
        return value

    def _tasks_created(self) -> int:
        if self.summary_tasks_created is not None:
            return self.summary_tasks_created
        return (
            sum(item.tasks_created for item in self.environment_snapshots.values())
            if self.environment_snapshots
            else self.snapshot.tasks_created
        )

    def _candidates_rejected(self) -> int:
        return (
            sum(item.candidates_rejected for item in self.environment_snapshots.values())
            if self.environment_snapshots
            else self.snapshot.candidates_rejected
        )

    def _candidates_failed(self) -> int:
        return (
            sum(item.candidates_failed for item in self.environment_snapshots.values())
            if self.environment_snapshots
            else self.snapshot.candidates_failed
        )

    def _tasks_text(self) -> Text:
        value = _single_line()
        value.append("Created ", style="green")
        value.append(str(self._tasks_created()), style="bold green")
        value.append(" · Candidates rejected ", style="dim")
        value.append(str(self._candidates_rejected()), style="bold yellow")
        value.append(" · failed ", style="dim")
        value.append(str(self._candidates_failed()), style="bold red")
        return value

    def _results_text(self) -> Text:
        value = _single_line()
        for index, (label, count, style) in enumerate(
            (
                ("Completed", self.environment_completed, "green"),
                ("Failed", self.environment_failed, "red"),
                ("Rejected", self.environment_rejected, "yellow"),
                ("Skipped", self.environment_skipped, "bright_blue"),
            )
        ):
            if index:
                value.append(" · ", style="dim")
            value.append(f"{label} ", style=style)
            value.append(str(count), style=f"bold {style}")
        return value

    def _activity_text(self, environment_id: str, now: datetime) -> Text:
        snapshot = self.environment_snapshots.get(environment_id)
        repository = self.environment_repositories.get(environment_id, environment_id)
        value = _single_line()
        work_item_current: str | None = None
        work_item_started_at: datetime | None = None
        value.append(_repository_label(repository), style="bold cyan")
        value.append(" · ", style="dim")
        if snapshot is None:
            value.append("Preparing", style="cyan")
        else:
            phase_style = {
                "waiting": "yellow",
                "retrying": "yellow",
                "failed": "red",
                "stopped": "yellow",
            }.get(snapshot.status, "bright_magenta")
            value.append(snapshot.phase_label, style=phase_style)
            if snapshot.total is not None:
                value.append(" · ", style="dim")
                value.append(
                    f"{snapshot.completed}/{snapshot.total} {snapshot.unit}",
                    style="bright_blue",
                )
            if snapshot.work_items:
                first = snapshot.work_items[0]
                value.append(" · ", style="dim")
                value.append(f"{len(snapshot.work_items)} active", style="bright_blue")
                value.append(" · ", style="dim")
                value.append(first.activity, style="bright_magenta")
                work_item_current = first.current or first.item_id
                work_item_started_at = self.environment_work_item_started_at.get(
                    (environment_id, first.item_id)
                )
        started_at = work_item_started_at or self.environment_activity_started_at.get(
            environment_id, self.active_environments[environment_id]
        )
        value.append(" · current ", style="dim")
        value.append(
            _duration(math.floor((now - started_at).total_seconds())),
            style="cyan",
        )
        current = work_item_current or (snapshot.current if snapshot is not None else None)
        if current:
            value.append(" · ", style="dim")
            value.append(_clip(current, limit=2000), style="bright_black")
        return value

    def _build_panel(
        self,
        *,
        heading: Text,
        progress_grid: Table,
        active: Sequence[str],
        now: datetime,
    ) -> Panel:
        details = Table.grid(padding=(0, 2), expand=True)
        details.add_column(width=13, style="dim", no_wrap=True)
        details.add_column(ratio=1, overflow="fold")
        details.add_row("Tasks", self._tasks_text())
        if self.environment_total is not None:
            details.add_row("Results", self._results_text())
        if active:
            for index, environment_id in enumerate(active):
                details.add_row(
                    "Active" if index == 0 else "",
                    self._activity_text(environment_id, now),
                )
        elif self.environment_total is None and self.snapshot.current:
            current = _single_line()
            current.append(self.snapshot.current, style="bright_black")
            details.add_row("Current", current)
        details.add_row("Status", self._status_elapsed_text(now))
        details.add_row("Usage", self.usage.render())
        output = _single_line()
        output.append(str(self.output_directory), style="bright_blue")
        details.add_row("Output", output)

        content: list[RenderableType] = [
            heading,
            Text(""),
            progress_grid,
            Text(""),
            details,
        ]
        if not self.batch_finished and self.snapshot.status not in (
            "completed",
            "failed",
            "stopped",
        ):
            content.extend((Text(""), Text("Ctrl+C stop safely", style="dim yellow")))
        status = self._status()
        border_style = {
            "completed": "green",
            "failed": "red",
            "stopped": "yellow",
            "retrying": "yellow",
        }.get(status, "cyan")
        return Panel(
            Group(*content),
            title=f"Generate · {self.snapshot.adapter_name}",
            border_style=border_style,
        )

    def _render(self) -> RenderableType:
        now = datetime.now(UTC)
        status = self._status()
        heading = Text(
            self._heading(),
            style=(
                "bold green"
                if status == "completed"
                else "bold red"
                if status == "failed"
                else "bold yellow"
                if status in ("retrying", "stopped", "waiting")
                else "bold cyan"
            ),
        )
        progress = self._progress()
        progress_grid = _progress_grid()
        if progress is None:
            progress_grid.add_row(
                Spinner("dots", text=self.snapshot.phase_label, style="cyan"),
                Text("Progress total unknown", style="yellow", no_wrap=True),
            )
        else:
            if self.environment_total is not None:
                terminal = (
                    self.environment_completed
                    + self.environment_failed
                    + self.environment_rejected
                    + self.environment_skipped
                )
                count = f"{terminal} / {self.environment_total} environments"
            elif self.snapshot.total is None:
                count = "Complete"
            else:
                count = f"{self.snapshot.completed} / {self.snapshot.total} {self.snapshot.unit}"
            label = _single_line()
            label.append(f"{progress:>3.0%}", style="bold cyan")
            label.append(" · ", style="dim")
            label.append(count, style="bright_blue")
            eta = self._eta(progress)
            if eta is not None:
                label.append(" · ETA ", style="dim")
                label.append(f"≈ {_duration(eta)}", style="yellow")
            elif progress < 1:
                label.append(" · ETA ", style="dim")
                label.append("estimating…", style="yellow")
            progress_grid.add_row(_progress_bar(progress), label)

        active = list(self.active_environments)
        return _fit_active_rows(
            self.console,
            len(active),
            lambda count: self._build_panel(
                heading=heading,
                progress_grid=progress_grid,
                active=active[:count],
                now=now,
            ),
        )
