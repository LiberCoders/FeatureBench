"""Cross-language Environment Build authoring and validation loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache
from math import ceil
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from codeworld.agents.factory import AgentFactory
from codeworld.agents.feedback import (
    AgentFeedback,
    AgentFeedbackError,
    FeedbackIssue,
    render_feedback,
)
from codeworld.config import AgentSettings, BaseImageSettings, EnvironmentSettings
from codeworld.environment.build_diagnostics import RepositoryDiagnostics
from codeworld.environment.diagnostics import (
    DEFAULT_FAILURE_DIAGNOSTICS,
    FailureDiagnosticContext,
)
from codeworld.environment.feedback import FeedbackCode, issue
from codeworld.environment.framework_validation import validate_framework_collection
from codeworld.environment.labels import normalize_ecosystem_label
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.planning import (
    SOURCE_MARKER,
    ArtifactRepairHistoryEntry,
    ClonePlan,
    EnvironmentAnalysis,
    EnvironmentArtifacts,
    ExecutionContract,
    analyze_environment,
    author_environment_artifacts,
)
from codeworld.environment.runner import (
    EnvironmentRunner,
    assert_stable_collection,
    probe_runner_failure_contract,
    probe_runner_large_request_transport,
    verified_run_test_points,
)
from codeworld.environment.runner_contract import (
    CollectedTestPoint,
    RunnerResponse,
    TestPointExecution,
)
from codeworld.errors import (
    DOCKER_HARNESS_STEP_MARKER,
    CommandTimeoutError,
    ConfigurationError,
    ContractError,
    DockerBuildError,
    DockerBuildOrigin,
    EnvironmentValidationError,
    InfrastructureError,
    RepositoryRejected,
    RunnerCollectionUnstableError,
)
from codeworld.events import Event, EventSink, NullEventSink
from codeworld.models import RunnerLimits, RuntimeResources, RuntimeSpec
from codeworld.runtime.container import DockerBuildProgress, DockerRuntime

_PLANNER_RUNTIME = RuntimeSpec(
    activation="#!/usr/bin/env bash\n# Planner image has no project environment yet.\ncd /testbed\n",
    runner=(
        "#!/usr/bin/env bash\n"
        "# The planner never invokes the test runner; a complete runner is authored later.\n"
        "echo 'planner runner is unavailable' >&2\nexit 2\n"
    ),
)

_DOCKER_BUILD_HEARTBEAT_SECONDS = 30.0
_VALIDATION_PHASE_LABELS = {
    "container_session": "Starting validation container",
    "activation": "Checking environment activation",
    "repository_baseline": "Checking repository baseline",
    "collect": "Collecting test points",
    "runner_transport": "Checking large Runner request transport",
    "quickcheck": "Running quickcheck",
    "baseline": "Running baseline tests",
    "container_cleanup": "Cleaning validation container",
}


@dataclass(slots=True)
class _ValidationAttemptMetrics:
    """Mutable measurements for one validation call; emitted exactly once by its wrapper."""

    started_monotonic: float = field(default_factory=time.monotonic)
    current_phase: str | None = "container_session"
    phase_started_monotonic: float = field(init=False)
    completed_phases: list[str] = field(default_factory=lambda: list[str]())
    phase_durations_seconds: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    observations: dict[str, object] = field(default_factory=lambda: dict[str, object]())

    def __post_init__(self) -> None:
        self.phase_started_monotonic = self.started_monotonic

    def complete_phase(self) -> float:
        if self.current_phase is None:
            raise RuntimeError("no active validation phase to complete")
        duration = max(time.monotonic() - self.phase_started_monotonic, 0)
        self.phase_durations_seconds[self.current_phase] = duration
        self.completed_phases.append(self.current_phase)
        self.current_phase = None
        return duration

    def begin_phase(self, phase: str) -> None:
        if self.current_phase is not None:
            raise RuntimeError(f"validation phase {self.current_phase!r} is still active")
        self.current_phase = phase
        self.phase_started_monotonic = time.monotonic()

    def current_phase_duration(self) -> float:
        if self.current_phase is None:
            raise RuntimeError("no active validation phase to measure")
        return max(time.monotonic() - self.phase_started_monotonic, 0)

    def event_data(
        self,
        *,
        status: str,
        error: BaseException | None,
    ) -> dict[str, object]:
        durations = dict(self.phase_durations_seconds)
        if self.current_phase is not None:
            durations[self.current_phase] = max(
                time.monotonic() - self.phase_started_monotonic,
                0,
            )
        return {
            "status": status,
            "failed_phase": self.current_phase if error is not None else None,
            "completed_phases": tuple(self.completed_phases),
            "total_duration_seconds": max(time.monotonic() - self.started_monotonic, 0),
            "phase_durations_seconds": durations,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error)[-5000:] if error is not None else None,
            **self.observations,
        }


def select_repair_runtime_spec(
    *,
    built_image_available: bool,
    current_runtime: RuntimeSpec,
    failure: AgentFeedback,
) -> RuntimeSpec:
    """Expose the real candidate environment unless activation itself needs repair."""
    implicated_fields = {
        issue_.path[0]
        for issue_ in failure.issues
        if issue_.path and isinstance(issue_.path[0], str)
    }
    activation_failed = any(
        issue_.context.get("failure_phase") == "activation" for issue_ in failure.issues
    ) or implicated_fields == {"activation"}
    if built_image_available and not activation_failed:
        return current_runtime
    return _PLANNER_RUNTIME


_BASE_RECIPE_DIRECTORY = Path(__file__).resolve().parent / "assets" / "base_images"
_DOCKERFILE_FROM = re.compile(
    r"^(?P<prefix>\s*FROM\s+(?:--platform=[^\s]+\s+)?)(?P<image>[^\s]+)(?P<suffix>.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_SYNTAX_DIRECTIVE = re.compile(r"^\s*#\s*syntax=.*$", re.IGNORECASE | re.MULTILINE)
_FORBIDDEN_BASE_RECIPE = re.compile(
    r"(?:pydantic[-_ ]?ai|openhands|openhindsight|codeworld|/opt/featurebench|"
    r"openai_api_key|anthropic_api_key|google_api_key|"
    r"^\s*(?:ADD|COPY)\s|^\s*(?:ARG|ENV)\s+[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))",
    re.IGNORECASE | re.MULTILINE,
)
_DEPENDENCY_RUNTIME_FAILURE = re.compile(
    r"\bincompatible (?:dependency|package|plugin|version)\b", re.IGNORECASE
)
_COLLECTION_SHORTFALL = re.compile(
    r"too few framework-addressable individual test cases.*required=(?P<required>\d+); "
    r"received=(?P<received>\d+)",
    re.IGNORECASE,
)
_OFFLINE_VALIDATION_NETWORK_FAILURE = re.compile(
    r"temporary failure in name resolution|unknown host|could not resolve host|"
    r"network is unreachable|failed to (?:fetch|download)|could not transfer artifact",
    re.IGNORECASE,
)
_ACTIVATION_SCRIPT_FAILURE = re.compile(
    r"activation timed out|activated command timed out|syntax error|unexpected end of file|"
    r"unbound variable|invalid option|"
    r"not a valid identifier|(?:^|\n).*:\s*line\s+\d+:\s*cd:\s*",
    re.IGNORECASE,
)
_ACTIVATION_PROVISIONING_FAILURE = re.compile(
    r"command not found|git is unavailable|no such file or directory|"
    r"(?:^|[/\\])\.?(?:venv|conda|node_modules)(?:[/\\]|$)",
    re.IGNORECASE,
)


def _activation_repair_fields(detail: str) -> tuple[str, ...]:
    """Classify whether a failed activation is script-local or may need provisioning."""
    if _ACTIVATION_SCRIPT_FAILURE.search(detail):
        return ("activation",)
    if _ACTIVATION_PROVISIONING_FAILURE.search(detail):
        return ("activation", "dockerfile")
    # A silent workdir assertion is an activation contract defect. Any other
    # unrecognized output remains ambiguous and may require Dockerfile provisioning.
    return ("activation",) if not detail.strip() else ("activation", "dockerfile")


def diagnose_environment_failure(
    error: DockerBuildError | InfrastructureError | ContractError,
    *,
    context: FailureDiagnosticContext | None = None,
) -> AgentFeedback:
    """Convert noisy runtime failures into one stable, actionable Agent feedback object."""
    detail = str(error)
    evidence = detail[-5000:]
    if isinstance(error, RunnerCollectionUnstableError):
        diagnostic = issue(
            FeedbackCode.ARTIFACT_COLLECTION_UNSTABLE,
            path=("runner",),
            evidence=evidence,
        )
    elif isinstance(error, DockerBuildError):
        if error.origin in (DockerBuildOrigin.HARNESS, DockerBuildOrigin.PLATFORM):
            raise error
        if error.timed_out:
            location = (
                f" in BuildKit stage {error.failed_step.stage!r}"
                if error.failed_step is not None
                else " before Docker reported an active BuildKit instruction"
            )
            diagnostic = issue(
                FeedbackCode.BUILD_COMMAND,
                path=("dockerfile",),
                expected=(
                    "every Agent-controlled Dockerfile instruction completes within the build deadline"
                ),
                actual="the Docker build timed out" + location,
                evidence=evidence,
                context={"failure_phase": "docker_build"},
            )
        else:
            if registered := DEFAULT_FAILURE_DIAGNOSTICS.diagnose(detail, context):
                return registered
            diagnostic = issue(
                FeedbackCode.BUILD_COMMAND,
                path=("dockerfile",),
                expected="every Agent-controlled Dockerfile instruction completes successfully",
                actual=(
                    "an Agent-controlled Dockerfile instruction failed"
                    if error.origin is DockerBuildOrigin.ARTIFACT
                    else "the Docker build failed without enough provenance to exclude the Dockerfile"
                ),
                evidence=evidence,
                context={"failure_phase": "docker_build"},
            )
    elif isinstance(error, InfrastructureError):
        if isinstance(error, CommandTimeoutError) and detail.startswith("Runner "):
            diagnostic = issue(
                FeedbackCode.VALIDATION_RUNTIME,
                path=("runner",),
                expected=(
                    "the requested Runner action completes within its configured action timeout "
                    "without narrowing the accepted suite"
                ),
                actual="the Runner action exceeded its configured timeout",
                evidence=evidence,
            )
        elif registered := DEFAULT_FAILURE_DIAGNOSTICS.diagnose(detail, context):
            # Some package managers surface strongly identifiable provisioning
            # failures outside the Docker wrapper. Registered ecosystem rules may
            # still authorize a precise artifact repair for those signatures.
            return registered
        else:
            # Typed host, source, and worker failures have already been attributed
            # outside the Agent-controlled artifact boundary.
            raise error
    else:
        if isinstance(error, EnvironmentValidationError):
            return AgentFeedback(
                issues=tuple(
                    issue(
                        FeedbackCode.VALIDATION_RUNTIME,
                        path=(field_name,),
                        expected="the environment satisfies the current validation phase contract",
                        actual=f"the {error.phase} phase failed",
                        evidence=evidence if index == 0 else None,
                        context={"failure_phase": error.phase},
                    )
                    for index, field_name in enumerate(error.repair_fields)
                )
            )
        if registered := DEFAULT_FAILURE_DIAGNOSTICS.diagnose(detail, context):
            return registered
        lowered = detail.casefold()
        if (
            "quickcheck environment health gate failed" in lowered
            or "baseline sweep produced too few verified passing test points" in lowered
        ):
            return AgentFeedback(
                issues=tuple(
                    issue(
                        FeedbackCode.VALIDATION_RUNTIME,
                        path=(field_name,),
                        expected=(
                            "the unchanged accepted tests execute through the Runner with enough "
                            "real passed outcomes to establish a usable environment"
                        ),
                        actual=(
                            "real Runner execution did not establish the required environment health"
                        ),
                        # All three paths authorize repairs for one shared failure;
                        # keep its potentially large evidence only once.
                        evidence=evidence if index == 0 else None,
                    )
                    for index, field_name in enumerate(("dockerfile", "activation", "runner"))
                )
            )
        if _OFFLINE_VALIDATION_NETWORK_FAILURE.search(detail) and "runner" in lowered:
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.VALIDATION_RUNTIME_MATERIALIZATION,
                        path=("dockerfile",),
                        expected=(
                            "all build, test, plugin, and transitive dependencies are materialized "
                            "during Docker build so validation succeeds with network disabled"
                        ),
                        actual=(
                            "the test framework attempted dependency resolution from the network "
                            "during offline validation"
                        ),
                        evidence=evidence,
                    ),
                )
            )
        if "runner did not produce" in lowered:
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected=(
                            "consume --request <request-json-path> and --response "
                            "<response-json-path>, read and write those option values, and exit "
                            "zero after writing a schema-valid bounded response"
                        ),
                        actual="the Runner process did not complete the V1 file contract",
                        evidence=evidence,
                    ),
                )
            )
        if "runner returned invalid json" in lowered:
            duplicate_detail = "duplicate test IDs" in detail
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected=(
                            "one unique stable framework-addressable ID per individual test case"
                            if duplicate_detail
                            else "a schema-valid bounded RunnerResponse JSON document"
                        ),
                        actual=(
                            "the response contains duplicate test IDs"
                            if duplicate_detail
                            else "the response failed RunnerResponse validation"
                        ),
                        evidence=evidence,
                    ),
                )
            )
        if "runner response" in lowered:
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected="a schema-valid bounded RunnerResponse JSON document",
                        actual="the Runner response violated the V1 contract",
                        evidence=evidence,
                    ),
                )
            )
        if "runner omitted or added selected test results" in lowered:
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected=(
                            "exactly one result for each requested test ID and no extra results"
                        ),
                        actual="the run response omitted or added selected test IDs",
                        evidence=evidence,
                    ),
                )
            )
        if "runner execution reported test ids absent from stable collection" in lowered:
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected=(
                            "every result returned by run uses an exact test ID from the "
                            "stable collect inventory"
                        ),
                        actual=(
                            "the execution response reported one or more IDs outside stable collection"
                        ),
                        evidence=evidence,
                    ),
                )
            )
        if collection_shortfall := _COLLECTION_SHORTFALL.search(detail):
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.VALIDATION_COLLECTION_GRANULARITY,
                        path=("runner",),
                        expected=(
                            ">= "
                            + collection_shortfall.group("required")
                            + " individual test case IDs"
                        ),
                        actual=(collection_shortfall.group("received") + " collected entries"),
                        evidence=evidence,
                    ),
                )
            )
        dependency_failure = bool(_DEPENDENCY_RUNTIME_FAILURE.search(detail)) or any(
            marker in lowered
            for marker in (
                "failed to download",
                "could not resolve",
                "temporary failure in name resolution",
                "command not found",
            )
        )
        if "activation failed" in lowered:
            paths = tuple((field_name,) for field_name in _activation_repair_fields(detail))
        elif dependency_failure:
            # Missing imports and explicit package/version incompatibilities are
            # provisioning defects. Changing response translation cannot repair them.
            paths = (("dockerfile",),)
        elif ("runner collect failed or returned fewer" in lowered) or any(
            marker in lowered
            for marker in (
                "runner collect returned a non-file",
                "runner collect returned an unsafe",
                "runner collect is not stable",
            )
        ):
            paths = (("runner",),)
        elif any(
            marker in lowered
            for marker in (
                "git baseline",
                "repository baseline is not clean",
                "forbidden codeworld control files",
            )
        ):
            paths = (("dockerfile",),)
        else:
            # Collection and test failures can originate in dependency provisioning or
            # response translation. Authorize either field and let repository evidence decide.
            paths = (("dockerfile",), ("runner",))
        return AgentFeedback(
            issues=tuple(
                issue(FeedbackCode.VALIDATION_RUNTIME, path=path, evidence=evidence)
                for path in paths
            )
        )
    return AgentFeedback(issues=(diagnostic,))


def select_quickcheck_test_ids(
    test_points: tuple[CollectedTestPoint, ...], *, commit: str, limit: int
) -> tuple[str, ...]:
    """Select a reproducible, language-neutral sample at test-point granularity."""
    ranked = sorted(
        test_points,
        key=lambda test: hashlib.sha256(f"{commit}\0{test.id}".encode()).hexdigest(),
    )
    return tuple(test.id for test in ranked[:limit])


def group_test_ids_by_path(
    test_points: tuple[CollectedTestPoint, ...],
    *,
    max_group_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Build bounded deterministic file-local batches for the complete baseline sweep."""
    if max_group_size < 1:
        raise ValueError("baseline group size must be at least one")
    grouped: dict[str, list[str]] = {}
    for test in test_points:
        grouped.setdefault(test.path, []).append(test.id)
    batches: list[tuple[str, ...]] = []
    for path in sorted(grouped):
        identifiers = sorted(grouped[path])
        batches.extend(
            tuple(identifiers[start : start + max_group_size])
            for start in range(0, len(identifiers), max_group_size)
        )
    return tuple(batches)


def summarize_runner_failure(
    response: RunnerResponse,
    *,
    expected_ids: tuple[str, ...] = (),
) -> str:
    """Render bounded, structured evidence without dropping failed test messages."""

    def clipped(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: limit - 1] + "…"

    def excerpt(value: str, limit: int = 3000) -> str:
        if len(value) <= limit:
            return value
        half = (limit - len("\n… output omitted …\n")) // 2
        return value[:half] + "\n… output omitted …\n" + value[-half:]

    returned = {result.id for result in response.results}
    expected = set(expected_ids)
    failed = [result for result in response.results if result.outcome != "passed"]
    evidence = {
        "exit_code": response.exit_code,
        "collected_count": len(response.collected),
        "result_count": len(response.results),
        "missing_test_ids": [clipped(item, 300) for item in sorted(expected - returned)[:20]],
        "unexpected_test_ids": [clipped(item, 300) for item in sorted(returned - expected)[:20]],
        "failed_results": [
            {
                "id": clipped(result.id, 300),
                "outcome": result.outcome,
                "message": clipped(result.message, 500),
            }
            for result in failed[:20]
        ],
        "errors": [clipped(item, 500) for item in response.errors[:20]],
        # Failure roots normally appear at the beginning while aggregate status and
        # cleanup failures appear at the end. Keeping only a tail loses the useful
        # exception whenever a framework emits a long backtrace.
        "output_excerpt": excerpt(response.output),
    }
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def baseline_timeout_response(
    test_ids: tuple[str, ...],
    error: CommandTimeoutError,
    *,
    total_timeout: bool = False,
) -> RunnerResponse:
    """Represent baseline work without real outcomes as conservative host-side errors."""
    detail = str(error)[-1000:] or "baseline Runner group timed out"
    message = (
        "baseline total timeout elapsed before the Runner produced real per-test outcomes"
        if total_timeout
        else "baseline group timed out before the Runner produced real per-test outcomes"
    )
    return RunnerResponse(
        action="run",
        exit_code=124,
        results=tuple(
            TestPointExecution(id=test_id, outcome="error", message=message) for test_id in test_ids
        ),
        errors=(message,),
        output=detail,
    )


class BaselineRunner(Protocol):
    async def run(
        self,
        test_ids: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> RunnerResponse: ...


class _IsolatedBaselineRunner:
    """Execute each baseline group in a fresh container made from the candidate image."""

    def __init__(
        self,
        runtime: DockerRuntime,
        *,
        image: str,
        runtime_spec: RuntimeSpec,
    ) -> None:
        self.runtime = runtime
        self.image = image
        self.runtime_spec = runtime_spec

    async def run(
        self,
        test_ids: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> RunnerResponse:
        if timeout_seconds is None or timeout_seconds <= 0:
            raise ValueError("an isolated baseline group requires a positive timeout")
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self.runtime.session(
                    self.image,
                    runtime_spec=self.runtime_spec,
                ) as session:
                    return await EnvironmentRunner(session).run(
                        test_ids, timeout_seconds=timeout_seconds
                    )
        except CommandTimeoutError:
            raise
        except TimeoutError as exc:
            raise CommandTimeoutError(
                f"isolated baseline group timed out after {timeout_seconds:g}s",
                timeout_seconds=timeout_seconds,
            ) from exc


async def run_baseline_groups(
    runner: BaselineRunner,
    group_ids: tuple[tuple[str, ...], ...],
    *,
    total_timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[RunnerResponse, ...]:
    """Run groups against one shared deadline, preserving only real outcomes."""
    if total_timeout_seconds <= 0:
        raise ValueError("baseline total timeout must be positive")
    reports: list[RunnerResponse] = []
    deadline = clock() + total_timeout_seconds
    for index, test_ids in enumerate(group_ids):
        remaining_seconds = deadline - clock()
        if remaining_seconds <= 0:
            error = CommandTimeoutError(
                f"baseline total timeout of {total_timeout_seconds:g}s elapsed"
            )
            reports.extend(
                baseline_timeout_response(
                    pending_ids,
                    error,
                    total_timeout=True,
                )
                for pending_ids in group_ids[index:]
            )
            break
        total_timeout_elapsed = False
        try:
            report = await runner.run(
                test_ids,
                timeout_seconds=remaining_seconds,
            )
        except CommandTimeoutError as exc:
            total_timeout_elapsed = True
            report = baseline_timeout_response(
                test_ids,
                exc,
                total_timeout=True,
            )
        reports.append(report)
        if total_timeout_elapsed:
            reports.extend(
                baseline_timeout_response(
                    pending_ids,
                    CommandTimeoutError(
                        f"baseline total timeout of {total_timeout_seconds:g}s elapsed"
                    ),
                    total_timeout=True,
                )
                for pending_ids in group_ids[index + 1 :]
            )
            break
    return tuple(reports)


def summarize_baseline_shortfall(
    *,
    required: int,
    verified: int,
    collected: int,
    group_ids: tuple[tuple[str, ...], ...],
    reports: tuple[RunnerResponse, ...],
) -> str:
    """Return bounded group-level evidence that is directly useful to a repair Agent."""

    def clipped(value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: limit - 1] + "…"

    candidates: list[tuple[int, tuple[str, ...], RunnerResponse]] = []
    for index, (test_ids, report) in enumerate(zip(group_ids, reports, strict=True)):
        if (
            report.exit_code != 0
            or report.errors
            or any(result.outcome != "passed" for result in report.results)
        ):
            candidates.append((index, test_ids, report))
    candidates.sort(
        key=lambda item: (
            not any("timed out" in error.casefold() for error in item[2].errors),
            not bool(item[2].errors),
            item[2].exit_code == 0,
            item[0],
        )
    )
    selected = candidates[:3]
    representative_groups: list[dict[str, object]] = [
        {
            "group_index": index,
            "requested_count": len(test_ids),
            "requested_test_ids": [clipped(test_id, 180) for test_id in test_ids[:2]],
            "exit_code": report.exit_code,
            "passed_count": len(report.passed_ids),
            "nonpassing_results": [
                {
                    "id": clipped(result.id, 180),
                    "outcome": result.outcome,
                    "message": clipped(result.message, 240),
                }
                for result in report.results
                if result.outcome != "passed"
            ][:2],
            "errors": [clipped(error, 300) for error in report.errors[:1]],
            "output_excerpt": clipped(report.output, 400),
        }
        for index, test_ids, report in selected
    ]
    evidence: dict[str, object] = {
        "required_verified_test_points": required,
        "observed_verified_test_points": verified,
        "collected_test_point_count": collected,
        "baseline_group_count": len(reports),
        "nonpassing_group_count": len(candidates),
        "omitted_nonpassing_group_count": max(0, len(candidates) - len(selected)),
        "representative_nonpassing_groups": representative_groups,
    }

    def render() -> str:
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))

    rendered = render()
    if len(rendered) <= 4500:
        return rendered

    compact_groups: list[dict[str, object]] = []
    for index, test_ids, report in selected[:2]:
        first_nonpassing = next(
            (result for result in report.results if result.outcome != "passed"),
            None,
        )
        compact_groups.append(
            {
                "group_index": index,
                "requested_count": len(test_ids),
                "requested_test_ids": [clipped(test_id, 100) for test_id in test_ids[:1]],
                "exit_code": report.exit_code,
                "passed_count": len(report.passed_ids),
                "nonpassing_results": (
                    [
                        {
                            "id": clipped(first_nonpassing.id, 100),
                            "outcome": first_nonpassing.outcome,
                            "message": clipped(first_nonpassing.message, 160),
                        }
                    ]
                    if first_nonpassing is not None
                    else []
                ),
                "errors": [clipped(error, 180) for error in report.errors[:1]],
            }
        )
    evidence["representative_nonpassing_groups"] = compact_groups
    evidence["feedback_compacted"] = True
    return render()


def assert_quickcheck_health(
    response: RunnerResponse,
    *,
    selected_ids: tuple[str, ...],
    minimum_pass_rate: float,
) -> tuple[int, int]:
    """Require enough real passes from a deterministic sample of collected IDs."""
    if not selected_ids:
        raise ValueError("quickcheck requires at least one selected test point")
    if not 0 < minimum_pass_rate <= 1:
        raise ValueError("quickcheck minimum pass rate must be in (0, 1]")
    required_passes = ceil(len(selected_ids) * minimum_pass_rate)
    observed_passes = len(response.passed_ids)
    if observed_passes < required_passes:
        raise ContractError(
            "quickcheck environment health gate failed: "
            f"required_passed={required_passes}; observed_passed={observed_passes}; "
            f"sampled={len(selected_ids)}; minimum_pass_rate={minimum_pass_rate}; "
            f"response={summarize_runner_failure(response, expected_ids=selected_ids)}"
        )
    return observed_passes, required_passes


def assert_collection_can_reach_verified_floor(
    response: RunnerResponse,
    *,
    minimum: int,
) -> None:
    """Stop immediately when collect cannot satisfy the configured environment floor."""
    if minimum < 1:
        raise ValueError("minimum verified test points must be at least one")
    if len(response.collected) < minimum:
        raise ContractError(
            "Runner collect returned too few framework-addressable individual test cases; "
            "test files, classes, and modules are not test points when the framework exposes "
            "finer case or node IDs: "
            f"required={minimum}; "
            f"received={len(response.collected)}; "
            f"response={summarize_runner_failure(response)}"
        )


def _read_base_recipe(selected: BaseImageSettings) -> str | None:
    if selected.recipe is None:
        return None
    return _read_base_recipe_asset(selected.recipe, selected.image)


@cache
def _read_base_recipe_asset(recipe_name: str, configured_image: str) -> str:
    recipe_path = _BASE_RECIPE_DIRECTORY / f"{recipe_name}.Dockerfile"
    try:
        recipe = recipe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read managed base-image recipe {recipe_name}: {exc}"
        ) from exc
    if SOURCE_MARKER in recipe or "/testbed" in recipe or _FORBIDDEN_BASE_RECIPE.search(recipe):
        raise ConfigurationError(
            f"managed base-image recipe contains task source or Agent controls: {recipe_path}"
        )
    recipe_matches = list(_DOCKERFILE_FROM.finditer(recipe))
    if len(recipe_matches) != 1:
        raise ConfigurationError("managed base-image recipes must contain exactly one FROM")
    recipe_image = recipe_matches[0].group("image")
    if recipe_image != configured_image:
        raise ConfigurationError(
            f"managed base-image recipe {recipe_name} starts FROM {recipe_image!r}, "
            f"but configured image is {configured_image!r}"
        )
    return recipe


def validate_base_image_recipes(settings: EnvironmentSettings) -> None:
    for selected in settings.base_images:
        _read_base_recipe(selected)


def _inline_base_recipe(dockerfile: str, selected: BaseImageSettings) -> str:
    recipe = _read_base_recipe(selected)
    if recipe is None:
        return dockerfile
    recipe_without_syntax = _SYNTAX_DIRECTIVE.sub("", recipe).strip()
    recipe_without_syntax = _DOCKERFILE_FROM.sub(
        lambda match: f"{match.group('prefix')}{match.group('image')} AS featurebench_managed_base",
        recipe_without_syntax,
        count=1,
    )
    artifact_without_syntax = _SYNTAX_DIRECTIVE.sub("", dockerfile).strip()
    replaced = 0

    def replace_selected(match: re.Match[str]) -> str:
        nonlocal replaced
        if match.group("image") != selected.image:
            return match.group(0)
        replaced += 1
        return f"{match.group('prefix')}featurebench_managed_base{match.group('suffix')}"

    artifact_without_syntax = _DOCKERFILE_FROM.sub(replace_selected, artifact_without_syntax)
    if replaced == 0:
        raise ContractError("artifact Dockerfile does not inherit its selected managed base image")
    return (
        "# syntax=docker/dockerfile:1.7\n\n"
        + recipe_without_syntax
        + "\n\n"
        + artifact_without_syntax
    )


def validate_portable_repository_url(repository_url: str) -> None:
    """Require a public URL that a credential-free standalone Dockerfile can clone."""
    parsed = urlsplit(repository_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"https?://[A-Za-z0-9._~:/@%+\-=]+", repository_url)
    ):
        raise ContractError(
            "self-contained Dockerfiles require a credential-free public HTTP(S) repository URL"
        )


def derive_clone_plan(repository: Path, *, repository_url: str, commit: str) -> ClonePlan:
    """Derive deterministic source materialization facts without asking an Agent."""
    recurse_submodules = (repository / ".gitmodules").is_file()
    fetch_lfs = False
    for attributes in repository.rglob(".gitattributes"):
        try:
            if "filter=lfs" in attributes.read_text(encoding="utf-8", errors="replace"):
                fetch_lfs = True
                break
        except OSError:
            continue
    return ClonePlan(
        repository_url=repository_url,
        commit=commit,
        destination="/testbed",
        strategy="partial",
        recurse_submodules=recurse_submodules,
        fetch_lfs=fetch_lfs,
    )


def render_final_dockerfile(
    artifacts: EnvironmentArtifacts,
    *,
    analysis: EnvironmentAnalysis,
    clone: ClonePlan,
    settings: EnvironmentSettings,
) -> str:
    """Replace the source marker with a pinned online clone and no local build context."""
    repository_url = clone.repository_url
    commit = clone.commit
    validate_portable_repository_url(repository_url)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("repository commit cannot be rendered safely")
    fallback_clone = (
        'git_transport clone "$REPOSITORY_URL" /testbed; \\\n'
        if clone.strategy == "full"
        else 'git_transport clone --depth=1 --no-checkout "$REPOSITORY_URL" /testbed; \\\n'
    )
    submodules = (
        "git_transport submodule update --init --recursive; \\\n"
        if clone.recurse_submodules
        else "true; \\\n"
    )
    lfs = (
        "git lfs version; git lfs install --local; git_transport lfs pull; "
        "git reset --hard HEAD; \\\n"
        if clone.fetch_lfs
        else "true; \\\n"
    )
    source = "".join(
        (
            "ARG GITHUB_PROXY_PREFIX=\n",
            f"ARG GIT_COMMAND_TIMEOUT_SECONDS={settings.clone_command_timeout_seconds}\n",
            f"ARG GIT_HTTP_LOW_SPEED_LIMIT={settings.clone_low_speed_bytes_per_second}\n",
            f"ARG GIT_HTTP_LOW_SPEED_TIME={settings.clone_low_speed_timeout_seconds}\n",
            "WORKDIR /\n",
            f"RUN {DOCKER_HARNESS_STEP_MARKER}; set -eux; \\\n",
            "    git_transport() { status=0; timeout --signal=TERM --kill-after=10s "
            '"${GIT_COMMAND_TIMEOUT_SECONDS}s" git '
            '-c "http.lowSpeedLimit=${GIT_HTTP_LOW_SPEED_LIMIT}" '
            '-c "http.lowSpeedTime=${GIT_HTTP_LOW_SPEED_TIME}" '
            "${GITHUB_PROXY_PREFIX:+-c} "
            "${GITHUB_PROXY_PREFIX:+url.${GITHUB_PROXY_PREFIX}https://github.com/.insteadOf=https://github.com/} "
            '"$@" || status=$?; '
            'if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then '
            'echo "Git command timed out after ${GIT_COMMAND_TIMEOUT_SECONDS}s" >&2; fi; '
            'return "$status"; }; \\\n',
            "    rm -rf /testbed; \\\n",
            "    git init --quiet /testbed; \\\n",
            f'    git -C /testbed remote add origin "{repository_url}"; \\\n',
            "    git -C /testbed config core.repositoryFormatVersion 1; \\\n",
            "    git -C /testbed config extensions.partialClone origin; \\\n",
            "    git -C /testbed config remote.origin.promisor true; \\\n",
            "    git -C /testbed config remote.origin.partialCloneFilter blob:none; \\\n",
            "    if git_transport -C /testbed fetch --force --no-tags --depth=1 "
            f'--filter=blob:none origin "{commit}" && \\\n',
            "       git_transport -C /testbed checkout --detach FETCH_HEAD; then \\\n",
            "        :; \\\n",
            "    else \\\n",
            "        rm -rf /testbed; \\\n",
            "        git init --quiet /testbed; \\\n",
            f'        git -C /testbed remote add origin "{repository_url}"; \\\n',
            "        if git_transport -C /testbed fetch --force --no-tags --depth=1 "
            f'origin "{commit}" && \\\n',
            "           git_transport -C /testbed checkout --detach FETCH_HEAD; then \\\n",
            "            :; \\\n",
            "        else \\\n",
            "            rm -rf /testbed; \\\n",
            "            ",
            fallback_clone.replace("$REPOSITORY_URL", repository_url),
            "            cd /testbed; \\\n",
            f'            git_transport checkout --detach "{commit}" || '
            f'(git_transport fetch --force --no-tags --depth=1 origin "{commit}" && '
            f"git_transport checkout --detach FETCH_HEAD); \\\n",
            "        fi; \\\n",
            "    fi; \\\n",
            "    cd /testbed; \\\n",
            f'    git remote set-url origin "{repository_url}"; \\\n',
            f'    test "$(git rev-parse HEAD)" = "{commit}"; \\\n',
            "    ",
            submodules,
            "    ",
            lfs,
            '    test -z "$(git status --porcelain --untracked-files=all)"\n',
            "WORKDIR /testbed",
        )
    )
    if artifacts.dockerfile.count(SOURCE_MARKER) != 1:
        raise ContractError("cannot inject pinned source: source marker is missing or duplicated")
    rendered = artifacts.dockerfile.replace(SOURCE_MARKER, source).rstrip()
    rendered = (
        rendered
        + "\n\nWORKDIR /testbed\n"
        + f"RUN {DOCKER_HARNESS_STEP_MARKER}; "
        + f'test "$(git rev-parse HEAD)" = "{commit}" && '
        + "git diff --no-ext-diff --exit-code && "
        + 'test -z "$(git status --porcelain --untracked-files=all)"\n'
    )
    selected = next(
        (
            image
            for image in settings.base_images
            if image.name == analysis.selected_base_image_name
        ),
        None,
    )
    if selected is None:
        raise ContractError("analysis selected an unavailable base image")
    return _inline_base_recipe(rendered, selected)


class EnvironmentBuilder:
    """Authors real artifacts and records only test points proven passing by execution."""

    def __init__(
        self,
        runtime: DockerRuntime,
        settings: EnvironmentSettings,
        *,
        agents: AgentFactory[AgentSettings],
        events: EventSink | None = None,
        run_id: str = "environment-build",
        task_id: str | None = None,
        diagnostics: RepositoryDiagnostics | None = None,
    ) -> None:
        validate_base_image_recipes(settings)
        self.runtime = runtime
        self.settings = settings
        self.agents = agents
        self.events = events or NullEventSink()
        self.run_id = run_id
        self.task_id = task_id
        self.diagnostics = diagnostics
        self._diagnostic_attempt: int | None = None

    async def _emit(self, kind: str, message: str, **data: object) -> None:
        if self.diagnostics is not None and kind == "validation_measured":
            self.diagnostics.record_validation(dict(data))
        await self.events.emit(
            Event(
                run_id=self.run_id,
                kind=kind,
                stage="environment-build",
                task_id=self.task_id,
                message=message,
                data=dict(data),
            )
        )

    async def _agent_progress(self, role: str, kind: str, data: dict[str, object]) -> None:
        if self.diagnostics is not None and kind in {
            "agent_run_started",
            "agent_run_metrics",
        }:
            self.diagnostics.record_agent_run(
                role=role,
                kind=kind,
                attempt=self._diagnostic_attempt,
                data=data,
            )
        if kind == "output_feedback":
            codes = data.get("codes")
            retry = data.get("retry")
            await self._emit(
                f"{role}_agent_output_feedback",
                f"{role} Agent output retry {retry}: {codes}",
                role=role,
                **data,
            )
            return
        if kind == "model_request_retry":
            await self._emit(
                f"{role}_agent_model_request_retry",
                f"{role} Agent retrying a transient model request",
                role=role,
                **data,
            )
            return
        tool = data.get("tool")
        action = "started" if kind.endswith("_call") else "completed"
        message = f"{role} Agent {action}"
        if tool:
            message += f" tool {tool}"
        await self._emit(f"{role}_agent_{kind}", message, role=role, **data)

    def _tag(self, kind: str, environment_id: str, attempt: int | None = None) -> str:
        run_suffix = hashlib.sha256(f"{self.run_id}\0{self.task_id or ''}".encode()).hexdigest()[
            :10
        ]
        suffix = f"-{attempt}" if attempt is not None else ""
        return f"codeworld/{kind}:{environment_id}-{run_suffix}{suffix}"

    @staticmethod
    def environment_id(repository_url: str, commit: str) -> str:
        return hashlib.sha256(f"{repository_url}@{commit}".encode()).hexdigest()[:24]

    async def _analyze(
        self,
        *,
        image: str,
        repository: Path | None,
        repository_url: str,
        commit: str,
    ) -> EnvironmentAnalysis:
        mounts = ((repository, "/testbed", True),) if repository is not None else ()
        async with self.runtime.session(
            image,
            runtime_spec=_PLANNER_RUNTIME,
            read_only=True,
            bind_mounts=mounts,
        ) as session:
            return await analyze_environment(
                agents=self.agents,
                session=session,
                settings=self.settings,
                repository_url=repository_url,
                commit=commit,
                progress=lambda kind, data: self._agent_progress("analysis", kind, data),
            )

    async def _author(
        self,
        *,
        image: str,
        repository: Path | None,
        runtime_spec: RuntimeSpec,
        analysis: EnvironmentAnalysis,
        execution_contract: ExecutionContract,
        repository_url: str,
        commit: str,
        current: EnvironmentArtifacts | None,
        failure: AgentFeedback | None,
        repair_history: tuple[ArtifactRepairHistoryEntry, ...] = (),
        accepted_constraints: AgentFeedback | None = None,
        validate_runner_transport: bool = False,
    ) -> EnvironmentArtifacts:
        mounts = ((repository, "/testbed", True),) if repository is not None else ()
        # Built candidates use Docker's disposable writable layer so diagnostic
        # commands observe the same filesystem semantics as validation. Repository
        # file tools remain read-only, and the container is destroyed after this run.
        candidate_repair = current is not None and repository is None
        async with self.runtime.session(
            image,
            runtime_spec=runtime_spec,
            read_only=not candidate_repair,
            bind_mounts=mounts,
        ) as session:
            return await author_environment_artifacts(
                agents=self.agents,
                session=session,
                settings=self.settings,
                analysis=analysis,
                execution_contract=execution_contract,
                repository_url=repository_url,
                commit=commit,
                current=current,
                failure=failure,
                repair_history=repair_history,
                accepted_constraints=accepted_constraints,
                validate_runner_transport=validate_runner_transport,
                progress=lambda kind, data: self._agent_progress("artifact", kind, data),
                review_progress=lambda kind, data: self._agent_progress(
                    "runner_review", kind, data
                ),
            )

    async def _build_final(
        self,
        *,
        environment_id: str,
        dockerfile: str,
        attempt: int,
        progress: Callable[[DockerBuildProgress], Awaitable[None]] | None = None,
    ) -> tuple[str, str]:
        with tempfile.TemporaryDirectory(prefix="featurebench-build-") as temporary_name:
            context = Path(temporary_name)
            (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            tag = self._tag("environment", environment_id, attempt)
            image = await self.runtime.build(
                context,
                tag=tag,
                pull=self.runtime.settings.pull,
                progress=progress,
            )
        return image, tag

    async def _build_with_progress(
        self,
        *,
        environment_id: str,
        dockerfile: str,
        attempt: int,
    ) -> tuple[str, str]:
        """Build once while projecting normalized BuildKit steps and liveness heartbeats."""
        started = time.monotonic()
        latest: DockerBuildProgress | None = None

        async def report(update: DockerBuildProgress) -> None:
            nonlocal latest
            latest = update
            await self._emit(
                "build_progress",
                "Docker build advanced to a new BuildKit step",
                attempt=attempt,
                max_attempts=self.settings.max_build_attempts,
                elapsed_seconds=max(time.monotonic() - started, 0),
                step_id=update.step_id,
                step_stage=update.stage,
                step_instruction=update.instruction[:500],
                heartbeat=False,
            )

        build = asyncio.create_task(
            self._build_final(
                environment_id=environment_id,
                dockerfile=dockerfile,
                attempt=attempt,
                progress=report,
            )
        )
        try:
            while True:
                completed, _ = await asyncio.wait(
                    (build,),
                    timeout=_DOCKER_BUILD_HEARTBEAT_SECONDS,
                )
                if build in completed:
                    return build.result()
                data: dict[str, object] = {
                    "attempt": attempt,
                    "max_attempts": self.settings.max_build_attempts,
                    "elapsed_seconds": max(time.monotonic() - started, 0),
                    "heartbeat": True,
                }
                if latest is not None:
                    data.update(
                        step_id=latest.step_id,
                        step_stage=latest.stage,
                        step_instruction=latest.instruction[:500],
                    )
                await self._emit(
                    "build_progress",
                    "Docker build is still running",
                    **data,
                )
        finally:
            if not build.done():
                build.cancel()
                await asyncio.gather(build, return_exceptions=True)

    async def _begin_validation_phase(
        self,
        metrics: _ValidationAttemptMetrics,
        phase: str,
    ) -> None:
        metrics.begin_phase(phase)
        label = _VALIDATION_PHASE_LABELS[phase]
        await self._emit(
            "validation_phase_started",
            label,
            attempt=self._diagnostic_attempt,
            phase=phase,
            phase_label=label,
        )

    def _ensure_eligible(self, analysis: EnvironmentAnalysis) -> str:
        if not analysis.has_unit_test_framework:
            raise RepositoryRejected(
                "no_unit_test_framework",
                analysis.rejection_reason or "repository has no identifiable unit-test framework",
            )
        if analysis.selected_base_image_name is None:
            raise RepositoryRejected(
                "no_compatible_base_image",
                analysis.rejection_reason or "no configured base image matches the repository",
            )
        if analysis.primary_language is None:
            raise RepositoryRejected(
                "missing_language_evidence",
                "a testable repository must identify its primary implementation language",
            )
        if analysis.gpu_requirement == "required":
            if not self.settings.allow_gpu_repositories:
                raise RepositoryRejected(
                    "required_gpu_disallowed",
                    "repository unit tests require a GPU but the profile disallows GPU-only repositories",
                )
            if self.runtime.settings.gpus is None:
                raise ConfigurationError(
                    "required-GPU repositories are allowed, but docker.gpus is not configured"
                )
        return analysis.primary_language

    def _runner_limits(self) -> RunnerLimits:
        return RunnerLimits(
            collect_timeout_seconds=self.settings.collect_timeout_seconds,
            baseline_total_timeout_seconds=self.settings.baseline_total_timeout_seconds,
        )

    def _runtime_resources(self, analysis: EnvironmentAnalysis) -> RuntimeResources:
        return RuntimeResources(
            network=self.runtime.settings.network,
            cpus=self.runtime.settings.cpus,
            memory=self.runtime.settings.memory,
            gpus=(self.runtime.settings.gpus if analysis.gpu_requirement == "required" else None),
        )

    def _execution_contract(self, analysis: EnvironmentAnalysis) -> ExecutionContract:
        return ExecutionContract(
            workdir=self.settings.workdir,
            resources=self._runtime_resources(analysis),
            limits=self._runner_limits(),
        )

    async def _remove_intermediate_image(self, image: str, *, attempt: int) -> None:
        """Best-effort cleanup that never replaces the build or cancellation root cause."""

        async def remove() -> None:
            try:
                await self.runtime.remove_image(image)
            except Exception as exc:
                with suppress(Exception):
                    await self._emit(
                        "intermediate_image_cleanup_failed",
                        "failed to remove an intermediate candidate image",
                        attempt=attempt,
                        image=image,
                        error=str(exc)[-5000:],
                    )

        cleanup = asyncio.create_task(remove())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            with suppress(asyncio.CancelledError):
                await cleanup
            raise

    async def build(
        self,
        *,
        repository: Path,
        repository_url: str,
        commit: str,
    ) -> EnvironmentRecord:
        if self.agents.settings.model is None:
            raise ConfigurationError(
                "cross-language environment authoring requires agent.model in YAML and its "
                "provider credential in .env"
            )
        environment_id = self.environment_id(repository_url, commit)
        clone = derive_clone_plan(
            repository,
            repository_url=repository_url,
            commit=commit,
        )
        await self._emit("planner_started", "preparing language-neutral analysis container")
        planner_image = await self.runtime.prepare_image(
            self.settings.planner_image,
            pull=self.runtime.settings.pull,
        )
        try:
            analysis = await self._analyze(
                image=planner_image,
                repository=repository,
                repository_url=repository_url,
                commit=commit,
            )
            if self.diagnostics is not None:
                self.diagnostics.record_analysis(analysis)
            primary_language = self._ensure_eligible(analysis)
            execution_contract = self._execution_contract(analysis)
            await self._emit(
                "analysis_completed",
                f"selected {analysis.selected_base_image_name} for {primary_language}",
                language=primary_language,
                base_image=next(
                    image.image
                    for image in self.settings.base_images
                    if image.name == analysis.selected_base_image_name
                ),
                build_systems=analysis.build_systems,
                package_managers=analysis.package_managers,
                ecosystem_modules=analysis.ecosystem_modules,
                test_frameworks=analysis.test_frameworks,
                gpu_requirement=analysis.gpu_requirement,
            )
            artifacts = await self._author(
                image=planner_image,
                repository=repository,
                runtime_spec=_PLANNER_RUNTIME,
                analysis=analysis,
                execution_contract=execution_contract,
                repository_url=repository_url,
                commit=commit,
                current=None,
                failure=None,
            )
            await self._emit(
                "artifacts_authored",
                f"authored {primary_language} Dockerfile, activation, and Runner V1",
                language=primary_language,
            )

            failure: AgentFeedback | None = None
            repair_history: list[ArtifactRepairHistoryEntry] = []
            repair_memory: dict[tuple[str, tuple[str | int, ...], str | None], FeedbackIssue] = {}
            for attempt in range(1, self.settings.max_build_attempts + 1):
                self._diagnostic_attempt = attempt
                runtime_spec = RuntimeSpec(
                    workdir="/testbed",
                    activation=artifacts.activation,
                    runner=artifacts.runner,
                    limits=execution_contract.limits,
                    resources=execution_contract.resources,
                )
                image: str | None = None
                retain_image = False
                dockerfile = ""
                final_tag = self._tag("environment", environment_id, attempt)
                try:
                    dockerfile = render_final_dockerfile(
                        artifacts,
                        analysis=analysis,
                        clone=clone,
                        settings=self.settings,
                    )
                    if self.diagnostics is not None:
                        self.diagnostics.record_attempt(
                            attempt=attempt,
                            dockerfile=dockerfile,
                            artifacts=artifacts,
                        )
                    await self._emit(
                        "build_started",
                        f"building environment attempt {attempt}/{self.settings.max_build_attempts}",
                        attempt=attempt,
                        max_attempts=self.settings.max_build_attempts,
                    )
                    image, final_tag = await self._build_with_progress(
                        environment_id=environment_id,
                        dockerfile=dockerfile,
                        attempt=attempt,
                    )
                    verified_test_points = await self.validate(
                        image=image,
                        runtime_spec=runtime_spec,
                        commit=commit,
                        test_frameworks=analysis.test_frameworks,
                    )
                    await self._emit(
                        "validation_completed",
                        f"verified {len(verified_test_points)} passing test points",
                        attempt=attempt,
                        verified_test_point_count=len(verified_test_points),
                        verified_test_file_count=len({test.path for test in verified_test_points}),
                    )
                    if self.diagnostics is not None:
                        self.diagnostics.append_event(
                            "attempt_succeeded",
                            attempt=attempt,
                            data={
                                "verified_test_point_count": len(verified_test_points),
                                "verified_test_file_count": len(
                                    {test.path for test in verified_test_points}
                                ),
                            },
                        )
                    retain_image = True
                    return EnvironmentRecord(
                        environment_id=environment_id,
                        repository=repository_url,
                        commit=commit,
                        language=primary_language,
                        gpu_requirement=analysis.gpu_requirement,
                        image=image,
                        dockerfile=dockerfile,
                        runtime=runtime_spec,
                        verified_test_points=verified_test_points,
                    )
                except (DockerBuildError, InfrastructureError, ContractError) as exc:
                    if isinstance(exc, AgentFeedbackError):
                        failure = exc.feedback
                    else:
                        try:
                            failure = diagnose_environment_failure(
                                exc,
                                context=FailureDiagnosticContext(
                                    primary_language=primary_language,
                                    build_systems=analysis.build_systems,
                                    package_managers=analysis.package_managers,
                                    test_frameworks=analysis.test_frameworks,
                                    runner_source=artifacts.runner,
                                ),
                            )
                        except (DockerBuildError, InfrastructureError):
                            if self.diagnostics is not None:
                                self.diagnostics.append_event(
                                    "infrastructure_failed",
                                    attempt=attempt,
                                    data={"error": str(exc)[-50_000:]},
                                )
                            await self._emit(
                                "infrastructure_failed",
                                "environment infrastructure failed outside Agent-repairable artifacts",
                                attempt=attempt,
                                error=str(exc)[-5000:],
                            )
                            raise
                    rendered_failure = render_feedback(failure)
                    failed_phase = "docker_build" if image is None else "validation"
                    for feedback_issue in failure.issues:
                        observed_phase = feedback_issue.context.get("failure_phase")
                        if isinstance(observed_phase, str) and observed_phase:
                            failed_phase = observed_phase
                            break
                    if self.diagnostics is not None:
                        self.diagnostics.record_feedback(
                            attempt=attempt,
                            feedback=failure,
                            failed_phase=failed_phase,
                        )
                    await self._emit(
                        "validation_failed",
                        f"attempt {attempt} failed; returning structured feedback to artifact Agent",
                        attempt=attempt,
                        error=rendered_failure,
                    )
                    if attempt >= self.settings.max_build_attempts:
                        raise ContractError(
                            f"environment failed after {attempt} complete build attempts: "
                            f"{rendered_failure}"
                        ) from exc
                    diagnostic_image = image or planner_image
                    diagnostic_runtime = select_repair_runtime_spec(
                        built_image_available=image is not None,
                        current_runtime=runtime_spec,
                        failure=failure,
                    )
                    artifacts = await self._author(
                        image=diagnostic_image,
                        repository=None if image is not None else repository,
                        runtime_spec=diagnostic_runtime,
                        analysis=analysis,
                        execution_contract=execution_contract,
                        repository_url=repository_url,
                        commit=commit,
                        current=artifacts,
                        failure=failure,
                        repair_history=tuple(repair_history),
                        accepted_constraints=(
                            AgentFeedback(issues=tuple(repair_memory.values()))
                            if repair_memory
                            else None
                        ),
                        validate_runner_transport=(
                            image is not None
                            and bool(failure.issues)
                            and all(issue_.path == ("runner",) for issue_ in failure.issues)
                        ),
                    )
                    repair_history.append(
                        ArtifactRepairHistoryEntry(
                            attempt=attempt,
                            failed_phase=failed_phase,
                            feedback=failure,
                        )
                    )
                    for remembered in failure.issues:
                        compact = remembered.model_copy(update={"evidence": None})
                        repair_memory[(compact.code, compact.path, compact.expected)] = compact
                finally:
                    if not retain_image and self.settings.cleanup_intermediate_images:
                        await self._remove_intermediate_image(final_tag, attempt=attempt)
            raise AssertionError("unreachable Environment Build loop")
        finally:
            await self._emit("planner_completed", "analysis container released")

    async def validate(
        self,
        *,
        image: str,
        runtime_spec: RuntimeSpec,
        commit: str,
        test_frameworks: tuple[str, ...] = (),
    ) -> tuple[CollectedTestPoint, ...]:
        metrics = _ValidationAttemptMetrics()
        await self._emit(
            "validation_phase_started",
            _VALIDATION_PHASE_LABELS["container_session"],
            attempt=self._diagnostic_attempt,
            phase="container_session",
            phase_label=_VALIDATION_PHASE_LABELS["container_session"],
        )
        try:
            verified = await self._validate_once(
                image=image,
                runtime_spec=runtime_spec,
                commit=commit,
                test_frameworks=test_frameworks,
                metrics=metrics,
            )
        except BaseException as exc:
            await self._emit(
                "validation_measured",
                "recorded failed environment validation attempt",
                **metrics.event_data(status="failed", error=exc),
            )
            raise
        await self._emit(
            "validation_measured",
            "recorded successful environment validation attempt",
            **metrics.event_data(status="succeeded", error=None),
        )
        return verified

    async def _assert_collected_test_paths(
        self,
        *,
        image: str,
        runtime_spec: RuntimeSpec,
        paths: tuple[str, ...],
    ) -> None:
        """Validate collected source paths against an untouched candidate-image filesystem."""
        script = r"""
set -euo pipefail
root=/testbed
for path do
    case "$path" in
        .git|.git/*|*/.git|*/.git/*)
            printf 'Git metadata is not a test source: %s\n' "$path" >&2
            exit 1
            ;;
    esac
    expected="$root/$path"
    if [ ! -f "$expected" ]; then
        printf 'not a regular file in the candidate image: %s\n' "$path" >&2
        exit 1
    fi
    resolved=$(realpath -e -- "$expected") || {
        printf 'cannot resolve test source: %s\n' "$path" >&2
        exit 1
    }
    case "$resolved" in
        "$root"/*) ;;
        *)
            printf 'test source escapes /testbed: %s -> %s\n' "$path" "$resolved" >&2
            exit 1
            ;;
    esac
    # Repository-owned symlinks are valid test boundaries (for example a monorepo
    # may expose a nested package's tests through its top-level tests directory).
    # Safety is determined from the resolved target below: it must remain under
    # /testbed, belong to a Git work tree there, and be tracked by that repository.
    source_dir=${resolved%/*}
    repository_root=$(git -C "$source_dir" rev-parse --show-toplevel 2>/dev/null) || {
        printf 'test source is outside a Git work tree: %s\n' "$path" >&2
        exit 1
    }
    case "$repository_root" in
        "$root"|"$root"/*) ;;
        *)
            printf 'test source belongs to a Git work tree outside /testbed: %s\n' "$path" >&2
            exit 1
            ;;
    esac
    repository_path=${resolved#"$repository_root"/}
    if ! git -C "$repository_root" ls-files --error-unmatch -- "$repository_path" \
        >/dev/null 2>&1; then
        printf 'test source is not tracked by its repository: %s\n' "$path" >&2
        exit 1
    fi
done
""".strip()
        # A new container proves that paths existed in the immutable candidate image;
        # Runner collect cannot manufacture a file and then cite it as source evidence.
        async with self.runtime.session(image, runtime_spec=runtime_spec) as path_session:
            for start in range(0, len(paths), 100):
                result = await path_session.exec(
                    (
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        script,
                        "featurebench-test-source-check",
                        *paths[start : start + 100],
                    ),
                    workdir="/",
                    timeout_seconds=runtime_spec.limits.collect_timeout_seconds,
                    check=False,
                )
                if result.exit_code != 0:
                    raise ContractError(
                        "Runner collect returned an unsafe or untracked test source path: "
                        f"{result.combined_output[-5000:]}"
                    )

    async def _validate_once(
        self,
        *,
        image: str,
        runtime_spec: RuntimeSpec,
        commit: str,
        test_frameworks: tuple[str, ...],
        metrics: _ValidationAttemptMetrics,
    ) -> tuple[CollectedTestPoint, ...]:
        verified: tuple[CollectedTestPoint, ...]
        async with self.runtime.session(
            image,
            runtime_spec=runtime_spec,
        ) as session:
            metrics.complete_phase()
            await self._begin_validation_phase(metrics, "activation")
            try:
                activation_check = await session.exec(
                    (
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        'set -euo pipefail; source /dev/stdin; test "$PWD" = /testbed',
                    ),
                    workdir="/",
                    input_bytes=runtime_spec.activation.encode("utf-8"),
                    check=False,
                )
            except CommandTimeoutError as exc:
                raise EnvironmentValidationError(
                    f"activation failed: activation timed out: {exc}",
                    phase="activation",
                    repair_fields=("activation",),
                ) from exc
            if activation_check.exit_code != 0:
                activation_detail = activation_check.combined_output[-5000:]
                raise EnvironmentValidationError(
                    "activation failed to enter /testbed: " + activation_detail,
                    phase="activation",
                    repair_fields=_activation_repair_fields(activation_detail),
                )
            try:
                activated_git = await session.exec(("git", "--version"), activated=True)
            except CommandTimeoutError as exc:
                raise EnvironmentValidationError(
                    f"activation failed: activated command timed out: {exc}",
                    phase="activation",
                    repair_fields=("activation",),
                ) from exc
            if activated_git.exit_code != 0:
                raise EnvironmentValidationError(
                    "activation failed: git is unavailable after environment activation",
                    phase="activation",
                    repair_fields=("activation", "dockerfile"),
                )
            metrics.complete_phase()

            await self._begin_validation_phase(metrics, "repository_baseline")
            git = await session.exec(("git", "rev-parse", "HEAD"), check=False)
            if git.exit_code != 0 or git.stdout.strip() != commit:
                raise ContractError("final image Git baseline is not the requested pinned commit")
            status = await session.exec(("git", "status", "--porcelain"), check=False)
            if status.exit_code != 0 or status.stdout.strip():
                raise ContractError("final image repository baseline is not clean")
            control_directory = await session.exec(
                ("test", "!", "-e", "/opt/featurebench"), workdir="/", check=False
            )
            if control_directory.exit_code != 0:
                raise ContractError("final image contains forbidden FeatureBench control files")
            metrics.complete_phase()

            await self._begin_validation_phase(metrics, "collect")
            runner = EnvironmentRunner(session)
            collected = await runner.collect(
                timeout_seconds=runtime_spec.limits.collect_timeout_seconds
            )
            metrics.observations.update(
                collected_test_point_count=len(collected.collected),
                minimum_verified_test_points=self.settings.min_verified_test_points,
                collection_stability_runs=self.settings.collect_stability_runs,
            )
            if collected.exit_code != 0 or collected.errors:
                raise ContractError("Runner collect failed: " + summarize_runner_failure(collected))
            validate_framework_collection(
                collected.collected,
                frameworks=test_frameworks,
            )
            # This eligibility gate deliberately precedes stability checks, path checks,
            # quickcheck, and baseline execution so an undersized inventory returns
            # immediate structured Runner feedback without spending more test time.
            assert_collection_can_reach_verified_floor(
                collected,
                minimum=self.settings.min_verified_test_points,
            )
            for _ in range(1, self.settings.collect_stability_runs):
                repeated = await runner.collect(
                    timeout_seconds=runtime_spec.limits.collect_timeout_seconds
                )
                assert_stable_collection(collected, repeated)
            unique_paths = tuple(dict.fromkeys(test.path for test in collected.collected))
            await self._assert_collected_test_paths(
                image=image,
                runtime_spec=runtime_spec,
                paths=unique_paths,
            )
            failure_probe = await probe_runner_failure_contract(
                runner,
                collected,
                timeout_seconds=runtime_spec.limits.collect_timeout_seconds,
            )
            metrics.observations.update(
                runner_failure_probe_exit_code=failure_probe.exit_code,
                runner_failure_probe_outcome=failure_probe.results[0].outcome,
            )
            collection_validation_duration_seconds = metrics.complete_phase()
            metrics.observations["collection_validation_duration_seconds"] = (
                collection_validation_duration_seconds
            )

            if any(
                normalize_ecosystem_label(framework) == "pytest" for framework in test_frameworks
            ):
                await self._begin_validation_phase(metrics, "runner_transport")
                try:
                    transport_probe = await probe_runner_large_request_transport(
                        runner,
                        collected,
                        timeout_seconds=runtime_spec.limits.collect_timeout_seconds,
                    )
                except (CommandTimeoutError, ContractError) as exc:
                    raise EnvironmentValidationError(
                        f"large Runner request transport failed: {exc}",
                        phase="runner_transport",
                        repair_fields=("runner",),
                    ) from exc
                metrics.observations.update(
                    runner_transport_probe_test_point_count=len(transport_probe.results),
                    runner_transport_probe_exit_code=transport_probe.exit_code,
                )
                metrics.complete_phase()

            selected_ids = select_quickcheck_test_ids(
                collected.collected,
                commit=commit,
                limit=self.settings.quickcheck_test_points,
            )
            metrics.observations.update(
                quickcheck_test_point_limit=self.settings.quickcheck_test_points,
                quickcheck_test_point_count=len(selected_ids),
                quickcheck_test_point_ids=selected_ids,
                quickcheck_min_pass_rate=self.settings.quickcheck_min_pass_rate,
            )
            await self._begin_validation_phase(metrics, "quickcheck")
            try:
                sample = await runner.run(
                    selected_ids,
                    timeout_seconds=runtime_spec.limits.collect_timeout_seconds,
                )
            except CommandTimeoutError as exc:
                raise EnvironmentValidationError(
                    f"quickcheck environment health gate failed: Runner run timed out: {exc}",
                    phase="quickcheck",
                    repair_fields=("dockerfile", "activation", "runner"),
                ) from exc
            quickcheck_duration_seconds = metrics.current_phase_duration()
            quickcheck_outcome_counts = {
                outcome: sum(result.outcome == outcome for result in sample.results)
                for outcome in ("passed", "failed", "skipped", "error")
            }
            metrics.observations.update(
                quickcheck_duration_seconds=quickcheck_duration_seconds,
                quickcheck_exit_code=sample.exit_code,
                quickcheck_outcomes=quickcheck_outcome_counts,
                quickcheck_error_count=len(sample.errors),
            )
            quickcheck_passed, quickcheck_required_passes = assert_quickcheck_health(
                sample,
                selected_ids=selected_ids,
                minimum_pass_rate=self.settings.quickcheck_min_pass_rate,
            )
            metrics.complete_phase()
            metrics.observations.update(
                quickcheck_required_passes=quickcheck_required_passes,
                quickcheck_passed=quickcheck_passed,
                quickcheck_observed_pass_rate=(quickcheck_passed / len(selected_ids)),
            )

            await self._begin_validation_phase(metrics, "baseline")
            baseline_group_ids = group_test_ids_by_path(
                collected.collected,
                max_group_size=self.settings.baseline_group_max_test_points,
            )
            baseline_reports = await run_baseline_groups(
                _IsolatedBaselineRunner(
                    self.runtime,
                    image=image,
                    runtime_spec=runtime_spec,
                ),
                baseline_group_ids,
                total_timeout_seconds=runtime_spec.limits.baseline_total_timeout_seconds,
            )
            baseline_duration_seconds = metrics.current_phase_duration()
            verified = verified_run_test_points(collected, baseline_reports)
            baseline_outcome_counts = {
                outcome: sum(
                    result.outcome == outcome
                    for report in baseline_reports
                    for result in report.results
                )
                for outcome in ("passed", "failed", "skipped", "error")
            }
            metrics.observations.update(
                verified_test_point_count=len(verified),
                baseline_group_count=len(baseline_reports),
                baseline_duration_seconds=baseline_duration_seconds,
                baseline_total_timeout_seconds=runtime_spec.limits.baseline_total_timeout_seconds,
                baseline_nonzero_group_count=sum(
                    report.exit_code != 0 for report in baseline_reports
                ),
                baseline_timeout_group_count=sum(
                    any("timed out" in error.casefold() for error in report.errors)
                    for report in baseline_reports
                ),
                baseline_total_timeout_group_count=sum(
                    any("total timeout" in error.casefold() for error in report.errors)
                    for report in baseline_reports
                ),
                baseline_outcomes=baseline_outcome_counts,
                baseline_error_count=sum(len(report.errors) for report in baseline_reports),
            )
            if len(verified) < self.settings.min_verified_test_points:
                shortfall_evidence = summarize_baseline_shortfall(
                    required=self.settings.min_verified_test_points,
                    verified=len(verified),
                    collected=len(collected.collected),
                    group_ids=baseline_group_ids,
                    reports=baseline_reports,
                )
                raise ContractError(
                    "baseline sweep produced too few verified passing test points: "
                    f"required={self.settings.min_verified_test_points}; "
                    f"received={len(verified)}; "
                    f"collected={len(collected.collected)}; "
                    f"groups={len(baseline_reports)}; "
                    f"evidence={shortfall_evidence}"
                )
            metrics.complete_phase()
            await self._begin_validation_phase(metrics, "container_cleanup")
        metrics.complete_phase()
        return verified
