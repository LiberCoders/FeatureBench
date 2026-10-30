"""Language-neutral test Runner contract executed as an external artifact."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import suppress
from math import ceil
from uuid import uuid4

from codeworld.environment.runner_contract import (
    CollectedTestPoint,
    RunnerAction,
    RunnerRequest,
    RunnerResponse,
)
from codeworld.errors import (
    CommandTimeoutError,
    ContractError,
    RunnerCollectionUnstableError,
    RunnerTimeoutError,
)
from codeworld.runtime.base import ContainerSessionProtocol

_MAX_RUNNER_RESPONSE_BYTES = 50_000_000
_LARGE_REQUEST_PROBE_SELECTION_BYTES = 3 * 1024 * 1024
_LARGE_REQUEST_PROBE_TEST_POINTS = 8192
_LARGE_REQUEST_PROBE_TIMEOUT_SECONDS = 30.0


class EnvironmentRunner:
    """Calls the image-owned runner without knowing its language or test framework."""

    def __init__(self, session: ContainerSessionProtocol) -> None:
        self.session = session

    def _timeout(self, action: RunnerAction, timeout_seconds: float | None) -> float:
        if timeout_seconds is not None:
            return timeout_seconds
        runtime_spec = self.session.runtime_spec
        if runtime_spec is None:
            raise ContractError("Runner execution requires an explicit timeout or runtime limits")
        if action == "collect":
            return runtime_spec.limits.collect_timeout_seconds
        raise ContractError("Runner run requires an explicit Adapter or validation timeout")

    @staticmethod
    def _remaining_deadline(deadline: float | None, *, phase: str) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise CommandTimeoutError(
                f"Runner {phase} exhausted its operation deadline",
                timeout_seconds=0,
            )
        return remaining

    async def execute(
        self,
        action: RunnerAction,
        *,
        test_ids: tuple[str, ...] = (),
        timeout_seconds: float | None = None,
        deadline: float | None = None,
    ) -> RunnerResponse:
        """Execute one request before an optional absolute event-loop deadline.

        Transport/setup deadline exhaustion remains infrastructure failure; only
        the image-owned process timeout is promoted to ``RunnerTimeoutError``.
        """
        if self.session.runtime_spec is None:
            raise ContractError("runtime artifacts are unavailable")
        request = RunnerRequest(action=action, test_ids=test_ids)
        request_bytes = request.model_dump_json().encode("utf-8")
        nonce = uuid4().hex
        request_path = f"/tmp/featurebench-request-{nonce}.json"
        response_path = f"/tmp/featurebench-response-{nonce}.json"
        try:
            await self.session.exec(
                ("tee", request_path),
                workdir="/",
                input_bytes=request_bytes,
                timeout_seconds=self._remaining_deadline(
                    deadline,
                    phase="request staging",
                ),
                check=True,
            )
            runner_timeout = self._timeout(action, timeout_seconds)
            remaining = self._remaining_deadline(deadline, phase="process startup")
            if remaining is not None:
                runner_timeout = min(runner_timeout, remaining)
            try:
                result = await self.session.run_script(
                    self.session.runtime_spec.runner,
                    ("--request", request_path, "--response", response_path),
                    activated=True,
                    timeout_seconds=runner_timeout,
                )
            except CommandTimeoutError as exc:
                raise RunnerTimeoutError(
                    f"Runner {action} timed out: {exc}",
                    argv=exc.argv,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    timeout_seconds=exc.timeout_seconds or runner_timeout,
                ) from exc
            response_timeout: float | None = None
            try:
                response_timeout = self._remaining_deadline(
                    deadline,
                    phase="response read",
                )
                if response_timeout is None:
                    response_bytes = await self.session.read_absolute_file(
                        response_path,
                        max_bytes=_MAX_RUNNER_RESPONSE_BYTES,
                    )
                else:
                    async with asyncio.timeout(response_timeout):
                        response_bytes = await self.session.read_absolute_file(
                            response_path,
                            max_bytes=_MAX_RUNNER_RESPONSE_BYTES,
                        )
            except TimeoutError as exc:
                raise CommandTimeoutError(
                    "Runner response read exhausted its operation deadline",
                    timeout_seconds=response_timeout,
                ) from exc
            except (ValueError, ContractError) as exc:
                try:
                    current_request = await self.session.read_absolute_file(
                        request_path,
                        max_bytes=max(len(request_bytes) + 1, 4096),
                    )
                except (ValueError, ContractError) as request_exc:
                    request_state = f"unreadable ({request_exc})"
                else:
                    request_state = (
                        "unchanged"
                        if current_request == request_bytes
                        else f"overwritten with {current_request[:1000]!r}"
                    )
                raise ContractError(
                    f"runner did not produce a bounded response for {action}: "
                    f"process_exit_code={result.exit_code}; "
                    f"process_output={result.combined_output[-5000:]!r}; "
                    f"request_file_state={request_state}; "
                    f"response_read_error={exc}"
                ) from exc
            try:
                raw = json.loads(response_bytes)
                response = RunnerResponse.model_validate(raw)
            except (ValueError, TypeError) as exc:
                response_tail = response_bytes[-5000:].decode("utf-8", errors="replace")
                raise ContractError(
                    f"runner returned an invalid response for {action}: "
                    f"process_exit_code={result.exit_code}; "
                    f"process_output={result.combined_output[-5000:]!r}; "
                    f"response_tail={response_tail!r}; "
                    f"validation_error={exc}"
                ) from exc
            if response.action != action:
                raise ContractError(
                    f"runner response action {response.action!r} does not match {action!r}"
                )
            if result.exit_code != 0:
                raise ContractError(
                    "runner process failed after producing a response: "
                    f"process={result.exit_code}, response={response.exit_code}; "
                    f"{result.combined_output[-5000:]}"
                )
            if action == "run":
                returned = {test.id for test in response.results}
                if returned != set(test_ids):
                    evidence = {
                        "requested_count": len(test_ids),
                        "returned_count": len(returned),
                        "missing_test_ids": sorted(set(test_ids) - returned)[:20],
                        "unexpected_test_ids": sorted(returned - set(test_ids))[:20],
                        "response_exit_code": response.exit_code,
                        "response_errors": list(response.errors[:10]),
                        "response_output_excerpt": response.output[-3000:],
                    }
                    raise ContractError(
                        "runner omitted or added selected test results: "
                        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
                    )
            if action == "collect":
                response = response.model_copy(
                    update={
                        "collected": tuple(
                            sorted(response.collected, key=lambda item: (item.id, item.path))
                        )
                    }
                )
            return response
        finally:
            cleanup_timeout = (
                None if deadline is None else deadline - asyncio.get_running_loop().time()
            )
            if cleanup_timeout is None:
                await self.session.exec(
                    ("rm", "-f", request_path, response_path),
                    workdir="/",
                    check=False,
                )
            elif cleanup_timeout > 0:
                # Preserve the operation outcome once its absolute deadline
                # is exhausted; the disposable container owns stale /tmp files.
                with suppress(CommandTimeoutError):
                    await self.session.exec(
                        ("rm", "-f", request_path, response_path),
                        workdir="/",
                        timeout_seconds=cleanup_timeout,
                        check=False,
                    )

    async def collect(
        self,
        *,
        timeout_seconds: float | None = None,
        deadline: float | None = None,
    ) -> RunnerResponse:
        return await self.execute(
            "collect",
            timeout_seconds=timeout_seconds,
            deadline=deadline,
        )

    async def run(
        self,
        test_ids: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        deadline: float | None = None,
    ) -> RunnerResponse:
        return await self.execute(
            "run",
            test_ids=test_ids,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
        )


async def probe_runner_failure_contract(
    runner: EnvironmentRunner,
    collected: RunnerResponse,
    *,
    timeout_seconds: float,
) -> RunnerResponse:
    """Exercise the Runner's missing-ID path before accepting it as an artifact."""

    known_ids = {test.id for test in collected.collected}
    missing_id = "__featurebench_missing_test_point__"
    while missing_id in known_ids:
        missing_id += "_"
    response = await runner.run((missing_id,), timeout_seconds=timeout_seconds)
    result = response.results[0]
    if response.exit_code == 0 or response.errors or result.outcome != "error":
        raise ContractError(
            "Runner failure-path probe did not report an auditable missing test: "
            f"exit_code={response.exit_code}; outcome={result.outcome}; "
            f"errors={list(response.errors)!r}; output={response.output[-3000:]!r}"
        )
    return response


def _large_request_probe_ids(collected: RunnerResponse) -> tuple[str, ...]:
    """Build a pytest-valid selection whose serialized request exceeds normal ARG_MAX."""

    collected_paths = {test.path for test in collected.collected}
    path = f"__featurebench_transport_probe_{uuid4().hex}.py"
    while path in collected_paths:
        path = "_" + path
    template = f"{path}::test_{{:08d}}_{{}}#0"
    fixed_bytes = len(template.format(0, "").encode("utf-8"))
    padding_bytes = max(
        ceil(_LARGE_REQUEST_PROBE_SELECTION_BYTES / _LARGE_REQUEST_PROBE_TEST_POINTS) - fixed_bytes,
        1,
    )
    padding = "x" * padding_bytes
    return tuple(
        template.format(index, padding) for index in range(_LARGE_REQUEST_PROBE_TEST_POINTS)
    )


async def probe_runner_large_request_transport(
    runner: EnvironmentRunner,
    collected: RunnerResponse,
    *,
    timeout_seconds: float,
) -> RunnerResponse:
    """Prove that a pytest Runner keeps an ARG_MAX-sized selection file-backed."""

    requested = _large_request_probe_ids(collected)
    selection_bytes = sum(len(test_id.encode("utf-8")) for test_id in requested)
    if selection_bytes < _LARGE_REQUEST_PROBE_SELECTION_BYTES:
        raise AssertionError("large Runner selection probe is smaller than its contract")
    try:
        response = await runner.run(
            requested,
            timeout_seconds=min(timeout_seconds, _LARGE_REQUEST_PROBE_TIMEOUT_SECONDS),
        )
    except (CommandTimeoutError, ContractError) as exc:
        raise ContractError(
            "Runner large-request transport probe failed before returning complete results: "
            f"requested_count={len(requested)}; selection_bytes={selection_bytes}; detail={exc}"
        ) from exc
    outcome_counts = {
        outcome: sum(result.outcome == outcome for result in response.results)
        for outcome in ("passed", "failed", "skipped", "error")
    }
    if response.exit_code == 0 or response.errors or outcome_counts["error"] != len(requested):
        raise ContractError(
            "Runner large-request transport probe did not fail closed for missing test IDs: "
            f"requested_count={len(requested)}; selection_bytes={selection_bytes}; "
            f"exit_code={response.exit_code}; outcomes={outcome_counts}; "
            f"errors={list(response.errors[:10])!r}; output={response.output[-3000:]!r}"
        )
    return response


def assert_stable_collection(
    baseline: RunnerResponse,
    repeated: RunnerResponse,
) -> None:
    """Enforce the collection invariant shared by preflight and final validation."""
    baseline_mapping = {test.id: test.path for test in baseline.collected}
    repeated_mapping = {test.id: test.path for test in repeated.collected}
    if repeated.exit_code != 0 or repeated.errors:
        raise RunnerCollectionUnstableError(
            "Runner collect is not stable across repeated executions: "
            f"repeated_exit_code={repeated.exit_code}; "
            f"repeated_errors={list(repeated.errors)!r}"
        )
    if repeated_mapping != baseline_mapping:
        baseline_ids = set(baseline_mapping)
        repeated_ids = set(repeated_mapping)
        path_mismatches = [
            (test_id, baseline_mapping[test_id], repeated_mapping[test_id])
            for test_id in sorted(baseline_ids & repeated_ids)
            if baseline_mapping[test_id] != repeated_mapping[test_id]
        ]
        raise RunnerCollectionUnstableError(
            "Runner collect is not stable across repeated executions: "
            f"baseline_count={len(baseline_mapping)}; "
            f"repeated_count={len(repeated_mapping)}; "
            f"missing_ids={sorted(baseline_ids - repeated_ids)[:10]!r}; "
            f"unexpected_ids={sorted(repeated_ids - baseline_ids)[:10]!r}; "
            f"path_mismatches={path_mismatches[:10]!r}"
        )


def verified_run_test_points(
    collected: RunnerResponse,
    runs: Sequence[RunnerResponse],
) -> tuple[CollectedTestPoint, ...]:
    """Return collected points explicitly reported passed by selected-test executions."""
    if collected.action != "collect" or any(run.action != "run" for run in runs):
        raise ValueError("verified test points require one collect response and run responses")
    inventory = {test.id: test for test in collected.collected}
    returned = {result.id for run in runs for result in run.results}
    unknown = returned - inventory.keys()
    if unknown:
        raise ContractError(
            "Runner execution reported test IDs absent from stable collection: "
            f"action=run; unexpected_ids={sorted(unknown)[:20]!r}"
        )
    verified = {result.id for run in runs for result in run.results if result.outcome == "passed"}
    return tuple(test for test in collected.collected if test.id in verified)
