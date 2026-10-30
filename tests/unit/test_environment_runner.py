from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import cast

import pytest

from codeworld.environment.frameworks.pytest import (
    format_pytest_public_id,
    parse_pytest_public_id,
    pytest_native_base,
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
)
from codeworld.environment.runner_contract import (
    TestPointExecution as RunnerTestPointExecution,
)
from codeworld.errors import CommandTimeoutError, ContractError, RunnerTimeoutError
from codeworld.models import RunnerLimits, RuntimeSpec
from codeworld.runtime.container import CommandResult, ContainerSession


def test_pytest_public_id_helper_uses_structural_base_and_occurrence() -> None:
    base = pytest_native_base(
        "tests/test_api.py::TestAPI::test_value[volatile-0x1234]",
        "test_value",
    )
    assert base == "tests/test_api.py::TestAPI::test_value"
    public_id = format_pytest_public_id(base, 2)
    assert public_id == "tests/test_api.py::TestAPI::test_value#2"
    parsed = parse_pytest_public_id(public_id)
    assert parsed.native_base == base
    assert parsed.occurrence == 2


@pytest.mark.parametrize(
    "value",
    ("tests/test_api.py::test_value", "tests/test_api.py::test_value#01", "#0"),
)
def test_pytest_public_id_helper_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="pytest public IDs"):
        parse_pytest_public_id(value)


class FakeRunnerSession:
    def __init__(self, *, process_exit_code: int) -> None:
        self.process_exit_code = process_exit_code
        self.last_timeout_seconds: float | None = None
        self.request_action = "collect"
        self.request_test_ids: tuple[str, ...] = ()
        self.runtime_spec = RuntimeSpec(
            activation="#!/usr/bin/env bash\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Fake external Runner used only to exercise host-side transport semantics.\n"
                "# Its implementation is deliberately supplied by the fake session below.\n"
                "true\n"
            ),
        )

    async def exec(
        self,
        arguments: Sequence[str],
        **kwargs: object,
    ) -> CommandResult:
        input_bytes = kwargs.get("input_bytes")
        if arguments and arguments[0] == "tee" and isinstance(input_bytes, bytes):
            request = json.loads(input_bytes)
            self.request_action = str(request["action"])
            self.request_test_ids = tuple(request.get("test_ids", ()))
        return CommandResult(
            argv=("docker", *arguments),
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0,
        )

    async def run_script(
        self,
        script: str,
        arguments: Sequence[str] = (),
        **kwargs: object,
    ) -> CommandResult:
        del script
        timeout_seconds = kwargs.get("timeout_seconds")
        self.last_timeout_seconds = (
            float(timeout_seconds) if isinstance(timeout_seconds, int | float) else None
        )
        return CommandResult(
            argv=("bash", *arguments),
            exit_code=self.process_exit_code,
            stdout="runner transport output",
            stderr="",
            duration_seconds=0,
        )

    async def read_absolute_file(self, path: str, *, max_bytes: int) -> bytes:
        del path, max_bytes
        return json.dumps(
            {
                "action": self.request_action,
                "exit_code": 1,
                "collected": [],
                "results": [
                    {"id": test_id, "outcome": "failed"} for test_id in self.request_test_ids
                ],
                "errors": [],
                "output": "one test failed",
            }
        ).encode()


class FakeMissingResponseSession(FakeRunnerSession):
    async def read_absolute_file(self, path: str, *, max_bytes: int) -> bytes:
        del max_bytes
        if "response" in path:
            raise ContractError("response is missing")
        return b'{"action":"collect","exit_code":0}'


class FakeTimeoutSession(FakeRunnerSession):
    async def run_script(
        self,
        script: str,
        arguments: Sequence[str] = (),
        **kwargs: object,
    ) -> CommandResult:
        del script, arguments, kwargs
        raise CommandTimeoutError("command timed out after 33s: docker exec example")


class FakeDeadlineTimeoutSession(FakeRunnerSession):
    async def exec(
        self,
        arguments: Sequence[str],
        **kwargs: object,
    ) -> CommandResult:
        if arguments and arguments[0] == "tee":
            await asyncio.sleep(0.005)
        return await super().exec(arguments, **kwargs)

    async def run_script(
        self,
        script: str,
        arguments: Sequence[str] = (),
        **kwargs: object,
    ) -> CommandResult:
        del script, arguments
        timeout_seconds = kwargs.get("timeout_seconds")
        assert isinstance(timeout_seconds, int | float)
        self.last_timeout_seconds = float(timeout_seconds)
        await asyncio.sleep(self.last_timeout_seconds)
        raise CommandTimeoutError(
            "Runner process exhausted its remaining operation deadline",
            timeout_seconds=self.last_timeout_seconds,
        )


class FakeRequestTimeoutSession(FakeRunnerSession):
    async def exec(
        self,
        arguments: Sequence[str],
        **kwargs: object,
    ) -> CommandResult:
        if arguments and arguments[0] == "tee":
            timeout_seconds = kwargs.get("timeout_seconds")
            raise CommandTimeoutError(
                "Runner request staging exhausted its operation deadline",
                timeout_seconds=(
                    float(timeout_seconds) if isinstance(timeout_seconds, int | float) else None
                ),
            )
        return await super().exec(arguments, **kwargs)


class FakeUnorderedCollectionSession(FakeRunnerSession):
    async def read_absolute_file(self, path: str, *, max_bytes: int) -> bytes:
        del path, max_bytes
        return (
            b'{"action":"collect","exit_code":0,"collected":['
            b'{"id":"test b","path":"tests/test_b.py"},'
            b'{"id":"test a","path":"tests/test_a.py"}]}'
        )


class FakeInvalidFailureResponseSession(FakeRunnerSession):
    async def read_absolute_file(self, path: str, *, max_bytes: int) -> bytes:
        del path, max_bytes
        return b'{"action":"collect","exit_code":1,"output":""}'


class FakeActionResponseSession(FakeRunnerSession):
    def __init__(self) -> None:
        super().__init__(process_exit_code=0)
        self.request_action = "collect"
        self.request_test_ids: tuple[str, ...] = ()
        self.last_max_bytes: int | None = None
        self.runtime_spec = self.runtime_spec.model_copy(
            update={
                "limits": RunnerLimits(
                    collect_timeout_seconds=11,
                    baseline_total_timeout_seconds=33,
                )
            }
        )

    async def exec(self, arguments: Sequence[str], **kwargs: object) -> CommandResult:
        input_bytes = kwargs.get("input_bytes")
        if arguments and arguments[0] == "tee" and isinstance(input_bytes, bytes):
            request = json.loads(input_bytes)
            self.request_action = str(request["action"])
            self.request_test_ids = tuple(request.get("test_ids", ()))
        return await super().exec(arguments, **kwargs)

    async def read_absolute_file(self, path: str, *, max_bytes: int) -> bytes:
        del path
        self.last_max_bytes = max_bytes
        return json.dumps(
            {
                "action": self.request_action,
                "exit_code": 0,
                "collected": [],
                "results": [
                    {"id": test_id, "outcome": "passed"} for test_id in self.request_test_ids
                ],
                "errors": [],
                "output": "",
            }
        ).encode()


@pytest.mark.asyncio
async def test_test_failure_is_reported_only_by_response_exit_code() -> None:
    runner = EnvironmentRunner(cast(ContainerSession, FakeRunnerSession(process_exit_code=0)))
    response = await runner.run(("test-id",), timeout_seconds=1)
    assert response.exit_code == 1


@pytest.mark.asyncio
async def test_nonzero_runner_process_is_a_transport_failure() -> None:
    runner = EnvironmentRunner(cast(ContainerSession, FakeRunnerSession(process_exit_code=2)))
    with pytest.raises(ContractError, match="runner process failed"):
        await runner.run(("test-id",), timeout_seconds=1)


@pytest.mark.asyncio
async def test_runner_timeout_preserves_the_action_for_failure_attribution() -> None:
    runner = EnvironmentRunner(cast(ContainerSession, FakeTimeoutSession(process_exit_code=0)))
    with pytest.raises(RunnerTimeoutError, match="Runner run timed out") as captured:
        await runner.run(("test-id",), timeout_seconds=33)
    assert captured.value.timeout_seconds == 33


@pytest.mark.asyncio
async def test_runner_deadline_deducts_request_setup_before_typed_process_timeout() -> None:
    session = FakeDeadlineTimeoutSession(process_exit_code=0)
    runner = EnvironmentRunner(cast(ContainerSession, session))
    loop = asyncio.get_running_loop()

    with pytest.raises(RunnerTimeoutError, match="Runner run timed out"):
        await runner.run(
            ("test-id",),
            timeout_seconds=1,
            deadline=loop.time() + 0.03,
        )

    assert session.last_timeout_seconds is not None
    assert 0 < session.last_timeout_seconds < 0.03


@pytest.mark.asyncio
async def test_runner_deadline_keeps_request_setup_timeout_as_infrastructure() -> None:
    session = FakeRequestTimeoutSession(process_exit_code=0)
    runner = EnvironmentRunner(cast(ContainerSession, session))
    deadline = asyncio.get_running_loop().time() + 1

    with pytest.raises(CommandTimeoutError) as captured:
        await runner.run(("test-id",), timeout_seconds=1, deadline=deadline)

    assert type(captured.value) is CommandTimeoutError
    assert "request staging" in str(captured.value)


@pytest.mark.asyncio
async def test_missing_response_reports_whether_runner_overwrote_request() -> None:
    runner = EnvironmentRunner(
        cast(ContainerSession, FakeMissingResponseSession(process_exit_code=0))
    )
    with pytest.raises(ContractError, match="request_file_state=overwritten"):
        await runner.collect(timeout_seconds=1)


@pytest.mark.asyncio
async def test_invalid_runner_response_preserves_transport_and_response_evidence() -> None:
    runner = EnvironmentRunner(
        cast(ContainerSession, FakeInvalidFailureResponseSession(process_exit_code=0))
    )

    with pytest.raises(ContractError) as captured:
        await runner.collect(timeout_seconds=1)

    message = str(captured.value)
    assert "process_exit_code=0" in message
    assert "runner transport output" in message
    assert '"exit_code":1' in message
    assert "nonzero response must preserve" in message


@pytest.mark.asyncio
async def test_failure_contract_probe_requires_auditable_missing_id_result() -> None:
    class ProbeRunner:
        def __init__(self, *, valid: bool) -> None:
            self.valid = valid
            self.requested: tuple[str, ...] = ()

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float | None = None,
        ) -> RunnerResponse:
            assert timeout_seconds == 5
            self.requested = test_ids
            return RunnerResponse(
                action="run",
                exit_code=4 if self.valid else 0,
                results=(
                    RunnerTestPointExecution(
                        id=test_ids[0],
                        outcome="error" if self.valid else "passed",
                    ),
                ),
                output="missing test point" if self.valid else "",
            )

    collected = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(
            CollectedTestPoint(
                id="__featurebench_missing_test_point__",
                path="tests/test_real.py",
            ),
        ),
    )
    valid = ProbeRunner(valid=True)

    response = await probe_runner_failure_contract(
        cast(EnvironmentRunner, valid),
        collected,
        timeout_seconds=5,
    )

    assert valid.requested == ("__featurebench_missing_test_point___",)
    assert response.exit_code == 4
    with pytest.raises(ContractError, match="failure-path probe"):
        await probe_runner_failure_contract(
            cast(EnvironmentRunner, ProbeRunner(valid=False)),
            collected,
            timeout_seconds=5,
        )


@pytest.mark.asyncio
async def test_large_request_probe_exceeds_arg_max_and_requires_conservative_results() -> None:
    class ProbeRunner:
        def __init__(self, *, valid: bool) -> None:
            self.valid = valid
            self.requested: tuple[str, ...] = ()
            self.timeout_seconds: float | None = None

        async def run(
            self,
            test_ids: tuple[str, ...],
            *,
            timeout_seconds: float | None = None,
        ) -> RunnerResponse:
            self.requested = test_ids
            self.timeout_seconds = timeout_seconds
            return RunnerResponse(
                action="run",
                exit_code=4 if self.valid else 0,
                results=tuple(
                    RunnerTestPointExecution(id=test_id, outcome="error") for test_id in test_ids
                ),
                output="pytest could not find the synthetic source" if self.valid else "",
            )

    collected = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(
            CollectedTestPoint(
                id="tests/test_real.py::test_real#0",
                path="__featurebench_transport_probe__.py",
            ),
        ),
    )
    valid = ProbeRunner(valid=True)

    response = await probe_runner_large_request_transport(
        cast(EnvironmentRunner, valid),
        collected,
        timeout_seconds=120,
    )

    serialized = json.dumps({"action": "run", "test_ids": valid.requested})
    assert len(serialized.encode()) >= 3 * 1024 * 1024
    assert len(valid.requested) == 8192
    probe_path = valid.requested[0].partition("::")[0]
    assert probe_path.startswith("__featurebench_transport_probe_")
    assert probe_path.endswith(".py")
    assert probe_path not in {test.path for test in collected.collected}
    assert all(test_id.startswith(probe_path + "::") for test_id in valid.requested)
    assert valid.timeout_seconds == 30
    assert response.exit_code == 4

    with pytest.raises(ContractError, match="did not fail closed"):
        await probe_runner_large_request_transport(
            cast(EnvironmentRunner, ProbeRunner(valid=False)),
            collected,
            timeout_seconds=5,
        )


@pytest.mark.asyncio
async def test_collect_order_is_canonicalized_at_the_host_boundary() -> None:
    runner = EnvironmentRunner(
        cast(ContainerSession, FakeUnorderedCollectionSession(process_exit_code=0))
    )
    response = await runner.collect(timeout_seconds=1)
    assert [item.id for item in response.collected] == ["test a", "test b"]


@pytest.mark.asyncio
async def test_runner_uses_collect_limit_and_requires_adapter_run_timeout() -> None:
    session = FakeActionResponseSession()
    runner = EnvironmentRunner(cast(ContainerSession, session))
    await runner.collect()
    assert session.last_timeout_seconds == 11
    with pytest.raises(ContractError, match="requires an explicit"):
        await runner.run(("test-id",))
    await runner.run(("test-id",), timeout_seconds=22)
    assert session.last_timeout_seconds == 22
    assert session.last_max_bytes == 50_000_000


@pytest.mark.parametrize(
    "path",
    (
        "/testbed/tests/test_api.py",
        "../tests/test_api.py",
        ".",
        ".git/config",
        "nested/.git/index",
        "tests/test_api.py\nforged",
    ),
)
def test_collected_test_paths_reject_unsafe_locations(path: str) -> None:
    with pytest.raises(ValueError):
        CollectedTestPoint(id="test_api", path=path)


def test_collected_test_paths_are_canonical_repository_relative_paths() -> None:
    point = CollectedTestPoint(id="test_api", path="tests/./unit/test_api.py")
    assert point.path == "tests/unit/test_api.py"


def test_collection_stability_compares_framework_ids_and_paths() -> None:
    baseline = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(CollectedTestPoint(id="test one", path="test.js"),),
    )
    assert_stable_collection(baseline, baseline)

    reordered = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(
            CollectedTestPoint(id="test two", path="two.js"),
            CollectedTestPoint(id="test one", path="test.js"),
        ),
    )
    reordered_baseline = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=tuple(reversed(reordered.collected)),
    )
    assert_stable_collection(reordered_baseline, reordered)

    changed = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(CollectedTestPoint(id="test two", path="test.js"),),
    )
    with pytest.raises(ContractError, match="missing_ids"):
        assert_stable_collection(baseline, changed)


def test_collection_stability_reports_repeat_command_failure() -> None:
    baseline = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(CollectedTestPoint(id="test one", path="test.js"),),
    )
    failed = RunnerResponse(
        action="collect",
        exit_code=2,
        errors=("collector failed",),
        output="collector stderr",
    )
    with pytest.raises(ContractError, match="repeated_exit_code=2"):
        assert_stable_collection(baseline, failed)


def test_run_reports_define_verified_inventory() -> None:
    collected = RunnerResponse(
        action="collect",
        exit_code=0,
        collected=(
            CollectedTestPoint(id="test one", path="test.js"),
            CollectedTestPoint(id="test two", path="test.js"),
        ),
    )
    first_run = RunnerResponse(
        action="run",
        exit_code=1,
        results=(
            RunnerTestPointExecution(id="test one", outcome="passed"),
            RunnerTestPointExecution(id="test two", outcome="failed"),
        ),
        output="one selected test failed",
    )
    assert verified_run_test_points(collected, (first_run,)) == (collected.collected[0],)

    wrong_action = RunnerResponse(
        action="collect",
        exit_code=0,
    )
    with pytest.raises(ValueError, match="one collect response and run responses"):
        verified_run_test_points(collected, (wrong_action,))

    fabricated = RunnerResponse(
        action="run",
        exit_code=0,
        results=(RunnerTestPointExecution(id="invented", outcome="passed"),),
    )
    with pytest.raises(ContractError, match="absent from stable collection"):
        verified_run_test_points(collected, (fabricated,))
