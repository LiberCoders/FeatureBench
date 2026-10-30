from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.providers import Provider
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
)

from codeworld.adapters.catalog import default_adapter_catalog
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.agents.compaction import (
    SUMMARY_PREFIX,
    RecoveringTieredCompaction,
    ReplacingSummarizingCompaction,
)
from codeworld.agents.factory import (
    AgentFactory,
    build_compaction_capability,
    configured_agent_model,
    configured_retries,
    container_file_key,
    is_transient_model_error,
    model_retry_delay,
    resumable_message_history,
    unlimited_usage_limits,
)
from codeworld.agents.feedback import from_validation_error, render_feedback
from codeworld.config import (
    AgentFactorySettings,
    AgentSettings,
    BaseImageSettings,
    EnvironmentSettings,
    FactorySettings,
)
from codeworld.environment.builder import validate_base_image_recipes
from codeworld.environment.planning import EnvironmentAnalysis
from codeworld.errors import AgentModelError, ConfigurationError, ContractError
from codeworld.events import Event
from codeworld.runtime.container import ContainerSession
from codeworld.state import JobRecord, RunLock, StateStore


def test_complete_yaml_config_and_dotenv_are_host_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FEATUREBENCH_TEST_SECRET=from-dotenv\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"workspace: {tmp_path / 'work'}\nenv_file: {env_file}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FEATUREBENCH_TEST_SECRET", raising=False)
    settings = FactorySettings.from_files((config,))
    assert settings.load_host_environment()
    assert os.environ["FEATUREBENCH_TEST_SECRET"] == "from-dotenv"
    assert settings.initialize_workspace().is_dir()


def test_yaml_rejects_stale_keys(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("legacy_frontend: true\n", encoding="utf-8")
    with pytest.raises(Exception, match="legacy_frontend"):
        FactorySettings.from_files((config,))


def test_agent_is_time_bounded_without_usage_budgets() -> None:
    settings = AgentSettings.model_validate(
        {
            "model": "test",
            "analysis_run_timeout_seconds": 30,
            "artifact_run_timeout_seconds": 80,
            "request_timeout_seconds": 30,
        }
    )
    assert settings.analysis_run_timeout_seconds == 30
    assert settings.artifact_run_timeout_seconds == 80
    assert settings.request_timeout_seconds == 30
    assert "max_requests" not in AgentSettings.model_fields
    assert "max_tokens" not in AgentSettings.model_fields
    assert settings.tool_retries == 10
    assert settings.output_retries == 10
    assert settings.transient_request_retries == 10
    assert settings.provider.tls_verify is True
    limits = unlimited_usage_limits()
    assert limits.request_limit is None
    assert limits.tool_calls_limit is None
    assert limits.input_tokens_limit is None
    assert limits.output_tokens_limit is None
    assert limits.total_tokens_limit is None
    retries = configured_retries(settings)
    assert retries.get("tools") == 10
    assert retries.get("output") == 10
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        AgentSettings.model_validate({"model": "test", "max_tokens": 1})
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        AgentSettings.model_validate({"model": "test", "default_run_timeout_seconds": 90})


@pytest.mark.asyncio
async def test_agent_factories_share_an_optional_run_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    active = 0
    maximum_active = 0

    async def run_configured(self: AgentFactory[AgentFactorySettings], **options: object) -> Output:
        nonlocal active, maximum_active
        del self, options
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return Output(value="ok")

    monkeypatch.setattr(AgentFactory, "_run_structured_configured", run_configured)
    semaphore = asyncio.Semaphore(2)
    factories: tuple[AgentFactory[AgentFactorySettings], ...] = tuple(
        AgentFactory(AgentFactorySettings(), run_semaphore=semaphore) for _ in range(3)
    )

    results = await asyncio.gather(
        *(
            factory.run_structured(
                session=cast("ContainerSession", object()),
                prompt="prompt",
                instructions="instructions",
                output_type=Output,
                run_timeout_seconds=1,
            )
            for factory in (*factories, *factories)
        )
    )

    assert results == [Output(value="ok")] * 6
    assert maximum_active == 2
    assert active == 0


@pytest.mark.asyncio
async def test_agent_run_capacity_is_released_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    first_entered = asyncio.Event()
    calls = 0

    async def run_configured(self: AgentFactory[AgentFactorySettings], **options: object) -> Output:
        nonlocal calls
        del self, options
        calls += 1
        if calls == 1:
            first_entered.set()
            await asyncio.Event().wait()
        return Output(value="ok")

    monkeypatch.setattr(AgentFactory, "_run_structured_configured", run_configured)
    factory = AgentFactory(AgentFactorySettings(), run_semaphore=asyncio.Semaphore(1))
    first = asyncio.create_task(
        factory.run_structured(
            session=cast("ContainerSession", object()),
            prompt="first",
            instructions="instructions",
            output_type=Output,
            run_timeout_seconds=1,
        )
    )
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = await asyncio.wait_for(
        factory.run_structured(
            session=cast("ContainerSession", object()),
            prompt="second",
            instructions="instructions",
            output_type=Output,
            run_timeout_seconds=1,
        ),
        timeout=1,
    )

    assert second == Output(value="ok")


@pytest.mark.asyncio
async def test_agent_provider_disables_tls_for_any_httpx_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *, verify: bool) -> None:
            observed["verify"] = verify

        async def __aenter__(self) -> FakeAsyncClient:
            observed["entered"] = True
            return self

        async def __aexit__(self, *args: object) -> None:
            del args
            observed["closed"] = True

    class FakeHttpProvider:
        def __init__(self, *, http_client: object) -> None:
            observed["http_client"] = http_client

        def _set_http_client(self, http_client: object) -> None:
            observed["replacement_client"] = http_client

    resolved_model = object()

    def fake_infer_model(model_name: str, *, provider_factory: object) -> object:
        observed["model_name"] = model_name
        observed["provider"] = cast("object", provider_factory)("anthropic")  # type: ignore[operator]
        return resolved_model

    def fake_infer_provider_class(_: str) -> type[FakeHttpProvider]:
        return FakeHttpProvider

    monkeypatch.setattr("codeworld.agents.factory.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "codeworld.agents.factory.infer_provider_class",
        fake_infer_provider_class,
    )
    monkeypatch.setattr("codeworld.agents.factory.infer_model", fake_infer_model)
    settings = AgentSettings.model_validate(
        {
            "model": "anthropic:claude-relay",
            "provider": {"tls_verify": False},
        }
    )

    async with configured_agent_model(settings) as model:
        assert model is resolved_model
        assert observed["entered"] is True
        assert observed.get("closed") is None

    assert observed["verify"] is False
    assert observed["model_name"] == "anthropic:claude-relay"
    assert observed["closed"] is True


@pytest.mark.asyncio
async def test_agent_tls_verification_cannot_silently_disable_for_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonHttpProvider:
        _set_http_client = Provider._set_http_client  # pyright: ignore[reportPrivateUsage]

    def fake_infer_provider_class(_: str) -> type[NonHttpProvider]:
        return NonHttpProvider

    monkeypatch.setattr(
        "codeworld.agents.factory.infer_provider_class",
        fake_infer_provider_class,
    )
    settings = AgentSettings.model_validate(
        {
            "model": "custom:model",
            "provider": {"tls_verify": False},
        }
    )
    with pytest.raises(ConfigurationError, match="does not expose an active HTTPX client"):
        async with configured_agent_model(settings):
            pass


def test_featurebench_agent_roles_inherit_provider_tls_policy() -> None:
    settings = FeatureBenchSettings.model_validate({"agent": {"provider": {"tls_verify": False}}})
    for role in (
        settings.agent.top,
        settings.agent.repair,
        settings.agent.issue_author,
    ):
        assert settings.agent.factory_settings(role).provider.tls_verify is False


def test_agent_retries_only_transient_provider_failures_until_run_timeout() -> None:
    import httpx
    from openai import APIError
    from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

    server_error = ModelHTTPError(502, "relay-model")
    throttled = ModelHTTPError(
        429,
        "relay-model",
        headers={"retry-after": "7"},
    )
    bad_request = ModelHTTPError(400, "relay-model")
    gateway_bad_request = ModelHTTPError(
        400,
        "relay-model",
        body=(
            "<html><head><title>400 Bad Request</title></head>"
            "<body><center>nginx</center><script src='cloudflare'></script></body></html>"
        ),
    )
    transport_error = ModelAPIError("relay-model", "connection reset")
    stream_error = APIError(
        "stream_read_error",
        request=httpx.Request("POST", "https://relay.invalid/v1/responses"),
        body=None,
    )
    incomplete_stream = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body",
        request=httpx.Request("POST", "https://relay.invalid/v1/responses"),
    )
    provider_namespace: dict[str, object] = {"__name__": "pydantic_ai.models.fake"}
    exec(
        "def fail():\n    raise TypeError(\"'NoneType' object is not iterable\")",
        provider_namespace,
    )
    empty_provider_response: TypeError | None = None
    try:
        cast("object", provider_namespace["fail"])()  # type: ignore[operator]
    except TypeError as exc:
        empty_provider_response = exc
    local_type_error = TypeError("'NoneType' object is not iterable")
    unanswered_protocol_error = UnexpectedModelBehavior("provider response was incomplete")
    completed_protocol_error = UnexpectedModelBehavior("structured output retries exhausted")

    assert is_transient_model_error(server_error)
    assert is_transient_model_error(throttled)
    assert is_transient_model_error(transport_error)
    assert is_transient_model_error(stream_error)
    assert is_transient_model_error(incomplete_stream)
    assert empty_provider_response is not None
    assert is_transient_model_error(
        empty_provider_response,
        request_unanswered=True,
    )
    assert not is_transient_model_error(empty_provider_response)
    assert not is_transient_model_error(local_type_error)
    assert is_transient_model_error(
        unanswered_protocol_error,
        request_unanswered=True,
    )
    assert not is_transient_model_error(completed_protocol_error)
    assert not is_transient_model_error(bad_request)
    assert is_transient_model_error(gateway_bad_request)
    assert model_retry_delay(throttled, 1) == 7
    assert model_retry_delay(server_error, 1) == 1
    assert model_retry_delay(server_error, 8) == 30


def test_interrupted_stream_replays_only_the_unanswered_model_request() -> None:
    request = ModelRequest(parts=[UserPromptPart("build the environment")])
    interrupted = ModelResponse(parts=[TextPart("partial")], state="interrupted")
    assert resumable_message_history([request, interrupted]) == [request]
    assert resumable_message_history([request, ModelResponse(parts=[TextPart("complete")])]) is None


def test_environment_requires_positive_test_point_counts() -> None:
    assert EnvironmentSettings().collect_timeout_seconds == 120
    assert EnvironmentSettings().min_verified_test_points == 10
    assert EnvironmentSettings().baseline_group_max_test_points == 100
    assert EnvironmentSettings().baseline_total_timeout_seconds == 3600
    assert EnvironmentSettings().total_timeout_seconds == 7_200
    assert EnvironmentSettings().quickcheck_test_points == 10
    assert EnvironmentSettings().quickcheck_min_pass_rate == 0.5
    with pytest.raises(Exception, match="greater than or equal to 1"):
        EnvironmentSettings(min_verified_test_points=0)
    with pytest.raises(Exception, match="greater than or equal to 1"):
        EnvironmentSettings(baseline_group_max_test_points=0)
    with pytest.raises(Exception, match="greater than 0"):
        EnvironmentSettings(baseline_total_timeout_seconds=0)
    with pytest.raises(Exception, match="greater than 0"):
        EnvironmentSettings(total_timeout_seconds=0)
    with pytest.raises(Exception, match="greater than or equal to 1"):
        EnvironmentSettings(quickcheck_test_points=0)
    with pytest.raises(Exception, match="greater than 0"):
        EnvironmentSettings(quickcheck_min_pass_rate=0)
    with pytest.raises(Exception, match="smoke_timeout_seconds"):
        EnvironmentSettings.model_validate({"smoke_timeout_seconds": 120})
    with pytest.raises(Exception, match="runner_response_max_bytes"):
        EnvironmentSettings.model_validate({"runner_response_max_bytes": 50_000_000})


def test_agent_uses_harness_tiered_compaction() -> None:
    settings = AgentSettings(model="test").compaction
    compaction = build_compaction_capability(settings)
    assert compaction.target_tokens == settings.target_tokens
    assert tuple(type(tier) for tier in compaction.tiers) == (
        ClampOversizedMessages,
        DeduplicateFileReads,
        ClearToolResults,
        ReplacingSummarizingCompaction,
    )
    assert (
        container_file_key(
            ToolCallPart(
                tool_name="read_file",
                args={"path": "pyproject.toml"},
                tool_call_id="read-1",
            )
        )
        == "pyproject.toml"
    )
    assert (
        container_file_key(ToolCallPart(tool_name="run_command", args={}, tool_call_id="shell-1"))
        is None
    )


@pytest.mark.asyncio
async def test_provider_overflow_forces_one_compaction_below_the_normal_trigger() -> None:
    class RecordingTier:
        calls = 0

        async def compact(
            self,
            messages: list[ModelMessage],
            ctx: RunContext[object],
        ) -> list[ModelMessage]:
            del ctx
            self.calls += 1
            return messages

    tier = RecordingTier()
    compaction = RecoveringTieredCompaction(
        tiers=(tier,),  # type: ignore[arg-type]
        target_tokens=240_000,
    )
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="x" * 600_000)])]
    request_context = cast("object", type("RequestContext", (), {"messages": messages})())
    context = cast(RunContext[object], object())

    await compaction.before_model_request(context, request_context)  # type: ignore[arg-type]
    assert tier.calls == 0
    compaction.request_emergency_compaction()
    await compaction.before_model_request(context, request_context)  # type: ignore[arg-type]
    assert tier.calls == 1
    assert compaction.target_tokens == 240_000
    compaction.request_emergency_compaction()
    await compaction.before_model_request(context, request_context)  # type: ignore[arg-type]
    assert tier.calls == 2
    assert compaction.target_tokens == 240_000


@pytest.mark.asyncio
async def test_provider_overflow_clears_tool_results_retained_by_normal_policy() -> None:
    compaction = build_compaction_capability(AgentSettings(model="test").compaction)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="classify the feature graph")])
    ]
    for index in range(5):
        tool_call_id = f"batch-{index}"
        messages.extend(
            (
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_next_top_batch",
                            args={"batch": index},
                            tool_call_id=tool_call_id,
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="get_next_top_batch",
                            content=f"batch-{index}:" + "x" * 80_000,
                            tool_call_id=tool_call_id,
                        )
                    ]
                ),
            )
        )
    request_context = cast("object", type("RequestContext", (), {"messages": messages})())
    context = cast(RunContext[object], object())

    compaction.request_emergency_compaction()
    await compaction.before_model_request(context, request_context)  # type: ignore[arg-type]

    compacted = cast("list[ModelMessage]", request_context.messages)  # type: ignore[attr-defined]
    tool_results = [
        part.content
        for message in compacted
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert tool_results.count("[tool result cleared]") == 3
    assert sum(isinstance(value, str) and len(value) > 10_000 for value in tool_results) == 2


@pytest.mark.asyncio
async def test_provider_overflow_resume_uses_a_fresh_physical_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    captured: list[ModelMessage] | None = None
    run_ids: list[str | None] = []
    conversation_ids: list[str | None] = []
    histories: list[list[ModelMessage] | None] = []
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def progress(kind: str, data: dict[str, object]) -> None:
        progress_events.append((kind, data))

    @contextmanager
    def fake_capture_run_messages():
        nonlocal captured
        messages: list[ModelMessage] = []
        captured = messages
        yield messages

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, user_prompt: str | None, **kwargs: object) -> object:
            run_ids.append(cast("str | None", kwargs.get("run_id")))
            conversation_ids.append(cast("str | None", kwargs.get("conversation_id")))
            histories.append(cast("list[ModelMessage] | None", kwargs.get("message_history")))
            if len(run_ids) == 1:
                assert user_prompt == "repair the mask"
                assert captured is not None
                captured.extend(
                    (
                        ModelRequest(parts=[UserPromptPart(content=user_prompt)]),
                        ModelResponse(parts=[TextPart(content="partial")], state="interrupted"),
                    )
                )
                raise RuntimeError("maximum context window exceeded")
            assert user_prompt is None
            return SimpleNamespace(output=Output(value="ok"))

    monkeypatch.setattr("codeworld.agents.factory.Agent", FakeAgent)
    monkeypatch.setattr("codeworld.agents.factory.capture_run_messages", fake_capture_run_messages)
    settings = AgentSettings.model_validate(
        {"model": "test", "metrics": {"enabled": False}, "trajectory": {"enabled": False}}
    )
    result = await AgentFactory(settings).run_structured(
        session=cast(
            "ContainerSession", object()
        ),  # container tools are disabled for this model-only test
        prompt="repair the mask",
        instructions="return Output",
        output_type=Output,
        container_tools=False,
        progress=progress,
        run_timeout_seconds=5,
        agent_name="mask_repair",
    )

    assert result == Output(value="ok")
    assert run_ids[0] is not None and run_ids[0].startswith("mask_repair-")
    assert run_ids[1] is None
    assert conversation_ids == [run_ids[0], run_ids[0]]
    assert histories[0] is None
    assert histories[1] is not None
    assert isinstance(histories[1][-1], ModelRequest)
    started = next(data for kind, data in progress_events if kind == "agent_run_started")
    assert started["model"] == "test"
    metrics = next(data for kind, data in progress_events if kind == "agent_run_metrics")
    assert metrics["model"] == "test"
    assert metrics["input_tokens"] == 0
    assert metrics["output_tokens"] == 0
    assert metrics["cache_read_tokens"] == 0
    assert metrics["cache_write_tokens"] == 0


@pytest.mark.asyncio
async def test_unanswered_model_protocol_failure_replays_the_physical_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    captured: list[ModelMessage] | None = None
    prompts: list[str | None] = []
    histories: list[list[ModelMessage] | None] = []
    progress_events: list[tuple[str, dict[str, object]]] = []

    @contextmanager
    def fake_capture_run_messages():
        nonlocal captured
        messages: list[ModelMessage] = []
        captured = messages
        yield messages

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, user_prompt: str | None, **kwargs: object) -> object:
            prompts.append(user_prompt)
            histories.append(cast("list[ModelMessage] | None", kwargs.get("message_history")))
            if len(prompts) == 1:
                assert captured is not None
                captured.append(ModelRequest(parts=[UserPromptPart(content=user_prompt or "")]))
                raise UnexpectedModelBehavior("provider ended before returning a response")
            return SimpleNamespace(output=Output(value="ok"))

    async def no_delay(_: float) -> None:
        return None

    async def progress(kind: str, data: dict[str, object]) -> None:
        progress_events.append((kind, data))

    monkeypatch.setattr("codeworld.agents.factory.Agent", FakeAgent)
    monkeypatch.setattr("codeworld.agents.factory.capture_run_messages", fake_capture_run_messages)
    monkeypatch.setattr("codeworld.agents.factory.asyncio.sleep", no_delay)
    settings = AgentSettings.model_validate(
        {"model": "test", "metrics": {"enabled": False}, "trajectory": {"enabled": False}}
    )

    result = await AgentFactory(settings).run_structured(
        session=cast("ContainerSession", object()),
        prompt="review the Runner",
        instructions="return Output",
        output_type=Output,
        container_tools=False,
        progress=progress,
        run_timeout_seconds=5,
        agent_name="environment_runner_review",
    )

    assert result == Output(value="ok")
    assert prompts == ["review the Runner", None]
    assert histories[0] is None
    assert histories[1] is not None
    assert isinstance(histories[1][-1], ModelRequest)
    retries = [data for kind, data in progress_events if kind == "model_request_retry"]
    assert len(retries) == 1
    assert retries[0]["resumed"] is True


@pytest.mark.asyncio
async def test_transient_model_request_retries_stop_at_the_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    captured: list[ModelMessage] | None = None
    calls = 0
    progress_events: list[tuple[str, dict[str, object]]] = []

    @contextmanager
    def fake_capture_run_messages():
        nonlocal captured
        messages: list[ModelMessage] = []
        captured = messages
        yield messages

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, user_prompt: str | None, **kwargs: object) -> object:
            nonlocal calls
            del kwargs
            calls += 1
            assert captured is not None
            captured.append(ModelRequest(parts=[UserPromptPart(content=user_prompt or "resume")]))
            raise UnexpectedModelBehavior("provider closed the response stream")

    async def no_delay(_: float) -> None:
        return None

    async def progress(kind: str, data: dict[str, object]) -> None:
        progress_events.append((kind, data))

    monkeypatch.setattr("codeworld.agents.factory.Agent", FakeAgent)
    monkeypatch.setattr("codeworld.agents.factory.capture_run_messages", fake_capture_run_messages)
    monkeypatch.setattr("codeworld.agents.factory.asyncio.sleep", no_delay)
    settings = AgentSettings.model_validate(
        {
            "model": "test",
            "transient_request_retries": 2,
            "metrics": {"enabled": False},
            "trajectory": {"enabled": False},
        }
    )

    with pytest.raises(AgentModelError, match="retry limit exhausted after 2 retries") as caught:
        await AgentFactory(settings).run_structured(
            session=cast("ContainerSession", object()),
            prompt="review the Runner",
            instructions="return Output",
            output_type=Output,
            container_tools=False,
            progress=progress,
            run_timeout_seconds=5,
            agent_name="environment_runner_review",
        )

    assert calls == 3
    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)
    retries = [data for kind, data in progress_events if kind == "model_request_retry"]
    assert [item["retry"] for item in retries] == [1, 2]
    exhausted = [data for kind, data in progress_events if kind == "model_request_retry_exhausted"]
    assert exhausted == [
        {
            "retries": 2,
            "maximum_retries": 2,
            "status_code": None,
            "error_type": "UnexpectedModelBehavior",
            "error": "provider closed the response stream",
            "resumed": True,
        }
    ]


@pytest.mark.asyncio
async def test_completed_model_failure_is_normalized_at_the_agent_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    captured: list[ModelMessage] | None = None

    @contextmanager
    def fake_capture_run_messages():
        nonlocal captured
        messages: list[ModelMessage] = []
        captured = messages
        yield messages

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, user_prompt: str | None, **kwargs: object) -> object:
            del kwargs
            assert captured is not None
            captured.extend(
                (
                    ModelRequest(parts=[UserPromptPart(content=user_prompt or "")]),
                    ModelResponse(parts=[TextPart(content="invalid structured output")]),
                )
            )
            raise UnexpectedModelBehavior("structured output retries exhausted")

    monkeypatch.setattr("codeworld.agents.factory.Agent", FakeAgent)
    monkeypatch.setattr("codeworld.agents.factory.capture_run_messages", fake_capture_run_messages)
    settings = AgentSettings.model_validate(
        {"model": "test", "metrics": {"enabled": False}, "trajectory": {"enabled": False}}
    )

    with pytest.raises(
        AgentModelError, match="environment_runner_review Agent model run failed"
    ) as caught:
        await AgentFactory(settings).run_structured(
            session=cast("ContainerSession", object()),
            prompt="review the Runner",
            instructions="return Output",
            output_type=Output,
            container_tools=False,
            run_timeout_seconds=5,
            agent_name="environment_runner_review",
        )

    assert isinstance(caught.value.__cause__, UnexpectedModelBehavior)


def test_agent_feedback_preserves_pydantic_codes_and_field_paths() -> None:
    with pytest.raises(ValidationError) as caught:
        EnvironmentAnalysis.model_validate({})
    feedback = from_validation_error(caught.value)
    assert any(issue.path == ("test_frameworks",) for issue in feedback.issues)
    assert all(issue.code.startswith("schema.") for issue in feedback.issues)
    rendered = render_feedback(feedback)
    assert '"protocol"' not in rendered
    assert '"issues":[' in rendered


@pytest.mark.asyncio
async def test_candidate_validation_is_outside_agent_authoring_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Output(BaseModel):
        value: str

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.validator: Callable[[Output], Awaitable[Output]] | None = None

        def output_validator(
            self, validator: Callable[[Output], Awaitable[Output]]
        ) -> Callable[[Output], Awaitable[Output]]:
            self.validator = validator
            return validator

        async def run(self, user_prompt: str | None, **kwargs: object) -> object:
            del user_prompt, kwargs
            await asyncio.sleep(0.05)
            output = Output(value="ok")
            assert self.validator is not None
            await self.validator(output)
            await asyncio.sleep(0.05)
            return SimpleNamespace(output=output)

    async def validate_candidate(_: Output) -> None:
        # Longer than the complete authoring work and longer than the remaining
        # authoring budget. It must be bounded by its own checks, not this deadline.
        await asyncio.sleep(0.2)

    monkeypatch.setattr("codeworld.agents.factory.Agent", FakeAgent)
    settings = AgentSettings.model_validate(
        {"model": "test", "metrics": {"enabled": False}, "trajectory": {"enabled": False}}
    )

    result = await AgentFactory(settings).run_structured(
        session=cast("ContainerSession", object()),
        prompt="author candidate",
        instructions="return Output",
        output_type=Output,
        container_tools=False,
        candidate_validator=validate_candidate,
        run_timeout_seconds=0.15,
        agent_name="candidate_author",
    )

    assert result == Output(value="ok")


@pytest.mark.asyncio
async def test_incremental_compaction_keeps_exactly_one_latest_summary() -> None:
    class DeterministicSummary(ReplacingSummarizingCompaction):
        calls: int = 0

        async def _summarize(
            self,
            messages: list[ModelMessage],
            ctx: RunContext[object],
            *,
            previous_summary: str | None = None,
        ) -> str:
            del messages, ctx, previous_summary
            self.calls += 1
            return f"summary-{self.calls}"

    compaction = DeterministicSummary(max_messages=1, keep_messages=1)
    context = cast(RunContext[object], object())
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="stable instructions"),
                SystemPromptPart(content=f"{SUMMARY_PREFIX}old summary"),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content="original task")]),
        ModelResponse(parts=[TextPart(content="old work")]),
        ModelRequest(parts=[UserPromptPart(content="recent turn")]),
    ]
    once = await compaction.compact(history, context)
    twice = await compaction.compact(
        [
            *once,
            ModelResponse(parts=[TextPart(content="more work")]),
            ModelRequest(parts=[UserPromptPart(content="latest turn")]),
        ],
        context,
    )
    summaries = [
        part.content
        for message in twice
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.content.startswith(SUMMARY_PREFIX)
    ]
    assert summaries == [f"{SUMMARY_PREFIX}summary-2"]


def test_semantic_yaml_files_compose_without_ordered_overrides(tmp_path: Path) -> None:
    host = tmp_path / "host.yaml"
    build = tmp_path / "build.yaml"
    duplicate = tmp_path / "duplicate.yaml"
    host.write_text(f"workspace: {tmp_path / 'work'}\n", encoding="utf-8")
    build.write_text(
        "concurrency: 2\nenvironment:\n  allow_gpu_repositories: false\nagent:\n  model: test\n",
        encoding="utf-8",
    )
    duplicate.write_text("environment:\n  workdir: /testbed\n", encoding="utf-8")
    settings = FactorySettings.from_files((host, build))
    assert settings.concurrency == 2
    assert settings.agent.model == "test"
    with pytest.raises(Exception, match="appears in both"):
        FactorySettings.from_files((build, duplicate))


def test_yaml_duplicate_keys_are_never_silently_replaced(tmp_path: Path) -> None:
    config = tmp_path / "duplicate-key.yaml"
    config.write_text("concurrency: 2\nconcurrency: 3\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        FactorySettings.from_files((config,))


def test_adapter_owns_its_config_schema(tmp_path: Path) -> None:
    assert "featurebench" not in FactorySettings.model_fields
    config = tmp_path / "featurebench.yaml"
    config.write_text("featurebench:\n  unknown_option: true\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="featurebench"):
        FactorySettings.from_files((config,))
    with pytest.raises(ConfigurationError, match="unknown_option"):
        FactorySettings.from_files((config,), adapter_catalog=default_adapter_catalog())

    config.write_text("featurebench: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="environment_concurrency"):
        FactorySettings.from_files((config,), adapter_catalog=default_adapter_catalog())


def test_default_semantic_files_and_managed_base_recipes_are_complete() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "default"
    files = tuple(sorted(root.rglob("*.yaml")))
    assert {path.relative_to(root).as_posix() for path in files} == {
        "host.yaml",
        "discover.yaml",
        "build.yaml",
        "adapters/featurebench.yaml",
        "sources.yaml",
    }
    catalog = default_adapter_catalog()
    settings = FactorySettings.from_files(files, adapter_catalog=catalog)
    featurebench = FeatureBenchSettings.model_validate(settings.adapters["featurebench"])
    assert featurebench == FeatureBenchSettings()
    assert settings.workspace == Path(".featurebench")
    assert settings.concurrency == 4
    assert settings.refresh_per_second == 8
    assert settings.jsonl_fsync_interval == 1
    assert settings.env_file == Path(".env")
    assert settings.agent.model == "openai:gpt-5.6-luna"
    assert settings.agent.provider.tls_verify is True
    assert featurebench.maximum_tasks_per_environment is None
    assert featurebench.environment_concurrency == 8
    assert featurebench.maximum_p2p_test_points == 3000
    assert FeatureBenchSettings(maximum_p2p_test_points=None).maximum_p2p_test_points is None
    assert featurebench.maximum_selected_p2p_test_files == 64
    assert featurebench.maximum_f2p_ratio == 0.7
    assert featurebench.gold_patch_line_minimum == 200
    assert featurebench.gold_patch_line_preferred_minimum == 600
    assert featurebench.gold_patch_line_maximum == 1000
    assert featurebench.top_batch_size == 20
    assert featurebench.top_batch_concurrency == 1
    assert featurebench.seed_concurrency == 16
    assert featurebench.agent_concurrency == 4
    assert featurebench.trace_concurrency == 24
    assert featurebench.seed_ordering == "easy_first"
    assert featurebench.compatibility_timeout_seconds == 120
    assert featurebench.repair.strategy == "rollback"
    assert featurebench.repair.optimization_target == "first_accepted"
    assert featurebench.repair.maximum_validations == 10
    assert featurebench.repair.timeout_seconds == 3600
    assert featurebench.agent.top.model == "openai:gpt-5.6-terra"
    assert featurebench.agent.repair.model == "openai:gpt-5.6-terra"
    assert featurebench.agent.issue_author.model == "openai:gpt-5.6-terra"
    assert featurebench.agent.provider.tls_verify is True
    assert featurebench.agent.metrics.enabled is True
    assert featurebench.agent.trajectory.enabled is True
    assert featurebench.agent.trajectory.retention == "failures"
    assert featurebench.audit.retention == "failures"
    validate_base_image_recipes(settings.environment)
    assert {image.runtime_family for image in settings.environment.base_images} >= {
        "python",
        "node",
        "go",
        "rust",
        "jvm",
        "cuda",
    }
    conda_service = settings.sources.services["conda"]
    assert all("conda-forge" in endpoint.url for endpoint in conda_service.endpoints)
    assert settings.sources.services["github_git"].maximum_attempts == 3
    miniforge_service = settings.sources.services["miniforge"]
    assert all("miniforge" in endpoint.url.casefold() for endpoint in miniforge_service.endpoints)
    pytorch_cpu_service = settings.sources.services["pytorch_cpu"]
    assert pytorch_cpu_service.endpoints[0].url == ("https://mirrors.aliyun.com/pytorch-wheels/cpu")
    assert pytorch_cpu_service.endpoints[-1].canonical is True
    huggingface_service = settings.sources.services["huggingface"]
    assert huggingface_service.endpoints[0].url == "https://hf-mirror.com"
    assert huggingface_service.endpoints[-1].url == "https://huggingface.co"
    assert huggingface_service.endpoints[-1].canonical is True
    python_versions = tuple(
        image.runtime_version
        for image in settings.environment.base_images
        if image.runtime_family == "python"
    )
    assert python_versions == ("3.14", "3.13", "3.12", "3.11", "3.10")
    for version in python_versions:
        assert version is not None
        recipe = (
            Path(__file__).resolve().parents[2]
            / "codeworld"
            / "environment"
            / "assets"
            / "base_images"
            / f"python-{version}.Dockerfile"
        ).read_text(encoding="utf-8")
        assert "ARG MINIFORGE_DIST_URL=" in recipe
        assert "ARG CONDA_CHANNEL_URL=" in recipe
        assert "ARG UV_INSTALLER_GITHUB_BASE_URL=https://github.com" in recipe
        assert "ARG UV_VERSION=0.11.32" in recipe
        assert "ARG UV_CONCURRENT_DOWNLOADS=8" in recipe
        assert "ARG UV_CONCURRENT_INSTALLS=8" in recipe
        assert "https://astral.sh" not in recipe
        assert "/astral-sh/uv/releases/download/${UV_VERSION}/" in recipe
        miniforge_step = recipe.index("id=env-base-miniforge-installers")
        conda_step = recipe.index("id=env-base-conda-pkgs")
        uv_step = recipe.index("id=env-base-uv-archives")
        assert miniforge_step < conda_step < uv_step
        assert "conda install" not in recipe[miniforge_step:conda_step]
        assert "MINIFORGE_DIST_URL" not in recipe[conda_step:uv_step]
        assert "uv_target" not in recipe[conda_step:uv_step]
        assert "id=env-base-conda-pkgs" not in recipe[uv_step:]
        assert "id=env-base-uv-archives,target=/var/cache/uv-archives,sharing=locked" in recipe
        assert 'uv_cache_dir="/var/cache/uv-archives/${UV_VERSION}"' in recipe
        assert 'uv_download="${uv_archive}.download.$$"' in recipe
        assert 'tar -tzf "$uv_download"' in recipe
        assert 'tar -tzf "$uv_archive"' in recipe
        assert 'mv -f "$uv_download" "$uv_archive"' in recipe
        assert 'test -x "$uv_tmp/uv-${uv_target}/uv"' in recipe
        assert 'test -x "$uv_tmp/uv-${uv_target}/uvx"' in recipe
        assert "--override-channels" in recipe
        assert '--channel "$CONDA_CHANNEL_URL"' in recipe


def test_managed_base_recipe_must_match_its_configured_image() -> None:
    settings = EnvironmentSettings(
        base_images=(
            BaseImageSettings(
                name="python-3.12",
                image="python:3.12-slim-bookworm",
                languages=("python",),
                runtime_family="python",
                runtime_version="3.12",
                recipe="python-3.12",
            ),
        )
    )
    with pytest.raises(ConfigurationError, match=r"starts FROM 'ubuntu:24\.04'"):
        validate_base_image_recipes(settings)


@pytest.mark.asyncio
async def test_state_store_persists_structured_events(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    store.create_run("run", "test", {})
    await store.emit(Event(run_id="run", kind="started", stage="unit", message="hello"))
    store.upsert_job("run", "job", "unit", "running", {"value": 1})
    job = store.get_job("run", "job")
    assert job is not None
    assert isinstance(job, JobRecord)
    assert job.status == "running"
    assert job.payload == {"value": 1}
    store.finish_run("run")
    store.close()


def test_state_store_rejects_an_unknown_persisted_job_status(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    store = StateStore(path)
    store.create_run("run", "test", {})
    store.upsert_job("run", "job", "unit", "running", {})
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE jobs SET status = 'unknown' WHERE job_id = 'job'")

    store = StateStore(path)
    with pytest.raises(ContractError, match="invalid job checkpoint run/job"):
        store.get_job("run", "job")
    store.close()


def test_state_store_resume_refreshes_its_cached_resolved_config(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    store.create_run("run", "discover", {"value": 1})
    store.finish_run("run", error="interrupted")
    store.resume_run("run", "discover", {"value": 2})
    _, interrupted_revision = store.load_run_config("run", "discover")
    assert interrupted_revision == {"value": 2}
    store.finish_run("run")
    store.resume_run("run", "discover", {"value": 3})
    _, completed_revision = store.load_run_config("run", "discover")
    assert completed_revision == {"value": 3}
    store.close()


def test_run_lock_prevents_a_second_writer_and_can_be_reacquired(tmp_path: Path) -> None:
    path = tmp_path / "run" / "run.lock"
    with (
        RunLock(path),
        pytest.raises(Exception, match="already active"),
        RunLock(path),
    ):
        pass
    with RunLock(path):
        assert path.read_text(encoding="utf-8").startswith("pid=")
