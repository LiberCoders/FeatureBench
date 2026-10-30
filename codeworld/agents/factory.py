"""Focused host-side agents built with Pydantic AI and Harness capabilities."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar, cast
from uuid import uuid4

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries, ModelRetry, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import (
    AgentRunError,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider_class
from pydantic_ai.providers import gateway as pydantic_ai_gateway
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
)
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence

from codeworld.agents.compaction import (
    RecoveringTieredCompaction,
    ReplacingSummarizingCompaction,
)
from codeworld.agents.feedback import (
    AgentFeedback,
    AgentFeedbackError,
    FeedbackIssue,
    StructuredOutputFeedback,
    render_feedback,
)
from codeworld.agents.step_store import PrefixCompactingJsonStepStore
from codeworld.agents.telemetry import append_agent_metric, make_agent_metric
from codeworld.agents.workspace import ContainerWorkspace
from codeworld.config import AgentCompactionSettings, AgentFactorySettings
from codeworld.errors import (
    AgentContextWindowError,
    AgentModelError,
    AgentTimeoutError,
    ConfigurationError,
    ContractError,
)
from codeworld.runtime.base import ContainerSessionProtocol

OutputT = TypeVar("OutputT", bound=BaseModel)
AgentProgress = Callable[[str, dict[str, object]], Awaitable[None]]

_CONTEXT_WINDOW_MARKERS = (
    "context window",
    "context length",
    "maximum context",
    "too many input tokens",
)
_TRANSIENT_RELAY_MARKERS = ("not supported by any configured account in this group",)
_TRANSIENT_HTML_GATEWAY_MARKERS = (
    "<html",
    "<title>400 bad request</title>",
    "cloudflare",
    "nginx",
)
_MAX_CONTEXT_COMPACTION_ATTEMPTS = 3


def is_context_window_error(error: BaseException) -> bool:
    """Recognize permanent size failures even from loosely compatible relays."""
    message = str(error).lower()
    return any(marker in message for marker in _CONTEXT_WINDOW_MARKERS)


def _originates_in_model_provider(error: BaseException) -> bool:
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith(
            (
                "openai",
                "httpx",
                "httpcore",
                "pydantic_ai.models",
                "pydantic_ai.providers",
            )
        ):
            return True
        traceback = traceback.tb_next
    return False


def is_transient_model_error(
    error: BaseException,
    *,
    request_unanswered: bool = False,
) -> bool:
    """Identify provider failures for which replaying the physical request is safe."""
    if is_context_window_error(error):
        return False
    if any(marker in str(error).lower() for marker in _TRANSIENT_RELAY_MARKERS):
        return True
    if isinstance(error, UnexpectedModelBehavior):
        # Pydantic AI uses this broad exception for both response-protocol failures
        # and exhausted semantic output retries. Only the former leaves the most
        # recent ModelRequest unanswered and is therefore safe to replay.
        return request_unanswered
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, TypeError) and request_unanswered and _originates_in_model_provider(error):
        # A few provider integrations still leak parser TypeErrors instead of a
        # provider-neutral exception. Request state, not exception text, determines
        # whether replay is safe.
        return True
    if isinstance(error, ModelHTTPError):
        if error.status_code == 400 and isinstance(error.body, str):
            normalized_body = error.body.lower()
            # OpenAI-compatible relays sometimes return an intermediary's HTML
            # error page instead of the provider's structured client error.  It
            # carries no actionable request feedback and is safe to replay.
            if all(
                marker in normalized_body for marker in _TRANSIENT_HTML_GATEWAY_MARKERS[:2]
            ) and any(marker in normalized_body for marker in _TRANSIENT_HTML_GATEWAY_MARKERS[2:]):
                return True
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    if isinstance(error, ModelAPIError):
        return True
    # OpenAI-compatible providers may surface a mid-stream SDK error before Pydantic AI
    # can wrap it. Status-bearing client errors remain non-retryable unless transient.
    mro_names = {item.__name__ for item in type(error).__mro__}
    if "APIStatusError" in mro_names:
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (
            status_code in {408, 409, 429} or status_code >= 500
        )
    return "APIConnectionError" in mro_names or (
        "APIError" in mro_names and type(error).__module__.startswith("openai")
    )


def model_retry_delay(error: BaseException, attempt: int) -> float:
    """Honor provider Retry-After and otherwise use a bounded exponential delay."""
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, (int, float)):
        return min(max(float(retry_after), 1.0), 60.0)
    return min(2.0 ** max(0, attempt - 1), 30.0)


def resumable_message_history(messages: list[ModelMessage]) -> list[ModelMessage] | None:
    """Drop only an incomplete streamed response so Pydantic AI can replay its request."""
    recovered = list(messages)
    while (
        recovered
        and isinstance(recovered[-1], ModelResponse)
        and recovered[-1].state == "interrupted"
    ):
        recovered.pop()
    return recovered if recovered and isinstance(recovered[-1], ModelRequest) else None


def configured_retries(settings: AgentFactorySettings) -> AgentRetries:
    """Keep Pydantic AI's independent per-tool and global output retry semantics."""
    return {
        "tools": settings.tool_retries,
        "output": settings.output_retries,
    }


def unlimited_usage_limits() -> UsageLimits:
    """Override Pydantic AI's otherwise implicit 50-request default limit."""
    return UsageLimits(
        request_limit=None,
        tool_calls_limit=None,
        input_tokens_limit=None,
        output_tokens_limit=None,
        total_tokens_limit=None,
    )


def provider_with_http_client(
    provider_name: str,
    http_client: httpx.AsyncClient,
) -> Provider[Any]:
    """Instantiate a Pydantic AI provider through its declared HTTPX injection point."""
    if provider_name.startswith("gateway/"):
        upstream_provider = provider_name.removeprefix("gateway/")
        if pydantic_ai_gateway.normalize_gateway_provider(upstream_provider) == "bedrock":
            raise ConfigurationError(
                "agent.provider.tls_verify cannot configure gateway/bedrock because "
                "that provider uses the AWS SDK instead of HTTPX"
            )
        gateway_factory = cast(
            Callable[..., Provider[Any]],
            vars(pydantic_ai_gateway)["gateway_provider"],
        )
        return gateway_factory(upstream_provider, http_client=http_client)

    try:
        provider_class = infer_provider_class(provider_name)
    except ValueError as exc:
        raise ConfigurationError(f"unknown Agent model provider: {provider_name}") from exc
    supports_http_client = "http_client" in inspect.signature(provider_class).parameters
    replaces_http_client = inspect.getattr_static(
        provider_class, "_set_http_client"
    ) is not inspect.getattr_static(Provider, "_set_http_client")
    if not supports_http_client or not replaces_http_client:
        raise ConfigurationError(
            f"agent.provider.tls_verify cannot configure provider {provider_name!r}: "
            "its Pydantic AI integration does not expose an active HTTPX client"
        )
    constructor = cast(Callable[..., Provider[Any]], provider_class)
    return constructor(http_client=http_client)


@asynccontextmanager
async def configured_agent_model(
    settings: AgentFactorySettings,
) -> AsyncGenerator[Model | str, None]:
    """Resolve an Agent model and own any explicitly configured provider client."""
    model_name = settings.model
    if model_name is None:
        raise ConfigurationError(
            "an Agent is required but agent.model is unset; set it in the YAML configuration"
        )
    if settings.provider.tls_verify:
        yield model_name
        return

    provider_name, separator, _ = model_name.partition(":")
    if not separator:
        raise ConfigurationError(
            "agent.provider.tls_verify=false requires a provider:model identifier"
        )

    async with httpx.AsyncClient(verify=False) as http_client:
        provider = provider_with_http_client(provider_name, http_client)

        def provider_factory(_: str) -> Provider[Any]:
            return provider

        yield infer_model(model_name, provider_factory=provider_factory)


def container_file_key(call: ToolCallPart) -> str | None:
    """Identify repeat reads for Harness without guessing about unrelated tools."""
    if call.tool_name != "read_file":
        return None
    path = call.args_as_dict().get("path")
    return path if isinstance(path, str) else None


def build_compaction_capability(
    settings: AgentCompactionSettings,
) -> RecoveringTieredCompaction:
    """Build Harness's cheap-to-expensive context compaction policy."""
    return RecoveringTieredCompaction(
        tiers=(
            ClampOversizedMessages(
                max_part_tokens=settings.max_part_tokens,
                keep_head_chars=settings.keep_head_chars,
                keep_tail_chars=settings.keep_tail_chars,
            ),
            DeduplicateFileReads(file_key=container_file_key),
            ClearToolResults(
                max_tokens=1,
                keep_pairs=settings.keep_tool_pairs,
                min_clear_tokens=settings.min_clear_tokens,
            ),
            ReplacingSummarizingCompaction(
                max_messages=1,
                keep_messages=settings.keep_messages,
            ),
        ),
        target_tokens=settings.target_tokens,
    )


class AgentFactory[FactorySettingsT: AgentFactorySettings]:
    """Creates short-lived typed agents; model credentials remain on the host."""

    def __init__(
        self,
        settings: FactorySettingsT | None = None,
        *,
        run_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.settings: FactorySettingsT = (
            settings if settings is not None else cast(FactorySettingsT, AgentFactorySettings())
        )
        self.run_semaphore = run_semaphore

    async def run_structured(
        self,
        *,
        session: ContainerSessionProtocol,
        prompt: str,
        instructions: str,
        output_type: type[OutputT],
        writable: bool = False,
        container_tools: bool = True,
        planning: bool = False,
        extra_capabilities: Sequence[AbstractCapability[object]] = (),
        output_validator: Callable[[OutputT], Awaitable[None] | None] | None = None,
        candidate_validator: Callable[[OutputT], Awaitable[None] | None] | None = None,
        progress: AgentProgress | None = None,
        run_timeout_seconds: float,
        agent_name: str = "agent",
        metadata: dict[str, str] | None = None,
    ) -> OutputT:
        if self.run_semaphore is not None:
            async with self.run_semaphore:
                return await self._run_structured_configured(
                    session=session,
                    prompt=prompt,
                    instructions=instructions,
                    output_type=output_type,
                    writable=writable,
                    container_tools=container_tools,
                    planning=planning,
                    extra_capabilities=extra_capabilities,
                    output_validator=output_validator,
                    candidate_validator=candidate_validator,
                    progress=progress,
                    run_timeout_seconds=run_timeout_seconds,
                    agent_name=agent_name,
                    metadata=metadata,
                )
        return await self._run_structured_configured(
            session=session,
            prompt=prompt,
            instructions=instructions,
            output_type=output_type,
            writable=writable,
            container_tools=container_tools,
            planning=planning,
            extra_capabilities=extra_capabilities,
            output_validator=output_validator,
            candidate_validator=candidate_validator,
            progress=progress,
            run_timeout_seconds=run_timeout_seconds,
            agent_name=agent_name,
            metadata=metadata,
        )

    async def _run_structured_configured(
        self,
        *,
        session: ContainerSessionProtocol,
        prompt: str,
        instructions: str,
        output_type: type[OutputT],
        writable: bool,
        container_tools: bool,
        planning: bool,
        extra_capabilities: Sequence[AbstractCapability[object]],
        output_validator: Callable[[OutputT], Awaitable[None] | None] | None,
        candidate_validator: Callable[[OutputT], Awaitable[None] | None] | None,
        progress: AgentProgress | None,
        run_timeout_seconds: float,
        agent_name: str,
        metadata: dict[str, str] | None,
    ) -> OutputT:
        """Acquire model resources only after any external capacity wait has completed."""
        async with configured_agent_model(self.settings) as model:
            return await self._run_structured_with_model(
                session=session,
                prompt=prompt,
                instructions=instructions,
                output_type=output_type,
                model=model,
                writable=writable,
                container_tools=container_tools,
                planning=planning,
                extra_capabilities=extra_capabilities,
                output_validator=output_validator,
                candidate_validator=candidate_validator,
                progress=progress,
                run_timeout_seconds=run_timeout_seconds,
                agent_name=agent_name,
                metadata=metadata,
            )

    async def _run_structured_with_model(
        self,
        *,
        session: ContainerSessionProtocol,
        prompt: str,
        instructions: str,
        output_type: type[OutputT],
        model: Model | str,
        writable: bool = False,
        container_tools: bool = True,
        planning: bool = False,
        extra_capabilities: Sequence[AbstractCapability[object]] = (),
        output_validator: Callable[[OutputT], Awaitable[None] | None] | None = None,
        candidate_validator: Callable[[OutputT], Awaitable[None] | None] | None = None,
        progress: AgentProgress | None = None,
        run_timeout_seconds: float,
        agent_name: str = "agent",
        metadata: dict[str, str] | None = None,
    ) -> OutputT:
        feedback_count = 0

        async def report_feedback(feedback: AgentFeedback) -> None:
            nonlocal feedback_count
            feedback_count += 1
            if progress is None:
                return
            payload = feedback.model_dump(mode="json")
            codes = [issue.code for issue in feedback.issues]
            await progress(
                "output_feedback",
                {
                    "retry": feedback_count,
                    "codes": codes,
                    "feedback": payload,
                },
            )

        capabilities: list[AbstractCapability[object]] = [
            StructuredOutputFeedback(on_feedback=report_feedback),
        ]
        capabilities.extend(extra_capabilities)
        if container_tools:
            capabilities.append(
                ContainerWorkspace(
                    session=session,
                    writable=writable,
                    default_timeout_seconds=self.settings.tool_timeout_seconds,
                )
            )
        compaction = build_compaction_capability(self.settings.compaction)
        capabilities.append(compaction)
        run_id = f"{agent_name}-{uuid4().hex}"
        conversation_id = run_id
        trajectory_path: str | None = None
        trajectory_store: PrefixCompactingJsonStepStore | None = None
        if self.settings.trajectory.enabled:
            trajectory_path = str((self.settings.trajectory.directory / run_id).resolve())
            trajectory_store = PrefixCompactingJsonStepStore(trajectory_path)
        if planning:
            capabilities.append(Planning())
        agent = Agent(
            model,
            output_type=output_type,
            instructions=instructions,
            retries=configured_retries(self.settings),
            tool_timeout=self.settings.tool_timeout_seconds,
            capabilities=capabilities,
        )
        run_timeout_scope: asyncio.Timeout | None = None

        async def apply_validator(
            validator: Callable[[OutputT], Awaitable[None] | None],
            output: OutputT,
        ) -> None:
            validation = validator(output)
            if inspect.isawaitable(validation):
                await validation

        if output_validator is not None or candidate_validator is not None:

            async def validate_output(output: OutputT) -> OutputT:
                try:
                    if output_validator is not None:
                        await apply_validator(output_validator, output)
                    if candidate_validator is not None:
                        # Candidate checks may include an independent review Agent and
                        # real container execution. Keep Pydantic AI's ModelRetry
                        # conversation, but do not charge that host-side work to the
                        # Artifact Agent's authoring deadline.
                        scope = run_timeout_scope
                        loop = asyncio.get_running_loop()
                        deadline = scope.when() if scope is not None else None
                        if scope is not None and deadline is not None:
                            remaining = deadline - loop.time()
                            if remaining <= 0:
                                raise TimeoutError
                            scope.reschedule(None)
                            try:
                                await apply_validator(candidate_validator, output)
                            finally:
                                scope.reschedule(loop.time() + remaining)
                        else:
                            await apply_validator(candidate_validator, output)
                except AgentFeedbackError as exc:
                    await report_feedback(exc.feedback)
                    raise ModelRetry(render_feedback(exc.feedback)) from exc
                except (ContractError, ValueError) as exc:
                    await report_feedback(
                        AgentFeedback(
                            issues=(
                                FeedbackIssue(
                                    code="contract.unstructured",
                                    message=str(exc) or type(exc).__name__,
                                    correction=(
                                        "Correct the reported contract violation and return the complete "
                                        "structured object again."
                                    ),
                                ),
                            )
                        )
                    )
                    raise ModelRetry(str(exc)) from exc
                return output

            agent.output_validator(validate_output)

        async def handle_events(_context: object, stream: AsyncIterable[object]) -> None:
            async for event in stream:
                if progress is None:
                    continue
                kind = str(getattr(event, "event_kind", "agent_event"))
                if kind not in (
                    "function_tool_call",
                    "function_tool_result",
                    "output_tool_call",
                    "output_tool_result",
                ):
                    continue
                part = getattr(event, "part", None)
                tool_name = getattr(part, "tool_name", None)
                payload: dict[str, object] = (
                    {"tool": str(tool_name)} if tool_name is not None else {}
                )
                if (
                    kind == "function_tool_call"
                    and isinstance(tool_name, str)
                    and isinstance(part, ToolCallPart)
                ):
                    try:
                        arguments = cast(dict[str, object], part.args_as_dict())
                    except (AttributeError, TypeError, ValueError):
                        arguments = {}
                    if tool_name == "run_command":
                        command = arguments.get("command")
                        if isinstance(command, str):
                            payload["command"] = command[:2000]
                    elif tool_name in {
                        "read_file",
                        "write_file",
                        "edit_file",
                        "list_directory",
                        "search_files",
                        "find_files",
                        "file_info",
                        "create_directory",
                    }:
                        path = arguments.get("path")
                        if isinstance(path, str):
                            payload["path"] = path[:1000]
                await progress(
                    kind,
                    payload,
                )

        run_timeout = run_timeout_seconds
        usage = RunUsage()
        started_at = datetime.now(UTC)
        started = asyncio.get_running_loop().time()

        async def record_metrics(
            status: str,
            error: BaseException | None,
        ) -> None:
            ended_at = datetime.now(UTC)
            metric = make_agent_metric(
                run_id=run_id,
                conversation_id=conversation_id,
                agent_name=agent_name,
                model=self.settings.model or "",
                output_type=output_type.__name__,
                metadata=metadata,
                status=status,  # pyright: ignore[reportArgumentType]
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=asyncio.get_running_loop().time() - started,
                usage=usage,
                pricing=self.settings.pricing,
                trajectory_path=trajectory_path,
                error=error,
            )
            if self.settings.metrics.enabled:
                await append_agent_metric(self.settings.metrics.path, metric)
            if progress is not None:
                await progress(
                    "agent_run_metrics",
                    {
                        "run_id": run_id,
                        "trajectory_path": trajectory_path,
                        "status": status,
                        "duration_seconds": metric.duration_seconds,
                        "model": metric.model,
                        "input_tokens": metric.input_tokens,
                        "output_tokens": metric.output_tokens,
                        "cache_read_tokens": metric.cache_read_tokens,
                        "cache_write_tokens": metric.cache_write_tokens,
                        "cost_usd": metric.cost_usd,
                    },
                )

        if progress is not None:
            await progress(
                "agent_run_started",
                {
                    "run_id": run_id,
                    "model": self.settings.model or "",
                    "trajectory_path": trajectory_path,
                    "metadata": dict(metadata or {}),
                },
            )
        try:
            async with asyncio.timeout(run_timeout) as run_timeout_scope:
                transport_attempt = 0
                context_compaction_attempts = 0
                physical_attempt = 0
                message_history: list[ModelMessage] | None = None
                while True:
                    attempt_prompt = prompt if message_history is None else None
                    captured_messages: list[ModelMessage] = []
                    try:
                        run_capabilities: tuple[StepPersistence[object], ...] | None = None
                        if trajectory_store is not None:
                            run_capabilities = (
                                StepPersistence(
                                    store=trajectory_store,
                                    agent_name=agent_name,
                                    run_id=f"attempt-{physical_attempt:04d}",
                                    metadata={
                                        "logical_run_id": run_id,
                                        **dict(metadata or {}),
                                    },
                                ),
                            )
                        with capture_run_messages() as captured_messages:
                            result = await agent.run(
                                attempt_prompt,
                                message_history=message_history,
                                model_settings={"timeout": self.settings.request_timeout_seconds},
                                usage_limits=unlimited_usage_limits(),
                                usage=usage,
                                # A resumed request includes the preceding physical run ID
                                # in message_history. Pydantic AI requires every subsequent
                                # physical run to receive a fresh ID; omitting it delegates
                                # that ID generation while conversation_id keeps the logical
                                # Agent run correlated across retries.
                                run_id=run_id if physical_attempt == 0 else None,
                                conversation_id=conversation_id,
                                event_stream_handler=(
                                    handle_events if progress is not None else None
                                ),
                                capabilities=run_capabilities,
                            )
                        break
                    except Exception as exc:
                        recovered = resumable_message_history(captured_messages)
                        if is_context_window_error(exc):
                            if (
                                context_compaction_attempts < _MAX_CONTEXT_COMPACTION_ATTEMPTS
                                and recovered is not None
                            ):
                                context_compaction_attempts += 1
                                physical_attempt += 1
                                message_history = recovered
                                compaction.request_emergency_compaction()
                                if progress is not None:
                                    await progress(
                                        "context_compaction_retry",
                                        {
                                            "attempt": context_compaction_attempts,
                                            "maximum_attempts": (_MAX_CONTEXT_COMPACTION_ATTEMPTS),
                                            "error_type": type(exc).__name__,
                                            "error": (str(exc) or repr(exc))[:1000],
                                            "target_tokens": compaction.target_tokens,
                                        },
                                    )
                                continue
                            raise AgentContextWindowError(str(exc)) from exc
                        if not is_transient_model_error(
                            exc,
                            request_unanswered=recovered is not None,
                        ):
                            raise
                        if transport_attempt >= self.settings.transient_request_retries:
                            exhausted = AgentModelError(
                                f"{agent_name} Agent transient model request retry limit "
                                f"exhausted after {transport_attempt} retries: "
                                f"{type(exc).__name__}: {str(exc) or repr(exc)}"
                            )
                            if progress is not None:
                                await progress(
                                    "model_request_retry_exhausted",
                                    {
                                        "retries": transport_attempt,
                                        "maximum_retries": (
                                            self.settings.transient_request_retries
                                        ),
                                        "status_code": getattr(exc, "status_code", None),
                                        "error_type": type(exc).__name__,
                                        "error": (str(exc) or repr(exc))[:1000],
                                        "resumed": recovered is not None,
                                    },
                                )
                            raise exhausted from exc
                        transport_attempt += 1
                        physical_attempt += 1
                        delay = model_retry_delay(exc, transport_attempt)
                        message_history = recovered
                        if progress is not None:
                            await progress(
                                "model_request_retry",
                                {
                                    "retry": transport_attempt,
                                    "status_code": (getattr(exc, "status_code", None)),
                                    "error_type": type(exc).__name__,
                                    "error": (str(exc) or repr(exc))[:1000],
                                    "delay_seconds": delay,
                                    "resumed": message_history is not None,
                                },
                            )
                        await asyncio.sleep(delay)
        except asyncio.CancelledError as exc:
            await asyncio.shield(record_metrics("failed", exc))
            raise
        except TimeoutError as exc:
            timeout_error = AgentTimeoutError(f"Agent run exceeded {run_timeout:g} seconds")
            await record_metrics("timed_out", timeout_error)
            raise timeout_error from exc
        except AgentRunError as exc:
            model_error = AgentModelError(
                f"{agent_name} Agent model run failed: {str(exc) or type(exc).__name__}"
            )
            await record_metrics("failed", model_error)
            raise model_error from exc
        except Exception as exc:
            await record_metrics("failed", exc)
            raise
        await record_metrics("succeeded", None)
        return result.output
