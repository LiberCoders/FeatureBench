"""Durable per-run Agent metrics independent of optional full trajectory capture."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Literal

from genai_prices import calc_price
from pydantic import Field
from pydantic_ai.usage import RunUsage

from codeworld.config import AgentPricingSettings
from codeworld.models import StrictModel
from codeworld.storage.jsonl import append_jsonl_record


class AgentRunMetric(StrictModel):
    run_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    output_type: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    status: Literal["succeeded", "failed", "timed_out"]
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(ge=0)
    requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    usage_details: dict[str, int]
    catalog_cost_usd: float | None = Field(default=None, ge=0)
    configured_cost_usd: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cost_source: Literal["configured", "pydantic_catalog"] | None = None
    trajectory_path: str | None = None
    error_type: str | None = None
    error: str | None = None


def catalog_cost_usd(model: str, usage: RunUsage) -> float | None:
    """Use Pydantic's genai-prices catalog when the provider/model is known."""
    provider, separator, model_name = model.partition(":")
    if not separator or not provider or not model_name:
        return None
    try:
        calculation = calc_price(usage, model_name, provider_id=provider)
    except (AssertionError, LookupError, ValueError):
        return None
    return float(calculation.total_price)


def configured_cost_usd(pricing: AgentPricingSettings, usage: RunUsage) -> float | None:
    """Apply explicit relay rates, avoiding double charging cached inclusive input tokens."""
    if pricing.input_per_million_usd is None or pricing.output_per_million_usd is None:
        return None
    cache_read_rate = (
        pricing.cache_read_per_million_usd
        if pricing.cache_read_per_million_usd is not None
        else pricing.input_per_million_usd
    )
    cache_write_rate = (
        pricing.cache_write_per_million_usd
        if pricing.cache_write_per_million_usd is not None
        else pricing.input_per_million_usd
    )
    uncached_input = max(
        0,
        usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens,
    )
    total = (
        uncached_input * pricing.input_per_million_usd
        + usage.cache_read_tokens * cache_read_rate
        + usage.cache_write_tokens * cache_write_rate
        + usage.output_tokens * pricing.output_per_million_usd
    ) / 1_000_000
    return total


def make_agent_metric(
    *,
    run_id: str,
    conversation_id: str,
    agent_name: str,
    model: str,
    output_type: str,
    metadata: dict[str, str] | None = None,
    status: Literal["succeeded", "failed", "timed_out"],
    started_at: datetime,
    ended_at: datetime,
    duration_seconds: float,
    usage: RunUsage,
    pricing: AgentPricingSettings,
    trajectory_path: str | None,
    error: BaseException | None,
) -> AgentRunMetric:
    catalog = catalog_cost_usd(model, usage)
    configured = configured_cost_usd(pricing, usage)
    selected = configured if configured is not None else catalog
    source: Literal["configured", "pydantic_catalog"] | None = None
    if configured is not None:
        source = "configured"
    elif catalog is not None:
        source = "pydantic_catalog"
    return AgentRunMetric(
        run_id=run_id,
        conversation_id=conversation_id,
        agent_name=agent_name,
        model=model,
        output_type=output_type,
        metadata=dict(metadata or {}),
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        requests=usage.requests,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        usage_details=dict(usage.details),
        catalog_cost_usd=catalog,
        configured_cost_usd=configured,
        cost_usd=selected,
        cost_source=source,
        trajectory_path=trajectory_path,
        error_type=type(error).__name__ if error is not None else None,
        error=str(error)[-20_000:] if error is not None else None,
    )


async def append_agent_metric(path: Path, metric: AgentRunMetric) -> None:
    await asyncio.to_thread(
        append_jsonl_record,
        path,
        metric,
        parse=AgentRunMetric.model_validate_json,
        serialize=lambda record: record.model_dump_json(exclude_none=True),
    )
