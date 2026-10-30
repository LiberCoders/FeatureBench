from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.usage import RunUsage

from codeworld.agents.factory import is_context_window_error, is_transient_model_error
from codeworld.agents.telemetry import (
    AgentRunMetric,
    append_agent_metric,
    catalog_cost_usd,
    configured_cost_usd,
    make_agent_metric,
)
from codeworld.config import AgentPricingSettings


def usage() -> RunUsage:
    return RunUsage(
        requests=3,
        tool_calls=4,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=20,
        cache_write_tokens=10,
        details={"reasoning_tokens": 7},
    )


def test_costs_support_pydantic_catalog_and_explicit_relay_rates() -> None:
    observed = usage()
    assert (catalog_cost_usd("openai:gpt-5.4-mini", observed) or 0) > 0
    configured = configured_cost_usd(
        AgentPricingSettings(
            input_per_million_usd=1,
            output_per_million_usd=2,
            cache_read_per_million_usd=0.1,
            cache_write_per_million_usd=0.5,
        ),
        observed,
    )
    assert configured == pytest.approx(0.000177)
    assert configured_cost_usd(AgentPricingSettings(), observed) is None


def test_context_window_errors_are_permanent_even_without_http_status() -> None:
    context_error = ModelAPIError(
        "relay-model", "Your input exceeds the context window of this model."
    )
    transient_error = ModelAPIError("relay-model", "temporary malformed provider response")

    assert is_context_window_error(context_error)
    assert not is_transient_model_error(context_error)
    assert is_transient_model_error(transient_error)


def test_relay_account_group_404_is_retryable() -> None:
    error = ModelHTTPError(
        status_code=404,
        model_name="gpt-5.4-mini",
        body={
            "message": (
                'Model "gpt-5.4-mini" is not supported by any configured account in this group'
            )
        },
    )

    assert is_transient_model_error(error)


@pytest.mark.asyncio
async def test_metric_jsonl_repairs_only_a_torn_tail(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    metric = make_agent_metric(
        run_id="analysis-1",
        conversation_id="analysis-1",
        agent_name="analysis",
        model="openai:gpt-5.4-mini",
        output_type="EnvironmentAnalysis",
        metadata={"repository_url": "https://github.com/owner/repo.git"},
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_seconds=0,
        usage=usage(),
        pricing=AgentPricingSettings(),
        trajectory_path=None,
        error=None,
    )
    path = tmp_path / "agent-metrics.jsonl"
    await append_agent_metric(path, metric)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"torn"')
    await append_agent_metric(path, metric.model_copy(update={"run_id": "analysis-2"}))
    records = [AgentRunMetric.model_validate_json(line) for line in path.read_text().splitlines()]
    assert [record.run_id for record in records] == ["analysis-1", "analysis-2"]
    assert records[0].metadata["repository_url"].endswith("owner/repo.git")
