"""Small correctness adapter around Harness context compaction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
    estimate_token_count,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

SUMMARY_PREFIX = "Summary of previous conversation:\n\n"


@dataclass
class RecoveringTieredCompaction(TieredCompaction[object]):
    """Harness tiering with one explicit provider-overflow recovery hook."""

    _emergency_requested: bool = field(default=False, init=False, repr=False)
    _emergency_level: int = field(default=0, init=False, repr=False)

    def request_emergency_compaction(self) -> None:
        """Force every configured Harness tier once after a real provider overflow."""
        self._emergency_requested = True
        self._emergency_level += 1

    async def before_model_request(
        self,
        ctx: RunContext[object],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if not self._emergency_requested:
            return await super().before_model_request(ctx, request_context)
        self._emergency_requested = False
        messages = list(request_context.messages)
        for tier in self.tiers:
            messages = await tier.compact(messages, ctx)
        # A provider may reject a request even when the local estimate is below the
        # configured target (tool schemas and provider tokenizers are not represented
        # exactly). Normal retention can also make every configured tier a no-op, e.g.
        # five huge tool pairs with keep_pairs=6. Escalate only after a real overflow,
        # using Harness primitives and progressively retaining less on repeated failures.
        configured_tool_pairs = max(
            (tier.keep_pairs for tier in self.tiers if isinstance(tier, ClearToolResults)),
            default=3,
        )
        level = min(self._emergency_level, 3)
        emergency_tool_pairs = max(0, configured_tool_pairs * (3 - level) // 6)
        messages = await ClearToolResults(
            max_tokens=1,
            keep_pairs=emergency_tool_pairs,
            min_clear_tokens=None,
        ).compact(messages, ctx)
        if estimate_token_count(messages, self.tokenizer) > self.target_tokens:
            configured_messages = max(
                (
                    tier.keep_messages
                    for tier in self.tiers
                    if isinstance(tier, SummarizingCompaction)
                ),
                default=20,
            )
            emergency_messages = max(2, configured_messages // (2**level))
            messages = await ReplacingSummarizingCompaction(
                max_messages=1,
                keep_messages=emergency_messages,
            ).compact(messages, ctx)
        request_context.messages = messages
        return request_context


@dataclass
class ReplacingSummarizingCompaction(SummarizingCompaction[object]):
    """Ensure incremental Harness summaries replace, rather than accumulate with, predecessors.

    Harness 0.12.0 correctly feeds its preceding summary into the next summary request, but its
    leading-system-prompt preservation can also retain that preceding summary beside the new one.
    This adapter leaves Harness's summarization and safe-cutoff logic intact and normalizes its
    result to exactly one latest summary.
    """

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[object],
    ) -> list[ModelMessage]:
        compacted = await super().compact(messages, ctx)
        summary_positions = [
            (message_index, part_index)
            for message_index, message in enumerate(compacted)
            if isinstance(message, ModelRequest)
            for part_index, part in enumerate(message.parts)
            if isinstance(part, SystemPromptPart) and part.content.startswith(SUMMARY_PREFIX)
        ]
        if len(summary_positions) <= 1:
            return compacted

        latest = summary_positions[-1]
        normalized: list[ModelMessage] = []
        for message_index, message in enumerate(compacted):
            if not isinstance(message, ModelRequest):
                normalized.append(message)
                continue
            parts = [
                part
                for part_index, part in enumerate(message.parts)
                if not (
                    isinstance(part, SystemPromptPart)
                    and part.content.startswith(SUMMARY_PREFIX)
                    and (message_index, part_index) != latest
                )
            ]
            if parts:
                normalized.append(replace(message, parts=parts))
        return normalized
