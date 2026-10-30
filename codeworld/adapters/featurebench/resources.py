"""Invocation-owned execution capacity shared by FeatureBench Environments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from codeworld.adapters.featurebench.config import FeatureBenchSettings


@dataclass(frozen=True, slots=True)
class FeatureBenchRunResources:
    """Independent capacity pools for local work and model-backed work."""

    trace_slots: asyncio.Semaphore
    seed_slots: asyncio.Semaphore
    agent_slots: asyncio.Semaphore

    @classmethod
    def from_settings(cls, settings: FeatureBenchSettings) -> FeatureBenchRunResources:
        return cls(
            trace_slots=asyncio.Semaphore(settings.trace_concurrency),
            seed_slots=asyncio.Semaphore(settings.seed_concurrency),
            agent_slots=asyncio.Semaphore(settings.agent_concurrency),
        )
