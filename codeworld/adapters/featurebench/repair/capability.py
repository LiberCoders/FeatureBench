"""FeatureBench-owned Pydantic AI submit capability."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from codeworld.adapters.featurebench.repair.service import MaskRepairService


@dataclass
class MaskRepairSubmission(AbstractCapability[object]):
    """Register one adapter-specific host validation tool with a generic Agent."""

    service: MaskRepairService = field(repr=False)
    timeout_seconds: float

    def get_instructions(self) -> str:
        return (
            "submit_repair_file(path) discards scratch repository edits, applies the proposal "
            "increment to its referenced attempt, rebuilds /testbed, and runs authoritative real "
            'validation. It returns {"accepted": bool, "attempt_id": str, "feedback_dir": str}.'
        )

    def get_toolset(self) -> FunctionToolset[object]:
        toolset: FunctionToolset[object] = FunctionToolset(id="featurebench_mask_repair")

        async def submit_repair_file(path: str) -> str:
            """Submit one proposal file for host-side Mask Plan application and real tests.

            Args:
                path: Proposal path under the mounted repair outbox.
            """
            result = await self.service.submit_file(path)
            return result.model_dump_json()

        toolset.add_function(
            submit_repair_file,
            sequential=True,
            timeout=self.timeout_seconds,
        )
        return toolset
