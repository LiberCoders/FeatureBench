"""Host-owned incremental Mask Plan application and real-validation service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from pydantic import ValidationError

from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.masking import FeatureSlicePlanner, MaskExecutor
from codeworld.adapters.featurebench.models import MaskPlan, QualityReport, TraceBundle
from codeworld.adapters.featurebench.p2p import TracePointIndex
from codeworld.adapters.featurebench.repair.models import (
    MaskRepairProposal,
    MaskRepairProposalError,
    RepairSubmitResult,
    RepairViolation,
)
from codeworld.adapters.featurebench.repair.workspace import MaskRepairWorkspace
from codeworld.errors import CandidateRejected, CommandTimeoutError, ContractError
from codeworld.runtime.base import ContainerSessionProtocol

PlanValidator = Callable[[MaskPlan, int], Awaitable[QualityReport]]


class MaskRepairService:
    """Mutable state behind one adapter-registered submit capability."""

    def __init__(
        self,
        *,
        workspace: MaskRepairWorkspace,
        session: ContainerSessionProtocol,
        index: RepositoryIndex,
        planner: FeatureSlicePlanner,
        traces: Mapping[str, TraceBundle],
        initial_plan: MaskPlan,
        initial_quality: QualityReport,
        point_index: TracePointIndex,
        validate_plan: PlanValidator,
    ) -> None:
        self.workspace = workspace
        self.session = session
        self.index = index
        self.planner = planner
        self.traces = traces
        self.validate_plan = validate_plan
        self.executor = MaskExecutor(index)
        self.current_plan = initial_plan
        self.current_quality = initial_quality
        self.current_attempt_index = 0
        self.current_attempt_id = self._attempt_id(0)
        self.accepted_attempt_id: str | None = None
        self.learned_protected_ids: set[str] = set()
        self.plans_by_attempt: dict[str, MaskPlan] = {self.current_attempt_id: initial_plan}
        self.workspace.prepare(
            index=index,
            initial_plan=initial_plan,
            point_index=point_index,
        )
        self.current_feedback_dir = self.workspace.write_feedback(
            attempt_id=self.current_attempt_id,
            based_on_attempt=None,
            plan=initial_plan,
            quality=initial_quality,
            previous_plan=None,
            previous_quality=None,
            proposal=None,
            learned_protected_ids=frozenset(),
        )

    @staticmethod
    def _attempt_id(index: int) -> str:
        return f"attempt-{index:04d}"

    async def _record_rejected_submit(
        self,
        *,
        based_on_attempt: str,
        proposal: MaskRepairProposal | None,
        violations: tuple[RepairViolation, ...],
    ) -> RepairSubmitResult:
        previous_plan = self.current_plan
        previous_quality = self.current_quality
        await self.executor.apply(self.session, self.current_plan)
        self.current_attempt_index += 1
        self.current_attempt_id = self._attempt_id(self.current_attempt_index)
        self.current_feedback_dir = self.workspace.write_feedback(
            attempt_id=self.current_attempt_id,
            based_on_attempt=based_on_attempt,
            plan=self.current_plan,
            quality=self.current_quality,
            previous_plan=previous_plan,
            previous_quality=previous_quality,
            proposal=proposal,
            learned_protected_ids=frozenset(self.learned_protected_ids),
            extra_violations=violations,
        )
        self.plans_by_attempt[self.current_attempt_id] = self.current_plan
        return RepairSubmitResult(
            accepted=False,
            attempt_id=self.current_attempt_id,
            feedback_dir=self.current_feedback_dir,
        )

    async def submit_file(self, submitted_path: str) -> RepairSubmitResult:
        """Parse one Agent file, apply its increment, run tests, and publish feedback."""
        if self.accepted_attempt_id is not None:
            return RepairSubmitResult(
                accepted=True,
                attempt_id=self.accepted_attempt_id,
                feedback_dir=self.current_feedback_dir,
            )
        based_on_attempt = self.current_attempt_id
        try:
            path = self.workspace.proposal_host_path(submitted_path)
            if not path.is_file():
                raise ValueError(f"proposal file does not exist: {submitted_path}")
            if path.stat().st_size > 1_000_000:
                raise ValueError("proposal file exceeds 1000000 bytes")
            proposal = MaskRepairProposal.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, ValidationError) as exc:
            return await self._record_rejected_submit(
                based_on_attempt=based_on_attempt,
                proposal=None,
                violations=(
                    RepairViolation(
                        code="repair.proposal_invalid",
                        message=str(exc) or type(exc).__name__,
                        correction=(
                            "Write valid JSON containing only based_on_attempt, "
                            "must_restore_object_ids, must_mask_object_ids, and rationale."
                        ),
                    ),
                ),
            )
        based_on_plan = self.plans_by_attempt.get(proposal.based_on_attempt)
        if based_on_plan != self.current_plan:
            return await self._record_rejected_submit(
                based_on_attempt=proposal.based_on_attempt,
                proposal=proposal,
                violations=(
                    RepairViolation(
                        code="repair.stale_attempt",
                        field="based_on_attempt",
                        message=(
                            f"Proposal is based on {proposal.based_on_attempt}, but the current "
                            f"attempt is {self.current_attempt_id}."
                        ),
                        correction=(
                            "Use the based_on_attempt value in the refreshed outbox proposal. An "
                            "older attempt is accepted only while its Mask Plan is still identical."
                        ),
                    ),
                ),
            )
        try:
            revised = self.planner.apply_increment(
                index=self.index,
                plan=self.current_plan,
                proposal=proposal,
                traces=self.traces,
                learned_protected_ids=self.learned_protected_ids,
            )
        except MaskRepairProposalError as exc:
            return await self._record_rejected_submit(
                based_on_attempt=proposal.based_on_attempt,
                proposal=proposal,
                violations=exc.violations,
            )

        previous_plan = self.current_plan
        previous_quality = self.current_quality
        next_index = self.current_attempt_index + 1
        try:
            quality = await self.validate_plan(revised, next_index)
        except (CandidateRejected, CommandTimeoutError, ContractError) as exc:
            return await self._record_rejected_submit(
                based_on_attempt=proposal.based_on_attempt,
                proposal=proposal,
                violations=(
                    RepairViolation(
                        code="repair.validation_failed",
                        message=str(exc) or type(exc).__name__,
                        correction=(
                            "Inspect the activated repository, then restore the smallest implicated "
                            "object set or choose a different replacement mask."
                        ),
                    ),
                ),
            )
        # Repair increments are monotonic: once an Agent explicitly restores an
        # object and the host successfully applies and validates that revised
        # plan, later increments must not mask the same object again.  This
        # prevents long repair runs from cycling between a runtime fix and a
        # scale-recovery mask after context compaction, without encoding any
        # framework-specific object names in the host.
        self.learned_protected_ids.update(proposal.must_restore_object_ids)

        self.current_attempt_index = next_index
        self.current_attempt_id = self._attempt_id(next_index)
        self.current_plan = revised
        self.current_quality = quality
        self.current_feedback_dir = self.workspace.write_feedback(
            attempt_id=self.current_attempt_id,
            based_on_attempt=proposal.based_on_attempt,
            plan=revised,
            quality=quality,
            previous_plan=previous_plan,
            previous_quality=previous_quality,
            proposal=proposal,
            learned_protected_ids=frozenset(self.learned_protected_ids),
        )
        self.plans_by_attempt[self.current_attempt_id] = revised
        if quality.accepted:
            self.accepted_attempt_id = self.current_attempt_id
        return RepairSubmitResult(
            accepted=quality.accepted,
            attempt_id=self.current_attempt_id,
            feedback_dir=self.current_feedback_dir,
        )
