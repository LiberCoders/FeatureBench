"""Typed files and tool results for FeatureBench Mask repair."""

from __future__ import annotations

from pydantic import Field, model_validator

from codeworld.models import StrictModel


class MaskRepairProposal(StrictModel):
    """One incremental constraint set written by the Repair Agent."""

    based_on_attempt: str = Field(min_length=1)
    must_restore_object_ids: tuple[str, ...] = ()
    must_mask_object_ids: tuple[str, ...] = ()
    rationale: str

    @model_validator(mode="after")
    def useful_disjoint_increment(self) -> MaskRepairProposal:
        restore = set(self.must_restore_object_ids)
        mask = set(self.must_mask_object_ids)
        if len(restore) != len(self.must_restore_object_ids):
            raise ValueError("must_restore_object_ids contains duplicates")
        if len(mask) != len(self.must_mask_object_ids):
            raise ValueError("must_mask_object_ids contains duplicates")
        overlap = restore & mask
        if overlap:
            raise ValueError(
                f"the same object cannot be restored and masked: {sorted(overlap)[:5]}"
            )
        if not restore and not mask:
            raise ValueError("the proposal must restore or mask at least one object")
        return self


class RepairViolation(StrictModel):
    """One concise, actionable feedback record written to violations.jsonl."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    correction: str | None = None
    field: str | None = None
    object_id: str | None = None
    object_ids: tuple[str, ...] | None = None
    diagnostic_file: str | None = None


class MaskRepairProposalError(ValueError):
    """A syntactically valid increment that cannot apply to its base Mask Plan."""

    def __init__(self, violations: tuple[RepairViolation, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(item.message for item in violations))


class RepairSubmitResult(StrictModel):
    """Small tool result; detailed evidence remains in the feedback directory."""

    accepted: bool
    attempt_id: str
    feedback_dir: str


class MaskRepairCompletion(StrictModel):
    """Final Agent output may only reference an accepted submit attempt."""

    accepted_attempt: str = Field(min_length=1)
    rationale: str


class RepairResultFile(StrictModel):
    """Authoritative compact status for one real or rejected submit attempt."""

    attempt_id: str
    based_on_attempt: str | None
    accepted: bool
    collection_ok: bool
    runtime_ok: bool
    test_exit_code: int
    f2p: int = Field(ge=0)
    p2p: int = Field(ge=0)
    f2p_ratio: float = Field(ge=0, le=1)
    gold_patch_lines: int = Field(ge=0)
    gold_patch_files: int = Field(ge=0)
    masked_objects: int = Field(ge=0)
    delta_f2p: int | None = None
    delta_p2p: int | None = None
    delta_gold_patch_lines: int | None = None
    delta_gold_patch_files: int | None = None
    delta_masked_objects: int | None = None
    violation_codes: tuple[str, ...] = ()
