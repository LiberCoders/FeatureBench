"""Strict public and internal models for FeatureBench Level 1 generation."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.models import StrictModel

ObjectKind = Literal["function", "async_function", "class"]
GoldPatchTier = Literal["preferred", "fallback"]


class GoldPatchMetrics(StrictModel):
    """Observed size of the final reverse patch."""

    lines: int = Field(ge=1)
    files: int = Field(ge=1)


class TestPointMetrics(StrictModel):
    """Unique final pytest node-ID counts."""

    __test__: ClassVar[bool] = False
    f2p: int = Field(ge=1)
    p2p: int = Field(ge=1)


class TopObjectMetrics(StrictModel):
    """Final Top count and its line coverage of the Gold Patch."""

    objects: int = Field(ge=1)
    gold_patch_lines: int = Field(ge=0)
    gold_patch_ratio: float = Field(ge=0, le=1)


class FeatureBenchMetrics(StrictModel):
    """Stable, queryable quality metrics embedded in every final task."""

    gold_patch: GoldPatchMetrics
    tests: TestPointMetrics
    top: TopObjectMetrics

    @model_validator(mode="after")
    def consistent_top_coverage(self) -> FeatureBenchMetrics:
        if self.top.gold_patch_lines > self.gold_patch.lines:
            raise ValueError("Top Gold Patch lines cannot exceed total Gold Patch lines")
        expected = self.top.gold_patch_lines / self.gold_patch.lines
        if not math.isclose(self.top.gold_patch_ratio, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Top Gold Patch ratio must equal top lines divided by total lines")
        return self


class FeatureBenchPayload(StrictModel):
    """Self-contained data required to reconstruct and grade one Level 1 task."""

    type: Literal["featurebench"]
    prompt: str = Field(min_length=1)
    gold_patch: str = Field(min_length=1)
    hidden_tests_patch: str = Field(min_length=1)
    fail_to_pass: tuple[CollectedTestPoint, ...] = Field(min_length=1)
    pass_to_pass: tuple[CollectedTestPoint, ...] = Field(min_length=1)
    test_timeout_seconds: float = Field(gt=0)
    metrics: FeatureBenchMetrics

    @field_validator("fail_to_pass", "pass_to_pass")
    @classmethod
    def canonical_test_points(
        cls, values: tuple[CollectedTestPoint, ...]
    ) -> tuple[CollectedTestPoint, ...]:
        identifiers = [item.id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("FeatureBench test-point IDs must be unique within each outcome")
        return tuple(sorted(values, key=lambda item: (item.id, item.path)))

    @model_validator(mode="after")
    def disjoint_tests(self) -> FeatureBenchPayload:
        overlap = set(self.fail_to_pass_ids) & set(self.pass_to_pass_ids)
        if overlap:
            raise ValueError(f"F2P and P2P must be disjoint: {sorted(overlap)[:3]}")
        if self.metrics.tests.f2p != len(self.fail_to_pass):
            raise ValueError("metrics.tests.f2p must equal the unique F2P test-point count")
        if self.metrics.tests.p2p != len(self.pass_to_pass):
            raise ValueError("metrics.tests.p2p must equal the unique P2P test-point count")
        patch_lines = sum(
            line.startswith("+") and not line.startswith("+++")
            for line in self.gold_patch.splitlines()
        )
        patch_files = sum(line.startswith("diff --git ") for line in self.gold_patch.splitlines())
        if self.metrics.gold_patch.lines != patch_lines:
            raise ValueError("metrics.gold_patch.lines must match the Gold Patch")
        if self.metrics.gold_patch.files != patch_files:
            raise ValueError("metrics.gold_patch.files must match the Gold Patch")
        return self

    @property
    def fail_to_pass_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.fail_to_pass)

    @property
    def pass_to_pass_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.pass_to_pass)

    @property
    def required_test_ids(self) -> tuple[str, ...]:
        return (*self.fail_to_pass_ids, *self.pass_to_pass_ids)

    @property
    def protected_test_paths(self) -> frozenset[str]:
        return frozenset(item.path for item in (*self.fail_to_pass, *self.pass_to_pass))


class TestReport(StrictModel):
    __test__: ClassVar[bool] = False
    exit_code: int
    collected: tuple[str, ...]
    outcomes: dict[str, Literal["passed", "failed", "skipped"]]
    collection_errors: tuple[str, ...] = ()
    output: str = ""

    @property
    def passed(self) -> tuple[str, ...]:
        return tuple(sorted(node for node, outcome in self.outcomes.items() if outcome == "passed"))

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(sorted(node for node, outcome in self.outcomes.items() if outcome == "failed"))


class TraceRecord(StrictModel):
    """One repository function observed while a test file was running."""

    path: str
    qualname: str
    line: int = Field(ge=1)
    calls: int = Field(ge=1)
    callers: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    process_ids: tuple[int, ...] = ()

    @property
    def object_id(self) -> str:
        return f"{self.path}::{self.qualname}::{self.line}"


class TraceBundle(StrictModel):
    """Reusable dynamic evidence for one test file or collection run."""

    test_file: str
    records: tuple[TraceRecord, ...]
    edges: tuple[TraceEdge, ...] = ()
    imports: tuple[TraceImport, ...] = ()
    passed_test_ids: tuple[str, ...] = ()


class TraceEdge(StrictModel):
    """One runtime call edge projected onto repository-owned frames."""

    caller: str = Field(min_length=1)
    callee: str = Field(min_length=1)
    calls: int = Field(ge=1)
    test_ids: tuple[str, ...] = ()
    process_ids: tuple[int, ...] = ()
    thread_ids: tuple[int, ...] = ()
    provenance: tuple[
        Literal["direct", "third_party_bridge", "subprocess", "framework_plugin"], ...
    ] = ()


class TraceImport(StrictModel):
    """Observed repository import binding used to complement the static index."""

    importer_path: str
    importer_line: int = Field(ge=1)
    imported_name: str
    defined_path: str
    defined_name: str
    defined_line: int | None = Field(default=None, ge=1)
    test_ids: tuple[str, ...] = ()


class SourceObject(StrictModel):
    """A source definition matched to dynamic evidence."""

    object_id: str
    path: str
    qualname: str
    name: str
    kind: ObjectKind
    line: int = Field(ge=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    parent_qualname: str | None = None
    signature: str
    docstring: str = ""
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    calls: int = Field(default=1, ge=0)
    callers: tuple[str, ...] = ()
    classification: Literal["candidate", "top", "specific"] = "candidate"

    @model_validator(mode="after")
    def valid_span_and_identity(self) -> SourceObject:
        if not self.start_line <= self.line <= self.end_line:
            raise ValueError("source object line must fall inside its deletion span")
        expected = f"{self.path}::{self.qualname}::{self.line}"
        if self.object_id != expected:
            raise ValueError(f"object_id must be {expected!r}")
        return self

    @property
    def removed_lines(self) -> int:
        return self.end_line - self.start_line + 1


class ImportBinding(StrictModel):
    """A statically resolved import statement that may need localized rewriting."""

    importer_path: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    module: str
    level: int = Field(default=0, ge=0)
    names: tuple[tuple[str, str | None], ...]
    resolved_path: str | None = None


class DefinitionTimeReference(StrictModel):
    """A reference evaluated while a module or class creates declarations."""

    referenced_object_id: str = Field(min_length=1)
    owner_object_id: str | None = None


class ImportEdit(StrictModel):
    importer_path: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    remove_names: tuple[str, ...] = Field(min_length=1)


class RepositoryIndexData(StrictModel):
    """Bounded metadata produced by the container-side Python source indexer."""

    definitions: tuple[SourceObject, ...]
    imports: tuple[ImportBinding, ...] = ()
    exports: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    extension_references: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    definition_time_references: tuple[DefinitionTimeReference, ...] = ()


class TopCandidateDecision(StrictModel):
    object_id: str = Field(min_length=1)
    classification: Literal["top", "traverse"]
    rationale: str


class TopBatchClassification(StrictModel):
    """One complete structured judgment for one deterministic BFS batch."""

    decisions: tuple[TopCandidateDecision, ...] = Field(min_length=1)


class P2PSelection(StrictModel):
    test_ids: tuple[str, ...]
    top_object_ids: tuple[str, ...]


class MaskOptimizationMetrics(StrictModel):
    """Auditable metrics from deterministic joint Mask/P2P optimization."""

    status: str = Field(min_length=1)
    solve_seconds: float = Field(ge=0)
    minimum_lines: int = Field(ge=1)
    maximum_lines: int = Field(ge=1)
    center_distance: float = Field(ge=0)
    modified_files: int = Field(ge=1)
    eligible_p2p_test_points: int = Field(ge=0)
    excluded_p2p_test_points: int = Field(ge=0)
    selected_p2p_test_points: int = Field(ge=0)
    selected_p2p_files: int = Field(ge=0)
    estimated_seed_test_points: int = Field(ge=0)
    estimated_seed_affected_test_points: int = Field(ge=0)
    estimated_f2p_ratio: float = Field(ge=0, le=1)
    estimated_seed_pass_rate: float = Field(ge=0, le=1)
    candidate_units: int = Field(ge=0)
    selected_units: int = Field(ge=0)
    signature_groups: int = Field(ge=0)


class MaskPlan(StrictModel):
    """The complete, reproducible feature slice removed from a clean workspace."""

    seed_test_file: str
    gold_patch_tier: GoldPatchTier = "preferred"
    revision: int = Field(default=0, ge=0)
    top_objects: tuple[SourceObject, ...] = Field(min_length=1)
    specific_objects: tuple[SourceObject, ...] = ()
    protected_objects: tuple[str, ...] = ()
    protection_reasons: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    dynamic_imports: tuple[TraceImport, ...] = ()
    import_edits: tuple[ImportEdit, ...] = ()
    candidate_object_ids: tuple[str, ...] = ()
    p2p_test_ids: tuple[str, ...] = ()
    optimization: MaskOptimizationMetrics | None = None

    @model_validator(mode="after")
    def disjoint_unique_objects(self) -> MaskPlan:
        top = [item.object_id for item in self.top_objects]
        specific = [item.object_id for item in self.specific_objects]
        if len(top) != len(set(top)) or len(specific) != len(set(specific)):
            raise ValueError("mask object IDs must be unique")
        overlap = set(top) & set(specific)
        if overlap:
            raise ValueError(f"top and specific objects overlap: {sorted(overlap)[:3]}")
        targets = set(top) | set(specific)
        protected = set(self.protected_objects)
        protected_targets = targets & protected
        if protected_targets:
            raise ValueError(
                f"mask targets include protected objects: {sorted(protected_targets)[:3]}"
            )
        reason_ids = set(self.protection_reasons)
        if reason_ids and reason_ids != protected:
            raise ValueError("protection reasons must describe every and only protected object")
        return self

    @property
    def targets(self) -> tuple[SourceObject, ...]:
        return (*self.top_objects, *self.specific_objects)

    @property
    def removed_lines(self) -> int:
        spans = {(target.path, target.start_line, target.end_line) for target in self.targets}
        return sum(end - start + 1 for _, start, end in spans)

    @property
    def patch_files(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {target.path for target in self.targets}
                | {edit.importer_path for edit in self.import_edits}
            )
        )


class QualityReport(StrictModel):
    accepted: bool
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    seed_pass_rate: float = Field(default=1.0, ge=0, le=1)
    f2p_ratio: float = Field(default=1.0, ge=0, le=1)
    gold_patch_lines: int = Field(default=0, ge=0)
    gold_patch_files: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()
    test_exit_code: int
    collection_errors: tuple[str, ...] = ()
    test_output: str = ""


class InterfaceContract(StrictModel):
    """One generated docstring bound to one final Top object."""

    object_id: str = Field(min_length=1)
    docstring: str


class IssueDraft(StrictModel):
    """Official-style task statement and exhaustive final-Top contracts."""

    task_statement: str
    interface_contracts: tuple[InterfaceContract, ...] = Field(min_length=1)


class InterfaceDescription(StrictModel):
    path: str
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
