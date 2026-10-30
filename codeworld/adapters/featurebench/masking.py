"""FeatureBench Top/Specific/P2P planning and full-object structural deletion."""

from __future__ import annotations

import ast
import asyncio
import json
import random
import re
import textwrap
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from codeworld.adapters.featurebench.indexing import RepositoryIndex, is_test_path
from codeworld.adapters.featurebench.models import (
    GoldPatchTier,
    ImportEdit,
    MaskOptimizationMetrics,
    MaskPlan,
    P2PSelection,
    SourceObject,
    TopBatchClassification,
    TraceBundle,
    TraceImport,
)
from codeworld.adapters.featurebench.optimization import (
    DeletionInfeasibilityDiagnostics,
    DeletionOptimizationInfeasibleError,
    DeletionUnit,
    optimize_deletions,
)
from codeworld.adapters.featurebench.p2p import (
    TracePointIndex,
    seed_import_object_ids,
)
from codeworld.adapters.featurebench.repair.models import (
    MaskRepairProposal,
    MaskRepairProposalError,
    RepairViolation,
)
from codeworld.adapters.featurebench.top_agent import (
    FeatureGraphTraversal,
    TopClassificationBatch,
    feature_graph_root_ids,
    load_top_prompt,
)
from codeworld.agents.factory import AgentFactory, AgentProgress
from codeworld.agents.feedback import AgentFeedback, AgentFeedbackError
from codeworld.config import AgentFactorySettings
from codeworld.environment.frameworks.pytest import parse_pytest_public_id
from codeworld.errors import CandidateRejected, ContractError
from codeworld.runtime.base import ContainerSessionProtocol


@dataclass(frozen=True, slots=True)
class TopCandidate:
    object: SourceObject
    depth: int


@dataclass(frozen=True, slots=True)
class SeedFeasibilityProfile:
    """Deterministic optimistic bounds used before any Seed Agent call."""

    test_file: str
    top_candidate_count: int
    estimated_top_batches: int
    optimistic_removed_lines: int
    available_p2p_test_points: int
    minimum_p2p_test_points: int
    estimated_validation_files: int
    estimated_validation_points: int
    preferred_code_scale: bool
    capped_line_capacity: int
    rejection_reasons: tuple[str, ...] = ()
    protected_object_ids: frozenset[str] = frozenset()

    @property
    def rejected(self) -> bool:
        return bool(self.rejection_reasons)

    def easy_first_key(self) -> tuple[int, int, int, int, int, int, str]:
        """Prefer likely acceptance, then cheaper Agent and validation work."""
        p2p_is_tight = self.available_p2p_test_points < 2 * self.minimum_p2p_test_points
        return (
            int(not self.preferred_code_scale),
            int(p2p_is_tight),
            self.estimated_top_batches,
            self.estimated_validation_files,
            self.estimated_validation_points,
            -self.capped_line_capacity,
            self.test_file,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "test_file": self.test_file,
            "top_candidate_count": self.top_candidate_count,
            "estimated_top_batches": self.estimated_top_batches,
            "optimistic_removed_lines": self.optimistic_removed_lines,
            "available_p2p_test_points": self.available_p2p_test_points,
            "minimum_p2p_test_points": self.minimum_p2p_test_points,
            "estimated_validation_files": self.estimated_validation_files,
            "estimated_validation_points": self.estimated_validation_points,
            "preferred_code_scale": self.preferred_code_scale,
            "capped_line_capacity": self.capped_line_capacity,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass(frozen=True, slots=True)
class JointOptimizationOptions:
    """Runtime-free settings for one deterministic Mask/P2P optimization."""

    point_index: TracePointIndex
    minimum_p2p_test_points: int
    maximum_p2p_test_points: int | None
    maximum_selected_p2p_test_files: int | None
    minimum_f2p_test_points: int
    maximum_f2p_test_points: int | None
    maximum_f2p_ratio: float
    maximum_seed_pass_rate: float
    timeout_seconds: float
    workers: int
    minimum_code_lines: int | None = None
    maximum_code_lines: int | None = None
    gold_patch_tier: GoldPatchTier = "preferred"


class MaskIntervalInfeasible(CandidateRejected):
    """One Gold Patch line tier was proven infeasible for a Seed."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[DeletionInfeasibilityDiagnostics, ...] = (),
    ) -> None:
        self.diagnostics = diagnostics
        super().__init__(message)


def _binding_groups(index: RepositoryIndex) -> dict[tuple[str, str], tuple[SourceObject, ...]]:
    return index.binding_groups


def _object_ids_for_tests(
    index: RepositoryIndex,
    traces: Iterable[TraceBundle],
    test_ids: Iterable[str],
) -> frozenset[str]:
    selected = frozenset(test_ids)
    if not selected:
        return frozenset()
    return frozenset(
        item.object_id
        for bundle in traces
        for record in bundle.records
        if selected.intersection(record.test_ids)
        if (item := index.resolve_trace(record)) is not None
    )


@dataclass(frozen=True, slots=True)
class MaskProtection:
    """One explainable fixed-point protection set used by every mask decision."""

    protected_ids: frozenset[str]
    reasons: Mapping[str, tuple[str, ...]]

    @classmethod
    def build(
        cls,
        *,
        index: RepositoryIndex,
        p2p_object_ids: Iterable[str],
        collection_object_ids: Iterable[str],
        critical_magic_methods: frozenset[str],
        seed_collection_object_ids: Iterable[str] = (),
        additional_protected_ids: Iterable[str] = (),
    ) -> MaskProtection:
        reasons: dict[str, set[str]] = defaultdict(set)

        def protect(object_id: str, reason: str) -> bool:
            if object_id not in index.by_id:
                return False
            before = len(reasons[object_id])
            reasons[object_id].add(reason)
            return before == 0

        for object_id in p2p_object_ids:
            protect(object_id, "p2p_trace")
        for object_id in collection_object_ids:
            protect(object_id, "collection_trace")
        for object_id in seed_collection_object_ids:
            protect(object_id, "seed_collection_import")
        for object_id in index.protected_ids(critical_magic_methods):
            protect(object_id, "static_contract")
        for object_id in additional_protected_ids:
            protect(object_id, "previous_plan")
        for item in index.data.definitions:
            if is_test_path(item.path):
                protect(item.object_id, "test_code")

        groups = _binding_groups(index)
        queue = deque(sorted(reasons))
        expanded: set[str] = set()
        while queue:
            object_id = queue.popleft()
            if object_id in expanded:
                continue
            expanded.add(object_id)
            item = index.by_id[object_id]
            for peer in groups[(item.path, item.qualname)]:
                if peer.object_id == object_id:
                    continue
                if protect(peer.object_id, "binding_contract"):
                    queue.append(peer.object_id)
            for ancestor in index.ancestors(item):
                if protect(ancestor.object_id, "structural_ancestor"):
                    queue.append(ancestor.object_id)

        canonical = {
            object_id: tuple(sorted(object_reasons))
            for object_id, object_reasons in sorted(reasons.items())
        }
        return cls(
            protected_ids=frozenset(canonical),
            reasons=MappingProxyType(canonical),
        )


def _expand_binding_ids(
    index: RepositoryIndex,
    object_ids: Iterable[str],
) -> tuple[SourceObject, ...]:
    groups = _binding_groups(index)
    expanded: dict[str, SourceObject] = {}
    for object_id in object_ids:
        item = index.by_id.get(object_id)
        if item is None:
            continue
        for peer in groups[(item.path, item.qualname)]:
            expanded[peer.object_id] = peer
    return tuple(
        sorted(expanded.values(), key=lambda item: (item.path, item.start_line, item.object_id))
    )


def _span_contains(outer: SourceObject, inner: SourceObject) -> bool:
    return (
        outer.path == inner.path
        and outer.start_line <= inner.start_line
        and inner.end_line <= outer.end_line
    )


def _collapse_nested(objects: Iterable[SourceObject]) -> tuple[SourceObject, ...]:
    unique = {item.object_id: item for item in objects}
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.path, item.start_line, -item.end_line, item.object_id),
        )
    )
    return tuple(
        item
        for item in ordered
        if not any(
            other.object_id != item.object_id and _span_contains(other, item) for other in ordered
        )
    )


def _unique_span_lines(objects: Iterable[SourceObject]) -> int:
    return sum(item.removed_lines for item in _collapse_nested(objects))


def _deletion_ids(index: RepositoryIndex, targets: Iterable[SourceObject]) -> frozenset[str]:
    """Resolve atomic bindings and every declaration nested in their source spans."""
    roots = _expand_binding_ids(index, (item.object_id for item in targets))
    return frozenset(
        candidate.object_id
        for candidate in index.data.definitions
        if any(_span_contains(root, candidate) for root in roots)
    )


class FeatureSlicePlanner:
    """Construct one official-style feature slice from cached static/dynamic evidence."""

    def __init__(
        self,
        *,
        top_max_depth: int,
        top_batch_size: int,
        minimum_code_lines: int,
        maximum_code_lines: int,
        random_seed: int,
        critical_magic_methods: frozenset[str],
    ) -> None:
        self.top_max_depth = top_max_depth
        self.top_batch_size = top_batch_size
        self.minimum_code_lines = minimum_code_lines
        self.maximum_code_lines = maximum_code_lines
        self.random_seed = random_seed
        self.critical_magic_methods = critical_magic_methods

    @staticmethod
    def top_deletion_impacts(
        index: RepositoryIndex,
        top_object_ids: Iterable[str],
    ) -> dict[str, tuple[str, ...]]:
        """Return every declaration erased when each atomic Top binding is deleted."""
        groups = _binding_groups(index)
        impacts: dict[str, tuple[str, ...]] = {}
        for object_id in top_object_ids:
            item = index.by_id.get(object_id)
            if item is None:
                continue
            atomic_roots = groups[(item.path, item.qualname)]
            impacts[object_id] = tuple(
                sorted(index.structural_test_impact_ids(_deletion_ids(index, atomic_roots)))
            )
        return impacts

    @staticmethod
    def _trace_graph(
        index: RepositoryIndex, trace: TraceBundle
    ) -> tuple[dict[str, SourceObject], dict[str, set[str]]]:
        objects = {item.object_id: item for item in index.traced_objects(trace.records)}
        children: dict[str, set[str]] = defaultdict(set)
        if trace.edges:
            for edge in trace.edges:
                caller = index.normalize_runtime_id(edge.caller)
                callee = index.normalize_runtime_id(edge.callee)
                if caller in objects and callee in objects:
                    children[caller].add(callee)
            return objects, children
        for item in objects.values():
            for caller in item.callers:
                children[caller].add(item.object_id)
        return objects, children

    def _top_candidates(
        self,
        index: RepositoryIndex,
        trace: TraceBundle,
        test_ids: tuple[str, ...],
        protected_ids: frozenset[str],
        trace_graph: tuple[dict[str, SourceObject], dict[str, set[str]]] | None = None,
    ) -> tuple[TopCandidate, ...]:
        objects, children = trace_graph or self._trace_graph(index, trace)
        roots = feature_graph_root_ids(
            index=index,
            trace=trace,
            test_ids=test_ids,
            objects=objects,
            children=children,
        )
        queue = deque((root, 0) for root in sorted(roots))
        distances: dict[str, int] = dict.fromkeys(roots, 0)
        while queue:
            current, depth = queue.popleft()
            if depth >= self.top_max_depth:
                continue
            for child in sorted(children.get(current, ())):
                if child in distances and distances[child] <= depth + 1:
                    continue
                distances[child] = depth + 1
                queue.append((child, depth + 1))
        candidates = [
            TopCandidate(item, distances[object_id])
            for object_id, item in objects.items()
            if 0 < distances.get(object_id, self.top_max_depth + 1) <= self.top_max_depth
            and not is_test_path(item.path)
            and item.object_id not in protected_ids
            and item.name not in self.critical_magic_methods
        ]
        candidates.sort(
            key=lambda value: (value.depth, -value.object.calls, value.object.object_id)
        )
        return tuple(candidates)

    def _top_protection(
        self,
        *,
        index: RepositoryIndex,
        trace: TraceBundle,
        collection_trace: TraceBundle | None,
    ) -> MaskProtection:
        collection_ids = (
            ()
            if collection_trace is None
            else (item.object_id for item in index.traced_objects(collection_trace.records))
        )
        return MaskProtection.build(
            index=index,
            p2p_object_ids=(),
            collection_object_ids=collection_ids,
            critical_magic_methods=self.critical_magic_methods,
            seed_collection_object_ids=seed_import_object_ids(
                index=index,
                test_file=trace.test_file,
                dynamic_imports=trace.imports,
            ),
        )

    def top_candidates(
        self, index: RepositoryIndex, trace: TraceBundle
    ) -> tuple[TopCandidate, ...]:
        protection = self._top_protection(
            index=index,
            trace=trace,
            collection_trace=None,
        )
        return self._top_candidates(
            index,
            trace,
            trace.passed_test_ids,
            protection.protected_ids,
        )

    def eligible_top_candidates(
        self,
        *,
        index: RepositoryIndex,
        trace: TraceBundle,
        collection_trace: TraceBundle,
        test_ids: tuple[str, ...] | None = None,
    ) -> tuple[TopCandidate, ...]:
        """Return the exact deterministic candidate inventory seen by the Top Agent."""
        protection = self._top_protection(
            index=index,
            trace=trace,
            collection_trace=collection_trace,
        )
        return self._top_candidates(
            index,
            trace,
            test_ids or trace.passed_test_ids,
            protection.protected_ids,
        )

    def seed_feasibility_profile(
        self,
        *,
        index: RepositoryIndex,
        trace: TraceBundle,
        collection_trace: TraceBundle,
        test_ids: tuple[str, ...],
        point_index: TracePointIndex,
        minimum_code_lines: int,
        preferred_minimum_code_lines: int,
        maximum_code_lines: int,
        minimum_p2p_test_points: int,
        maximum_p2p_test_points: int | None,
        maximum_selected_p2p_test_files: int | None,
    ) -> SeedFeasibilityProfile:
        """Build sound rejection bounds plus deterministic scheduling estimates."""
        protection = self._top_protection(
            index=index,
            trace=trace,
            collection_trace=collection_trace,
        )
        trace_graph = self._trace_graph(index, trace)
        candidates = self._top_candidates(
            index,
            trace,
            test_ids,
            protection.protected_ids,
            trace_graph,
        )
        objects, children = trace_graph
        reachable_ids = {candidate.object.object_id for candidate in candidates}
        queue = deque(sorted(reachable_ids))
        while queue:
            current = queue.popleft()
            for child in sorted(children.get(current, ())):
                if child in reachable_ids:
                    continue
                reachable_ids.add(child)
                queue.append(child)
        optimistic_targets = _expand_binding_ids(
            index,
            (
                object_id
                for object_id in reachable_ids
                if object_id in objects and not is_test_path(objects[object_id].path)
            ),
        )
        optimistic_removed_lines = _unique_span_lines(optimistic_targets)
        candidates_by_depth: dict[int, int] = defaultdict(int)
        for candidate in candidates:
            candidates_by_depth[candidate.depth] += 1
        estimated_top_batches = sum(
            (count + self.top_batch_size - 1) // self.top_batch_size
            for count in candidates_by_depth.values()
        )

        candidate_mask = point_index.candidate_test_mask(trace.test_file)
        eligible_mask = point_index.limit_mask_to_best_files(
            candidate_mask,
            maximum_selected_p2p_test_files,
        )
        available_p2p = eligible_mask.bit_count()
        estimated_selected_p2p = (
            available_p2p
            if maximum_p2p_test_points is None
            else min(available_p2p, maximum_p2p_test_points)
        )
        estimated_validation_files = point_index.minimum_files_for_test_count(
            eligible_mask,
            estimated_selected_p2p,
        )
        seed_test_points = point_index.file_test_masks.get(trace.test_file, 0).bit_count()
        rejection_reasons: list[str] = []
        if not candidates:
            rejection_reasons.append(f"no maskable Top objects traced by {trace.test_file}")
        if optimistic_removed_lines < minimum_code_lines:
            rejection_reasons.append(
                f"optimistic Mask line capacity is {optimistic_removed_lines}, below "
                f"gold_patch_line_minimum={minimum_code_lines}"
            )
        if available_p2p < minimum_p2p_test_points:
            rejection_reasons.append(
                f"only {available_p2p} optimistic P2P points remain, below "
                f"minimum_remaining_p2p_test_points={minimum_p2p_test_points}"
            )
        return SeedFeasibilityProfile(
            test_file=trace.test_file,
            top_candidate_count=len(candidates),
            estimated_top_batches=estimated_top_batches,
            optimistic_removed_lines=optimistic_removed_lines,
            available_p2p_test_points=available_p2p,
            minimum_p2p_test_points=minimum_p2p_test_points,
            estimated_validation_files=estimated_validation_files or 0,
            estimated_validation_points=seed_test_points + estimated_selected_p2p,
            preferred_code_scale=(optimistic_removed_lines >= preferred_minimum_code_lines),
            capped_line_capacity=min(optimistic_removed_lines, maximum_code_lines),
            rejection_reasons=tuple(rejection_reasons),
            protected_object_ids=protection.protected_ids,
        )

    async def select_top(
        self,
        *,
        index: RepositoryIndex,
        trace: TraceBundle,
        collection_trace: TraceBundle,
        test_ids: tuple[str, ...],
        agents: AgentFactory[AgentFactorySettings],
        session: ContainerSessionProtocol,
        run_timeout_seconds: float,
        batch_concurrency: int = 1,
        progress: AgentProgress | None = None,
        protected_object_ids: frozenset[str] | None = None,
        candidate_count: int | None = None,
    ) -> tuple[SourceObject, ...]:
        if batch_concurrency < 1:
            raise ValueError("batch_concurrency must be at least 1")
        if protected_object_ids is None:
            protected_object_ids = self._top_protection(
                index=index,
                trace=trace,
                collection_trace=collection_trace,
            ).protected_ids
        if candidate_count is None:
            candidate_count = len(
                self._top_candidates(
                    index,
                    trace,
                    test_ids,
                    protected_object_ids,
                )
            )
        if candidate_count < 1:
            raise CandidateRejected(f"no maskable Top objects traced by {trace.test_file}")
        traversal = FeatureGraphTraversal(
            index=index,
            trace=trace,
            test_ids=test_ids,
            max_depth=self.top_max_depth,
            batch_size=self.top_batch_size,
            critical_magic_methods=self.critical_magic_methods,
            protected_object_ids=protected_object_ids,
        )
        allowed = traversal.objects
        batch_number = 0

        async def classify(
            current_batch: TopClassificationBatch,
            current_batch_number: int,
            semaphore: asyncio.Semaphore,
        ) -> TopBatchClassification:
            async def validate(result: TopBatchClassification) -> None:
                issues = traversal.validation_issues(current_batch, result)
                if issues:
                    raise AgentFeedbackError(AgentFeedback(issues=issues))

            async with semaphore:
                return await agents.run_structured(
                    session=session,
                    prompt=(
                        load_top_prompt("batch")
                        + "\n\n"
                        + json.dumps(
                            current_batch.payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                    instructions=load_top_prompt("instructions"),
                    output_type=TopBatchClassification,
                    output_validator=validate,
                    progress=progress,
                    writable=False,
                    container_tools=False,
                    run_timeout_seconds=run_timeout_seconds,
                    agent_name="featurebench_top_classifier",
                    metadata={
                        "seed_test_file": trace.test_file,
                        "bfs_depth": str(current_batch.depth),
                        "batch_number": str(current_batch_number),
                    },
                )

        while not traversal.done:
            layer_batches = traversal.layer_batches()
            semaphore = asyncio.Semaphore(batch_concurrency)
            tasks = tuple(
                asyncio.create_task(classify(batch, batch_number + index, semaphore))
                for index, batch in enumerate(layer_batches, start=1)
            )
            try:
                classifications = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for batch, result in zip(layer_batches, classifications, strict=True):
                traversal.apply(batch, result)
            batch_number += len(layer_batches)
        selected_ids = traversal.selected_ids()
        if not selected_ids:
            raise CandidateRejected(f"no Top objects selected for {trace.test_file}")
        return tuple(allowed[object_id] for object_id in selected_ids)

    def create_plan(
        self,
        *,
        index: RepositoryIndex,
        seed_trace: TraceBundle,
        all_traces: Mapping[str, TraceBundle],
        collection_trace: TraceBundle,
        top_objects: Sequence[SourceObject],
        p2p: P2PSelection,
        p2p_object_ids: Iterable[str] | None = None,
        joint_optimization: JointOptimizationOptions | None = None,
    ) -> MaskPlan:
        objects, children = self._trace_graph(index, seed_trace)
        minimum_code_lines = (
            joint_optimization.minimum_code_lines
            if joint_optimization is not None and joint_optimization.minimum_code_lines is not None
            else self.minimum_code_lines
        )
        maximum_code_lines = (
            joint_optimization.maximum_code_lines
            if joint_optimization is not None and joint_optimization.maximum_code_lines is not None
            else self.maximum_code_lines
        )
        dynamic_imports = tuple(
            item
            for bundle in (seed_trace, collection_trace, *all_traces.values())
            for item in bundle.imports
        )
        p2p_ids: frozenset[str] = (
            frozenset()
            if joint_optimization is not None
            else (
                frozenset(p2p_object_ids)
                if p2p_object_ids is not None
                else _object_ids_for_tests(index, all_traces.values(), p2p.test_ids)
            )
        )
        collection_ids = frozenset(
            item.object_id for item in index.traced_objects(collection_trace.records)
        )
        protection = MaskProtection.build(
            index=index,
            p2p_object_ids=p2p_ids,
            collection_object_ids=collection_ids,
            critical_magic_methods=self.critical_magic_methods,
            seed_collection_object_ids=seed_import_object_ids(
                index=index,
                test_file=seed_trace.test_file,
                dynamic_imports=seed_trace.imports,
            ),
        )

        effective_top_ids = set(p2p.top_object_ids)
        top_protection_conflicts = {
            item.object_id: tuple(sorted(_deletion_ids(index, (item,)) & protection.protected_ids))
            for item in top_objects
            if item.object_id in effective_top_ids
        }
        selected_top = {
            item.object_id: item
            for item in top_objects
            if item.object_id in effective_top_ids and not top_protection_conflicts[item.object_id]
        }
        raw_top = tuple(
            selected_top.get(item.object_id, item).model_copy(update={"classification": "top"})
            for item in _expand_binding_ids(index, selected_top)
        )
        top = _collapse_nested(raw_top)
        if not top:
            protected_top_ids = tuple(
                sorted(
                    object_id
                    for object_id, conflicts in top_protection_conflicts.items()
                    if conflicts
                )
            )
            rendered = "; ".join(
                f"{object_id} deletes "
                + ", ".join(
                    f"{conflict} ({', '.join(protection.reasons[conflict])})"
                    for conflict in top_protection_conflicts[object_id][:3]
                )
                for object_id in protected_top_ids[:5]
            )
            raise CandidateRejected(
                "runtime contracts protected every selected Top object"
                + (f": {rendered}" if rendered else "")
            )
        top_dependency_blockers = index.definition_time_blockers(_deletion_ids(index, top))
        if top_dependency_blockers:
            rendered = [
                f"{object_id} required by " + ", ".join(owner or "<module>" for owner in owners)
                for object_id, owners in tuple(top_dependency_blockers.items())[:5]
            ]
            raise CandidateRejected(
                "Top deletion leaves definition-time references alive: " + "; ".join(rendered)
            )

        top_ids = {item.object_id for item in top}
        distance: dict[str, int] = {object_id: 0 for object_id in top_ids if object_id in objects}
        queue = deque((object_id, 0) for object_id in sorted(distance))
        while queue:
            current, depth = queue.popleft()
            for child in sorted(children.get(current, ())):
                if child in distance and distance[child] <= depth + 1:
                    continue
                distance[child] = depth + 1
                queue.append((child, depth + 1))

        groups = _binding_groups(index)
        candidate_units: dict[tuple[str, str], tuple[SourceObject, ...]] = {}
        candidate_impacts: dict[tuple[str, str], frozenset[str]] = {}
        candidate_scores: dict[tuple[str, str], tuple[int, int, str, str]] = {}
        for object_id, depth in distance.items():
            if depth == 0 or object_id not in objects:
                continue
            dynamic_item = objects[object_id]
            key = (dynamic_item.path, dynamic_item.qualname)
            if key in candidate_units:
                continue
            unit = tuple(objects.get(item.object_id, item) for item in groups[key])
            deletion_ids = _deletion_ids(index, unit)
            if deletion_ids & (protection.protected_ids | top_ids):
                continue
            if any(
                _span_contains(target, item) or _span_contains(item, target)
                for target in top
                for item in unit
            ):
                continue
            reachable = [distance[item.object_id] for item in unit if item.object_id in distance]
            dynamic_calls = sum(
                objects[item.object_id].calls for item in unit if item.object_id in objects
            )
            candidate_units[key] = unit
            candidate_impacts[key] = deletion_ids
            candidate_scores[key] = (
                min(reachable),
                -dynamic_calls,
                dynamic_item.path,
                dynamic_item.qualname,
            )

        owner_units = {
            item.object_id: key for key, unit in candidate_units.items() for item in unit
        }
        top_deletion_ids = _deletion_ids(index, top)
        selected_keys: set[tuple[str, str]] = set()

        def dependency_closure(
            seed_key: tuple[str, str],
            current_keys: set[tuple[str, str]],
        ) -> set[tuple[str, str]] | None:
            prospective = {*current_keys, seed_key}
            while True:
                deleted = top_deletion_ids | frozenset(
                    object_id for key in prospective for object_id in candidate_impacts[key]
                )
                added = False
                for owners in index.definition_time_blockers(deleted).values():
                    for owner in owners:
                        if owner is None or owner in deleted:
                            continue
                        owner_key = owner_units.get(owner)
                        if owner_key is None:
                            return None
                        if owner_key not in prospective:
                            prospective.add(owner_key)
                            added = True
                if not added:
                    return prospective

        maskable_keys = {
            key for key in candidate_units if dependency_closure(key, set()) is not None
        }
        optimization_metrics: MaskOptimizationMetrics | None = None
        if joint_optimization is None:
            line_budget = random.Random(
                f"{self.random_seed}:{seed_trace.test_file}:line-budget"
            ).randint(minimum_code_lines, maximum_code_lines)
            removed_lines = _unique_span_lines(top)
            for key in sorted(maskable_keys, key=candidate_scores.__getitem__):
                if removed_lines >= line_budget:
                    break
                if key in selected_keys:
                    continue
                prospective_keys = dependency_closure(key, selected_keys)
                if prospective_keys is None:
                    continue
                prospective_targets = _collapse_nested(
                    (
                        *top,
                        *(
                            item
                            for prospective_key in prospective_keys
                            for item in candidate_units[prospective_key]
                        ),
                    )
                )
                prospective_lines = _unique_span_lines(prospective_targets)
                if prospective_lines > line_budget:
                    continue
                selected_keys = prospective_keys
                removed_lines = prospective_lines
        else:
            closures = {
                key: closure
                for key in maskable_keys
                if (closure := dependency_closure(key, set())) is not None
            }
            maskable_keys = {key for key, closure in closures.items() if closure <= closures.keys()}
            stable_keys = {
                key: json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                for key in maskable_keys
            }
            ordered_keys = tuple(sorted(maskable_keys))
            point_index = joint_optimization.point_index
            base_candidate_mask = point_index.candidate_test_mask(seed_trace.test_file)
            top_impact_ids = index.structural_test_impact_ids(top_deletion_ids)
            top_test_mask = point_index.test_mask_for_objects(top_impact_ids)
            base_eligible_mask = base_candidate_mask & ~top_test_mask
            seed_test_mask = point_index.file_test_masks.get(seed_trace.test_file, 0)
            fixed_seed_affected_mask = seed_test_mask & point_index.observed_test_mask_for_objects(
                top_impact_ids
            )
            fixed_removed_lines = _unique_span_lines(top)
            possible_removed_lines = _unique_span_lines(
                (
                    *top,
                    *(item for key in ordered_keys for item in candidate_units[key]),
                )
            )
            if possible_removed_lines < minimum_code_lines:
                diagnostics = DeletionInfeasibilityDiagnostics(
                    status="PRECHECK_INFEASIBLE",
                    sufficient_constraint_core=("gold_patch_minimum_upper_bound",),
                    fixed_removed_lines=fixed_removed_lines,
                    candidate_removed_lines_upper_bound=possible_removed_lines,
                    minimum_removed_lines=minimum_code_lines,
                    maximum_removed_lines=maximum_code_lines,
                    candidate_units=len(ordered_keys),
                    base_eligible_p2p_test_points=base_eligible_mask.bit_count(),
                    minimum_p2p_test_points=joint_optimization.minimum_p2p_test_points,
                    maximum_p2p_test_points=joint_optimization.maximum_p2p_test_points,
                    seed_test_points=seed_test_mask.bit_count(),
                    fixed_seed_affected_test_points=fixed_seed_affected_mask.bit_count(),
                    minimum_f2p_test_points=joint_optimization.minimum_f2p_test_points,
                    maximum_f2p_test_points=joint_optimization.maximum_f2p_test_points,
                    required_seed_affected_test_points=None,
                    maximum_f2p_ratio=joint_optimization.maximum_f2p_ratio,
                )
                raise MaskIntervalInfeasible(
                    f"joint Mask/P2P line upper-bound precheck failed for "
                    f"{seed_trace.test_file}: {diagnostics.render()}",
                    diagnostics=(diagnostics,),
                )
            overlap_pairs: list[tuple[str, str]] = []
            for position, left_key in enumerate(ordered_keys):
                for right_key in ordered_keys[position + 1 :]:
                    if any(
                        _span_contains(left, right) or _span_contains(right, left)
                        for left in candidate_units[left_key]
                        for right in candidate_units[right_key]
                    ):
                        overlap_pairs.append((stable_keys[left_key], stable_keys[right_key]))
            units = tuple(
                DeletionUnit(
                    key=stable_keys[key],
                    removed_lines=_unique_span_lines(candidate_units[key]),
                    object_count=len(_collapse_nested(candidate_units[key])),
                    test_mask=point_index.test_mask_for_objects(
                        index.structural_test_impact_ids(candidate_impacts[key])
                    ),
                    seed_test_mask=(
                        seed_test_mask
                        & point_index.observed_test_mask_for_objects(
                            index.structural_test_impact_ids(candidate_impacts[key])
                        )
                    ),
                    files=tuple(
                        sorted(
                            {item.path for item in candidate_units[key]}
                            | {
                                edit.importer_path
                                for edit in self.import_edits(
                                    index,
                                    candidate_units[key],
                                    dynamic_imports,
                                )
                            }
                        )
                    ),
                    dependencies=tuple(
                        sorted(
                            stable_keys[dependency]
                            for dependency in closures[key] - {key}
                            if dependency in stable_keys
                        )
                    ),
                )
                for key in ordered_keys
            )
            try:
                optimized = optimize_deletions(
                    units=units,
                    overlap_pairs=tuple(overlap_pairs),
                    mandatory_unit_keys=frozenset(),
                    base_eligible_test_mask=base_eligible_mask,
                    fixed_removed_lines=fixed_removed_lines,
                    fixed_removed_objects=len(top),
                    fixed_files=frozenset(
                        {item.path for item in top}
                        | {
                            edit.importer_path
                            for edit in self.import_edits(index, top, dynamic_imports)
                        }
                    ),
                    seed_test_mask=seed_test_mask,
                    fixed_seed_affected_test_mask=fixed_seed_affected_mask,
                    minimum_removed_lines=minimum_code_lines,
                    maximum_removed_lines=maximum_code_lines,
                    minimum_f2p_test_points=joint_optimization.minimum_f2p_test_points,
                    maximum_f2p_test_points=joint_optimization.maximum_f2p_test_points,
                    maximum_f2p_ratio=joint_optimization.maximum_f2p_ratio,
                    maximum_seed_pass_rate=joint_optimization.maximum_seed_pass_rate,
                    minimum_p2p_test_points=joint_optimization.minimum_p2p_test_points,
                    maximum_p2p_test_points=joint_optimization.maximum_p2p_test_points,
                    timeout_seconds=joint_optimization.timeout_seconds,
                    workers=joint_optimization.workers,
                    random_seed=self.random_seed,
                )
            except DeletionOptimizationInfeasibleError as exc:
                raise MaskIntervalInfeasible(
                    f"joint Mask/P2P optimization failed for {seed_trace.test_file}: {exc}",
                    diagnostics=(exc.diagnostics,),
                ) from exc
            except ValueError as exc:
                raise CandidateRejected(
                    f"joint Mask/P2P optimization failed for {seed_trace.test_file}: {exc}"
                ) from exc
            keys_by_stable = {stable: key for key, stable in stable_keys.items()}
            selected_keys = {
                keys_by_stable[stable_key] for stable_key in optimized.selected_unit_keys
            }
            deleted_object_ids = top_deletion_ids | frozenset(
                object_id for key in selected_keys for object_id in candidate_impacts[key]
            )
            estimated_seed_affected_mask = (
                seed_test_mask
                & point_index.observed_test_mask_for_objects(
                    index.structural_test_impact_ids(deleted_object_ids)
                )
            )
            estimated_seed_test_points = seed_test_mask.bit_count()
            estimated_seed_affected_test_points = estimated_seed_affected_mask.bit_count()
            estimated_seed_pass_rate = (
                1 - estimated_seed_affected_test_points / estimated_seed_test_points
                if estimated_seed_test_points
                else 1.0
            )
            eligible_p2p_mask = point_index.limit_mask_to_best_files(
                optimized.eligible_test_mask,
                joint_optimization.maximum_selected_p2p_test_files,
            )
            selected_p2p_test_points = min(
                optimized.selected_p2p_test_points,
                eligible_p2p_mask.bit_count(),
            )
            if selected_p2p_test_points < joint_optimization.minimum_p2p_test_points:
                raise CandidateRejected(
                    f"{seed_trace.test_file} has only {selected_p2p_test_points} P2P points "
                    "after the physical-file limit"
                )
            estimated_f2p_ratio = (
                estimated_seed_affected_test_points
                / (estimated_seed_affected_test_points + selected_p2p_test_points)
                if estimated_seed_affected_test_points + selected_p2p_test_points
                else 1.0
            )
            if estimated_f2p_ratio > joint_optimization.maximum_f2p_ratio:
                raise CandidateRejected(
                    f"trace-estimated F2P ratio {estimated_f2p_ratio:.3f} exceeds "
                    f"{joint_optimization.maximum_f2p_ratio:.3f} for "
                    f"{seed_trace.test_file}"
                )
            optimized_p2p_test_ids = point_index.representative_test_ids(
                eligible_test_mask=eligible_p2p_mask,
                count=selected_p2p_test_points,
                deleted_object_ids=deleted_object_ids,
                maximum_files=joint_optimization.maximum_selected_p2p_test_files,
                random_seed=self.random_seed,
                seed_test_file=seed_trace.test_file,
            )
            p2p = P2PSelection(
                test_ids=optimized_p2p_test_ids,
                top_object_ids=tuple(sorted(effective_top_ids)),
            )
            p2p_ids = point_index.object_ids_for_tests(optimized_p2p_test_ids)
            protection = MaskProtection.build(
                index=index,
                p2p_object_ids=p2p_ids,
                collection_object_ids=collection_ids,
                critical_magic_methods=self.critical_magic_methods,
                seed_collection_object_ids=seed_import_object_ids(
                    index=index,
                    test_file=seed_trace.test_file,
                    dynamic_imports=seed_trace.imports,
                ),
            )
            selected_files = {
                point_index.test_paths[point_index.test_ids.index(test_id)]
                for test_id in optimized_p2p_test_ids
            }
            optimization_metrics = MaskOptimizationMetrics(
                status=optimized.status,
                solve_seconds=optimized.solve_seconds,
                minimum_lines=optimized.minimum_lines,
                maximum_lines=optimized.maximum_lines,
                center_distance=optimized.center_distance,
                modified_files=optimized.modified_files,
                eligible_p2p_test_points=optimized.eligible_test_points,
                excluded_p2p_test_points=optimized.excluded_test_points,
                selected_p2p_test_points=len(optimized_p2p_test_ids),
                selected_p2p_files=len(selected_files),
                estimated_seed_test_points=estimated_seed_test_points,
                estimated_seed_affected_test_points=estimated_seed_affected_test_points,
                estimated_f2p_ratio=estimated_f2p_ratio,
                estimated_seed_pass_rate=estimated_seed_pass_rate,
                candidate_units=len(units),
                selected_units=len(selected_keys),
                signature_groups=optimized.signature_groups,
            )
        selected = tuple(
            item.model_copy(update={"classification": "specific"})
            for item in _collapse_nested(
                item for key in selected_keys for item in candidate_units[key]
            )
        )
        targets = (*top, *selected)
        dependency_blockers = index.definition_time_blockers(_deletion_ids(index, targets))
        if dependency_blockers:
            raise ContractError(
                "optimized Mask violates definition-time dependency closure: "
                f"{tuple(dependency_blockers)[:5]!r}"
            )
        protected_conflicts = sorted(_deletion_ids(index, targets) & protection.protected_ids)
        if protected_conflicts:
            raise ContractError(
                f"optimized Mask intersects final P2P protection: {protected_conflicts[:5]!r}"
            )
        import_edits = self.import_edits(index, targets, dynamic_imports)
        candidate_ids = tuple(
            sorted(item.object_id for key in maskable_keys for item in candidate_units[key])
        )
        return MaskPlan(
            seed_test_file=seed_trace.test_file,
            gold_patch_tier=(
                joint_optimization.gold_patch_tier
                if joint_optimization is not None
                else "preferred"
            ),
            top_objects=top,
            specific_objects=selected,
            protected_objects=tuple(sorted(protection.protected_ids)),
            protection_reasons=dict(protection.reasons),
            dynamic_imports=dynamic_imports,
            import_edits=import_edits,
            candidate_object_ids=candidate_ids,
            p2p_test_ids=p2p.test_ids,
            optimization=optimization_metrics,
        )

    @staticmethod
    def import_edits(
        index: RepositoryIndex,
        targets: Iterable[SourceObject],
        dynamic_imports: Iterable[TraceImport] = (),
    ) -> tuple[ImportEdit, ...]:
        top_level_by_path: dict[str, set[str]] = defaultdict(set)
        for target in targets:
            if target.parent_qualname is None:
                top_level_by_path[target.path].add(target.name)
        removals: dict[tuple[str, int, int], set[str]] = defaultdict(set)
        for binding in index.data.imports:
            if is_test_path(binding.importer_path):
                continue
            deleted = top_level_by_path.get(binding.resolved_path or "", set())
            removals[(binding.importer_path, binding.line, binding.end_line)].update(
                name for name, _ in binding.names if name in deleted
            )
        bindings_by_location = {
            (binding.importer_path, binding.line): binding for binding in index.data.imports
        }
        for observed in dynamic_imports:
            if is_test_path(observed.importer_path):
                continue
            if observed.defined_name.rsplit(".", 1)[-1] not in top_level_by_path.get(
                observed.defined_path, set()
            ):
                continue
            binding = bindings_by_location.get((observed.importer_path, observed.importer_line))
            if binding is None or not any(
                imported_name == observed.imported_name for imported_name, _ in binding.names
            ):
                continue
            removals[(binding.importer_path, binding.line, binding.end_line)].add(
                observed.imported_name
            )
        return tuple(
            ImportEdit(
                importer_path=path,
                line=line,
                end_line=end_line,
                remove_names=tuple(sorted(names)),
            )
            for (path, line, end_line), names in sorted(removals.items())
            if names
        )

    def apply_increment(
        self,
        *,
        index: RepositoryIndex,
        plan: MaskPlan,
        proposal: MaskRepairProposal,
        traces: Mapping[str, TraceBundle] | None = None,
        learned_protected_ids: Iterable[str] = (),
    ) -> MaskPlan:
        """Apply one sparse Agent increment and derive the smallest valid structural closure."""
        violations: list[RepairViolation] = []
        current_target_ids = {item.object_id for item in plan.targets}
        currently_deleted = _deletion_ids(index, plan.targets)
        known_ids = set(index.by_id)
        eligible_ids = set(plan.candidate_object_ids) | current_target_ids

        for object_id in proposal.must_restore_object_ids:
            if object_id not in known_ids:
                violations.append(
                    RepairViolation(
                        code="repair.object_unknown",
                        field="must_restore_object_ids",
                        object_id=object_id,
                        message=f"Unknown repository object ID: {object_id}",
                        correction="Copy an exact ID from mask-objects.jsonl.",
                    )
                )
            elif object_id not in current_target_ids:
                violations.append(
                    RepairViolation(
                        code="repair.restore_not_current_target",
                        field="must_restore_object_ids",
                        object_id=object_id,
                        message=f"Object is not an explicit target in the base attempt: {object_id}",
                        correction="Restore only IDs listed in the base feedback mask-objects.jsonl.",
                    )
                )
        for object_id in proposal.must_mask_object_ids:
            if object_id not in known_ids:
                violations.append(
                    RepairViolation(
                        code="repair.object_unknown",
                        field="must_mask_object_ids",
                        object_id=object_id,
                        message=f"Unknown repository object ID: {object_id}",
                        correction="Copy an exact eligible ID from catalog/objects.jsonl.",
                    )
                )
            elif object_id in currently_deleted:
                violations.append(
                    RepairViolation(
                        code="repair.mask_already_effective",
                        field="must_mask_object_ids",
                        object_id=object_id,
                        message=f"Object is already deleted by the base Mask Plan: {object_id}",
                        correction="Choose a currently unmasked eligible object.",
                    )
                )
            elif object_id not in eligible_ids:
                violations.append(
                    RepairViolation(
                        code="repair.mask_not_eligible",
                        field="must_mask_object_ids",
                        object_id=object_id,
                        message=f"Object is outside the base attempt's Mask candidate inventory: {object_id}",
                        correction="Choose an object whose catalog record has eligible_for_mask=true.",
                    )
                )
        if violations:
            raise MaskRepairProposalError(tuple(violations))

        restore_objects = _expand_binding_ids(index, proposal.must_restore_object_ids)
        mask_objects = _expand_binding_ids(index, proposal.must_mask_object_ids)
        restore = {item.object_id for item in restore_objects}
        requested_mask = {item.object_id for item in mask_objects}
        expanded_overlap = restore & requested_mask
        if expanded_overlap:
            raise MaskRepairProposalError(
                (
                    RepairViolation(
                        code="repair.atomic_binding_conflict",
                        message=(
                            "Restore and mask requests expand to the same atomic binding: "
                            f"{sorted(expanded_overlap)[:5]}"
                        ),
                        correction="Keep every overload or atomic binding group on one side only.",
                    ),
                )
            )

        p2p_test_ids = plan.p2p_test_ids
        p2p_protected = _object_ids_for_tests(
            index,
            traces.values() if traces is not None else (),
            p2p_test_ids,
        )
        collection_protected = {
            object_id
            for object_id, reasons in plan.protection_reasons.items()
            if "collection_trace" in reasons
        }
        protection = MaskProtection.build(
            index=index,
            p2p_object_ids=p2p_protected,
            collection_object_ids=collection_protected,
            critical_magic_methods=self.critical_magic_methods,
            additional_protected_ids=(
                *(plan.protected_objects if traces is None else ()),
                *learned_protected_ids,
            ),
        )
        protected_mask = requested_mask & protection.protected_ids
        if protected_mask:
            raise MaskRepairProposalError(
                tuple(
                    RepairViolation(
                        code="repair.mask_protected",
                        field="must_mask_object_ids",
                        object_id=object_id,
                        message=f"Object is protected by the fixed test/runtime contract: {object_id}",
                        correction="Choose a feature-specific unprotected replacement object.",
                    )
                    for object_id in sorted(protected_mask)
                )
            )
        previous = {item.object_id: item for item in plan.targets}

        def materialize(
            object_ids: Iterable[str], classification: Literal["top", "specific"]
        ) -> tuple[SourceObject, ...]:
            groups = _binding_groups(index)
            selected: dict[str, SourceObject] = {}
            seen_groups: set[tuple[str, str]] = set()
            for object_id in object_ids:
                item = index.by_id.get(object_id)
                if item is None:
                    continue
                key = (item.path, item.qualname)
                if key in seen_groups:
                    continue
                seen_groups.add(key)
                unit = groups[key]
                if any(peer.object_id in protection.protected_ids for peer in unit):
                    continue
                for peer in unit:
                    selected[peer.object_id] = previous.get(peer.object_id, peer).model_copy(
                        update={"classification": classification}
                    )
            return tuple(
                sorted(
                    selected.values(),
                    key=lambda item: (item.path, item.start_line, item.object_id),
                )
            )

        retained_top_ids = {
            item.object_id for item in plan.top_objects if item.object_id not in restore
        }
        promoted_mask_ids: set[str] = requested_mask if not retained_top_ids else set()
        top = list(materialize(retained_top_ids | promoted_mask_ids, "top"))
        top = list(_collapse_nested(top))
        top_ids = {item.object_id for item in top}
        retained_specific_ids = {
            item.object_id
            for item in plan.specific_objects
            if item.object_id not in restore and item.object_id not in top_ids
        }
        specific_ids = set(retained_specific_ids) | (requested_mask - promoted_mask_ids)
        specific = list(materialize(specific_ids - top_ids, "specific"))
        specific = list(_collapse_nested(specific))
        specific = [
            item
            for item in specific
            if not any(
                _span_contains(target, item) or _span_contains(item, target) for target in top
            )
        ]
        if not top:
            raise MaskRepairProposalError(
                (
                    RepairViolation(
                        code="repair.empty_top",
                        message="The increment removes every Top object without a valid replacement.",
                        correction=(
                            "Keep one current Top object or add an eligible must_mask object that can "
                            "be promoted to Top."
                        ),
                    ),
                )
            )
        targets = (*top, *specific)
        resulting_deletions = _deletion_ids(index, targets)
        missing_requested = requested_mask - resulting_deletions
        if missing_requested:
            raise MaskRepairProposalError(
                tuple(
                    RepairViolation(
                        code="repair.mask_not_materialized",
                        field="must_mask_object_ids",
                        object_id=object_id,
                        message=f"Requested object could not be materialized in the Mask Plan: {object_id}",
                        correction="Choose an unprotected object outside conflicting structural spans.",
                    )
                    for object_id in sorted(missing_requested)
                )
            )
        blockers = index.definition_time_blockers(resulting_deletions)
        if blockers:
            examples = tuple(blockers.items())[:5]
            raise MaskRepairProposalError(
                (
                    RepairViolation(
                        code="repair.definition_time_blocker",
                        message=f"Increment violates definition-time dependency closure: {examples!r}",
                        correction="Restore the blocked definition or choose a different mask object.",
                    ),
                )
            )
        revised = plan.model_copy(
            update={
                "revision": plan.revision + 1,
                "top_objects": tuple(
                    item.model_copy(update={"classification": "top"}) for item in top
                ),
                "specific_objects": tuple(specific),
                "protected_objects": tuple(sorted(protection.protected_ids)),
                "protection_reasons": dict(protection.reasons),
                "import_edits": self.import_edits(index, targets, plan.dynamic_imports),
                "p2p_test_ids": p2p_test_ids,
                "candidate_object_ids": tuple(
                    sorted(eligible_ids | current_target_ids | requested_mask)
                ),
                "optimization": None,
            }
        )
        if revised.model_copy(update={"revision": plan.revision}) == plan:
            raise MaskRepairProposalError(
                (
                    RepairViolation(
                        code="repair.no_effect",
                        message="The increment produces no effective Mask Plan change.",
                        correction="Change at least one currently effective object constraint.",
                    ),
                )
            )
        return revised


def _covered(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(left <= start and end <= right for left, right in ranges)


def _suite_repairs(source: str, ranges: Sequence[tuple[int, int]]) -> dict[int, str]:
    """Insert pass only when full declaration removal empties a surviving Python suite."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines(keepends=True)
    repairs: dict[int, str] = {}

    def inspect_suite(owner_start: int, owner_end: int, body: list[ast.stmt]) -> None:
        if not body or _covered(owner_start, owner_end, ranges):
            return
        if all(_covered(item.lineno, item.end_lineno or item.lineno, ranges) for item in body):
            line = body[0].lineno
            raw = lines[line - 1] if line <= len(lines) else ""
            indentation = raw[: len(raw) - len(raw.lstrip())]
            newline = "\n" if raw.endswith("\n") else ""
            repairs[line] = f"{indentation}pass{newline}"

    for node in ast.walk(tree):
        for attribute in ("body", "orelse", "finalbody"):
            body_value: object = getattr(node, attribute, None)
            if isinstance(node, ast.stmt) and isinstance(body_value, list):
                body_items = cast(list[object], body_value)
                if not body_items or not all(isinstance(item, ast.stmt) for item in body_items):
                    continue
                inspect_suite(
                    node.lineno,
                    node.end_lineno or node.lineno,
                    cast(list[ast.stmt], body_items),
                )
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                inspect_suite(handler.lineno, handler.end_lineno or handler.lineno, handler.body)
        if isinstance(node, ast.Match):
            for case in node.cases:
                if case.body:
                    end = case.body[-1].end_lineno or case.body[-1].lineno
                    inspect_suite(case.pattern.lineno, end, case.body)
    return repairs


def _rewrite_import(source: str, edit: ImportEdit) -> str | None:
    lines = source.splitlines(keepends=True)
    segment = "".join(lines[edit.line - 1 : edit.end_line])
    try:
        node = ast.parse(textwrap.dedent(segment)).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(node, ast.ImportFrom):
        return None
    remaining = [alias for alias in node.names if alias.name not in set(edit.remove_names)]
    if not remaining:
        return ""
    indentation = lines[edit.line - 1][
        : len(lines[edit.line - 1]) - len(lines[edit.line - 1].lstrip())
    ]
    module = "." * node.level + (node.module or "")
    rendered_names = ", ".join(
        alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in remaining
    )
    newline = "\n" if segment.endswith("\n") else ""
    return f"{indentation}from {module} import {rendered_names}{newline}"


def _transform_declarations(
    sources: Mapping[str, str],
    *,
    targets: Sequence[SourceObject],
    import_edits: Sequence[ImportEdit] = (),
) -> dict[str, str]:
    """Delete exact declarations and validate every changed Python source."""
    result = dict(sources)
    targets_by_path: dict[str, list[SourceObject]] = defaultdict(list)
    edits_by_path: dict[str, list[ImportEdit]] = defaultdict(list)
    for target in targets:
        targets_by_path[target.path].append(target)
    for edit in import_edits:
        edits_by_path[edit.importer_path].append(edit)
    for path in sorted(set(targets_by_path) | set(edits_by_path)):
        source = sources.get(path)
        if source is None:
            raise CandidateRejected(f"mask source disappeared: {path}")
        lines = source.splitlines(keepends=True)
        ranges = sorted({(item.start_line, item.end_line) for item in targets_by_path[path]})
        collapsed: list[tuple[int, int]] = []
        for start, end in ranges:
            if any(
                left <= start and end <= right
                for left, right in ranges
                if (left, right) != (start, end)
            ):
                continue
            collapsed.append((start, end))
        replacements: list[tuple[int, int, str]] = []
        for edit in edits_by_path[path]:
            rendered = _rewrite_import(source, edit)
            if rendered is not None:
                replacements.append((edit.line, edit.end_line, rendered))
        for start, end in collapsed:
            replacements.append((start, end, ""))
        blank_ranges = list(collapsed)
        for start, end, replacement in replacements:
            if not replacement:
                blank_ranges.append((start, end))
        repairs = _suite_repairs(source, blank_ranges)
        for start, end, replacement in sorted(replacements, reverse=True):
            if (
                any(left <= start and end <= right for left, right in collapsed)
                and (start, end) not in collapsed
            ):
                continue
            newline_count = end - start + 1
            if not replacement:
                replacement_lines = [
                    "\n" if lines[index].endswith("\n") else "" for index in range(start - 1, end)
                ]
            else:
                replacement_lines = [replacement]
                replacement_lines.extend("\n" for _ in range(max(0, newline_count - 1)))
            lines[start - 1 : end] = replacement_lines
        for line, replacement in repairs.items():
            lines[line - 1] = replacement
        transformed = "".join(lines)
        try:
            ast.parse(transformed, filename=path)
        except SyntaxError as exc:
            raise CandidateRejected(f"structural deletion made {path} invalid: {exc}") from exc
        result[path] = transformed
    transformed_index = RepositoryIndex.from_sources(result)
    remaining = {item.object_id for item in transformed_index.data.definitions}
    leaked = [item.object_id for item in targets if item.object_id in remaining]
    if leaked:
        raise CandidateRejected(
            f"structural deletion left target declarations behind: {leaked[:5]}"
        )
    return result


def transform_sources(sources: Mapping[str, str], plan: MaskPlan) -> dict[str, str]:
    """Apply a production mask without ever leaving target declarations behind."""
    return _transform_declarations(
        sources,
        targets=plan.targets,
        import_edits=plan.import_edits,
    )


def _test_declaration(
    index: RepositoryIndex,
    test_id: str,
    test_path: str,
    test_source_paths: Mapping[str, str],
) -> SourceObject | None:
    """Resolve a pytest node ID to its longest indexed declaration prefix."""
    # The public occurrence suffix addresses one collected case, not a Python
    # declaration. The canonical parser keeps this consumer aligned with the
    # Environment Runner and the tracing plugin.
    declaration_id = parse_pytest_public_id(test_id).native_base
    parts = declaration_id.split("::")
    if len(parts) < 2:
        return None
    source_path = test_source_paths.get(test_path)
    if source_path is None:
        return None
    names = [re.sub(r"\[.*$", "", part) for part in parts[1:]]
    for stop in range(len(names), 0, -1):
        item = index.by_qualname.get((source_path, ".".join(names[:stop])))
        if item is not None and item.kind in {"function", "async_function"}:
            return item
    if len(names) < 2:
        return None

    class_item = index.by_qualname.get((source_path, ".".join(names[:-1])))
    if class_item is None or class_item.kind != "class":
        return None
    method_name = names[-1]
    visited: set[str] = set()

    def inherited_method(owner: SourceObject) -> SourceObject | None:
        if owner.object_id in visited:
            return None
        visited.add(owner.object_id)
        parent = owner.qualname.rpartition(".")[0]
        for base in owner.bases:
            if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", base) is None:
                continue
            candidates = tuple(
                dict.fromkeys(
                    (
                        base,
                        f"{parent}.{base}" if parent else base,
                        base.rsplit(".", 1)[-1],
                    )
                )
            )
            base_item = next(
                (
                    candidate
                    for qualname in candidates
                    if (candidate := index.by_qualname.get((source_path, qualname))) is not None
                    and candidate.kind == "class"
                ),
                None,
            )
            if base_item is None:
                continue
            method = index.by_qualname.get((base_item.path, f"{base_item.qualname}.{method_name}"))
            if method is not None and method.kind in {"function", "async_function"}:
                return method
            inherited = inherited_method(base_item)
            if inherited is not None:
                return inherited
        return None

    return inherited_method(class_item)


def hide_test_points(
    sources: Mapping[str, str],
    *,
    index: RepositoryIndex,
    test_ids: Sequence[str],
    test_paths: Mapping[str, str],
    test_source_paths: Mapping[str, str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Hide only declarations addressed by F2P node IDs, never whole test files."""
    targets: dict[str, SourceObject] = {}
    unresolved: list[str] = []
    for test_id in test_ids:
        test_path = test_paths.get(test_id)
        item = (
            _test_declaration(index, test_id, test_path, test_source_paths)
            if test_path is not None
            else None
        )
        if item is None or not is_test_path(item.path):
            unresolved.append(test_id)
        else:
            targets[item.object_id] = item
    if unresolved:
        raise CandidateRejected(
            f"F2P node IDs could not be mapped to removable test declarations: {unresolved[:10]}"
        )
    if not targets:
        raise CandidateRejected("no F2P test declarations were selected for hiding")
    transformed = _transform_declarations(
        sources,
        targets=tuple(targets.values()),
    )
    return transformed, tuple(sorted({item.path for item in targets.values()}))


class MaskExecutor:
    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    async def apply(self, session: ContainerSessionProtocol, plan: MaskPlan) -> None:
        await session.clean_workspace()
        transformed = transform_sources(self.index.sources, plan)
        for path in plan.patch_files:
            if path in transformed and transformed[path] != self.index.sources[path]:
                await session.write_file(path, transformed[path].encode("utf-8"))
