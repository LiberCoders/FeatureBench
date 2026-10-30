"""Deterministic BFS scheduling and bounded evidence for Top classification."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from codeworld.adapters.featurebench.indexing import RepositoryIndex, is_test_path
from codeworld.adapters.featurebench.models import (
    SourceObject,
    TopBatchClassification,
    TopCandidateDecision,
    TraceBundle,
)
from codeworld.agents.feedback import FeedbackIssue
from codeworld.environment.frameworks.pytest import parse_pytest_public_id

_PROMPT_DIRECTORY = Path(__file__).resolve().parent / "assets" / "prompts"


@cache
def load_top_prompt(name: str) -> str:
    """Load one immutable Top-classification prompt asset."""
    path = _PROMPT_DIRECTORY / f"top.{name}.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read FeatureBench Top prompt asset {path}: {exc}") from exc
    if not content:
        raise ValueError(f"FeatureBench Top prompt asset is empty: {path}")
    return content


@dataclass(slots=True)
class EdgeEvidence:
    calls: int = 0
    test_ids: set[str] = field(default_factory=lambda: set[str]())
    provenance: set[str] = field(default_factory=lambda: set[str]())


@dataclass(frozen=True, slots=True)
class TopClassificationBatch:
    """One immutable model input produced by the deterministic traversal."""

    depth: int
    candidate_ids: tuple[str, ...]
    payload: dict[str, object]


def feature_graph_root_ids(
    *,
    index: RepositoryIndex,
    trace: TraceBundle,
    test_ids: tuple[str, ...],
    objects: Mapping[str, SourceObject],
    children: Mapping[str, set[str]],
) -> frozenset[str]:
    """Resolve test functions and independently invoked local fixtures as graph roots."""
    expected_qualnames = {
        ".".join(
            part.split("[", 1)[0]
            for part in parse_pytest_public_id(test_id).native_base.split("::")[1:]
        )
        for test_id in test_ids
    }
    roots = {
        item.object_id
        for item in objects.values()
        if item.path == trace.test_file and item.qualname in expected_qualnames
    }
    selected_test_ids = frozenset(test_ids)
    attributed_local_ids = {
        item.object_id
        for record in trace.records
        if selected_test_ids.intersection(record.test_ids)
        if (item := index.resolve_trace(record)) is not None
        if item.object_id in objects and item.path == trace.test_file
    }
    called_ids = {callee for callees in children.values() for callee in callees}
    roots.update(attributed_local_ids - called_ids)
    if roots:
        return frozenset(roots)
    return frozenset(
        item.object_id
        for item in objects.values()
        if item.path == trace.test_file and item.qualname.rsplit(".", 1)[-1].startswith("test")
    )


class FeatureGraphTraversal:
    """Own BFS order and stopping rules; the model only labels supplied candidates."""

    def __init__(
        self,
        *,
        index: RepositoryIndex,
        trace: TraceBundle,
        test_ids: tuple[str, ...],
        max_depth: int,
        batch_size: int,
        critical_magic_methods: frozenset[str],
        protected_object_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.index = index
        self.trace = trace
        self.test_ids = test_ids
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.critical_magic_methods = critical_magic_methods
        self.objects = {item.object_id: item for item in self.index.traced_objects(trace.records)}
        self.children: dict[str, set[str]] = defaultdict(set)
        self.edge_evidence: dict[tuple[str, str], EdgeEvidence] = {}
        for edge in trace.edges:
            caller = index.normalize_runtime_id(edge.caller)
            callee = index.normalize_runtime_id(edge.callee)
            if caller not in self.objects or callee not in self.objects:
                continue
            self.children[caller].add(callee)
            evidence = self.edge_evidence.setdefault((caller, callee), EdgeEvidence())
            evidence.calls += edge.calls
            evidence.test_ids.update(edge.test_ids)
            evidence.provenance.update(edge.provenance)
        if not trace.edges:
            for item in self.objects.values():
                for raw_caller in item.callers:
                    caller = index.normalize_runtime_id(raw_caller) or raw_caller
                    if caller in self.objects:
                        self.children[caller].add(item.object_id)
        self.protected = set(index.protected_ids(critical_magic_methods)) | set(
            protected_object_ids
        )
        self.roots = feature_graph_root_ids(
            index=index,
            trace=trace,
            test_ids=test_ids,
            objects=self.objects,
            children=self.children,
        )
        self.frontier = tuple(sorted(self.roots))
        self.visited = set(self.roots)
        self.depth = 0
        self.layer_nodes: tuple[str, ...] = ()
        self.remaining_candidates: list[str] = []
        self.decisions: dict[str, TopCandidateDecision] = {}
        self.parent: dict[str, str] = {}
        self.done = False
        self._prepare_layer()

    def _classifiable(self, item: SourceObject) -> bool:
        return (
            not is_test_path(item.path)
            and item.object_id not in self.protected
            and item.name not in self.critical_magic_methods
        )

    def _prepare_layer(self) -> None:
        while self.frontier and self.depth < self.max_depth:
            next_nodes: set[str] = set()
            for caller in self.frontier:
                for child in self.children.get(caller, ()):
                    if child in self.visited:
                        continue
                    self.visited.add(child)
                    self.parent[child] = caller
                    next_nodes.add(child)
            if not next_nodes:
                self.done = True
                return
            self.layer_nodes = tuple(sorted(next_nodes))
            self.remaining_candidates = sorted(
                node
                for node in next_nodes
                if node in self.objects and self._classifiable(self.objects[node])
            )
            if self.remaining_candidates:
                return
            self.frontier = self.layer_nodes
            self.depth += 1
        self.done = True

    def _advance_layer(self) -> None:
        selected_here = {
            node
            for node in self.layer_nodes
            if self.decisions.get(node) is not None and self.decisions[node].classification == "top"
        }
        self.frontier = tuple(node for node in self.layer_nodes if node not in selected_here)
        self.depth += 1
        self.layer_nodes = ()
        self.remaining_candidates = []
        self._prepare_layer()

    def _call_path(self, object_id: str) -> tuple[str, ...]:
        result = [object_id]
        cursor = object_id
        while cursor in self.parent:
            cursor = self.parent[cursor]
            result.append(cursor)
        result.reverse()
        return tuple(result)

    def _source(self, item: SourceObject) -> str:
        source = self.index.sources.get(item.path, "")
        return "\n".join(source.splitlines()[item.start_line - 1 : item.end_line])

    def _candidate_payload(self, object_id: str) -> dict[str, object]:
        item = self.objects[object_id]
        calls = 0
        traced_test_ids: set[str] = set()
        provenance: set[str] = set()
        for caller in sorted(
            candidate for candidate, children in self.children.items() if object_id in children
        ):
            raw = self.edge_evidence.get((caller, object_id), EdgeEvidence())
            calls += raw.calls
            traced_test_ids.update(raw.test_ids)
            provenance.update(raw.provenance)
        return {
            "object_id": item.object_id,
            "source": self._source(item),
            "call_path": self._call_path(object_id),
            "dynamic_evidence": {
                "calls": calls,
                "test_points": len(traced_test_ids),
                "provenance": sorted(provenance),
            },
        }

    def selected_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                object_id
                for object_id, decision in self.decisions.items()
                if decision.classification == "top"
            )
        )

    def _test_context(self) -> dict[str, object]:
        return {
            "test_file": self.trace.test_file,
            "test_point_count": len(self.test_ids),
            "test_source": self.index.sources.get(self.trace.test_file, ""),
        }

    def test_context(self) -> dict[str, object]:
        """Return the bounded context shared by all batches for audit and tests."""
        return self._test_context()

    def layer_batches(self) -> tuple[TopClassificationBatch, ...]:
        """Materialize every immutable batch in the current BFS layer."""
        if self.done:
            raise RuntimeError("Top traversal is complete")
        test_context = self._test_context()
        chunks = tuple(
            tuple(self.remaining_candidates[offset : offset + self.batch_size])
            for offset in range(0, len(self.remaining_candidates), self.batch_size)
        )
        return tuple(
            TopClassificationBatch(
                depth=self.depth + 1,
                candidate_ids=candidate_ids,
                payload={
                    "test_context": test_context,
                    "classification_batch": {
                        "depth": self.depth + 1,
                        "remaining_candidates_at_depth": len(self.remaining_candidates),
                        "batch_index": index,
                        "batch_count": len(chunks),
                        "candidates": [
                            self._candidate_payload(object_id) for object_id in candidate_ids
                        ],
                    },
                },
            )
            for index, candidate_ids in enumerate(chunks, start=1)
        )

    def validation_issues(
        self,
        batch: TopClassificationBatch,
        classification: TopBatchClassification,
    ) -> tuple[FeedbackIssue, ...]:
        expected = set(batch.candidate_ids)
        actual = [decision.object_id for decision in classification.decisions]
        issues: list[FeedbackIssue] = []
        if batch.depth != self.depth + 1 or not expected.issubset(self.remaining_candidates):
            raise RuntimeError("Top classification batch is no longer current")
        if len(actual) != len(set(actual)):
            issues.append(
                FeedbackIssue(
                    code="featurebench.top.duplicate_decision",
                    message="The batch contains duplicate object decisions.",
                    correction="Return exactly one decision for every supplied object_id.",
                )
            )
        if set(actual) != expected:
            issues.append(
                FeedbackIssue(
                    code="featurebench.top.batch_mismatch",
                    message=f"Expected {sorted(expected)}, received {sorted(set(actual))}.",
                    correction="Copy every exact object_id from the input batch once.",
                )
            )
        return tuple(issues)

    def apply(
        self,
        batch: TopClassificationBatch,
        classification: TopBatchClassification,
    ) -> None:
        issues = self.validation_issues(batch, classification)
        if issues:
            raise ValueError("cannot apply an invalid Top classification batch")
        for decision in classification.decisions:
            self.decisions[decision.object_id] = decision
            self.remaining_candidates.remove(decision.object_id)
        if not self.remaining_candidates:
            self._advance_layer()
