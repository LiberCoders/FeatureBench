"""Point-grained P2P selection over a compact inverted trace bitmap index."""

from __future__ import annotations

import ast
import random
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from heapq import heappop, heappush
from types import MappingProxyType

from codeworld.adapters.featurebench.indexing import RepositoryIndex, is_test_path
from codeworld.adapters.featurebench.models import P2PSelection, TraceBundle, TraceImport
from codeworld.errors import CandidateRejected, ContractError


def _is_cached_definition(decorators: Iterable[str]) -> bool:
    """Recognize standard memoization decorators without importing repository code."""
    cache_names = {"cache", "lru_cache", "cached_property"}
    for decorator in decorators:
        expression = decorator.split("(", 1)[0].strip()
        if expression.rsplit(".", 1)[-1] in cache_names:
            return True
    return False


def _statically_accessed_attribute_names(source: str) -> frozenset[str]:
    """Find object attributes whose existence a test can require without calling them."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    reflective_calls = {"getattr", "hasattr", "setattr", "delattr"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        callable_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        attribute = node.args[1]
        if (
            callable_name in reflective_calls
            and isinstance(attribute, ast.Constant)
            and isinstance(attribute.value, str)
        ):
            names.add(attribute.value)
    return frozenset(names)


def seed_import_object_ids(
    *,
    index: RepositoryIndex,
    test_file: str,
    dynamic_imports: Iterable[TraceImport] = (),
) -> frozenset[str]:
    """Resolve declarations that must exist for one test module to import."""
    definitions_by_path_name: dict[tuple[str, str], set[str]] = {}
    for item in index.data.definitions:
        if item.parent_qualname is None:
            definitions_by_path_name.setdefault((item.path, item.name), set()).add(item.object_id)
    protected: set[str] = set()
    for binding in index.data.imports:
        if binding.importer_path != test_file or binding.resolved_path is None:
            continue
        for imported_name, _alias in binding.names:
            names = (
                index.data.exports.get(binding.resolved_path, ())
                if imported_name == "*"
                else (imported_name,)
            )
            for name in names:
                protected.update(definitions_by_path_name.get((binding.resolved_path, name), ()))
    protected.update(
        trace_import_object_ids(
            index=index,
            dynamic_imports=(
                imported for imported in dynamic_imports if imported.importer_path == test_file
            ),
        )
    )
    return frozenset(protected)


def trace_import_object_ids(
    *,
    index: RepositoryIndex,
    dynamic_imports: Iterable[TraceImport],
) -> frozenset[str]:
    """Resolve exact repository declarations observed by runtime import tracing."""
    protected: set[str] = set()
    for imported in dynamic_imports:
        item = index.by_qualname.get((imported.defined_path, imported.defined_name))
        if item is None and imported.defined_line is not None:
            item = index.by_runtime_key.get(
                (imported.defined_path, imported.defined_name, imported.defined_line)
            )
        if item is not None:
            protected.add(item.object_id)
    return frozenset(protected)


@dataclass(frozen=True, slots=True)
class TracePointIndex:
    """Immutable object-to-test-point incidence index backed by integer bitmaps."""

    test_ids: tuple[str, ...]
    object_test_masks: Mapping[str, int]
    observed_object_test_masks: Mapping[str, int]
    file_test_masks: Mapping[str, int]
    test_paths: tuple[str, ...]
    all_test_mask: int
    traced_associations: int

    @classmethod
    def build(
        cls,
        *,
        index: RepositoryIndex,
        traces: Iterable[TraceBundle],
        candidate_test_ids: Iterable[str],
        test_paths: Mapping[str, str],
    ) -> TracePointIndex:
        """Build one canonical index for every successfully traced test point."""
        test_ids = tuple(sorted(candidate_test_ids))
        if len(test_ids) != len(set(test_ids)):
            raise ContractError("P2P trace-point inventory contains duplicate test IDs")
        missing_paths = tuple(test_id for test_id in test_ids if test_id not in test_paths)
        if missing_paths:
            raise ContractError(f"P2P trace-point inventory has no path for {missing_paths[:20]!r}")

        test_bits = {test_id: 1 << ordinal for ordinal, test_id in enumerate(test_ids)}
        file_test_masks: dict[str, int] = {}
        for test_id, bit in test_bits.items():
            path = test_paths[test_id]
            file_test_masks[path] = file_test_masks.get(path, 0) | bit

        object_test_masks: dict[str, int] = {}
        observed_object_test_masks: dict[str, int] = {}
        for bundle in traces:
            shared_file_mask = file_test_masks.get(bundle.test_file, 0)
            resolved_records: dict[str, object] = {}
            for record in bundle.records:
                item = index.resolve_trace(record)
                if item is None or is_test_path(item.path):
                    continue
                resolved_records[item.object_id] = item
                mask = 0
                for test_id in record.test_ids:
                    mask |= test_bits.get(test_id, 0)
                if mask:
                    observed_object_test_masks[item.object_id] = (
                        observed_object_test_masks.get(item.object_id, 0) | mask
                    )
                    object_test_masks[item.object_id] = (
                        object_test_masks.get(item.object_id, 0) | mask
                    )

            # A frame may be present in the merged trace without point IDs while
            # its incoming/outgoing call edge still has precise point attribution.
            # Top traversal consumes those edges, so the P2P incidence index must
            # consume the same evidence or the two views can disagree about whether
            # a selected Top object is traceable.
            for edge in bundle.edges:
                mask = 0
                for test_id in edge.test_ids:
                    mask |= test_bits.get(test_id, 0)
                if not mask:
                    continue
                for runtime_id in (edge.caller, edge.callee):
                    object_id = index.normalize_runtime_id(runtime_id)
                    if object_id is None:
                        continue
                    item = index.by_id[object_id]
                    if is_test_path(item.path):
                        continue
                    observed_object_test_masks[object_id] = (
                        observed_object_test_masks.get(object_id, 0) | mask
                    )
                    object_test_masks[object_id] = object_test_masks.get(object_id, 0) | mask

            # Batched traces can hide cold-start dependencies behind a cache warmed
            # by an earlier point.  Only memoized roots and their observed dynamic
            # descendants become file-shared; unrelated per-point calls stay precise.
            children: dict[str, set[str]] = {}
            for edge in bundle.edges:
                caller = index.normalize_runtime_id(edge.caller)
                callee = index.normalize_runtime_id(edge.callee)
                if caller is not None and callee is not None:
                    children.setdefault(caller, set()).add(callee)
            shared_objects = {
                object_id
                for object_id in resolved_records
                if _is_cached_definition(index.by_id[object_id].decorators)
            }
            pending = list(shared_objects)
            while pending:
                current = pending.pop()
                for child in children.get(current, ()):
                    if child not in shared_objects:
                        shared_objects.add(child)
                        pending.append(child)

            # Attribute lookup, monkeypatching, and reflection can require a
            # declaration to exist without executing its body, so no dynamic frame
            # can reveal the dependency.  Project such test-file references onto all
            # matching repository declarations conservatively.
            attribute_names = _statically_accessed_attribute_names(
                index.sources.get(bundle.test_file, "")
            )
            shared_objects.update(
                item.object_id
                for item in index.data.definitions
                if not is_test_path(item.path) and item.name in attribute_names
            )
            if shared_file_mask:
                for object_id in shared_objects:
                    object_test_masks[object_id] = (
                        object_test_masks.get(object_id, 0) | shared_file_mask
                    )
            for imported in bundle.imports:
                if not is_test_path(imported.importer_path):
                    continue
                imported_object = index.by_qualname.get(
                    (imported.defined_path, imported.defined_name)
                )
                if imported_object is None and imported.defined_line is not None:
                    imported_object = index.by_runtime_key.get(
                        (
                            imported.defined_path,
                            imported.defined_name,
                            imported.defined_line,
                        )
                    )
                if imported_object is None:
                    continue
                mask = 0
                for test_id in imported.test_ids:
                    mask |= test_bits.get(test_id, 0)
                if not mask:
                    mask = file_test_masks.get(imported.importer_path, 0)
                if mask:
                    object_test_masks[imported_object.object_id] = (
                        object_test_masks.get(imported_object.object_id, 0) | mask
                    )

        definitions_by_path_name: dict[tuple[str, str], list[str]] = {}
        for item in index.data.definitions:
            if item.parent_qualname is None:
                definitions_by_path_name.setdefault((item.path, item.name), []).append(
                    item.object_id
                )
        for binding in index.data.imports:
            if not is_test_path(binding.importer_path) or binding.resolved_path is None:
                continue
            mask = file_test_masks.get(binding.importer_path, 0)
            if not mask:
                continue
            for imported_name, _alias in binding.names:
                names = (
                    index.data.exports.get(binding.resolved_path, ())
                    if imported_name == "*"
                    else (imported_name,)
                )
                for name in names:
                    for object_id in definitions_by_path_name.get(
                        (binding.resolved_path, name), ()
                    ):
                        object_test_masks[object_id] = object_test_masks.get(object_id, 0) | mask

        return cls(
            test_ids=test_ids,
            object_test_masks=MappingProxyType(object_test_masks),
            observed_object_test_masks=MappingProxyType(observed_object_test_masks),
            file_test_masks=MappingProxyType(file_test_masks),
            test_paths=tuple(test_paths[test_id] for test_id in test_ids),
            all_test_mask=(1 << len(test_ids)) - 1,
            traced_associations=sum(mask.bit_count() for mask in object_test_masks.values()),
        )

    def _test_ids_from_mask(self, mask: int) -> list[str]:
        selected: list[str] = []
        while mask:
            least_significant_bit = mask & -mask
            ordinal = least_significant_bit.bit_length() - 1
            selected.append(self.test_ids[ordinal])
            mask ^= least_significant_bit
        return selected

    def object_ids_for_tests(self, test_ids: Iterable[str]) -> frozenset[str]:
        """Project selected points back to objects without rescanning raw traces."""
        selected_mask = self.test_mask_for_ids(test_ids)
        return frozenset(
            object_id
            for object_id, object_mask in self.object_test_masks.items()
            if object_mask & selected_mask
        )

    def test_mask_for_ids(self, test_ids: Iterable[str]) -> int:
        """Return the bitmap for exact known test IDs."""
        selected_mask = 0
        unknown: list[str] = []
        for test_id in test_ids:
            ordinal = bisect_left(self.test_ids, test_id)
            if ordinal >= len(self.test_ids) or self.test_ids[ordinal] != test_id:
                unknown.append(test_id)
                continue
            selected_mask |= 1 << ordinal
        if unknown:
            raise ContractError(f"P2P index has no trace for test points {unknown[:20]!r}")
        return selected_mask

    def candidate_test_mask(self, seed_test_file: str) -> int:
        """Return every successfully traced point outside the Seed file."""
        return self.all_test_mask & ~self.file_test_masks.get(seed_test_file, 0)

    def minimum_files_for_test_count(self, test_mask: int, count: int) -> int | None:
        """Return the optimistic minimum physical files needed for ``count`` points."""
        if count < 0:
            raise ValueError("test-point count must not be negative")
        if count == 0:
            return 0
        masks_by_path: dict[str, int] = {}
        for test_index in self._test_indexes_from_mask(test_mask & self.all_test_mask):
            path = self.test_paths[test_index]
            masks_by_path[path] = masks_by_path.get(path, 0) | (1 << test_index)
        accumulated = 0
        for files, mask in enumerate(
            sorted(masks_by_path.values(), key=int.bit_count, reverse=True),
            start=1,
        ):
            accumulated += mask.bit_count()
            if accumulated >= count:
                return files
        return None

    def test_mask_for_objects(self, object_ids: Iterable[str]) -> int:
        """Return the union of test points that executed any supplied object."""
        mask = 0
        for object_id in object_ids:
            mask |= self.object_test_masks.get(object_id, 0)
        return mask

    def observed_test_mask_for_objects(self, object_ids: Iterable[str]) -> int:
        """Return exact per-point frames before conservative cold-start expansion."""
        mask = 0
        for object_id in object_ids:
            mask |= self.observed_object_test_masks.get(object_id, 0)
        return mask

    def representative_test_ids(
        self,
        *,
        eligible_test_mask: int,
        count: int,
        deleted_object_ids: Iterable[str],
        maximum_files: int | None,
        random_seed: int,
        seed_test_file: str,
    ) -> tuple[str, ...]:
        """Greedily maximize coverage using an exact lazy priority queue."""
        if count < 1 or (maximum_files is not None and maximum_files < 1):
            raise ValueError("representative P2P count and maximum_files must be positive")
        eligible_test_mask &= self.all_test_mask
        if eligible_test_mask.bit_count() < count:
            raise CandidateRejected(
                f"{seed_test_file} has only {eligible_test_mask.bit_count()} optimized P2P "
                f"points; requires {count}"
            )
        if eligible_test_mask.bit_count() == count:
            # The common unbounded case is an inventory projection, not a
            # subset-selection problem.  Running iterative coverage scoring here
            # would turn "keep every P2P point" into quadratic work.
            return tuple(self._test_ids_from_mask(eligible_test_mask))
        deleted = frozenset(deleted_object_ids)
        surviving_objects = tuple(
            object_id for object_id in sorted(self.object_test_masks) if object_id not in deleted
        )
        coverage_by_test: dict[int, int] = {}
        for object_index, object_id in enumerate(surviving_objects):
            mask = self.object_test_masks[object_id] & eligible_test_mask
            while mask:
                bit = mask & -mask
                test_index = bit.bit_length() - 1
                coverage_by_test[test_index] = coverage_by_test.get(test_index, 0) | (
                    1 << object_index
                )
                mask ^= bit

        eligible_indexes = self._test_indexes_from_mask(eligible_test_mask)
        remaining = set(eligible_indexes)
        selected: list[int] = []
        selected_files: set[str] = set()
        covered_objects = 0
        total_coverage = {
            test_index: coverage_by_test.get(test_index, 0).bit_count()
            for test_index in eligible_indexes
        }
        stable_tie_break = {
            test_index: -int.from_bytes(
                sha256(
                    f"{random_seed}:{seed_test_file}:{self.test_ids[test_index]}".encode()
                ).digest()[:8],
                "big",
            )
            for test_index in eligible_indexes
        }
        indexes_by_file: dict[str, list[int]] = {}
        for test_index in eligible_indexes:
            indexes_by_file.setdefault(self.test_paths[test_index], []).append(test_index)

        # Coverage gains can only decrease as objects become covered, so a stale
        # score is an upper bound. File affinity is the one increasing component;
        # refresh that file eagerly when it first enters the selection.
        versions = [0] * len(self.test_ids)
        queue: list[tuple[int, int, int, int, int, int]] = []

        def score(test_index: int) -> tuple[int, int, int, int]:
            coverage = coverage_by_test.get(test_index, 0)
            return (
                (coverage & ~covered_objects).bit_count(),
                int(self.test_paths[test_index] in selected_files),
                total_coverage[test_index],
                stable_tie_break[test_index],
            )

        def enqueue(test_index: int) -> None:
            versions[test_index] += 1
            current = score(test_index)
            heappush(
                queue,
                (
                    -current[0],
                    -current[1],
                    -current[2],
                    -current[3],
                    test_index,
                    versions[test_index],
                ),
            )

        for test_index in eligible_indexes:
            enqueue(test_index)

        while len(selected) < count:
            chosen: int | None = None
            while queue:
                *cached, test_index, version = heappop(queue)
                if test_index not in remaining or version != versions[test_index]:
                    continue
                path = self.test_paths[test_index]
                if (
                    path not in selected_files
                    and maximum_files is not None
                    and len(selected_files) >= maximum_files
                ):
                    continue
                current = score(test_index)
                if tuple(cached) != tuple(-component for component in current):
                    enqueue(test_index)
                    continue
                chosen = test_index
                break
            if chosen is None:
                raise CandidateRejected(
                    f"{seed_test_file} cannot select {count} P2P points within "
                    f"{maximum_files} physical files"
                )
            selected.append(chosen)
            remaining.remove(chosen)
            covered_objects |= coverage_by_test.get(chosen, 0)
            selected_path = self.test_paths[chosen]
            if selected_path not in selected_files:
                selected_files.add(selected_path)
                for sibling in indexes_by_file[selected_path]:
                    if sibling in remaining:
                        enqueue(sibling)
        return tuple(sorted(self.test_ids[test_index] for test_index in selected))

    def limit_mask_to_best_files(
        self,
        eligible_test_mask: int,
        maximum_files: int | None,
    ) -> int:
        """Keep the point-maximizing physical-file subset for one fixed plan."""
        eligible_test_mask &= self.all_test_mask
        if maximum_files is None:
            return eligible_test_mask
        masks_by_path: dict[str, int] = {}
        for test_index in self._test_indexes_from_mask(eligible_test_mask):
            path = self.test_paths[test_index]
            masks_by_path[path] = masks_by_path.get(path, 0) | (1 << test_index)
        selected_paths = sorted(
            masks_by_path,
            key=lambda path: (-masks_by_path[path].bit_count(), path),
        )[:maximum_files]
        limited = 0
        for path in selected_paths:
            limited |= masks_by_path[path]
        return limited

    @staticmethod
    def _test_indexes_from_mask(mask: int) -> list[int]:
        selected: list[int] = []
        while mask:
            bit = mask & -mask
            selected.append(bit.bit_length() - 1)
            mask ^= bit
        return selected

    def select_p2p(
        self,
        *,
        seed_test_file: str,
        top_object_ids: Sequence[str],
        minimum_test_points: int,
        maximum_test_points: int | None,
        random_seed: int,
        top_object_impacts: Mapping[str, Iterable[str]] | None = None,
    ) -> P2PSelection:
        """Select points whose trace avoids every retained Top deletion footprint."""
        top = set(top_object_ids)
        if not top:
            raise CandidateRejected(f"{seed_test_file} has no Top objects for P2P selection")
        missing_top = tuple(sorted(top - self.object_test_masks.keys()))
        if missing_top:
            raise CandidateRejected(
                f"{seed_test_file} has selected Top objects without point-grained "
                f"P2P evidence: {missing_top[:20]!r}"
            )

        candidate_mask = self.all_test_mask & ~self.file_test_masks.get(seed_test_file, 0)
        top_masks: dict[str, int] = {}
        for object_id in top:
            impact_ids = (
                top_object_impacts.get(object_id, (object_id,))
                if top_object_impacts is not None
                else (object_id,)
            )
            impact_mask = 0
            for impact_id in impact_ids:
                impact_mask |= self.object_test_masks.get(impact_id, 0)
            top_masks[object_id] = impact_mask
        removal_order = sorted(
            top,
            key=lambda object_id: (
                (top_masks[object_id] & candidate_mask).bit_count(),
                object_id,
            ),
            reverse=True,
        )

        def eligible_mask() -> int:
            conflicting = 0
            for object_id in top:
                conflicting |= top_masks[object_id]
            return candidate_mask & ~conflicting

        eligible = eligible_mask()
        removal_index = 0
        while eligible.bit_count() < minimum_test_points and len(top) > 1:
            while removal_order[removal_index] not in top:
                removal_index += 1
            top.remove(removal_order[removal_index])
            removal_index += 1
            eligible = eligible_mask()

        eligible_count = eligible.bit_count()
        if eligible_count < minimum_test_points:
            raise CandidateRejected(
                f"{seed_test_file} has only {eligible_count} non-overlapping P2P test points; "
                f"requires {minimum_test_points}"
            )

        eligible_ids = self._test_ids_from_mask(eligible)
        if maximum_test_points is None or maximum_test_points >= len(eligible_ids):
            selected_ids = eligible_ids
        else:
            random.Random(f"{random_seed}:{seed_test_file}").shuffle(eligible_ids)
            selected_ids = sorted(eligible_ids[:maximum_test_points])
        return P2PSelection(
            test_ids=tuple(selected_ids),
            top_object_ids=tuple(sorted(top)),
        )
