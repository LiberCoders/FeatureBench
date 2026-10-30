"""Deterministic, scale-preserving rollback for rejected FeatureBench Mask plans."""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.masking import FeatureSlicePlanner
from codeworld.adapters.featurebench.models import (
    MaskPlan,
    QualityReport,
    SourceObject,
    TraceBundle,
)
from codeworld.adapters.featurebench.p2p import TracePointIndex
from codeworld.adapters.featurebench.quality import (
    has_f2p_hideability_error,
    reapply_gold_patch_tier_gates,
)
from codeworld.adapters.featurebench.repair.models import (
    MaskRepairProposal,
    MaskRepairProposalError,
)
from codeworld.errors import CandidateRejected, FactoryError

PlanValidator = Callable[[MaskPlan, int], Awaitable[QualityReport]]
RollbackProgress = Callable[[str, dict[str, object]], Awaitable[None]]


class _ValidationLimitReached(Exception):
    """Stop deterministic search after its unique real-validation budget is exhausted."""


@dataclass(frozen=True, slots=True)
class RollbackResult:
    """Accepted plan plus compact deterministic-search statistics."""

    plan: MaskPlan
    quality: QualityReport
    validations: int
    cache_hits: int
    duration_seconds: float
    validation_seconds: float
    engine_seconds: float
    restored_object_ids: tuple[str, ...]
    replacement_object_ids: tuple[str, ...]


def _plan_key(plan: MaskPlan) -> tuple[object, ...]:
    """Identify executable plan content while deliberately ignoring revision metadata."""
    return (
        plan.seed_test_file,
        plan.gold_patch_tier,
        tuple(sorted(item.object_id for item in plan.top_objects)),
        tuple(sorted(item.object_id for item in plan.specific_objects)),
        tuple(plan.p2p_test_ids),
    )


def _runtime_ok(quality: QualityReport) -> bool:
    return (
        quality.test_exit_code in (0, 1)
        and not quality.collection_errors
        and not any(error.startswith("isolated runs omitted ") for error in quality.errors)
    )


def _chunks_by_weight(
    object_ids: Sequence[str],
    *,
    parts: int,
    weights: Mapping[str, int],
) -> tuple[tuple[str, ...], ...]:
    """Partition deterministically while balancing estimated lost Gold Patch lines."""
    buckets: list[list[str]] = [[] for _ in range(min(parts, len(object_ids)))]
    totals = [0] * len(buckets)
    for object_id in sorted(object_ids, key=lambda item: (-weights[item], item)):
        bucket = min(range(len(buckets)), key=lambda index: (totals[index], index))
        buckets[bucket].append(object_id)
        totals[bucket] += weights[object_id]
    return tuple(tuple(sorted(bucket)) for bucket in buckets if bucket)


class DeterministicMaskRollback:
    """Repair a real-test failure without a model or repository-specific branches.

    The initial Mask plan remains the immutable search root. Runtime evidence narrows the
    first restore batch; weighted delta debugging then removes unnecessary restores. If
    the smallest operational rollback loses too much scale, safe replacement masks are
    admitted in batches and split only when a real validation rejects the batch.
    """

    def __init__(
        self,
        *,
        index: RepositoryIndex,
        planner: FeatureSlicePlanner,
        traces: Mapping[str, TraceBundle],
        initial_plan: MaskPlan,
        initial_quality: QualityReport,
        point_index: TracePointIndex,
        settings: FeatureBenchSettings,
        validate_plan: PlanValidator,
        timeout_seconds: float,
        progress: RollbackProgress | None = None,
    ) -> None:
        self.index = index
        self.planner = planner
        self.traces = traces
        self.initial_plan = initial_plan
        self.initial_quality = initial_quality
        self.point_index = point_index
        self.settings = settings
        self.validate_plan = validate_plan
        self.timeout_seconds = timeout_seconds
        self.progress = progress
        self._cache: dict[tuple[object, ...], QualityReport | None] = {
            _plan_key(initial_plan): initial_quality
        }
        self._validations = 0
        self._cache_hits = 0
        self._validation_seconds = 0.0
        self._initial_target_ids = frozenset(item.object_id for item in initial_plan.targets)
        self._best: tuple[MaskPlan, QualityReport] | None = None
        self._best_fallback: tuple[MaskPlan, QualityReport] | None = None
        self._remember_accepted(initial_plan, initial_quality)
        self._weights = {
            item.object_id: max(1, item.removed_lines) for item in initial_plan.targets
        }
        definitions_by_path: dict[str, list[SourceObject]] = defaultdict(list)
        for item in index.data.definitions:
            definitions_by_path[item.path].append(item)
        self._definitions_by_path = {
            path: tuple(sorted(items, key=lambda item: (item.start_line, item.object_id)))
            for path, items in definitions_by_path.items()
        }
        self._footprints: dict[str, tuple[str, ...]] = {}

    @property
    def _stops_at_first_accepted(self) -> bool:
        return self.settings.repair.optimization_target == "first_accepted"

    async def run(self) -> RollbackResult:
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                plan, quality = await self._search()
        except (TimeoutError, _ValidationLimitReached) as exc:
            if self._best is not None:
                plan, quality = self._best
            elif self._best_fallback is not None:
                plan, quality = self._best_fallback
            else:
                if isinstance(exc, _ValidationLimitReached):
                    reason = str(exc)
                else:
                    reason = (
                        f"deterministic Mask rollback exceeded {self.timeout_seconds:g}s "
                        "without an accepted plan"
                    )
                failure = CandidateRejected(reason)
                await self._emit_failure(started=started, reason=str(failure))
                raise failure from exc
        except CandidateRejected as exc:
            await self._emit_failure(started=started, reason=str(exc))
            raise
        return self._result(plan, quality, duration_seconds=time.monotonic() - started)

    async def _emit_failure(self, *, started: float, reason: str) -> None:
        duration = time.monotonic() - started
        await self._emit(
            "failed",
            seed_test_file=self.initial_plan.seed_test_file,
            validations=self._validations,
            cache_hits=self._cache_hits,
            duration_seconds=round(duration, 6),
            validation_seconds=round(self._validation_seconds, 6),
            engine_seconds=round(max(0.0, duration - self._validation_seconds), 6),
            reason=reason,
        )

    def _result(
        self,
        plan: MaskPlan,
        quality: QualityReport,
        *,
        duration_seconds: float,
    ) -> RollbackResult:
        final_ids = frozenset(item.object_id for item in plan.targets)
        return RollbackResult(
            plan=plan,
            quality=quality,
            validations=self._validations,
            cache_hits=self._cache_hits,
            duration_seconds=duration_seconds,
            validation_seconds=self._validation_seconds,
            engine_seconds=max(0.0, duration_seconds - self._validation_seconds),
            restored_object_ids=tuple(sorted(self._initial_target_ids - final_ids)),
            replacement_object_ids=tuple(sorted(final_ids - self._initial_target_ids)),
        )

    async def _emit(self, kind: str, **data: object) -> None:
        if self.progress is not None:
            await self.progress(kind, data)

    def _accepted_score(self, plan: MaskPlan, quality: QualityReport) -> tuple[object, ...]:
        minimum = self.settings.gold_patch_minimum_for_tier(plan.gold_patch_tier)
        maximum = self.settings.gold_patch_maximum_for_tier(plan.gold_patch_tier)
        center = (minimum + maximum) / 2
        return (
            abs(quality.gold_patch_lines - center),
            -quality.gold_patch_files,
            -len(quality.pass_to_pass),
            len(self._initial_target_ids - {item.object_id for item in plan.targets}),
            _plan_key(plan),
        )

    def _remember_accepted(self, plan: MaskPlan, quality: QualityReport) -> None:
        if quality.accepted:
            if self._best is None or self._accepted_score(plan, quality) < self._accepted_score(
                *self._best
            ):
                self._best = (plan, quality)
            return
        if plan.gold_patch_tier != "preferred":
            return
        fallback_plan = plan.model_copy(update={"gold_patch_tier": "fallback"})
        fallback_quality = reapply_gold_patch_tier_gates(
            quality,
            settings=self.settings,
            tier="fallback",
        )
        if not fallback_quality.accepted:
            return
        candidate = (fallback_plan, fallback_quality)
        if self._best_fallback is None or self._accepted_score(*candidate) < self._accepted_score(
            *self._best_fallback
        ):
            self._best_fallback = candidate

    def _behavior_deficit(self, quality: QualityReport) -> tuple[float, ...]:
        """Order incomplete behavioral gates so real improvements can be compounded."""
        maximum_f2p = self.settings.maximum_f2p_test_points
        return (
            float(
                max(
                    0,
                    self.settings.minimum_remaining_p2p_test_points - len(quality.pass_to_pass),
                )
            ),
            max(0.0, quality.f2p_ratio - self.settings.maximum_f2p_ratio),
            float(
                max(0, len(quality.fail_to_pass) - maximum_f2p) if maximum_f2p is not None else 0
            ),
            float(
                max(
                    0,
                    self.settings.minimum_f2p_test_points - len(quality.fail_to_pass),
                )
            ),
            max(0.0, quality.seed_pass_rate - self.settings.maximum_seed_pass_rate),
        )

    def _gold_deficit(self, plan: MaskPlan, quality: QualityReport) -> int:
        minimum = self.settings.gold_patch_minimum_for_tier(plan.gold_patch_tier)
        maximum = self.settings.gold_patch_maximum_for_tier(plan.gold_patch_tier)
        if quality.gold_patch_lines < minimum:
            return minimum - quality.gold_patch_lines
        if quality.gold_patch_lines > maximum:
            return quality.gold_patch_lines - maximum
        return 0

    def _replacement_makes_progress(
        self,
        *,
        base_plan: MaskPlan,
        base_quality: QualityReport,
        revised_plan: MaskPlan,
        revised_quality: QualityReport,
    ) -> bool:
        """Keep an incomplete batch only when its real result adds usable information."""
        if revised_quality.accepted:
            return True
        if base_quality.accepted or not _runtime_ok(revised_quality):
            return False
        if revised_quality.gold_patch_lines > self.settings.gold_patch_maximum_for_tier(
            revised_plan.gold_patch_tier
        ):
            return False
        base_behavior = self._behavior_deficit(base_quality)
        revised_behavior = self._behavior_deficit(revised_quality)
        if revised_behavior < base_behavior:
            return True
        if revised_behavior != base_behavior or any(revised_behavior):
            return False
        return self._gold_deficit(revised_plan, revised_quality) < self._gold_deficit(
            base_plan, base_quality
        )

    async def _validate(self, plan: MaskPlan) -> QualityReport | None:
        key = _plan_key(plan)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        maximum = self.settings.repair.maximum_validations
        if self._validations >= maximum:
            raise _ValidationLimitReached(
                f"deterministic Mask rollback reached its limit of {maximum} unique real "
                "validations without an accepted plan"
            )
        self._validations += 1
        await self._emit(
            "validation_started",
            validation=self._validations,
            masked_objects=len(plan.targets),
            removed_lines=plan.removed_lines,
        )
        validation_started = time.monotonic()
        try:
            try:
                quality = await self.validate_plan(plan, self._validations)
            except FactoryError:
                quality = None
        finally:
            self._validation_seconds += time.monotonic() - validation_started
        self._cache[key] = quality
        if quality is not None:
            self._remember_accepted(plan, quality)
        await self._emit(
            "validation_completed",
            validation=self._validations,
            accepted=bool(quality and quality.accepted),
            runtime_ok=bool(quality and _runtime_ok(quality)),
            gold_patch_lines=quality.gold_patch_lines if quality else 0,
            gold_patch_files=quality.gold_patch_files if quality else 0,
            f2p=len(quality.fail_to_pass) if quality else 0,
            p2p=len(quality.pass_to_pass) if quality else 0,
        )
        return quality

    def _increment(
        self,
        base: MaskPlan,
        *,
        restore: Iterable[str] = (),
        mask: Iterable[str] = (),
        learned_protected_ids: Iterable[str] = (),
    ) -> MaskPlan | None:
        restore_ids = self._restore_closure(base, restore)
        mask_ids = tuple(sorted(set(mask)))
        if not restore_ids and not mask_ids:
            return base
        try:
            return self.planner.apply_increment(
                index=self.index,
                plan=base,
                proposal=MaskRepairProposal(
                    based_on_attempt="deterministic-rollback",
                    must_restore_object_ids=restore_ids,
                    must_mask_object_ids=mask_ids,
                    rationale="Deterministic evidence-guided rollback.",
                ),
                traces=self.traces,
                learned_protected_ids=learned_protected_ids,
            )
        except MaskRepairProposalError:
            return None

    def _footprint_ids(self, object_id: str) -> tuple[str, ...]:
        cached = self._footprints.get(object_id)
        if cached is not None:
            return cached
        item = self.index.by_id[object_id]
        footprint = tuple(
            candidate.object_id
            for candidate in self._definitions_by_path.get(item.path, ())
            if item.start_line <= candidate.start_line and candidate.end_line <= item.end_line
        )
        self._footprints[object_id] = footprint
        return footprint

    def _deleted_ids(self, plan: MaskPlan) -> frozenset[str]:
        return frozenset(
            object_id
            for target in plan.targets
            for object_id in self._footprint_ids(target.object_id)
        )

    def _restore_closure(
        self,
        plan: MaskPlan,
        restore_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Close a restore request over dependencies evaluated at definition time."""
        current = {item.object_id: item for item in plan.targets}
        selected = {object_id for object_id in restore_ids if object_id in current}
        if not selected:
            return ()
        roots_by_deleted_id: dict[str, set[str]] = defaultdict(set)
        for target in plan.targets:
            for deleted_id in self._footprint_ids(target.object_id):
                roots_by_deleted_id[deleted_id].add(target.object_id)
        while True:
            deleted = frozenset(
                deleted_id
                for object_id, target in current.items()
                if object_id not in selected
                for deleted_id in self._footprint_ids(target.object_id)
            )
            blockers = self.index.definition_time_blockers(deleted)
            additions = {
                root_id
                for blocked_id in blockers
                for root_id in roots_by_deleted_id.get(blocked_id, ())
                if root_id not in selected
            }
            if not additions:
                return tuple(sorted(selected))
            selected.update(additions)

    def _impact_mask(self, object_id: str) -> int:
        return self.point_index.test_mask_for_objects(self._footprint_ids(object_id))

    def _rank_restore_ids(
        self,
        plan: MaskPlan,
        quality: QualityReport,
    ) -> tuple[str, ...]:
        diagnostic = "\n".join((*quality.collection_errors, quality.test_output)).lower()
        failed_p2p = set(plan.p2p_test_ids) - set(quality.pass_to_pass)
        failed_p2p_mask = self.point_index.test_mask_for_ids(failed_p2p) if failed_p2p else 0
        seed_mask = self.point_index.file_test_masks.get(plan.seed_test_file, 0)
        top_ids = {item.object_id for item in plan.top_objects}
        targets = tuple(plan.targets)

        def evidence(item: SourceObject) -> tuple[int, int, int, int, int, str]:
            path_match = int(item.path.lower() in diagnostic)
            tokens = {item.name.lower(), item.qualname.rsplit(".", 1)[-1].lower()}
            symbol_match = int(
                any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", diagnostic) for token in tokens)
            )
            impact = self._impact_mask(item.object_id)
            failed_impact = (impact & failed_p2p_mask).bit_count()
            seed_impact = (impact & seed_mask).bit_count()
            # Specific objects are safer to restore. Among equally strong evidence,
            # prefer fewer lost lines and less Seed coverage.
            return (
                path_match,
                symbol_match,
                failed_impact,
                int(item.object_id not in top_ids),
                -seed_impact,
                item.object_id,
            )

        def restore_key(item: SourceObject) -> tuple[int, int, int, int, int, int, str]:
            path_match, symbol_match, failed_impact, specific, negative_seed, object_id = evidence(
                item
            )
            return (
                -path_match,
                -symbol_match,
                -failed_impact,
                -specific,
                -negative_seed,
                self._weights.get(item.object_id, item.removed_lines),
                object_id,
            )

        ordered = sorted(targets, key=restore_key)
        return tuple(item.object_id for item in ordered)

    def _focused_restore_ids(
        self,
        plan: MaskPlan,
        quality: QualityReport,
        ranked: Sequence[str],
    ) -> tuple[str, ...]:
        diagnostic = "\n".join((*quality.collection_errors, quality.test_output)).lower()
        failed_p2p = set(plan.p2p_test_ids) - set(quality.pass_to_pass)
        failed_mask = self.point_index.test_mask_for_ids(failed_p2p) if failed_p2p else 0
        path_and_symbol: list[str] = []
        symbol_matches: list[str] = []
        failed_impact: list[str] = []
        path_matches: list[str] = []
        for object_id in ranked:
            item = self.index.by_id[object_id]
            symbols = {item.name.lower(), item.qualname.rsplit(".", 1)[-1].lower()}
            path_match = item.path.lower() in diagnostic
            symbol_match = any(
                re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", diagnostic) for symbol in symbols
            )
            if path_match and symbol_match:
                path_and_symbol.append(object_id)
            elif symbol_match:
                symbol_matches.append(object_id)
            elif self._impact_mask(object_id) & failed_mask:
                failed_impact.append(object_id)
            elif path_match:
                path_matches.append(object_id)
        # Precise traceback evidence must not be diluted by every other masked
        # declaration in the same physical file. Broader evidence is used only
        # when the more informative tier is empty.
        return tuple(path_and_symbol or symbol_matches or failed_impact or path_matches)

    def _can_restore(self, plan: MaskPlan, restore_ids: Iterable[str]) -> bool:
        restoring = set(restore_ids)
        return bool({item.object_id for item in plan.top_objects} - restoring)

    async def _restore_from_root(
        self,
        restore_ids: Iterable[str],
    ) -> tuple[MaskPlan, QualityReport] | None:
        selected = tuple(sorted(set(restore_ids)))
        if not selected or not self._can_restore(self.initial_plan, selected):
            return None
        plan = self._increment(self.initial_plan, restore=selected)
        if plan is None:
            return None
        quality = await self._validate(plan)
        return (plan, quality) if quality is not None else None

    async def _minimal_operational_restore(
        self,
        restore_ids: Sequence[str],
    ) -> tuple[MaskPlan, QualityReport] | None:
        """Use weighted ddmin; every tested state is cached by executable content."""
        current = tuple(sorted(set(restore_ids)))
        initial = await self._restore_from_root(current)
        if initial is None or not _runtime_ok(initial[1]):
            return None
        if self._stops_at_first_accepted and initial[1].accepted:
            return initial
        current_result = initial
        granularity = 2
        while len(current) >= 2:
            chunks = _chunks_by_weight(current, parts=granularity, weights=self._weights)
            changed = False

            def cheap_chunk_key(chunk: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
                return sum(self._weights[item] for item in chunk), len(chunk), chunk

            def expensive_chunk_key(chunk: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
                return -sum(self._weights[item] for item in chunk), chunk

            # Cheap restore subsets are tried first because they preserve more Gold scale.
            for subset in sorted(chunks, key=cheap_chunk_key):
                result = await self._restore_from_root(subset)
                if result is not None and _runtime_ok(result[1]):
                    if self._stops_at_first_accepted and result[1].accepted:
                        return result
                    current = subset
                    current_result = result
                    granularity = max(2, granularity - 1)
                    changed = True
                    break
            if changed:
                continue
            # Then test complements, dropping the most expensive chunks first.
            for chunk in sorted(chunks, key=expensive_chunk_key):
                complement = tuple(item for item in current if item not in set(chunk))
                result = await self._restore_from_root(complement)
                if result is not None and _runtime_ok(result[1]):
                    if self._stops_at_first_accepted and result[1].accepted:
                        return result
                    current = complement
                    current_result = result
                    granularity = max(2, granularity - 1)
                    changed = True
                    break
            if changed:
                continue
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
        return current_result

    async def _follow_diagnostic_chain(self) -> tuple[MaskPlan, QualityReport] | None:
        """Consume newly revealed runtime failures before falling back to broad ddmin."""
        current_plan = self.initial_plan
        current_quality = self.initial_quality
        restored: set[str] = set()
        initial_top_ids = {item.object_id for item in self.initial_plan.top_objects}
        while not _runtime_ok(current_quality):
            ranked = self._rank_restore_ids(current_plan, current_quality)
            focused = list(self._focused_restore_ids(current_plan, current_quality, ranked))
            if not focused:
                return None
            prospective = restored | set(focused)
            if not (initial_top_ids - prospective):
                retained_top = min(
                    (object_id for object_id in focused if object_id in initial_top_ids),
                    key=lambda object_id: (self._weights.get(object_id, 1), object_id),
                    default=None,
                )
                if retained_top is None:
                    return None
                focused.remove(retained_top)
            additions = set(focused) - restored
            if not additions:
                return None
            restored.update(additions)
            result = await self._restore_from_root(restored)
            if result is None:
                return None
            current_plan, current_quality = result
            if _runtime_ok(current_quality):
                return await self._minimal_operational_restore(tuple(restored))
        return current_plan, current_quality

    async def _establish_operational_plan(self) -> tuple[MaskPlan, QualityReport]:
        if _runtime_ok(self.initial_quality):
            return self.initial_plan, self.initial_quality
        diagnostic_result = await self._follow_diagnostic_chain()
        if diagnostic_result is not None:
            return diagnostic_result
        ranked = self._rank_restore_ids(self.initial_plan, self.initial_quality)
        top_ids = {item.object_id for item in self.initial_plan.top_objects}
        specific = tuple(item for item in ranked if item not in top_ids)
        pools: list[tuple[str, ...]] = []
        if specific and specific not in pools:
            pools.append(specific)
        # Top restoration is a last resort. Try every possible retained interface;
        # otherwise choosing one arbitrary Top can miss the actual broken declaration.
        for keep_top in sorted(top_ids, key=lambda item: (self._weights.get(item, 1), item)):
            broad = tuple(item for item in ranked if item != keep_top)
            if broad and broad not in pools:
                pools.append(broad)
        for pool in pools:
            result = await self._minimal_operational_restore(pool)
            if result is not None:
                return result
        raise CandidateRejected(
            f"deterministic rollback could not restore an operational Mask plan for "
            f"{self.initial_plan.seed_test_file}"
        )

    async def _restore_quality(
        self,
        base_plan: MaskPlan,
        base_quality: QualityReport,
    ) -> tuple[MaskPlan, QualityReport]:
        """Restore high-impact objects in exponential batches until hard gates recover."""
        if base_quality.accepted:
            return base_plan, base_quality
        ranked = self._rank_restore_ids(base_plan, base_quality)
        top_ids = {item.object_id for item in base_plan.top_objects}
        # Preserve every interface while Specific restorations can recover quality;
        # if they cannot, permit Top rollback as a deterministic last resort while
        # retaining at least one interface for a valid FeatureBench task.
        remaining: list[str] = [item for item in ranked if item not in top_ids]
        remaining.extend(item for item in ranked if item in top_ids)
        current_plan, current_quality = base_plan, base_quality
        batch_size = 1
        while remaining and (
            current_quality.f2p_ratio > self.settings.maximum_f2p_ratio
            or len(current_quality.pass_to_pass) < self.settings.minimum_remaining_p2p_test_points
            or has_f2p_hideability_error(current_quality)
            or (
                self.settings.maximum_f2p_test_points is not None
                and len(current_quality.fail_to_pass) > self.settings.maximum_f2p_test_points
            )
            or current_quality.gold_patch_lines
            > self.settings.gold_patch_maximum_for_tier(current_plan.gold_patch_tier)
        ):
            batch_candidates = remaining[:batch_size]
            del remaining[:batch_size]
            current_top_ids = {item.object_id for item in current_plan.top_objects}
            batch: list[str] = []
            for object_id in batch_candidates:
                if object_id in current_top_ids and len(current_top_ids - set(batch)) <= 1:
                    continue
                batch.append(object_id)
            if not batch:
                continue
            revised = self._increment(current_plan, restore=batch)
            if revised is None:
                continue
            quality = await self._validate(revised)
            if quality is None or not _runtime_ok(quality):
                continue
            current_plan, current_quality = revised, quality
            current_ids = {item.object_id for item in current_plan.targets}
            remaining = [object_id for object_id in remaining if object_id in current_ids]
            if quality.accepted:
                return current_plan, current_quality
            batch_size = min(max(1, batch_size * 2), len(remaining) or 1)
        return current_plan, current_quality

    def _rank_replacements(
        self,
        plan: MaskPlan,
        *,
        permanently_restored: frozenset[str],
        need_seed_break: bool,
    ) -> tuple[str, ...]:
        current_ids = self._deleted_ids(plan)
        current_paths = set(plan.patch_files)
        seed_mask = self.point_index.file_test_masks.get(plan.seed_test_file, 0)
        p2p_mask = self.point_index.test_mask_for_ids(plan.p2p_test_ids)
        candidates: list[SourceObject] = []
        for object_id in plan.candidate_object_ids:
            item = self.index.by_id.get(object_id)
            if (
                item is None
                or object_id in current_ids
                or object_id in permanently_restored
                or object_id in plan.protected_objects
            ):
                continue
            candidates.append(item)

        def replacement_key(item: SourceObject) -> tuple[object, ...]:
            impact = self._impact_mask(item.object_id)
            seed_impact = (impact & seed_mask).bit_count()
            p2p_impact = (impact & p2p_mask).bit_count()
            new_file = int(item.path not in current_paths)
            if need_seed_break:
                return (
                    int(seed_impact == 0),
                    int(p2p_impact > 0),
                    p2p_impact,
                    -seed_impact,
                    -new_file,
                    -item.removed_lines,
                    item.path,
                    item.start_line,
                    item.object_id,
                )
            return (
                int(p2p_impact > 0),
                -new_file,
                p2p_impact,
                -item.removed_lines,
                item.path,
                item.start_line,
                item.object_id,
            )

        return tuple(item.object_id for item in sorted(candidates, key=replacement_key))

    async def _admit_replacements(
        self,
        base_plan: MaskPlan,
        base_quality: QualityReport,
        replacement_ids: Sequence[str],
        *,
        permanently_restored: frozenset[str],
    ) -> tuple[MaskPlan, QualityReport]:
        """Try one large batch, recursively splitting only a rejected real execution."""
        if not replacement_ids:
            return base_plan, base_quality
        revised = self._increment(
            base_plan,
            mask=replacement_ids,
            learned_protected_ids=permanently_restored,
        )
        if revised is None:
            if len(replacement_ids) == 1:
                return base_plan, base_quality
            midpoint = len(replacement_ids) // 2
            left_plan, left_quality = await self._admit_replacements(
                base_plan,
                base_quality,
                replacement_ids[:midpoint],
                permanently_restored=permanently_restored,
            )
            if self._stops_at_first_accepted and left_quality.accepted:
                return left_plan, left_quality
            return await self._admit_replacements(
                left_plan,
                left_quality,
                replacement_ids[midpoint:],
                permanently_restored=permanently_restored,
            )
        quality = await self._validate(revised)
        if quality is not None and self._replacement_makes_progress(
            base_plan=base_plan,
            base_quality=base_quality,
            revised_plan=revised,
            revised_quality=quality,
        ):
            return revised, quality
        if len(replacement_ids) == 1:
            return base_plan, base_quality
        midpoint = len(replacement_ids) // 2
        left_plan, left_quality = await self._admit_replacements(
            base_plan,
            base_quality,
            replacement_ids[:midpoint],
            permanently_restored=permanently_restored,
        )
        if self._stops_at_first_accepted and left_quality.accepted:
            return left_plan, left_quality
        return await self._admit_replacements(
            left_plan,
            left_quality,
            replacement_ids[midpoint:],
            permanently_restored=permanently_restored,
        )

    async def _refill_scale(
        self,
        plan: MaskPlan,
        quality: QualityReport,
    ) -> tuple[MaskPlan, QualityReport]:
        permanently_restored = self._initial_target_ids - {item.object_id for item in plan.targets}
        need_seed_break = (
            quality.seed_pass_rate > self.settings.maximum_seed_pass_rate
            or len(quality.fail_to_pass) < self.settings.minimum_f2p_test_points
        )
        remaining = list(
            self._rank_replacements(
                plan,
                permanently_restored=frozenset(permanently_restored),
                need_seed_break=need_seed_break,
            )
        )
        current_plan, current_quality = plan, quality
        while remaining:
            minimum = self.settings.gold_patch_minimum_for_tier(current_plan.gold_patch_tier)
            maximum = self.settings.gold_patch_maximum_for_tier(current_plan.gold_patch_tier)
            center = (minimum + maximum) // 2
            if current_quality.accepted and (
                self._stops_at_first_accepted or current_quality.gold_patch_lines >= center
            ):
                return current_plan, current_quality
            desired_lines = max(0, center - current_quality.gold_patch_lines)
            line_capacity = max(
                0,
                maximum - current_quality.gold_patch_lines,
            )
            need_seed_break = (
                current_quality.seed_pass_rate > self.settings.maximum_seed_pass_rate
                or len(current_quality.fail_to_pass) < self.settings.minimum_f2p_test_points
            )
            selected: list[str] = []
            estimated_lines = 0
            selected_seed_impact = False
            while remaining:
                object_id = remaining.pop(0)
                item = self.index.by_id[object_id]
                if estimated_lines + item.removed_lines > line_capacity:
                    continue
                selected.append(object_id)
                estimated_lines += item.removed_lines
                selected_seed_impact = selected_seed_impact or bool(
                    self._impact_mask(object_id)
                    & self.point_index.file_test_masks.get(current_plan.seed_test_file, 0)
                )
                if estimated_lines >= desired_lines and (
                    not need_seed_break or selected_seed_impact
                ):
                    break
            if not selected:
                break
            previous_key = _plan_key(current_plan)
            current_plan, current_quality = await self._admit_replacements(
                current_plan,
                current_quality,
                tuple(selected),
                permanently_restored=frozenset(permanently_restored),
            )
            if current_quality.accepted and (
                self._stops_at_first_accepted or current_quality.gold_patch_lines >= center
            ):
                return current_plan, current_quality
            if _plan_key(current_plan) == previous_key and not remaining:
                break
        return current_plan, current_quality

    async def _search(self) -> tuple[MaskPlan, QualityReport]:
        await self._emit(
            "started",
            seed_test_file=self.initial_plan.seed_test_file,
            initial_masked_objects=len(self.initial_plan.targets),
            initial_removed_lines=self.initial_plan.removed_lines,
        )
        plan, quality = await self._establish_operational_plan()
        if self._stops_at_first_accepted and quality.accepted:
            return plan, quality
        if not quality.accepted:
            plan, quality = await self._restore_quality(plan, quality)
        if self._stops_at_first_accepted and quality.accepted:
            return plan, quality
        plan, quality = await self._refill_scale(plan, quality)
        if quality.accepted:
            if self._stops_at_first_accepted:
                return plan, quality
            assert self._best is not None
            return self._best
        if self._best is not None:
            return self._best
        if self._best_fallback is not None:
            return self._best_fallback
        raise CandidateRejected(
            f"deterministic rollback exhausted scale-preserving candidates for "
            f"{self.initial_plan.seed_test_file}: {'; '.join(quality.errors)}"
        )
