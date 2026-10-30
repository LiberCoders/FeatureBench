"""Deterministic multi-objective planning for FeatureBench masks."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from ortools.sat.python import cp_model


@dataclass(frozen=True, slots=True)
class DeletionInfeasibilityDiagnostics:
    """Compact, machine-readable evidence for one proven-infeasible model."""

    status: str
    sufficient_constraint_core: tuple[str, ...]
    fixed_removed_lines: int
    candidate_removed_lines_upper_bound: int
    minimum_removed_lines: int
    maximum_removed_lines: int
    candidate_units: int
    base_eligible_p2p_test_points: int
    minimum_p2p_test_points: int
    maximum_p2p_test_points: int | None
    seed_test_points: int
    fixed_seed_affected_test_points: int
    minimum_f2p_test_points: int
    maximum_f2p_test_points: int | None
    required_seed_affected_test_points: int | None
    maximum_f2p_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostic data for events and persisted ledgers."""
        return {
            "status": self.status,
            "sufficient_constraint_core": self.sufficient_constraint_core,
            "removed_lines": {
                "fixed": self.fixed_removed_lines,
                "candidate_upper_bound": self.candidate_removed_lines_upper_bound,
                "minimum": self.minimum_removed_lines,
                "maximum": self.maximum_removed_lines,
            },
            "candidate_units": self.candidate_units,
            "p2p_test_points": {
                "base_eligible": self.base_eligible_p2p_test_points,
                "minimum": self.minimum_p2p_test_points,
                "maximum": self.maximum_p2p_test_points,
            },
            "seed_test_points": {
                "total": self.seed_test_points,
                "fixed_affected": self.fixed_seed_affected_test_points,
                "minimum_f2p": self.minimum_f2p_test_points,
                "maximum_f2p": self.maximum_f2p_test_points,
                "required_affected": self.required_seed_affected_test_points,
            },
            "maximum_f2p_ratio": self.maximum_f2p_ratio,
        }

    def render(self) -> str:
        """Render the decisive conflict and its main quantitative bounds."""
        core = ", ".join(self.sufficient_constraint_core) or "structural constraints"
        return (
            f"sufficient conflicting constraints: {core}; removed lines "
            f"fixed={self.fixed_removed_lines}, possible<="
            f"{self.candidate_removed_lines_upper_bound}, required="
            f"{self.minimum_removed_lines}..{self.maximum_removed_lines}; P2P "
            f"eligible={self.base_eligible_p2p_test_points}, required>="
            f"{self.minimum_p2p_test_points}; Seed points={self.seed_test_points}"
        )


class DeletionOptimizationInfeasibleError(ValueError):
    """The complete CP-SAT model proved that an interval has no feasible plan."""

    def __init__(self, diagnostics: DeletionInfeasibilityDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            f"deletion optimizer has no feasible solution: {diagnostics.status}; "
            f"{diagnostics.render()}"
        )


@dataclass(frozen=True, slots=True)
class DeletionUnit:
    """One atomic deletion choice and its trace-predicted effects."""

    key: str
    removed_lines: int
    object_count: int
    test_mask: int
    dependencies: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    seed_test_mask: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("deletion unit key must not be empty")
        if self.removed_lines < 1:
            raise ValueError("deletion unit removed_lines must be positive")
        if self.object_count < 1:
            raise ValueError("deletion unit object_count must be positive")
        if self.test_mask < 0 or self.seed_test_mask < 0:
            raise ValueError("deletion unit test masks must not be negative")
        if len(self.files) != len(set(self.files)) or any(not path for path in self.files):
            raise ValueError("deletion unit files must be unique non-empty paths")


@dataclass(frozen=True, slots=True)
class DeletionOptimizationResult:
    """One optimized deletion choice and its trace-predicted quality."""

    selected_unit_keys: tuple[str, ...]
    eligible_test_mask: int
    eligible_test_points: int
    excluded_test_points: int
    selected_p2p_test_points: int
    estimated_f2p_test_points: int
    removed_lines: int
    removed_objects: int
    modified_files: int
    minimum_lines: int
    maximum_lines: int
    center_distance_twice: int
    solve_seconds: float
    status: str
    signature_groups: int

    @property
    def center_distance(self) -> float:
        """Distance from the configured interval midpoint, including half-lines."""
        return self.center_distance_twice / 2


def _iter_bit_indexes(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _signature_counts(
    *,
    units: tuple[DeletionUnit, ...],
    inventory_mask: int,
    mask_attribute: str,
) -> Counter[int]:
    signatures = [0] * inventory_mask.bit_length()
    for unit_index, unit in enumerate(units):
        unit_mask = getattr(unit, mask_attribute) & inventory_mask
        for test_index in _iter_bit_indexes(unit_mask):
            signatures[test_index] |= 1 << unit_index
    return Counter(signatures[index] for index in _iter_bit_indexes(inventory_mask))


def _or_variable(
    model: cp_model.CpModel,
    *,
    name: str,
    members: list[cp_model.IntVar],
) -> cp_model.IntVar:
    value = model.new_bool_var(name)
    model.add_max_equality(value, members)
    return value


def optimize_deletions(
    *,
    units: tuple[DeletionUnit, ...],
    overlap_pairs: tuple[tuple[str, str], ...],
    mandatory_unit_keys: frozenset[str],
    base_eligible_test_mask: int,
    fixed_removed_lines: int,
    fixed_removed_objects: int,
    fixed_files: frozenset[str] = frozenset(),
    seed_test_mask: int = 0,
    fixed_seed_affected_test_mask: int = 0,
    minimum_removed_lines: int,
    maximum_removed_lines: int,
    minimum_f2p_test_points: int = 0,
    maximum_f2p_test_points: int | None = None,
    maximum_f2p_ratio: float | None = None,
    maximum_seed_pass_rate: float | None = None,
    minimum_p2p_test_points: int,
    maximum_p2p_test_points: int | None = None,
    timeout_seconds: float,
    workers: int,
    random_seed: int,
) -> DeletionOptimizationResult:
    """Optimize predictable mask quality in the user's declared priority order.

    The model first requires the restored production lines to lie inside the
    configured interval.  Its objective then prefers symmetric proximity to the
    interval midpoint, more modified production files, and finally more P2P
    points.  F2P constraints are trace estimates here; real pytest outcomes are
    always checked again by the post-mask quality gate.
    """
    if base_eligible_test_mask < 0 or seed_test_mask < 0:
        raise ValueError("test inventory masks must not be negative")
    if fixed_seed_affected_test_mask & ~seed_test_mask:
        raise ValueError("fixed affected Seed tests must belong to the Seed inventory")
    if fixed_removed_lines < 1 or fixed_removed_objects < 1:
        raise ValueError("fixed deletion size must be positive")
    if minimum_removed_lines < 1 or maximum_removed_lines < minimum_removed_lines:
        raise ValueError("invalid removed-line interval")
    if minimum_f2p_test_points < 0:
        raise ValueError("minimum F2P test points must not be negative")
    if maximum_f2p_test_points is not None and (maximum_f2p_test_points < minimum_f2p_test_points):
        raise ValueError("maximum F2P test points must not be below the minimum")
    if maximum_f2p_ratio is not None and not 0 < maximum_f2p_ratio < 1:
        raise ValueError("maximum F2P ratio must lie strictly between zero and one")
    if maximum_seed_pass_rate is not None and not 0 <= maximum_seed_pass_rate < 1:
        raise ValueError("maximum Seed pass rate must lie in [0, 1)")
    if minimum_p2p_test_points < 1:
        raise ValueError("minimum P2P test points must be positive")
    if maximum_p2p_test_points is not None and (maximum_p2p_test_points < minimum_p2p_test_points):
        raise ValueError("maximum P2P test points must not be below the minimum")
    if timeout_seconds <= 0 or workers < 1:
        raise ValueError("optimizer timeout and workers must be positive")

    started = time.monotonic()
    base_count = base_eligible_test_mask.bit_count()
    ordered_units = tuple(sorted(units, key=lambda unit: unit.key))
    keys = {unit.key for unit in ordered_units}
    if len(keys) != len(ordered_units):
        raise ValueError("deletion unit keys must be unique")
    unknown_mandatory = mandatory_unit_keys - keys
    if unknown_mandatory:
        raise ValueError(f"unknown mandatory deletion units: {sorted(unknown_mandatory)!r}")
    for unit in ordered_units:
        unknown_dependencies = set(unit.dependencies) - keys
        if unknown_dependencies:
            raise ValueError(
                f"deletion unit {unit.key!r} has unknown dependencies: "
                f"{sorted(unknown_dependencies)!r}"
            )
    for left, right in overlap_pairs:
        if left not in keys or right not in keys or left == right:
            raise ValueError(f"invalid overlap pair: {(left, right)!r}")

    model = cp_model.CpModel()
    assumption_names: dict[int, str] = {}

    def require(name: str, constraint: cp_model.Constraint) -> None:
        assumption = model.new_bool_var(f"require_{name}")
        constraint.only_enforce_if(assumption)  # pyright: ignore[reportUnknownMemberType]
        model.add_assumption(assumption)
        assumption_names[assumption.index] = name

    variables = {
        unit.key: model.new_bool_var(f"delete_{index}") for index, unit in enumerate(ordered_units)
    }
    for unit in ordered_units:
        for dependency in unit.dependencies:
            model.add(variables[unit.key] <= variables[dependency])
    for left, right in overlap_pairs:
        model.add(variables[left] + variables[right] <= 1)
    for key in mandatory_unit_keys:
        model.add(variables[key] == 1)

    line_expression = fixed_removed_lines + sum(
        unit.removed_lines * variables[unit.key] for unit in ordered_units
    )
    require("gold_patch_minimum", model.add(line_expression >= minimum_removed_lines))
    require("gold_patch_maximum", model.add(line_expression <= maximum_removed_lines))

    files_to_units: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for unit in ordered_units:
        for path in unit.files:
            if path not in fixed_files:
                files_to_units[path].append(variables[unit.key])
    active_file_variables = [
        _or_variable(model, name=f"modify_file_{index}", members=members)
        for index, (_path, members) in enumerate(sorted(files_to_units.items()))
    ]
    file_expression = len(fixed_files) + sum(active_file_variables)

    p2p_signature_counts = _signature_counts(
        units=ordered_units,
        inventory_mask=base_eligible_test_mask,
        mask_attribute="test_mask",
    )
    always_eligible = p2p_signature_counts.pop(0, 0)
    remaining_terms: list[cp_model.LinearExprT] = []
    excluded_terms: list[cp_model.LinearExprT] = []
    for group_index, (signature, count) in enumerate(sorted(p2p_signature_counts.items())):
        excluded = _or_variable(
            model,
            name=f"exclude_p2p_signature_{group_index}",
            members=[
                variables[ordered_units[unit_index].key]
                for unit_index in _iter_bit_indexes(signature)
            ],
        )
        excluded_terms.append(count * excluded)
        remaining_terms.append(count * (1 - excluded))
    remaining_expression = always_eligible + cp_model.LinearExpr.sum(  # pyright: ignore[reportUnknownMemberType]
        remaining_terms
    )
    p2p_upper_bound = base_count
    selected_p2p = model.new_int_var(
        0,
        p2p_upper_bound,
        "selected_p2p_test_points",
    )
    model.add(selected_p2p <= remaining_expression)
    require("minimum_p2p", model.add(selected_p2p >= minimum_p2p_test_points))
    if maximum_p2p_test_points is not None:
        require("maximum_p2p", model.add(selected_p2p <= maximum_p2p_test_points))

    fixed_seed_affected_mask = fixed_seed_affected_test_mask & seed_test_mask
    variable_seed_mask = seed_test_mask & ~fixed_seed_affected_mask
    seed_signature_counts = _signature_counts(
        units=ordered_units,
        inventory_mask=variable_seed_mask,
        mask_attribute="seed_test_mask",
    )
    # Signature zero never becomes F2P from a candidate deletion.
    seed_signature_counts.pop(0, None)
    affected_terms: list[cp_model.LinearExprT] = []
    for group_index, (signature, count) in enumerate(sorted(seed_signature_counts.items())):
        affected = _or_variable(
            model,
            name=f"affect_seed_signature_{group_index}",
            members=[
                variables[ordered_units[unit_index].key]
                for unit_index in _iter_bit_indexes(signature)
            ],
        )
        affected_terms.append(count * affected)
    estimated_f2p = fixed_seed_affected_mask.bit_count() + cp_model.LinearExpr.sum(  # pyright: ignore[reportUnknownMemberType]
        affected_terms
    )
    required_affected: int | None = None
    if seed_test_mask:
        require("minimum_f2p", model.add(estimated_f2p >= minimum_f2p_test_points))
        if maximum_f2p_test_points is not None:
            require("maximum_f2p", model.add(estimated_f2p <= maximum_f2p_test_points))
        if maximum_seed_pass_rate is not None:
            required_affected = math.ceil(seed_test_mask.bit_count() * (1 - maximum_seed_pass_rate))
            require("seed_break", model.add(estimated_f2p >= required_affected))
        if maximum_f2p_ratio is not None:
            ratio = Fraction(str(maximum_f2p_ratio)).limit_denominator(10_000)
            # f / (f + p) <= n / d  <=>  (d - n)f <= np
            require(
                "maximum_f2p_ratio",
                model.add(
                    (ratio.denominator - ratio.numerator) * estimated_f2p
                    <= ratio.numerator * selected_p2p
                ),
            )

    midpoint_twice = minimum_removed_lines + maximum_removed_lines
    possible_line_upper = fixed_removed_lines + sum(unit.removed_lines for unit in ordered_units)
    maximum_distance = max(
        abs(2 * fixed_removed_lines - midpoint_twice),
        abs(2 * possible_line_upper - midpoint_twice),
    )
    distance = model.new_int_var(0, maximum_distance, "gold_patch_center_distance_twice")
    model.add_abs_equality(distance, 2 * line_expression - midpoint_twice)

    # Exact lexicographic scalarization.  Once the line interval is feasible,
    # one midpoint-distance step outweighs every possible file and P2P gain;
    # one additional modified file outweighs every possible P2P gain.
    file_priority = p2p_upper_bound + 1
    maximum_files = len(fixed_files) + len(active_file_variables)
    distance_priority = (maximum_files + 1) * file_priority
    model.maximize(-distance * distance_priority + file_expression * file_priority + selected_p2p)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = random_seed
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    if status == cp_model.INFEASIBLE:
        sufficient_core = tuple(
            sorted(
                assumption_names.get(index, f"unknown_assumption_{index}")
                for index in solver.sufficient_assumptions_for_infeasibility()
            )
        )
        raise DeletionOptimizationInfeasibleError(
            DeletionInfeasibilityDiagnostics(
                status=solver.status_name(status),
                sufficient_constraint_core=sufficient_core,
                fixed_removed_lines=fixed_removed_lines,
                candidate_removed_lines_upper_bound=possible_line_upper,
                minimum_removed_lines=minimum_removed_lines,
                maximum_removed_lines=maximum_removed_lines,
                candidate_units=len(ordered_units),
                base_eligible_p2p_test_points=base_count,
                minimum_p2p_test_points=minimum_p2p_test_points,
                maximum_p2p_test_points=maximum_p2p_test_points,
                seed_test_points=seed_test_mask.bit_count(),
                fixed_seed_affected_test_points=fixed_seed_affected_mask.bit_count(),
                minimum_f2p_test_points=minimum_f2p_test_points,
                maximum_f2p_test_points=maximum_f2p_test_points,
                required_seed_affected_test_points=required_affected,
                maximum_f2p_ratio=maximum_f2p_ratio,
            )
        )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(
            f"deletion optimizer did not produce a feasible solution: {solver.status_name(status)}"
        )

    selected = tuple(
        unit.key for unit in ordered_units if solver.boolean_value(variables[unit.key])
    )
    conflicts = 0
    seed_conflicts = fixed_seed_affected_mask
    removed_lines = fixed_removed_lines
    removed_objects = fixed_removed_objects
    modified_files = set(fixed_files)
    for unit in ordered_units:
        if unit.key not in selected:
            continue
        conflicts |= unit.test_mask
        seed_conflicts |= unit.seed_test_mask
        removed_lines += unit.removed_lines
        removed_objects += unit.object_count
        modified_files.update(unit.files)
    eligible_mask = base_eligible_test_mask & ~conflicts
    return DeletionOptimizationResult(
        selected_unit_keys=selected,
        eligible_test_mask=eligible_mask,
        eligible_test_points=eligible_mask.bit_count(),
        excluded_test_points=base_count - eligible_mask.bit_count(),
        selected_p2p_test_points=solver.value(selected_p2p),
        estimated_f2p_test_points=(seed_conflicts & seed_test_mask).bit_count(),
        removed_lines=removed_lines,
        removed_objects=removed_objects,
        modified_files=len(modified_files),
        minimum_lines=minimum_removed_lines,
        maximum_lines=maximum_removed_lines,
        center_distance_twice=solver.value(distance),
        solve_seconds=time.monotonic() - started,
        status=solver.status_name(status),
        signature_groups=len(p2p_signature_counts) + len(seed_signature_counts),
    )
