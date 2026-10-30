"""Output-driven work selection shared by every resumable stage."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from enum import StrEnum
from functools import cache

from codeworld.errors import ContractError


class WorkState(StrEnum):
    """Canonical durable outcome of one logical work item."""

    UNSEEN = "unseen"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class WorkSelection:
    """One named predicate over states recovered from durable outputs."""

    name: str
    states: frozenset[WorkState]

    def accepts(self, state: WorkState) -> bool:
        return state in self.states


class WorkSelectionRegistry:
    """Named policies plus canonical combinations of durable non-success states."""

    def __init__(self) -> None:
        self._selections: dict[str, WorkSelection] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._selections))

    def register(self, selection: WorkSelection) -> None:
        name = selection.name.strip()
        if not name or name != selection.name:
            raise ValueError("work selection name must be non-empty without surrounding whitespace")
        if name in self._selections:
            raise ContractError(f"work selection is already registered: {name}")
        self._selections[name] = selection

    def resolve(self, name: str) -> WorkSelection:
        registered = self._selections.get(name)
        if registered is not None:
            return registered

        atoms = name.split("+")
        if not atoms or any(not atom.strip() for atom in atoms):
            raise ContractError(
                f"invalid work selection expression {name!r}; join states with one '+'"
            )
        selectable = (
            WorkState.UNSEEN,
            WorkState.FAILED,
            WorkState.REJECTED,
            WorkState.INTERRUPTED,
        )
        by_name = {state.value: state for state in selectable}
        states: set[WorkState] = set()
        for raw_atom in atoms:
            atom = raw_atom.strip()
            state = by_name.get(atom)
            if state is None:
                suggestion = get_close_matches(atom, tuple(by_name), n=1)
                hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
                available = ", ".join((*self.names, *by_name))
                raise ContractError(
                    f"unknown work selection state {atom!r} in {name!r}{hint}; "
                    f"available policies and states: {available}"
                )
            states.add(state)
        canonical = "+".join(state.value for state in selectable if state in states)
        return WorkSelection(canonical, frozenset(states))


class OutcomeIndex:
    """In-memory projection of append-only outputs keyed by stable aliases."""

    def __init__(self) -> None:
        self._states: dict[str, WorkState] = {}

    def record(self, aliases: str | tuple[str, ...], state: WorkState) -> None:
        keys = (aliases,) if isinstance(aliases, str) else aliases
        for key in keys:
            if not key:
                raise ValueError("work outcome alias must be non-empty")
            previous = self._states.get(key)
            # A valid success artifact is authoritative even if an older or newer
            # diagnostic failure for the same logical object also remains append-only.
            if previous is WorkState.SUCCEEDED and state is not WorkState.SUCCEEDED:
                continue
            self._states[key] = state

    def state_of(self, aliases: str | tuple[str, ...]) -> WorkState:
        keys = (aliases,) if isinstance(aliases, str) else aliases
        observed = [self._states[key] for key in keys if key in self._states]
        if WorkState.SUCCEEDED in observed:
            return WorkState.SUCCEEDED
        return observed[-1] if observed else WorkState.UNSEEN


@cache
def default_work_selection_registry() -> WorkSelectionRegistry:
    """Return the shared registry extensible by future stage or plugin packages."""

    registry = WorkSelectionRegistry()
    registry.register(
        WorkSelection(
            name="unseen",
            states=frozenset((WorkState.UNSEEN,)),
        )
    )
    registry.register(
        WorkSelection(
            name="non-success",
            states=frozenset(
                (
                    WorkState.UNSEEN,
                    WorkState.FAILED,
                    WorkState.REJECTED,
                    WorkState.INTERRUPTED,
                )
            ),
        )
    )
    return registry
