from __future__ import annotations

import pytest

from codeworld.errors import ContractError
from codeworld.selection import (
    OutcomeIndex,
    WorkSelection,
    WorkSelectionRegistry,
    WorkState,
    default_work_selection_registry,
)


def test_default_selection_only_accepts_unseen_work() -> None:
    selection = default_work_selection_registry().resolve("unseen")

    assert selection.accepts(WorkState.UNSEEN)
    assert not selection.accepts(WorkState.SUCCEEDED)
    assert not selection.accepts(WorkState.FAILED)
    assert not selection.accepts(WorkState.REJECTED)
    assert not selection.accepts(WorkState.INTERRUPTED)


def test_non_success_selection_accepts_every_state_except_success() -> None:
    selection = default_work_selection_registry().resolve("non-success")

    assert selection.accepts(WorkState.FAILED)
    assert selection.accepts(WorkState.REJECTED)
    assert selection.accepts(WorkState.INTERRUPTED)
    assert selection.accepts(WorkState.UNSEEN)
    assert not selection.accepts(WorkState.SUCCEEDED)


def test_selection_composes_and_canonicalizes_non_success_states() -> None:
    selection = default_work_selection_registry().resolve(
        "rejected + unseen + rejected + interrupted"
    )

    assert selection.name == "unseen+rejected+interrupted"
    assert selection.states == frozenset(
        {
            WorkState.UNSEEN,
            WorkState.REJECTED,
            WorkState.INTERRUPTED,
        }
    )
    assert not selection.accepts(WorkState.FAILED)
    assert not selection.accepts(WorkState.SUCCEEDED)


@pytest.mark.parametrize("expression", ("unseen+", "+failed", "unseen++failed"))
def test_selection_rejects_empty_expression_atoms(expression: str) -> None:
    with pytest.raises(ContractError, match="invalid work selection expression"):
        default_work_selection_registry().resolve(expression)


def test_selection_reports_a_close_state_name_for_typos() -> None:
    with pytest.raises(ContractError, match="did you mean 'rejected'"):
        default_work_selection_registry().resolve("unseen+rejcted")


def test_selection_does_not_reprocess_durable_successes() -> None:
    with pytest.raises(ContractError, match="unknown work selection state 'succeeded'"):
        default_work_selection_registry().resolve("succeeded")


def test_selection_registry_is_extensible_without_scheduler_changes() -> None:
    registry = WorkSelectionRegistry()
    registry.register(WorkSelection("failed-only", frozenset((WorkState.FAILED,))))

    assert registry.resolve("failed-only").accepts(WorkState.FAILED)
    with pytest.raises(ContractError, match="unknown work selection"):
        registry.resolve("missing")


def test_success_dominates_append_only_failure_history_for_every_alias() -> None:
    index = OutcomeIndex()
    index.record(("github:1", "repository:one"), WorkState.FAILED)
    index.record("repository:one", WorkState.SUCCEEDED)
    index.record("repository:one", WorkState.REJECTED)

    assert index.state_of("repository:one") is WorkState.SUCCEEDED
    assert index.state_of(("github:1", "repository:one")) is WorkState.SUCCEEDED
