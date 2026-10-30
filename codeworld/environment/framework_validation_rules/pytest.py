"""Pytest public-ID inventory validation."""

from __future__ import annotations

import json
from collections import defaultdict

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.feedback import FeedbackCode, issue
from codeworld.environment.framework_validation import (
    FrameworkCollectionValidationContext,
    FrameworkValidationRegistry,
)
from codeworld.environment.frameworks.pytest import parse_pytest_public_id


def _pytest_public_ids(
    context: FrameworkCollectionValidationContext,
) -> AgentFeedback | None:
    occurrences: dict[tuple[str, str], set[int]] = defaultdict(set)
    malformed: list[str] = []
    for point in context.collected:
        try:
            parsed = parse_pytest_public_id(point.id)
        except ValueError:
            malformed.append(point.id)
            continue
        occurrences[(point.path, parsed.native_base)].add(parsed.occurrence)

    non_contiguous = [
        {
            "path": path,
            "native_base": native_base,
            "observed": sorted(observed),
            "expected": list(range(len(observed))),
        }
        for (path, native_base), observed in sorted(occurrences.items())
        if observed != set(range(len(observed)))
    ]
    if not malformed and not non_contiguous:
        return None
    evidence = {
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:20],
        "non_contiguous_count": len(non_contiguous),
        "non_contiguous_examples": non_contiguous[:10],
    }
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_PUBLIC_ID,
                path=("runner",),
                expected=(
                    "every pytest item uses <native_base>#<zero-based occurrence>; replace the "
                    "final item leaf with item.originalname when pytest provides it, then number "
                    "items sharing that base in collection order from 0 without gaps"
                ),
                actual=(
                    f"{len(malformed)} malformed IDs and "
                    f"{len(non_contiguous)} non-contiguous occurrence groups"
                ),
                evidence=json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
            ),
        )
    )


def register(registry: FrameworkValidationRegistry) -> None:
    registry.register_collection(
        "pytest.public-ids",
        _pytest_public_ids,
        frameworks=("pytest",),
    )
