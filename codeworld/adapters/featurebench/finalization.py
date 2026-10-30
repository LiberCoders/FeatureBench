"""Build the one canonical, self-contained FeatureBench task record."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from codeworld.adapters.featurebench.models import (
    FeatureBenchPayload,
    MaskPlan,
    QualityReport,
)
from codeworld.adapters.featurebench.quality import build_featurebench_metrics
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.models import TaskRecord


def assemble_task_record(
    *,
    environment: EnvironmentRecord,
    prompt: str,
    plan: MaskPlan,
    quality: QualityReport,
    verified_points: Mapping[str, CollectedTestPoint],
    gold_patch: str,
    hidden_tests_patch: str,
    test_timeout_seconds: float,
) -> TaskRecord:
    """Return the exact JSONL record, excluding all generation-only state."""
    task_id = (
        "featurebench-"
        + hashlib.sha256(
            (
                environment.environment_id
                + "\0"
                + plan.seed_test_file
                + "\0"
                + gold_patch
                + "\0"
                + hidden_tests_patch
            ).encode()
        ).hexdigest()[:24]
    )
    payload = FeatureBenchPayload(
        type="featurebench",
        prompt=prompt,
        gold_patch=gold_patch,
        hidden_tests_patch=hidden_tests_patch,
        fail_to_pass=tuple(verified_points[test_id] for test_id in quality.fail_to_pass),
        pass_to_pass=tuple(verified_points[test_id] for test_id in quality.pass_to_pass),
        test_timeout_seconds=test_timeout_seconds,
        metrics=build_featurebench_metrics(
            gold_patch=gold_patch,
            quality=quality,
            plan=plan,
        ),
    )
    record = TaskRecord(
        task_id=task_id,
        environment_id=environment.environment_id,
        image=environment.image,
        runtime=environment.runtime,
        adapter=payload.model_dump(mode="json"),
    )
    # The Gold closure must consume the same representation written to JSONL, not
    # richer generation-time objects that will not exist on another machine.
    return TaskRecord.model_validate_json(record.model_dump_json(exclude_none=True))
