"""FeatureBench's Adapter-owned Gold-closure quality gate."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import cast

from pydantic import BaseModel

from codeworld.adapters.base import GeneratorContext, QualityGateResult
from codeworld.adapters.featurebench.models import FeatureBenchPayload
from codeworld.adapters.featurebench.runtime import FeatureBenchTaskRuntime
from codeworld.environment.runner import EnvironmentRunner
from codeworld.models import TaskRecord


class FeatureBenchQualityGate:
    """Prove that Gold reconstructs correctly and passes every required test."""

    def __init__(self, runtime: FeatureBenchTaskRuntime | None = None) -> None:
        self.runtime = runtime or FeatureBenchTaskRuntime()

    async def check(
        self,
        task: TaskRecord,
        payload: BaseModel,
        context: GeneratorContext,
    ) -> QualityGateResult:
        typed = cast(FeatureBenchPayload, payload)
        gold_patch = typed.gold_patch
        async with context.runtime.session(
            task.image,
            runtime_spec=task.runtime,
        ) as session:
            await self.runtime.prepare(session, task, typed)
            if not gold_patch.strip():
                return QualityGateResult(passed=False, message="Gold patch is empty")
            protected_tests = typed.protected_test_paths
            patch_paths = self.candidate_patch_paths(gold_patch)
            test_like_paths = {
                path
                for path in patch_paths
                if "tests" in PurePosixPath(path).parts
                or PurePosixPath(path).name.startswith("test_")
                or PurePosixPath(path).name.endswith("_test.py")
            }
            if protected_tests & patch_paths or test_like_paths:
                return QualityGateResult(
                    passed=False,
                    message="Gold patch modifies protected tests",
                )
            if not await self.runtime.apply_patch(session, gold_patch):
                return QualityGateResult(
                    passed=False,
                    message="Gold patch does not apply to the masked workspace",
                )
            if not await self.runtime.restore_verification_assets(session, typed):
                return QualityGateResult(
                    passed=False,
                    message="Gold conflicts with hidden test restoration",
                )
            report = await EnvironmentRunner(session).run(
                typed.required_test_ids,
                timeout_seconds=typed.test_timeout_seconds,
            )
            required = set(typed.required_test_ids)
            passing = set(report.passed_ids)
            passed = report.exit_code == 0 and not report.errors and required <= passing
            diagnostics = "\n".join(
                detail
                for detail in (
                    "; ".join(report.errors[:10]),
                    report.output[-4000:],
                )
                if detail
            )
            return QualityGateResult(
                passed=passed,
                message="" if passed else diagnostics[-5000:] or "required tests did not pass",
                metrics={
                    "f2p_total": len(typed.fail_to_pass),
                    "p2p_total": len(typed.pass_to_pass),
                    "required_passed": len(required & passing),
                    "required_total": len(required),
                    "test_exit_code": report.exit_code,
                    "runner_errors": len(report.errors),
                },
            )

    @staticmethod
    def candidate_patch_paths(patch: str) -> set[str]:
        paths: set[str] = set()
        for line in patch.splitlines():
            if not line.startswith(("--- ", "+++ ")):
                continue
            path = line[4:]
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                paths.add(path[2:])
        return paths
