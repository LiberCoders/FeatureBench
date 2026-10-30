"""Deterministic reconstruction of a public FeatureBench task workspace."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from codeworld.adapters.featurebench.models import FeatureBenchPayload
from codeworld.errors import ContractError
from codeworld.models import TaskRecord
from codeworld.runtime.base import ContainerSessionProtocol


class FeatureBenchTaskRuntime:
    """Prepare or transition an ephemeral task container without calling a model."""

    @staticmethod
    async def apply_patch(
        session: ContainerSessionProtocol,
        patch: str,
        *,
        reverse: bool = False,
    ) -> bool:
        arguments = ["git", "apply", "--binary", "--whitespace=nowarn"]
        if reverse:
            arguments.append("--reverse")
        arguments.append("-")
        result = await session.exec(arguments, input_bytes=patch.encode(), check=False)
        return result.exit_code == 0

    async def prepare(
        self,
        session: ContainerSessionProtocol,
        task: TaskRecord,
        payload: BaseModel,
    ) -> None:
        """Expose only the masked production code and public tests to a solution Agent."""
        typed = cast(FeatureBenchPayload, payload)
        await session.clean_workspace()
        if not await self.apply_patch(session, typed.gold_patch, reverse=True):
            raise ContractError(f"cannot reconstruct masked workspace for {task.task_id}")
        if not await self.apply_patch(session, typed.hidden_tests_patch, reverse=True):
            raise ContractError(f"cannot hide F2P tests for {task.task_id}")
        await session.seal_workspace()

    async def restore_verification_assets(
        self,
        session: ContainerSessionProtocol,
        payload: FeatureBenchPayload,
    ) -> bool:
        """Restore held-out tests only after the candidate has been applied."""
        return await self.apply_patch(session, payload.hidden_tests_patch)
