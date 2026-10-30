"""FeatureBench-owned Pydantic AI capability for Issue file submission."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from codeworld.adapters.featurebench.issue.service import IssueAuthorService


@dataclass
class IssueAuthorSubmission(AbstractCapability[object]):
    """Register the adapter's host validation tool with a generic Agent."""

    service: IssueAuthorService = field(repr=False)
    timeout_seconds: float

    def get_instructions(self) -> str:
        return (
            "submit_issue_file(path) reads one Issue JSON file under outbox/, validates its schema, "
            "exact interface coverage, and user-facing content, then returns only acceptance status "
            "and a relative feedback directory. Keep large Issue content in files."
        )

    def get_toolset(self) -> FunctionToolset[object]:
        toolset: FunctionToolset[object] = FunctionToolset(id="featurebench_issue_submission")

        async def submit_issue_file(path: str) -> str:
            """Submit an Issue JSON file for deterministic host validation.

            Args:
                path: Relative path under the workspace outbox directory.
            """
            result = await self.service.submit_file(path)
            return result.model_dump_json()

        toolset.add_function(
            submit_issue_file,
            sequential=True,
            timeout=self.timeout_seconds,
        )
        return toolset
