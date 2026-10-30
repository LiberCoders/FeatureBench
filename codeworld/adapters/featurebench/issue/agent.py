"""One file-working Pydantic AI conversation for FeatureBench Issue authoring."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic_ai_harness.filesystem import FileSystem

from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.issue.capability import IssueAuthorSubmission
from codeworld.adapters.featurebench.issue.models import IssueAuthorCompletion
from codeworld.adapters.featurebench.issue.service import IssueAuthorService
from codeworld.adapters.featurebench.issue.workspace import IssueAuthorWorkspace
from codeworld.adapters.featurebench.models import IssueDraft, MaskPlan
from codeworld.agents.factory import AgentFactory, AgentProgress
from codeworld.agents.feedback import AgentFeedback, AgentFeedbackError, FeedbackIssue
from codeworld.config import AgentFactorySettings
from codeworld.errors import ContractError
from codeworld.runtime.base import ContainerSessionProtocol

_PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "prompts"


@cache
def _prompt(name: str) -> str:
    path = _PROMPT_DIRECTORY / f"issue.{name}.md"
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"empty FeatureBench Issue prompt asset: {path}")
    return content


async def author_issue(
    *,
    agents: AgentFactory[AgentFactorySettings],
    session: ContainerSessionProtocol,
    index: RepositoryIndex,
    plan: MaskPlan,
    run_timeout_seconds: float,
    workspace_root: Path,
    metadata: dict[str, str],
    progress: AgentProgress | None = None,
) -> IssueDraft:
    """Let one Agent iteratively author and submit a file, then return accepted content."""
    workspace = IssueAuthorWorkspace(workspace_root)
    workspace.prepare(index=index, plan=plan)
    service = IssueAuthorService(workspace=workspace, plan=plan)
    environment = Environment(
        loader=FileSystemLoader(_PROMPT_DIRECTORY),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    user_prompt = environment.get_template("issue.user.md").render(
        top_count=len(plan.top_objects),
        source_file_count=len({item.path for item in plan.top_objects}),
        protocol_path=workspace.protocol_path,
        manifest_path=workspace.manifest_path,
        draft_path=workspace.draft_path,
    )

    async def validate_completion(output: IssueAuthorCompletion) -> None:
        if service.accepted_issue is None or output.path != service.accepted_path:
            raise AgentFeedbackError(
                AgentFeedback(
                    issues=(
                        FeedbackIssue(
                            code="featurebench.issue.unverified_completion",
                            path=("path",),
                            message="The referenced Issue file has not passed submit validation.",
                            correction=(
                                "Continue editing the Issue file and call submit_issue_file until "
                                "it returns accepted=true, then return that exact relative path."
                            ),
                            context={"draft_path": workspace.draft_path},
                        ),
                    )
                )
            )

    await agents.run_structured(
        session=session,
        prompt=user_prompt,
        instructions=_prompt("instructions"),
        output_type=IssueAuthorCompletion,
        container_tools=False,
        extra_capabilities=(
            FileSystem(
                root_dir=workspace.root,
                protected_patterns=(),
                max_read_lines=2_000,
                max_search_results=1_000,
                max_find_results=1_000,
            ),
            IssueAuthorSubmission(
                service=service,
                timeout_seconds=run_timeout_seconds,
            ),
        ),
        output_validator=validate_completion,
        progress=progress,
        writable=False,
        run_timeout_seconds=run_timeout_seconds,
        agent_name="featurebench_issue_author",
        metadata=metadata,
    )
    if service.accepted_issue is None:
        raise ContractError("Issue Author completed without an accepted Issue file")
    return service.accepted_issue
