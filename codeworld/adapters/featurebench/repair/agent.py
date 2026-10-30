"""One Pydantic AI run that iterates through real Mask validation submits."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from codeworld.adapters.featurebench.repair.capability import MaskRepairSubmission
from codeworld.adapters.featurebench.repair.models import MaskRepairCompletion
from codeworld.adapters.featurebench.repair.service import MaskRepairService
from codeworld.agents.factory import AgentFactory, AgentProgress
from codeworld.agents.feedback import AgentFeedback, AgentFeedbackError, FeedbackIssue
from codeworld.config import AgentFactorySettings
from codeworld.runtime.base import ContainerSessionProtocol

_PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "prompts"


@cache
def _prompt(name: str) -> str:
    path = _PROMPT_DIRECTORY / f"repair.{name}.md"
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"empty FeatureBench Repair prompt asset: {path}")
    return content


async def run_mask_repair_agent(
    *,
    agents: AgentFactory[AgentFactorySettings],
    session: ContainerSessionProtocol,
    service: MaskRepairService,
    run_timeout_seconds: float,
    metadata: dict[str, str],
    progress: AgentProgress | None = None,
) -> MaskRepairCompletion:
    """Run the complete inspect/edit/submit loop in one model conversation."""
    environment = Environment(
        loader=FileSystemLoader(_PROMPT_DIRECTORY),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    user_prompt = environment.get_template("repair.user.md").render(
        workspace_root=service.workspace.container_root,
        feedback_dir=service.current_feedback_dir,
        feedback_readme=f"{service.current_feedback_dir}/README.md",
        proposal_path=service.workspace.proposal_container_path,
    )

    async def validate_completion(output: MaskRepairCompletion) -> None:
        if (
            service.accepted_attempt_id is None
            or output.accepted_attempt != service.accepted_attempt_id
        ):
            raise AgentFeedbackError(
                AgentFeedback(
                    issues=(
                        FeedbackIssue(
                            code="featurebench.repair_unverified_completion",
                            path=("accepted_attempt",),
                            message="The referenced attempt has not passed submit validation.",
                            correction=(
                                "Continue editing the proposal and calling submit_repair_file until "
                                "the tool returns accepted=true, then copy that exact attempt ID."
                            ),
                            context={
                                "current_attempt": service.current_attempt_id,
                                "current_feedback_dir": service.current_feedback_dir,
                            },
                        ),
                    )
                )
            )

    return await agents.run_structured(
        session=session,
        prompt=user_prompt,
        instructions=_prompt("instructions"),
        output_type=MaskRepairCompletion,
        writable=True,
        extra_capabilities=(
            MaskRepairSubmission(
                service=service,
                timeout_seconds=run_timeout_seconds,
            ),
        ),
        output_validator=validate_completion,
        progress=progress,
        run_timeout_seconds=run_timeout_seconds,
        agent_name="featurebench_mask_repair",
        metadata=metadata,
    )
