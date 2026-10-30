"""Structured, centrally rendered feedback for model-correctable output errors."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import yaml
from pydantic import Field, ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import AbstractCapability, RawOutput
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext

from codeworld.errors import ContractError
from codeworld.models import StrictModel


class FeedbackTemplate(StrictModel):
    """One model-facing message loaded from a YAML asset."""

    code: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]+$")
    message: str = Field(min_length=1)
    correction: str = Field(min_length=1)


class FeedbackTemplateRegistry:
    """Extensible catalog of stable feedback codes and model-facing corrections."""

    def __init__(self) -> None:
        self._templates: dict[str, tuple[str, str]] = {}

    def register(self, code: str, message: str, correction: str) -> None:
        normalized = str(code)
        if not normalized or normalized in self._templates:
            raise ValueError(f"duplicate or empty feedback code: {normalized!r}")
        if not message.strip() or not correction.strip():
            raise ValueError(f"feedback template {normalized!r} must be non-empty")
        self._templates[normalized] = (message, correction)

    def resolve(self, code: str) -> tuple[str, str]:
        normalized = str(code)
        try:
            return self._templates[normalized]
        except KeyError as exc:
            raise ValueError(f"unregistered feedback code: {normalized!r}") from exc

    def __contains__(self, code: object) -> bool:
        return isinstance(code, str) and code in self._templates

    @classmethod
    def from_directory(cls, directory: Path) -> FeedbackTemplateRegistry:
        registry = cls()
        paths = tuple(sorted(directory.glob("*.yaml")))
        if not paths:
            raise ValueError(f"no feedback YAML assets found in {directory}")
        for path in paths:
            try:
                document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(document, list):
                    raise ValueError("feedback asset root must be a list")
                templates = tuple(
                    FeedbackTemplate.model_validate(item) for item in cast(list[object], document)
                )
            except (OSError, yaml.YAMLError, ValueError) as exc:
                raise ValueError(f"invalid feedback asset {path}: {exc}") from exc
            for template in templates:
                registry.register(
                    template.code,
                    template.message.strip(),
                    template.correction.strip(),
                )
        return registry


class FeedbackIssue(StrictModel):
    """One precise output correction that is useful to both code and a model."""

    code: str = Field(min_length=1)
    path: tuple[str | int, ...] = ()
    message: str = Field(min_length=1)
    correction: str = Field(min_length=1)
    expected: str | None = None
    actual: str | None = None
    evidence: str | None = None
    context: dict[str, str] = Field(default_factory=dict)


class AgentFeedback(StrictModel):
    """Structured feedback persisted and rendered through one path."""

    issues: tuple[FeedbackIssue, ...] = Field(min_length=1)


def from_validation_error(error: ValidationError) -> AgentFeedback:
    """Preserve Pydantic's typed locations and error codes in our common envelope."""
    issues: list[FeedbackIssue] = []
    for entry in error.errors(include_url=False, include_input=False):
        raw_context = entry.get("ctx") or {}
        context = {str(key): str(value) for key, value in raw_context.items()}
        issues.append(
            FeedbackIssue(
                code=f"schema.{entry['type']}",
                path=tuple(entry.get("loc", ())),
                message=str(entry["msg"]),
                correction="Correct this field and return the complete structured object again.",
                context=context,
            )
        )
    return AgentFeedback(issues=tuple(issues))


def render_feedback(feedback: AgentFeedback) -> str:
    """Render the sole text transport accepted by Pydantic AI's ModelRetry."""
    payload = feedback.model_dump(mode="json")
    return "Correct every issue in this structured feedback and retry:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


class AgentFeedbackError(ContractError):
    """Domain validation failure retaining its structured form until ModelRetry."""

    def __init__(self, feedback: AgentFeedback) -> None:
        self.feedback = feedback
        super().__init__(render_feedback(feedback))


def reject(issue_: FeedbackIssue) -> NoReturn:
    """Reject one Agent output with typed, catalog-backed feedback."""
    raise AgentFeedbackError(AgentFeedback(issues=(issue_,)))


@dataclass
class StructuredOutputFeedback(AbstractCapability[object]):
    """Use Pydantic AI's output-error hook to normalize schema retry feedback."""

    on_feedback: Callable[[AgentFeedback], Awaitable[None]] | None = None

    async def on_output_validate_error(
        self,
        ctx: RunContext[object],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        del ctx, output_context, output
        if isinstance(error, ValidationError):
            feedback = from_validation_error(error)
            if self.on_feedback is not None:
                await self.on_feedback(feedback)
            raise ModelRetry(render_feedback(feedback)) from error
        raise error
