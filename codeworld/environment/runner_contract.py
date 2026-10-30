"""Pure data contract exchanged with an environment-owned test Runner."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from codeworld.models import StrictModel

RunnerAction = Literal["collect", "run"]
TestOutcome = Literal["passed", "failed", "skipped", "error"]


class RunnerRequest(StrictModel):
    action: RunnerAction
    test_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def selection_matches_action(self) -> RunnerRequest:
        if self.action == "run" and not self.test_ids:
            raise ValueError("run requests require at least one test_id")
        if self.action != "run" and self.test_ids:
            raise ValueError("only run requests may contain test_ids")
        return self


class CollectedTestPoint(StrictModel):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def repository_relative_path(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("test paths may not contain control characters")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError("test paths must be repository-relative")
        if ".git" in path.parts:
            raise ValueError("test paths may not address Git metadata")
        return str(path)


class TestPointExecution(StrictModel):
    id: str = Field(min_length=1)
    outcome: TestOutcome
    duration_seconds: float | None = Field(default=None, ge=0)
    message: str = ""


class RunnerResponse(StrictModel):
    action: RunnerAction
    exit_code: int
    collected: tuple[CollectedTestPoint, ...] = ()
    results: tuple[TestPointExecution, ...] = ()
    errors: tuple[str, ...] = ()
    output: str = Field(
        default="",
        description=(
            "Bounded raw command-output tail. Required and nonempty whenever exit_code is nonzero; "
            "if the command emitted no bytes, provide a concise command-and-exit-code diagnostic."
        ),
    )

    @model_validator(mode="after")
    def unique_test_ids(self) -> RunnerResponse:
        collected_ids = [test.id for test in self.collected]
        result_ids = [test.id for test in self.results]
        if len(collected_ids) != len(set(collected_ids)):
            raise ValueError("collect response contains duplicate test IDs")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("run response contains duplicate test results")
        if self.action == "collect" and self.results:
            raise ValueError("collect responses may not contain execution results")
        if self.exit_code != 0 and not self.output.strip():
            raise ValueError("a nonzero response must preserve a bounded raw command-output tail")
        return self

    @property
    def passed_ids(self) -> tuple[str, ...]:
        return tuple(result.id for result in self.results if result.outcome == "passed")
