"""Typed host-side analysis and artifact authoring for cross-language environments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, create_model, field_validator, model_validator

from codeworld.agents.factory import AgentFactory
from codeworld.agents.feedback import (
    AgentFeedback,
    AgentFeedbackError,
    reject,
)
from codeworld.config import AgentSettings, BaseImageSettings, EnvironmentSettings
from codeworld.environment.artifact_validation import (
    DEFAULT_ARTIFACT_VALIDATORS,
    ArtifactRepairValidationContext,
    ArtifactValidationContext,
)
from codeworld.environment.ecosystems import (
    DEFAULT_ECOSYSTEM_PROMPTS,
    EcosystemPromptGuidance,
    ecosystem_prompt_guidance,
)
from codeworld.environment.feedback import FeedbackCode, issue
from codeworld.environment.framework_validation import validate_framework_collection
from codeworld.environment.prompt_assets import PromptAssetName, load_prompt_asset
from codeworld.environment.runner import (
    EnvironmentRunner,
    assert_stable_collection,
    probe_runner_failure_contract,
    probe_runner_large_request_transport,
)
from codeworld.environment.runner_contract import RunnerRequest, RunnerResponse
from codeworld.environment.runner_review import (
    RunnerIntegrityReview,
    RunnerReviewContext,
    review_runner_integrity,
    runner_review_feedback,
)
from codeworld.errors import (
    DOCKER_HARNESS_STEP_MARKER,
    CommandTimeoutError,
    ContractError,
    RunnerCollectionUnstableError,
)
from codeworld.models import RunnerLimits, RuntimeResources, RuntimeSpec, StrictModel
from codeworld.runtime.container import ContainerSession

_FROM = re.compile(
    r"^\s*FROM\s+(?:--platform=[^\s]+\s+)?([^\s]+)",
    re.IGNORECASE | re.MULTILINE,
)
SOURCE_MARKER = "# FEATUREBENCH_SOURCE"
_FORBIDDEN = re.compile(
    r"(?:pydantic[-_ ]?ai|openhands|openhindsight|openai_api_key|anthropic_api_key|google_api_key)",
    re.IGNORECASE,
)
_FORBIDDEN_DOCKERFILE = re.compile(
    r"/opt/featurebench",
    re.IGNORECASE | re.MULTILINE,
)
_CONTEXT_TRANSFER = re.compile(
    r"^\s*(?P<kind>ADD|COPY)(?:\s+--from=(?P<from>[^\s]+))?\s+",
    re.IGNORECASE | re.MULTILINE,
)
_SECRET_DECLARATION = re.compile(
    r"^\s*(?:ARG|ENV)\s+[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ARG_DECLARATION = re.compile(
    r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<default>\s*=\s*[^\r\n]*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_AUTOMATIC_BUILD_ARGUMENTS = frozenset(
    {
        "BUILDARCH",
        "BUILDOS",
        "BUILDPLATFORM",
        "BUILDVARIANT",
        "TARGETARCH",
        "TARGETOS",
        "TARGETPLATFORM",
        "TARGETVARIANT",
    }
)
_GIT_BASELINE_MUTATION = re.compile(
    r"(?:/testbed/\.git\b|"
    r"\bgit\s+(?:init|add|commit|reset|clean|checkout|restore|config|update-index|stash)\b)",
    re.IGNORECASE,
)
_ACTIVATION_CONTROL_FLOW = re.compile(
    r"(^|[;&|]\s*)(?:exit|exec|return)(?:\s|$)", re.IGNORECASE | re.MULTILINE
)
_ENV_PATH = re.compile(
    r"^\s*ENV\s+(?:[^\n]*?\s+)?PATH=(?P<value>[^\s]+)",
    re.IGNORECASE | re.MULTILINE,
)
_EXTERNAL_RUNTIME_REFERENCE = re.compile(
    r"(?:^|[^A-Za-z0-9_])/?testbed/(?:activate|runner)(?:[^A-Za-z0-9_]|$)",
    re.IGNORECASE,
)


def _relative_paths(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = value.strip()
        path = PurePosixPath(candidate)
        if (
            not candidate
            or any(character in candidate for character in ("\x00", "\n", "\r"))
            or path.is_absolute()
            or ".." in path.parts
            or str(path) == "."
        ):
            raise ValueError("repository evidence paths must be specific non-root relative paths")
        rendered = str(path)
        if rendered not in seen:
            seen.add(rendered)
            normalized.append(rendered)
    return tuple(normalized)


class ClonePlan(StrictModel):
    """Declarative clone input rendered by the data factory rather than Agent shell text."""

    repository_url: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    destination: Literal["/testbed"]
    strategy: Literal["full", "partial", "shallow"]
    recurse_submodules: bool
    fetch_lfs: bool


class EnvironmentAnalysis(StrictModel):
    """Repository facts and evidence-grounded choices requiring semantic inspection."""

    primary_language: str | None = Field(
        description=(
            "Lowercase primary implementation language used to build and test the repository; "
            "null only when a rejected repository contains no language evidence."
        ),
    )
    languages: tuple[str, ...] = Field(
        description=(
            "All lowercase languages required by one coherent accepted unit-test suite for the "
            "primary project; do not include independent optional subprojects with separate test "
            "stacks. Empty only when a rejected repository contains no language evidence."
        ),
    )
    runtime_requirement: str | None = Field(
        description=(
            "Repository-evidenced runtime version or version constraint, not the selected "
            "base image's version."
        )
    )
    build_systems: tuple[str, ...] = Field(
        description="Repository-evidenced build systems used by the accepted test path."
    )
    package_managers: tuple[str, ...] = Field(
        description="Repository-evidenced dependency managers used by the accepted test path."
    )
    test_frameworks: tuple[str, ...] = Field(
        description="Real unit-test frameworks used by the coherent accepted suite."
    )
    manifest_paths: tuple[str, ...] = Field(
        description="Inspected repository-relative dependency or build manifest paths."
    )
    ci_paths: tuple[str, ...] = Field(
        description="Inspected repository-relative CI or canonical automation paths."
    )
    test_evidence_paths: tuple[str, ...] = Field(
        description="Inspected repository-relative paths proving the test framework and suite."
    )
    gpu_requirement: Literal["none", "optional", "required"] = Field(
        description="Whether a meaningful accepted unit-test suite requires GPU hardware."
    )
    gpu_evidence_paths: tuple[str, ...] = Field(
        description="GPU-related repository evidence; empty when gpu_requirement is none."
    )
    gpu_reason: str | None = Field(
        description="Evidence-based CPU/GPU test-scope explanation; null when GPU is irrelevant."
    )
    selected_base_image_name: str | None = Field(
        description=(
            "Exact compatible base-image option name for an accepted repository; null when the "
            "repository is rejected because no unit-test framework or compatible option exists."
        ),
    )
    rejection_reason: str | None = Field(
        description=(
            "Concise evidence-based reason when selected_base_image_name is null; null for an "
            "accepted repository."
        ),
    )

    @field_validator("primary_language")
    @classmethod
    def normalized_primary_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip().lower():
            raise ValueError("primary_language must be lowercase without surrounding whitespace")
        return value

    @field_validator("languages")
    @classmethod
    def normalized_languages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            language = value.strip().lower()
            if not language:
                raise ValueError("languages may not contain empty values")
            if language not in seen:
                seen.add(language)
                normalized.append(language)
        return tuple(normalized)

    @field_validator("build_systems", "package_managers", "test_frameworks")
    @classmethod
    def normalized_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if any(not value for value in normalized):
            raise ValueError("analysis labels may not contain empty values")
        return normalized

    @field_validator("runtime_requirement", "gpu_reason", "rejection_reason")
    @classmethod
    def normalized_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("optional analysis text must be null or non-empty")
        return normalized

    @field_validator("selected_base_image_name")
    @classmethod
    def normalized_base_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError("selected base-image name may not have surrounding whitespace")
        if not value:
            raise ValueError("selected base-image name must be null or non-empty")
        return value

    @field_validator("manifest_paths", "ci_paths", "test_evidence_paths", "gpu_evidence_paths")
    @classmethod
    def repository_relative_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _relative_paths(values)

    @model_validator(mode="after")
    def primary_is_listed(self) -> EnvironmentAnalysis:
        if self.primary_language is None and self.languages:
            raise ValueError("non-empty languages requires primary_language")
        if self.primary_language is not None and self.primary_language not in self.languages:
            raise ValueError("primary_language must also appear in languages")
        if self.test_frameworks:
            if self.primary_language is None or not self.languages:
                raise ValueError("an identified unit-test framework requires language evidence")
            if not self.test_evidence_paths:
                raise ValueError("an identified unit-test framework requires repository evidence")
        else:
            if self.selected_base_image_name is not None:
                raise ValueError(
                    "a repository without a unit-test framework must not select a base image"
                )
        if self.selected_base_image_name is None and self.rejection_reason is None:
            raise ValueError("a rejected repository requires rejection_reason")
        if self.selected_base_image_name is not None and self.rejection_reason is not None:
            raise ValueError("an accepted repository must not include rejection_reason")
        if self.selected_base_image_name is not None and (
            self.primary_language is None or not self.languages
        ):
            raise ValueError("an accepted repository requires language evidence")
        if self.gpu_requirement == "none" and (
            self.gpu_evidence_paths or self.gpu_reason is not None
        ):
            raise ValueError(
                "GPU evidence and reason are only valid for optional or required GPU support"
            )
        if self.gpu_requirement != "none" and (not self.gpu_evidence_paths or not self.gpu_reason):
            raise ValueError("optional or required GPU support needs evidence paths and a reason")
        return self

    @property
    def has_unit_test_framework(self) -> bool:
        """Derive eligibility without asking the Agent to repeat the same decision."""
        return bool(self.test_frameworks)

    @property
    def ecosystem_modules(self) -> tuple[str, ...]:
        """Return deterministic prompt/validator modules implied by factual labels."""
        return tuple(
            module.module
            for module in DEFAULT_ECOSYSTEM_PROMPTS.select(
                languages=self.languages,
                build_systems=self.build_systems,
                package_managers=self.package_managers,
                test_frameworks=self.test_frameworks,
            )
        )


class EnvironmentArtifacts(StrictModel):
    """The only three controls required outside a clean task image."""

    dockerfile: str = Field(min_length=40)
    activation: str = Field(min_length=20)
    runner: str = Field(min_length=100)

    @field_validator("activation", "runner")
    @classmethod
    def executable_script_source(cls, value: str) -> str:
        if not value.startswith(("#!/bin/bash", "#!/usr/bin/env bash")):
            raise ValueError("activation and runner must be Bash-compatible scripts")
        return value


class EnvironmentArtifactRepair(StrictModel):
    """Sparse repair output; accepted fields are merged into the last complete artifact."""

    dockerfile: str | None = Field(default=None, min_length=40)
    activation: str | None = Field(default=None, min_length=20)
    runner: str | None = Field(default=None, min_length=100)

    @field_validator("activation", "runner")
    @classmethod
    def executable_script_source(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("#!/bin/bash", "#!/usr/bin/env bash")):
            raise ValueError("activation and runner must be Bash-compatible scripts")
        return value


_ARTIFACT_FIELDS = frozenset({"dockerfile", "activation", "runner"})
_ARTIFACT_FIELD_ORDER = ("dockerfile", "activation", "runner")
_ARTIFACT_MIN_LENGTHS = {"dockerfile": 40, "activation": 20, "runner": 100}
_ARTIFACT_USER_ASSETS: tuple[tuple[str, PromptAssetName], ...] = (
    ("dockerfile", "artifact.dockerfile.user"),
    ("activation", "artifact.activation.user"),
    ("runner", "artifact.runner.user"),
)


class ArtifactRepairContract(StrictModel):
    """Compact repair boundary; Pydantic AI supplies the field schema separately."""

    output_model: str = Field(min_length=1)
    authorized_fields: tuple[str, ...] = Field(min_length=1)
    directly_implicated_fields: tuple[str, ...] = Field(min_length=1)
    consistency_fields: tuple[str, ...]


def directly_implicated_artifact_fields(failure: AgentFeedback) -> set[str]:
    """Return fields named by feedback before consistency dependencies are added."""
    implicated = {
        issue_.path[0]
        for issue_ in failure.issues
        if issue_.path and isinstance(issue_.path[0], str) and issue_.path[0] in _ARTIFACT_FIELDS
    }
    if not implicated:
        implicated = {"dockerfile"}
    return implicated


def implicated_artifact_fields(failure: AgentFeedback) -> set[str]:
    """Close feedback paths over artifacts coupled to a Dockerfile environment."""
    implicated = directly_implicated_artifact_fields(failure)
    if "dockerfile" in implicated:
        implicated.update(("activation", "runner"))
    return implicated


def authorized_repair_output_type(failure: AgentFeedback) -> type[StrictModel]:
    """Expose only complete artifact values authorized by the current feedback."""
    directly_implicated = directly_implicated_artifact_fields(failure)
    implicated = sorted(implicated_artifact_fields(failure))
    multiple_direct_fields = len(directly_implicated) > 1
    fields: dict[str, Any] = {}
    for field_name in implicated:
        required = field_name in directly_implicated and not multiple_direct_fields
        annotation = str if required else str | None
        default = ... if required else None
        if field_name in directly_implicated:
            description = (
                f"Complete corrected {field_name} file content. This field MUST differ from "
                "current_artifacts when submitted."
            )
        else:
            description = (
                f"Optional complete corrected {field_name} file content; include it only when the "
                "direct repair requires a consistency update."
            )
        fields[field_name] = (
            annotation,
            Field(
                default=default,
                min_length=_ARTIFACT_MIN_LENGTHS[field_name],
                description=description,
            ),
        )
    suffix = "Or".join(name.title() for name in implicated)
    return create_model(
        f"Authorized{suffix}Repair",
        __base__=StrictModel,
        __doc__=(
            "Return only complete artifact field values authorized by the structured feedback."
        ),
        **fields,
    )


def materialize_artifact_repair(output: StrictModel) -> EnvironmentArtifactRepair:
    """Validate an authorized dynamic output as sparse complete artifact values."""
    return EnvironmentArtifactRepair.model_validate(output.model_dump(exclude_none=True))


def artifact_repair_contract(
    failure: AgentFeedback,
    output_type: type[StrictModel] | None = None,
) -> ArtifactRepairContract:
    """Describe the exact dynamic output model without maintaining a second prompt contract."""
    repair_type = output_type or authorized_repair_output_type(failure)
    authorized = tuple(repair_type.model_fields)
    directly_implicated = tuple(sorted(directly_implicated_artifact_fields(failure)))
    return ArtifactRepairContract(
        output_model=repair_type.__name__,
        authorized_fields=authorized,
        directly_implicated_fields=directly_implicated,
        consistency_fields=tuple(
            field_name for field_name in authorized if field_name not in directly_implicated
        ),
    )


def merge_artifact_repair(
    current: EnvironmentArtifacts,
    failure: AgentFeedback,
    repair: EnvironmentArtifactRepair,
) -> EnvironmentArtifacts:
    """Merge real authorized changes, treating repeated current values as omissions."""
    directly_implicated = directly_implicated_artifact_fields(failure)
    implicated = implicated_artifact_fields(failure)
    provided = {
        field_name: value
        for field_name in _ARTIFACT_FIELDS
        if (value := getattr(repair, field_name)) is not None
    }
    changed = {
        field_name: value
        for field_name, value in provided.items()
        if value != getattr(current, field_name)
    }
    unexpected = set(changed) - implicated
    if unexpected:
        field_name = sorted(unexpected)[0]
        reject(
            issue(
                FeedbackCode.ARTIFACT_UNRELATED_FIELD_CHANGED,
                path=(field_name,),
                expected="omit this field from the sparse repair",
                actual="an unrelated replacement was returned",
            )
        )
    updates = {
        field_name: value for field_name, value in changed.items() if field_name in implicated
    }
    if not updates:
        field_name = sorted(directly_implicated)[0]
        unchanged_fields = tuple(
            sorted(
                candidate
                for candidate, value in provided.items()
                if value == getattr(current, candidate)
            )
        )
        actual = (
            f"submitted field(s) {', '.join(unchanged_fields)} are byte-for-byte identical to "
            "current_artifacts"
            if unchanged_fields
            else "no authorized artifact field was submitted"
        )
        reject(
            issue(
                FeedbackCode.ARTIFACT_FAILED_FIELD_UNCHANGED,
                path=(field_name,),
                expected=(
                    "return a complete corrected value for at least one directly implicated field; "
                    "the value must differ byte-for-byte from current_artifacts"
                ),
                actual=actual,
            )
        )
    if not set(updates).intersection(directly_implicated):
        field_name = sorted(directly_implicated)[0]
        reject(
            issue(
                FeedbackCode.ARTIFACT_FAILED_FIELD_UNCHANGED,
                path=(field_name,),
                expected="change at least one field directly named by the failure feedback",
                actual="only consistency-dependent fields changed",
            )
        )
    return current.model_copy(update=updates)


def validate_analysis(
    analysis: EnvironmentAnalysis,
    settings: EnvironmentSettings,
) -> None:
    configured = {entry.name: entry for entry in settings.base_images}
    if analysis.selected_base_image_name is None:
        # Absence is a valid result for a repository with no real unit tests. Requiring
        # a manifest/CI/test path here would force the Agent to invent evidence for an
        # empty repository; rejection_reason carries the evidence-based conclusion.
        return
    selected = configured.get(analysis.selected_base_image_name)
    if selected is None:
        reject(
            issue(
                FeedbackCode.ANALYSIS_BASE_NAME,
                path=("selected_base_image_name",),
                expected=", ".join(configured),
                actual=analysis.selected_base_image_name,
            )
        )
    supported = {language.casefold() for language in selected.languages}
    observed = {language.casefold() for language in analysis.languages}
    if "generic" not in supported and not supported.intersection(observed):
        reject(
            issue(
                FeedbackCode.ANALYSIS_BASE_LANGUAGE,
                path=("selected_base_image_name",),
                expected=f"one of the detected languages: {', '.join(sorted(observed))}",
                actual=f"{selected.name}: {', '.join(selected.languages)}",
            )
        )
    if analysis.gpu_requirement == "required" and not selected.gpu_support:
        reject(
            issue(
                FeedbackCode.ANALYSIS_GPU_BASE_REQUIRED,
                path=("selected_base_image_name",),
                expected="a configured entry with gpu_support=true",
                actual=selected.name,
            )
        )
    if analysis.gpu_requirement != "required" and selected.gpu_support:
        reject(
            issue(
                FeedbackCode.ANALYSIS_CPU_BASE_REQUIRED,
                path=("selected_base_image_name",),
                expected="a configured entry with gpu_support=false",
                actual=selected.name,
            )
        )
    if not analysis.manifest_paths and not analysis.ci_paths and not analysis.test_evidence_paths:
        reject(issue(FeedbackCode.ANALYSIS_EVIDENCE_REQUIRED, path=("manifest_paths",)))


def validate_artifacts(
    artifacts: EnvironmentArtifacts,
    settings: EnvironmentSettings,
    analysis: EnvironmentAnalysis,
) -> None:
    selected_base = next(
        (item for item in settings.base_images if item.name == analysis.selected_base_image_name),
        None,
    )
    if selected_base is None:
        reject(
            issue(
                FeedbackCode.ANALYSIS_BASE_NAME,
                path=("selected_base_image_name",),
                actual=analysis.selected_base_image_name,
            )
        )
    if "featurebench_managed_base" in artifacts.dockerfile.casefold():
        reject(issue(FeedbackCode.ARTIFACT_RESERVED_BASE, path=("dockerfile",)))
    if DOCKER_HARNESS_STEP_MARKER in artifacts.dockerfile:
        reject(issue(FeedbackCode.ARTIFACT_AGENT_CONTENT, path=("dockerfile",)))
    sources = (artifacts.dockerfile, artifacts.activation, artifacts.runner)
    if any(_FORBIDDEN.search(source) for source in sources):
        reject(issue(FeedbackCode.ARTIFACT_AGENT_CONTENT))
    if _FORBIDDEN_DOCKERFILE.search(artifacts.dockerfile):
        reject(issue(FeedbackCode.ARTIFACT_CONTROL_CONTENT, path=("dockerfile",)))
    for name, source in (("activation", artifacts.activation), ("runner", artifacts.runner)):
        if _EXTERNAL_RUNTIME_REFERENCE.search(source):
            reject(
                issue(
                    FeedbackCode.ARTIFACT_EXTERNAL_RUNTIME_REFERENCE,
                    path=(name,),
                )
            )
    if _SECRET_DECLARATION.search(artifacts.dockerfile):
        reject(issue(FeedbackCode.ARTIFACT_SECRET, path=("dockerfile",)))
    defaulted_arguments = set(_AUTOMATIC_BUILD_ARGUMENTS)
    for declaration in _ARG_DECLARATION.finditer(artifacts.dockerfile):
        name = declaration.group("name").upper()
        if declaration.group("default") is not None:
            defaulted_arguments.add(name)
        elif name not in defaulted_arguments:
            reject(
                issue(
                    FeedbackCode.ARTIFACT_BUILD_ARG_DEFAULT,
                    path=("dockerfile",),
                    expected=f"ARG {name}=<canonical-default>",
                    actual=f"ARG {name}",
                )
            )
    for match in _ENV_PATH.finditer(artifacts.dockerfile):
        value = match.group("value")
        if "$PATH" not in value and "${PATH}" not in value:
            reject(
                issue(
                    FeedbackCode.ARTIFACT_PATH_OVERRIDE,
                    path=("dockerfile",),
                    expected="PATH extension containing $PATH or ${PATH}",
                    actual=value,
                )
            )
    if _GIT_BASELINE_MUTATION.search(artifacts.dockerfile):
        reject(issue(FeedbackCode.ARTIFACT_GIT_BASELINE, path=("dockerfile",)))
    from_matches = list(_FROM.finditer(artifacts.dockerfile))
    if not from_matches:
        reject(issue(FeedbackCode.ARTIFACT_FROM_REQUIRED, path=("dockerfile",)))
    if from_matches[0].group(1) != selected_base.image:
        reject(
            issue(
                FeedbackCode.ARTIFACT_BASE_REFERENCE,
                path=("dockerfile",),
                expected=selected_base.image,
                actual=from_matches[0].group(1),
            )
        )
    stage_names: set[str] = set()
    for line in artifacts.dockerfile.splitlines():
        match = _FROM.match(line)
        if match is None:
            continue
        image = match.group(1)
        if image != selected_base.image and image not in stage_names:
            reject(
                issue(
                    FeedbackCode.ARTIFACT_EXTERNAL_STAGE,
                    path=("dockerfile",),
                    expected=selected_base.image,
                    actual=image,
                )
            )
        tokens = line.split()
        if len(tokens) >= 4 and tokens[-2].upper() == "AS":
            stage_names.add(tokens[-1])
    for transfer in _CONTEXT_TRANSFER.finditer(artifacts.dockerfile):
        source_stage = transfer.group("from")
        if transfer.group("kind").upper() == "ADD" or source_stage is None:
            reject(issue(FeedbackCode.ARTIFACT_CONTEXT_TRANSFER, path=("dockerfile",)))
        if source_stage not in stage_names:
            reject(
                issue(
                    FeedbackCode.ARTIFACT_COPY_STAGE,
                    path=("dockerfile",),
                    actual=source_stage,
                )
            )
    if artifacts.dockerfile.count(SOURCE_MARKER) != 1:
        reject(
            issue(
                FeedbackCode.ARTIFACT_SOURCE_MARKER_COUNT,
                path=("dockerfile",),
                expected="1",
                actual=str(artifacts.dockerfile.count(SOURCE_MARKER)),
            )
        )
    if artifacts.dockerfile.index(SOURCE_MARKER) < from_matches[-1].start():
        reject(issue(FeedbackCode.ARTIFACT_SOURCE_MARKER_POSITION, path=("dockerfile",)))
    source_prefix = artifacts.dockerfile.partition(SOURCE_MARKER)[0]
    if re.search(r"(?<![A-Za-z0-9_.-])/testbed(?:/|\b)", source_prefix):
        reject(
            issue(
                FeedbackCode.ARTIFACT_SOURCE_PREFIX_TESTBED,
                path=("dockerfile",),
                expected="no /testbed reference before the source marker",
                actual="/testbed is used before the pinned checkout replaces that directory",
            )
        )
    if not re.search(
        r"(?:^|[;&|]\s*)cd\s+(?:--\s+)?['\"]?/testbed/?['\"]?(?:\s|$)",
        artifacts.activation,
        re.MULTILINE,
    ):
        reject(issue(FeedbackCode.ARTIFACT_ACTIVATION_WORKDIR, path=("activation",)))
    if _ACTIVATION_CONTROL_FLOW.search(artifacts.activation):
        reject(issue(FeedbackCode.ARTIFACT_ACTIVATION_CONTROL, path=("activation",)))
    if ecosystem_feedback := DEFAULT_ARTIFACT_VALIDATORS.validate(
        ArtifactValidationContext(
            dockerfile=artifacts.dockerfile,
            activation=artifacts.activation,
            runner=artifacts.runner,
        ),
        ecosystem_modules=analysis.ecosystem_modules,
    ):
        raise AgentFeedbackError(ecosystem_feedback)


def validate_artifact_repair_constraints(
    current: EnvironmentArtifacts,
    candidate: EnvironmentArtifacts,
    constraints: AgentFeedback,
    analysis: EnvironmentAnalysis,
) -> None:
    """Apply only the repair invariants registered for the selected ecosystems."""
    feedback = DEFAULT_ARTIFACT_VALIDATORS.validate_repair(
        ArtifactRepairValidationContext(
            current=ArtifactValidationContext(
                dockerfile=current.dockerfile,
                activation=current.activation,
                runner=current.runner,
            ),
            candidate=ArtifactValidationContext(
                dockerfile=candidate.dockerfile,
                activation=candidate.activation,
                runner=candidate.runner,
            ),
            constraints=constraints,
        ),
        ecosystem_modules=analysis.ecosystem_modules,
    )
    if feedback is not None:
        raise AgentFeedbackError(feedback)


class RunnerContract(StrictModel):
    invocation: str = Field(min_length=1)
    request_schema: dict[str, Any]
    response_schema: dict[str, Any]
    rules: tuple[str, ...] = Field(min_length=1)


class BaseImageOption(StrictModel):
    """Only base-image facts needed for the Analysis Agent's semantic choice."""

    name: str = Field(min_length=1)
    languages: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Primary runtime languages provided by the image, not an exhaustive whitelist of "
            "secondary compilers or build toolchains that artifacts may install."
        ),
    )
    runtime_family: str = Field(
        min_length=1,
        description="Primary runtime family that should match the accepted suite.",
    )
    runtime_version: str | None
    capabilities: tuple[str, ...] = Field(
        description=(
            "Preinstalled capabilities; build-toolchain permits installing evidenced secondary "
            "language toolchains during Docker build."
        )
    )
    gpu_support: bool
    description: str = ""

    @classmethod
    def from_settings(cls, settings: BaseImageSettings) -> BaseImageOption:
        return cls(
            name=settings.name,
            languages=settings.languages,
            runtime_family=settings.runtime_family,
            runtime_version=settings.runtime_version,
            capabilities=settings.capabilities,
            gpu_support=settings.gpu_support,
            description=settings.description,
        )


class SelectedBaseContract(StrictModel):
    """Resolved base-image contract available to the Artifact Agent."""

    image: str = Field(min_length=1)
    runtime_family: str = Field(min_length=1)
    runtime_version: str | None
    capabilities: tuple[str, ...]

    @classmethod
    def from_settings(cls, settings: BaseImageSettings) -> SelectedBaseContract:
        return cls(
            image=settings.image,
            runtime_family=settings.runtime_family,
            runtime_version=settings.runtime_version,
            capabilities=settings.capabilities,
        )


class ExecutionContract(StrictModel):
    """Exact runtime and validation constraints relevant to artifact authoring."""

    workdir: Literal["/testbed"]
    resources: RuntimeResources
    limits: RunnerLimits


class ArtifactRepositoryFacts(StrictModel):
    """Repository facts needed for authoring, without the duplicated base-image choice."""

    primary_language: str
    languages: tuple[str, ...]
    runtime_requirement: str | None
    build_systems: tuple[str, ...]
    package_managers: tuple[str, ...]
    test_frameworks: tuple[str, ...]
    manifest_paths: tuple[str, ...]
    ci_paths: tuple[str, ...]
    test_evidence_paths: tuple[str, ...]
    gpu_requirement: Literal["none", "optional", "required"]
    gpu_evidence_paths: tuple[str, ...]
    gpu_reason: str | None

    @classmethod
    def from_analysis(cls, analysis: EnvironmentAnalysis) -> ArtifactRepositoryFacts:
        if analysis.primary_language is None:
            raise ValueError("artifact authoring requires an accepted primary language")
        return cls(
            primary_language=analysis.primary_language,
            languages=analysis.languages,
            runtime_requirement=analysis.runtime_requirement,
            build_systems=analysis.build_systems,
            package_managers=analysis.package_managers,
            test_frameworks=analysis.test_frameworks,
            manifest_paths=analysis.manifest_paths,
            ci_paths=analysis.ci_paths,
            test_evidence_paths=analysis.test_evidence_paths,
            gpu_requirement=analysis.gpu_requirement,
            gpu_evidence_paths=analysis.gpu_evidence_paths,
            gpu_reason=analysis.gpu_reason,
        )


class AnalysisAgentContext(StrictModel):
    base_image_options: tuple[BaseImageOption, ...] = Field(min_length=1)


class ArtifactAgentContext(StrictModel):
    repository: ArtifactRepositoryFacts
    selected_base: SelectedBaseContract
    execution_contract: ExecutionContract
    runner_contract: RunnerContract
    ecosystem_guidance: tuple[EcosystemPromptGuidance, ...]


class ArtifactRepairCurrentArtifacts(StrictModel):
    """Only current file values that the active repair is authorized to replace."""

    dockerfile: str | None = None
    activation: str | None = None
    runner: str | None = None

    @classmethod
    def from_authorized_fields(
        cls,
        current: EnvironmentArtifacts,
        authorized_fields: tuple[str, ...],
    ) -> ArtifactRepairCurrentArtifacts:
        return cls(**{field_name: getattr(current, field_name) for field_name in authorized_fields})


class ArtifactRepairHistoryEntry(StrictModel):
    """One prior failed build attempt exposed to the next repair Agent."""

    attempt: int = Field(ge=1)
    failed_phase: str = Field(min_length=1)
    feedback: AgentFeedback


class ArtifactRepairAgentContext(StrictModel):
    """Field-scoped repair state without unrelated Runner or ecosystem contracts."""

    repository: ArtifactRepositoryFacts
    selected_base: SelectedBaseContract
    execution_contract: ExecutionContract
    runner_contract: RunnerContract | None
    ecosystem_guidance: tuple[EcosystemPromptGuidance, ...]
    current_artifacts: ArtifactRepairCurrentArtifacts
    build_or_validation_feedback: AgentFeedback
    repair_history: tuple[ArtifactRepairHistoryEntry, ...]
    accepted_repair_constraints: AgentFeedback | None
    repair_contract: ArtifactRepairContract


def _repair_ecosystem_guidance(
    modules: tuple[EcosystemPromptGuidance, ...],
    *,
    authorized_fields: tuple[str, ...],
) -> tuple[EcosystemPromptGuidance, ...]:
    authorized = frozenset(authorized_fields)
    include_dockerfile = "dockerfile" in authorized
    include_activation = "activation" in authorized
    include_runner = "runner" in authorized
    selected: list[EcosystemPromptGuidance] = []
    for module in modules:
        guidance = EcosystemPromptGuidance(
            module=module.module,
            dockerfile_guidance=(module.dockerfile_guidance if include_dockerfile else ()),
            activation_guidance=(module.activation_guidance if include_activation else ()),
            runner_guidance=module.runner_guidance if include_runner else (),
        )
        if guidance.dockerfile_guidance or guidance.activation_guidance or guidance.runner_guidance:
            selected.append(guidance)
    return tuple(selected)


def artifact_user_prompt(
    authorized_fields: tuple[str, ...],
    *,
    source_marker: str = SOURCE_MARKER,
) -> str:
    """Render only the stable artifact instructions represented by this run's schema."""
    requested = frozenset(authorized_fields)
    unknown = requested - _ARTIFACT_FIELDS
    if unknown or not requested:
        raise ValueError(
            "artifact prompt fields must be a non-empty subset of "
            + ", ".join(_ARTIFACT_FIELD_ORDER)
        )
    parts = [load_prompt_asset("artifact.common.user")]
    parts.extend(
        load_prompt_asset(asset).format(source_marker=source_marker)
        for field_name, asset in _ARTIFACT_USER_ASSETS
        if field_name in requested
    )
    return "\n\n".join(parts)


def render_artifact_agent_context(
    context: ArtifactAgentContext | ArtifactRepairAgentContext,
) -> str:
    """Render every effective contract value, including values supplied by defaults."""
    return context.model_dump_json(
        indent=2,
        exclude_none=True,
    )


def artifact_repair_agent_context(
    *,
    initial: ArtifactAgentContext,
    current: EnvironmentArtifacts,
    failure: AgentFeedback,
    repair_history: tuple[ArtifactRepairHistoryEntry, ...],
    accepted_constraints: AgentFeedback | None,
    repair_contract: ArtifactRepairContract,
) -> ArtifactRepairAgentContext:
    """Project a complete initial contract onto the fields authorized for this repair."""
    return ArtifactRepairAgentContext(
        repository=initial.repository,
        selected_base=initial.selected_base,
        execution_contract=initial.execution_contract,
        runner_contract=(
            initial.runner_contract if "runner" in repair_contract.authorized_fields else None
        ),
        ecosystem_guidance=_repair_ecosystem_guidance(
            initial.ecosystem_guidance,
            authorized_fields=repair_contract.authorized_fields,
        ),
        current_artifacts=ArtifactRepairCurrentArtifacts.from_authorized_fields(
            current,
            repair_contract.authorized_fields,
        ),
        build_or_validation_feedback=failure,
        repair_history=repair_history,
        accepted_repair_constraints=accepted_constraints,
        repair_contract=repair_contract,
    )


def runner_contract() -> RunnerContract:
    return RunnerContract(
        invocation=(
            "the host streams the Runner to Bash outside the image and passes "
            "--request REQUEST_JSON --response RESPONSE_JSON"
        ),
        request_schema=RunnerRequest.model_json_schema(),
        response_schema=RunnerResponse.model_json_schema(),
        rules=(
            "collect invokes the installed framework's real collector and returns one stable Runner ID plus "
            "repository-relative source path for every individual test case; follow any public-ID format required "
            "by applicable ecosystem guidance, otherwise use a native stable framework ID when available or a "
            "deterministic alias that run can resolve one-to-one to the current native ID; never hardcode or infer "
            "the inventory from filenames, and do not stop at files, classes, or modules when finer IDs exist",
            "collect and run obtain IDs and individual outcomes from framework-native structured hooks or "
            "reports whenever available; when a framework has no structured interface, use only version-stable, "
            "machine-oriented output explicitly permitted by applicable ecosystem guidance, never incidental "
            "human-facing presentation output, and never infer per-test outcomes from an aggregate exit code",
            "run executes exactly every requested ID in one framework invocation or a fixed bounded number; when "
            "a stable Runner ID aliases a nondeterministic native ID, restrict resolution to its requested source "
            "paths, require a one-to-one current native mapping, select and execute the mapped items inside that "
            "same framework invocation, and map machine evidence back; never replay a native ID resolved by a "
            "different process; return exactly one result per requested ID, preserve mixed outcomes, and support "
            "both singleton downstream verification and batched health checks",
            "when a real framework report omits a requested ID, return that ID conservatively as error; never "
            "convert missing execution evidence into passed, failed, or skipped; the host explicitly probes one "
            "missing ID, which must produce an error result, a nonzero response exit_code, no infrastructure "
            "errors, and nonempty bounded output evidence",
            "run treats IDs as unbounded structured input; preserve each exact ID in the request file or another "
            "selection file when delegating, and pass only bounded file paths through argv or environment variables; "
            "never serialize the arbitrary ID collection into argv, an environment-variable value, shell expansion, "
            "or generated plugin source",
            "collect is the authoritative inventory; the host samples collected IDs through run for an environment "
            "health gate, then groups the complete inventory by repository-relative test path, splits every path "
            "group into bounded deterministic chunks, and calls run on every chunk; only IDs explicitly reported "
            "passed by those real executions become downstream verified points",
            "for optional GPU repositories on CPU, the accepted suite is every evidenced CPU-runnable "
            "unit test and excludes only explicitly GPU-only tests",
            "the request and response values are file paths received from Bash options; read JSON from "
            "the request file and write JSON to the response file; when delegating to another process, "
            "explicitly forward both paths using that runtime's real argument semantics",
            "always write a schema-valid response, including on collection or test failure; ordinary test failures "
            "belong in per-test results, exit_code, and a bounded raw output tail, while errors is reserved for "
            "collection, protocol, or infrastructure failures and must not merely repeat an aggregate nonzero "
            "test status; every nonzero exit_code requires nonempty output, and if the underlying command emitted "
            "no bytes, synthesize a concise diagnostic containing the command and exit code",
            "after writing a valid response, the Runner process exits zero; response.exit_code alone reports the "
            "underlying framework command status",
            "collect returns stable IDs and the same repository-relative path for each ID across repeated "
            "executions; output order is not semantic because the host canonicalizes it",
        ),
    )


async def analyze_environment(
    *,
    agents: AgentFactory[AgentSettings],
    session: ContainerSession,
    settings: EnvironmentSettings,
    repository_url: str,
    commit: str,
    progress: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
) -> EnvironmentAnalysis:
    context = AnalysisAgentContext(
        base_image_options=tuple(
            BaseImageOption.from_settings(item) for item in settings.base_images
        ),
    )
    prompt = load_prompt_asset("analysis.user") + "\n\n" + context.model_dump_json(indent=2)

    async def validate_output(output: EnvironmentAnalysis) -> None:
        validate_analysis(output, settings)
        evidence_fields = (
            ("manifest_paths", output.manifest_paths),
            ("ci_paths", output.ci_paths),
            ("test_evidence_paths", output.test_evidence_paths),
            ("gpu_evidence_paths", output.gpu_evidence_paths),
        )
        evidence_owners = {
            path: field_name for field_name, paths in evidence_fields for path in paths
        }
        evidence_paths = tuple(evidence_owners)
        path_check = await session.exec(
            (
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'missing=0; for path do if ! test -e "$path"; then '
                'printf \'%s\\n\' "$path"; missing=1; fi; done; exit "$missing"',
                "featurebench-evidence-check",
                *evidence_paths,
            ),
            check=False,
        )
        if path_check.exit_code != 0:
            missing_paths = tuple(
                path for path in path_check.stdout.splitlines() if path in evidence_owners
            )
            raise AgentFeedbackError(
                AgentFeedback(
                    issues=tuple(
                        issue(
                            FeedbackCode.ANALYSIS_EVIDENCE_MISSING,
                            path=(evidence_owners[path],),
                            expected="an existing repository-relative path",
                            actual=path,
                        )
                        for path in missing_paths
                    )
                )
            )

    return await agents.run_structured(
        session=session,
        prompt=prompt,
        instructions=load_prompt_asset("analysis.instructions"),
        output_type=EnvironmentAnalysis,
        writable=False,
        output_validator=validate_output,
        progress=progress,
        run_timeout_seconds=agents.settings.analysis_run_timeout_seconds,
        agent_name="environment_analysis",
        metadata={"repository_url": repository_url, "commit": commit},
    )


async def author_environment_artifacts(
    *,
    agents: AgentFactory[AgentSettings],
    session: ContainerSession,
    settings: EnvironmentSettings,
    analysis: EnvironmentAnalysis,
    execution_contract: ExecutionContract,
    repository_url: str,
    commit: str,
    current: EnvironmentArtifacts | None,
    failure: AgentFeedback | None,
    repair_history: tuple[ArtifactRepairHistoryEntry, ...] = (),
    accepted_constraints: AgentFeedback | None = None,
    validate_runner_transport: bool = False,
    progress: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
    review_progress: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
) -> EnvironmentArtifacts:
    selected_base = next(
        image for image in settings.base_images if image.name == analysis.selected_base_image_name
    )
    repair_output_type = (
        authorized_repair_output_type(failure)
        if current is not None and failure is not None
        else None
    )
    repair_contract_value = (
        artifact_repair_contract(failure, repair_output_type)
        if failure is not None and repair_output_type is not None
        else None
    )
    selected_ecosystem_modules = ecosystem_prompt_guidance(
        languages=analysis.languages,
        build_systems=analysis.build_systems,
        package_managers=analysis.package_managers,
        test_frameworks=analysis.test_frameworks,
    )
    selected_runner_contract = runner_contract()
    initial_context = ArtifactAgentContext(
        repository=ArtifactRepositoryFacts.from_analysis(analysis),
        selected_base=SelectedBaseContract.from_settings(selected_base),
        execution_contract=execution_contract,
        runner_contract=selected_runner_contract,
        ecosystem_guidance=selected_ecosystem_modules,
    )
    context: ArtifactAgentContext | ArtifactRepairAgentContext
    if (
        current is not None
        and failure is not None
        and repair_output_type is not None
        and repair_contract_value is not None
    ):
        context = artifact_repair_agent_context(
            initial=initial_context,
            current=current,
            failure=failure,
            repair_history=repair_history,
            accepted_constraints=accepted_constraints,
            repair_contract=repair_contract_value,
        )
    else:
        context = initial_context
    prompt_fields = (
        repair_contract_value.authorized_fields
        if repair_contract_value is not None
        else _ARTIFACT_FIELD_ORDER
    )
    prompt = artifact_user_prompt(prompt_fields) + "\n\n" + render_artifact_agent_context(context)
    review_cache: dict[str, RunnerIntegrityReview] = {}

    async def validate_static(output: EnvironmentArtifacts) -> None:
        validate_artifacts(output, settings, analysis)
        for name, source in (("activation", output.activation), ("runner", output.runner)):
            syntax = await session.exec(
                ("bash", "--noprofile", "--norc", "-n"),
                workdir="/",
                input_bytes=source.encode("utf-8"),
                check=False,
            )
            if syntax.exit_code != 0:
                reject(
                    issue(
                        FeedbackCode.ARTIFACT_SCRIPT_SYNTAX,
                        path=(name,),
                        evidence=syntax.combined_output[-5000:],
                    )
                )

    async def validate_candidate(
        output: EnvironmentArtifacts,
        *,
        review_runner: bool = True,
    ) -> None:
        if review_runner:
            runner_digest = hashlib.sha256(output.runner.encode("utf-8")).hexdigest()
            review = review_cache.get(runner_digest)
            if review is None:
                review = await review_runner_integrity(
                    agents=agents,
                    session=session,
                    context=RunnerReviewContext(
                        test_frameworks=analysis.test_frameworks,
                        runner_contract=selected_runner_contract.model_dump(mode="json"),
                        applicable_runner_guidance=tuple(
                            guidance
                            for module in selected_ecosystem_modules
                            for guidance in module.runner_guidance
                        ),
                        runner_source=output.runner,
                    ),
                    progress=review_progress,
                    metadata={
                        "repository_url": repository_url,
                        "commit": commit,
                        "runner_sha256": runner_digest,
                    },
                )
                review_cache[runner_digest] = review
            if review.cheating_detected:
                raise AgentFeedbackError(
                    runner_review_feedback(
                        review,
                        applicable_runner_guidance=tuple(
                            guidance
                            for module in selected_ecosystem_modules
                            for guidance in module.runner_guidance
                        ),
                    )
                )
        if validate_runner_transport:
            previous_runtime_spec = session.runtime_spec
            session.runtime_spec = RuntimeSpec(
                workdir="/testbed",
                activation=output.activation,
                runner=output.runner,
            )
            try:
                runner = EnvironmentRunner(session)
                collected = await runner.collect(timeout_seconds=settings.collect_timeout_seconds)
                # A valid nonzero response proves transport correctness. Its command
                # failure may belong to the still-unrepaired Dockerfile and must be
                # attributed by the outer build state machine, not trapped here as a
                # Runner output retry.
                if collected.exit_code != 0 or collected.errors:
                    return
                if not collected.collected:
                    raise ContractError("Runner collect preflight returned no test points")
                validate_framework_collection(
                    collected.collected,
                    frameworks=analysis.test_frameworks,
                )
                if len(collected.collected) < settings.min_verified_test_points:
                    reject(
                        issue(
                            FeedbackCode.VALIDATION_COLLECTION_GRANULARITY,
                            path=("runner",),
                            expected=(
                                f">= {settings.min_verified_test_points} individual test case IDs"
                            ),
                            actual=f"{len(collected.collected)} collected entries",
                            evidence=(
                                "Runner repair preflight returned too few framework-addressable "
                                "individual test cases; files, classes, and modules are not test "
                                "points when the framework exposes finer IDs"
                            ),
                        )
                    )
                for _ in range(1, settings.collect_stability_runs):
                    repeated = await runner.collect(
                        timeout_seconds=settings.collect_timeout_seconds
                    )
                    assert_stable_collection(collected, repeated)
                await runner.run(
                    tuple(test.id for test in collected.collected[:2]),
                    timeout_seconds=settings.collect_timeout_seconds,
                )
                await probe_runner_failure_contract(
                    runner,
                    collected,
                    timeout_seconds=settings.collect_timeout_seconds,
                )
                if any(module.module == "pytest" for module in selected_ecosystem_modules):
                    await probe_runner_large_request_transport(
                        runner,
                        collected,
                        timeout_seconds=settings.collect_timeout_seconds,
                    )
            except RunnerCollectionUnstableError as exc:
                reject(
                    issue(
                        FeedbackCode.ARTIFACT_COLLECTION_UNSTABLE,
                        path=("runner",),
                        evidence=str(exc)[-5000:],
                    )
                )
            except (CommandTimeoutError, ContractError, ValueError) as exc:
                detail = str(exc)
                lowered = detail.casefold()
                expected = "a schema-valid Runner V1 response for every action"
                actual = "the candidate Runner violated its behavioral contract"
                if "duplicate test ids" in lowered:
                    expected = (
                        "one unique stable framework-addressable ID for every individual test case"
                    )
                    actual = "collect returned duplicate test IDs"
                elif "runner omitted or added selected test results" in lowered:
                    expected = "run returns exactly one result for every requested ID and no unrequested IDs"
                    actual = "the run response omitted requested IDs or added unrequested IDs"
                elif "large-request transport probe" in lowered:
                    expected = (
                        "run keeps arbitrary requested IDs in request or selection files and passes "
                        "only bounded file paths through argv and environment variables"
                    )
                    actual = "the candidate Runner failed the large-request file transport probe"
                reject(
                    issue(
                        FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                        path=("runner",),
                        expected=expected,
                        actual=actual,
                        evidence=detail[-5000:],
                    )
                )
            finally:
                session.runtime_spec = previous_runtime_spec

    async def validate_repair_static(output: EnvironmentArtifactRepair) -> None:
        assert current is not None and failure is not None
        merged = merge_artifact_repair(current, failure, output)
        for constraints in (failure, accepted_constraints):
            if constraints is not None:
                validate_artifact_repair_constraints(
                    current,
                    merged,
                    constraints,
                    analysis,
                )
        await validate_static(merged)

    async def validate_repair_candidate(output: EnvironmentArtifactRepair) -> EnvironmentArtifacts:
        assert current is not None and failure is not None
        merged = merge_artifact_repair(current, failure, output)
        await validate_candidate(merged, review_runner=merged.runner != current.runner)
        return merged

    metadata = {
        "repository_url": repository_url,
        "commit": commit,
    }
    if current is not None and failure is not None:
        assert repair_output_type is not None
        assert repair_contract_value is not None
        repair_contract = repair_contract_value
        runner_only_repair = directly_implicated_artifact_fields(failure) == {"runner"}
        accepted_candidate: EnvironmentArtifacts | None = None

        async def validate_authorized_repair(output: StrictModel) -> None:
            repair = materialize_artifact_repair(output)
            await validate_repair_static(repair)

        async def validate_authorized_candidate(output: StrictModel) -> None:
            nonlocal accepted_candidate
            repair = materialize_artifact_repair(output)
            accepted_candidate = await validate_repair_candidate(repair)

        await agents.run_structured(
            session=session,
            prompt=prompt,
            instructions=(
                load_prompt_asset("artifact.repair.instructions").format(
                    output_model=repair_contract.output_model,
                    authorized_fields=", ".join(repair_contract.authorized_fields),
                    directly_implicated_fields=", ".join(
                        repair_contract.directly_implicated_fields
                    ),
                    consistency_fields=(", ".join(repair_contract.consistency_fields) or "none"),
                )
                + (
                    "\n\n" + load_prompt_asset("artifact.runner-repair.instructions")
                    if runner_only_repair
                    else ""
                )
            ),
            output_type=repair_output_type,
            writable=False,
            container_tools=True,
            output_validator=validate_authorized_repair,
            candidate_validator=validate_authorized_candidate,
            progress=progress,
            run_timeout_seconds=agents.settings.artifact_run_timeout_seconds,
            agent_name="environment_artifact",
            metadata=metadata,
        )
        if accepted_candidate is None:
            raise AssertionError("validated artifact repair did not retain its candidate")
        return accepted_candidate
    return await agents.run_structured(
        session=session,
        prompt=prompt,
        instructions=load_prompt_asset("artifact.initial.instructions"),
        output_type=EnvironmentArtifacts,
        writable=False,
        output_validator=validate_static,
        candidate_validator=validate_candidate,
        progress=progress,
        run_timeout_seconds=agents.settings.artifact_run_timeout_seconds,
        agent_name="environment_artifact",
        metadata=metadata,
    )
