"""Environment-owned feedback codes and YAML-backed correction templates."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from codeworld.agents.feedback import (
    FeedbackIssue,
    FeedbackTemplateRegistry,
)

_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "feedback"


class FeedbackCode(StrEnum):
    """Stable machine-readable codes for Environment Agent contracts."""

    ANALYSIS_BASE_NAME = "analysis.base_name_unknown"
    ANALYSIS_BASE_LANGUAGE = "analysis.base_language_unsupported"
    ANALYSIS_GPU_BASE_REQUIRED = "analysis.gpu_base_required"
    ANALYSIS_CPU_BASE_REQUIRED = "analysis.cpu_base_required"
    ANALYSIS_EVIDENCE_REQUIRED = "analysis.evidence_required"
    ANALYSIS_EVIDENCE_MISSING = "analysis.evidence_path_missing"
    ARTIFACT_RESERVED_BASE = "artifact.reserved_base_stage"
    ARTIFACT_AGENT_CONTENT = "artifact.agent_content_forbidden"
    ARTIFACT_CONTROL_CONTENT = "artifact.control_content_forbidden"
    ARTIFACT_EXTERNAL_RUNTIME_REFERENCE = "artifact.external_runtime_reference"
    ARTIFACT_SECRET = "artifact.secret_declaration_forbidden"
    ARTIFACT_BUILD_ARG_DEFAULT = "artifact.build_arg_default_required"
    ARTIFACT_PATH_OVERRIDE = "artifact.base_path_overridden"
    ARTIFACT_SHELL_REQUIREMENT = "artifact.shell_requirement_unquoted"
    ARTIFACT_GIT_BASELINE = "artifact.git_baseline_mutation"
    ARTIFACT_FROM_REQUIRED = "artifact.from_required"
    ARTIFACT_BASE_REFERENCE = "artifact.base_reference_mismatch"
    ARTIFACT_EXTERNAL_STAGE = "artifact.external_stage_forbidden"
    ARTIFACT_CONTEXT_TRANSFER = "artifact.context_transfer_forbidden"
    ARTIFACT_COPY_STAGE = "artifact.copy_stage_unknown"
    ARTIFACT_SOURCE_MARKER_COUNT = "artifact.source_marker_count"
    ARTIFACT_SOURCE_MARKER_POSITION = "artifact.source_marker_position"
    ARTIFACT_SOURCE_PREFIX_TESTBED = "artifact.source_prefix_testbed"
    ARTIFACT_ACTIVATION_WORKDIR = "artifact.activation_workdir"
    ARTIFACT_ACTIVATION_CONTROL = "artifact.activation_control_flow"
    ARTIFACT_ACTIVATION_ENVIRONMENT = "artifact.activation_environment"
    ARTIFACT_RUNNER_CONTRACT = "artifact.runner_contract"
    ARTIFACT_RUNNER_PUBLIC_ID = "artifact.runner_public_id"
    ARTIFACT_RUNNER_INTEGRITY = "artifact.runner_integrity_rejected"
    ARTIFACT_COLLECTION_UNSTABLE = "artifact.collection_unstable"
    ARTIFACT_RUNTIME_MUTATION = "artifact.runtime_dependency_mutation"
    ARTIFACT_SCRIPT_SYNTAX = "artifact.script_syntax"
    ARTIFACT_FAILED_FIELD_UNCHANGED = "artifact.failed_field_unchanged"
    ARTIFACT_UNRELATED_FIELD_CHANGED = "artifact.unrelated_field_changed"
    ARTIFACT_REPAIR_REGRESSION = "artifact.repair_regressed_accepted_content"
    BUILD_DEPENDENCY_LOCK = "build.dependency_lock_incompatible"
    BUILD_DEPENDENCY_VERSION = "build.dependency_version_incompatible"
    BUILD_LOCAL_REACTOR_DEPENDENCY = "build.local_reactor_dependency_unmaterialized"
    BUILD_MAVEN_TEST_PROVIDER = "build.maven_test_provider_unmaterialized"
    BUILD_RUBY_DEPLOYMENT_LOCK = "build.ruby_deployment_without_lockfile"
    BUILD_RUBY_TEST_DEPENDENCY = "build.ruby_test_dependency_unmaterialized"
    BUILD_PROJECT_IMPORT_SHADOWED = "build.project_import_shadowed"
    BUILD_COMMAND = "build.command_failed"
    VALIDATION_COLLECTION_GRANULARITY = "validation.collection_granularity"
    VALIDATION_PYTEST_COLLECT_EXECUTED = "validation.pytest_collect_executed_tests"
    VALIDATION_RUNTIME_MATERIALIZATION = "validation.runtime_materialization"
    VALIDATION_RUNTIME = "validation.runtime_contract_failed"


FEEDBACK_TEMPLATES = FeedbackTemplateRegistry.from_directory(_ASSET_DIRECTORY)
_missing_codes = tuple(str(code) for code in FeedbackCode if code not in FEEDBACK_TEMPLATES)
if _missing_codes:
    raise ValueError("feedback assets are missing Environment codes: " + ", ".join(_missing_codes))


def issue(
    code: FeedbackCode,
    *,
    path: tuple[str | int, ...] = (),
    expected: str | None = None,
    actual: str | None = None,
    evidence: str | None = None,
    context: dict[str, str] | None = None,
) -> FeedbackIssue:
    """Construct one catalog-backed Environment issue."""
    message, correction = FEEDBACK_TEMPLATES.resolve(code)
    return FeedbackIssue(
        code=code,
        path=path,
        message=message,
        correction=correction,
        expected=expected,
        actual=actual,
        evidence=evidence,
        context=context or {},
    )
