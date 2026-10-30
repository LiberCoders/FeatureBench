from __future__ import annotations

import pytest

from codeworld.agents.feedback import (
    AgentFeedback,
    FeedbackIssue,
    FeedbackTemplateRegistry,
)
from codeworld.environment.artifact_validation import (
    DEFAULT_ARTIFACT_VALIDATORS,
    ArtifactValidationContext,
)
from codeworld.environment.diagnostics import (
    FailureDiagnosticContext,
    FailureDiagnosticRegistry,
)
from codeworld.environment.ecosystems import (
    DEFAULT_ECOSYSTEM_PROMPTS,
    EcosystemMatch,
    EcosystemPromptModule,
    EcosystemPromptRegistry,
)
from codeworld.environment.framework_validation import (
    DEFAULT_FRAMEWORK_VALIDATORS,
    FrameworkCollectionValidationContext,
)
from codeworld.environment.runner_contract import CollectedTestPoint


def test_prompt_registry_selects_only_matching_ecosystem_modules() -> None:
    selected = DEFAULT_ECOSYSTEM_PROMPTS.select(
        languages=("java",),
        build_systems=("maven",),
        package_managers=("maven",),
        test_frameworks=("junit 5",),
    )
    names = {module.module for module in selected}
    assert names == {"maven", "junit"}
    assert "pytest" not in names
    rendered = " ".join(guidance for module in selected for guidance in module.dockerfile_guidance)
    assert "reactor" in rendered
    assert "/root/.m2" in rendered

    normalized_alias = DEFAULT_ECOSYSTEM_PROMPTS.select(
        languages=("java",),
        build_systems=("Maven-Wrapper",),
    )
    assert {module.module for module in normalized_alias} == {"maven"}

    compact_framework_alias = DEFAULT_ECOSYSTEM_PROMPTS.select(
        test_frameworks=("junit4",),
    )
    assert {module.module for module in compact_framework_alias} == {"junit"}

    rust = DEFAULT_ECOSYSTEM_PROMPTS.select(
        languages=("python", "rust"),
        build_systems=("maturin",),
        package_managers=("uv", "cargo"),
        test_frameworks=("pytest",),
    )
    rust_guidance = " ".join(
        guidance
        for module in rust
        if module.module == "rust"
        for guidance in module.dockerfile_guidance
    )
    assert "selected_base.capabilities" in rust_guidance
    assert "before invoking maturin" in rust_guidance


def test_prompt_registry_accepts_new_modules_without_core_branch_changes() -> None:
    registry = EcosystemPromptRegistry()
    module = EcosystemPromptModule(
        module="elixir-exunit",
        match=EcosystemMatch(languages=("elixir",)),
        runner_guidance=("Use ExUnit's formatter contract.",),
    )
    registry.register(module)
    assert registry.select(languages=("elixir",)) == (module,)
    assert registry.select(languages=("python",)) == ()
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(module)


def test_feedback_templates_and_diagnostics_are_independently_registrable() -> None:
    templates = FeedbackTemplateRegistry()
    templates.register("elixir.mix-lock", "Mix lock failed.", "Repair the lock use.")
    assert templates.resolve("elixir.mix-lock") == (
        "Mix lock failed.",
        "Repair the lock use.",
    )

    diagnostics = FailureDiagnosticRegistry()

    def low_priority(_: str, __: FailureDiagnosticContext) -> AgentFeedback | None:
        return AgentFeedback(
            issues=(
                FeedbackIssue(
                    code="low",
                    message="low",
                    correction="low",
                ),
            )
        )

    def high_priority(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
        if "mix.lock" not in detail:
            return None
        return AgentFeedback(
            issues=(
                FeedbackIssue(
                    code="elixir.mix-lock",
                    message="Mix lock failed.",
                    correction="Repair the lock use.",
                ),
            )
        )

    diagnostics.register("fallback", low_priority, priority=0)
    diagnostics.register("elixir.mix-lock", high_priority, priority=100)
    feedback = diagnostics.diagnose("mix.lock is stale")
    assert feedback is not None
    assert feedback.issues[0].code == "elixir.mix-lock"


def test_diagnostic_registry_filters_scoped_rules_before_priority() -> None:
    registry = FailureDiagnosticRegistry()

    def feedback(code: str) -> AgentFeedback:
        return AgentFeedback(issues=(FeedbackIssue(code=code, message=code, correction=code),))

    registry.register(
        "java.maven",
        lambda _detail, _context: feedback("java.maven"),
        priority=100,
        build_systems=("maven",),
    )
    registry.register(
        "generic",
        lambda _detail, _context: feedback("generic"),
        priority=0,
    )
    selected = registry.diagnose(
        "failure",
        FailureDiagnosticContext(primary_language="python", build_systems=("uv",)),
    )
    assert selected is not None
    assert selected.issues[0].code == "generic"


def test_artifact_adapters_are_automatically_loaded_and_ecosystem_isolated() -> None:
    context = ArtifactValidationContext(
        dockerfile="FROM scratch\n",
        activation="#!/bin/bash\ncd /testbed\nnpm install\n",
        runner="#!/bin/bash\n",
    )
    node_feedback = DEFAULT_ARTIFACT_VALIDATORS.validate(
        context,
        ecosystem_modules=("node",),
    )
    assert node_feedback is not None
    assert node_feedback.issues[0].code == "artifact.runtime_dependency_mutation"

    python_feedback = DEFAULT_ARTIFACT_VALIDATORS.validate(
        context,
        ecosystem_modules=("python",),
    )
    assert python_feedback is None


def test_pytest_runner_rejects_post_collection_only_selection() -> None:
    post_collection_only = ArtifactValidationContext(
        dockerfile="FROM scratch\n",
        activation="#!/bin/bash\ncd /testbed\n",
        runner=(
            "#!/bin/bash\n"
            "def pytest_collection_modifyitems(config, items):\n"
            "    requested = set(load_ids())\n"
            "    items[:] = [item for item in items if item.nodeid in requested]\n"
            "command = ['python', '-m', 'pytest', '-p', 'plugin']\n"
        ),
    )

    feedback = DEFAULT_ARTIFACT_VALIDATORS.validate(
        post_collection_only,
        ecosystem_modules=("python", "pytest"),
    )

    assert feedback is not None
    assert feedback.issues[0].code == "artifact.runner_contract"
    assert "before unrelated test modules" in (feedback.issues[0].expected or "")
    assert "pytest>=8.2" in (feedback.issues[0].expected or "")
    assert "pytest_ignore_collect" in (feedback.issues[0].expected or "")

    argfile_pre_collection = ArtifactValidationContext(
        dockerfile=post_collection_only.dockerfile,
        activation=post_collection_only.activation,
        runner=post_collection_only.runner + "command.append('@' + str(ids_file))\n",
    )
    assert (
        DEFAULT_ARTIFACT_VALIDATORS.validate(
            argfile_pre_collection,
            ecosystem_modules=("python", "pytest"),
        )
        is None
    )

    legacy_pre_collection = ArtifactValidationContext(
        dockerfile=post_collection_only.dockerfile,
        activation=post_collection_only.activation,
        runner=(
            post_collection_only.runner
            + "def pytest_ignore_collect(collection_path, config):\n"
            + "    return collection_path not in load_paths_from_filter_file()\n"
        ),
    )
    assert (
        DEFAULT_ARTIFACT_VALIDATORS.validate(
            legacy_pre_collection,
            ecosystem_modules=("python", "pytest"),
        )
        is None
    )


@pytest.mark.parametrize(
    ("payload", "actual_fragment"),
    (
        ('env["CW_IDS"] = json.dumps(requested)\n', "environment-variable value"),
        ("command.extend(test_ids)\n", "pytest command argv"),
    ),
)
def test_pytest_runner_rejects_unbounded_selection_transport(
    payload: str,
    actual_fragment: str,
) -> None:
    context = ArtifactValidationContext(
        dockerfile="FROM scratch\n",
        activation="#!/bin/bash\ncd /testbed\n",
        runner="#!/bin/bash\n" + payload,
    )

    feedback = DEFAULT_ARTIFACT_VALIDATORS.validate(
        context,
        ecosystem_modules=("python", "pytest"),
    )

    assert feedback is not None
    assert feedback.issues[0].code == "artifact.runner_contract"
    assert actual_fragment in (feedback.issues[0].actual or "")
    assert "selection file" in (feedback.issues[0].expected or "")


def test_pytest_runner_allows_file_paths_in_environment_and_argv() -> None:
    context = ArtifactValidationContext(
        dockerfile="FROM scratch\n",
        activation="#!/bin/bash\ncd /testbed\n",
        runner=(
            "#!/bin/bash\n"
            'env["CW_REQUEST_PATH"] = request_path\n'
            'env["CW_REPORT_PATH"] = report_path\n'
            "command.extend(['@' + str(paths_file)])\n"
        ),
    )

    assert (
        DEFAULT_ARTIFACT_VALIDATORS.validate(
            context,
            ecosystem_modules=("python", "pytest"),
        )
        is None
    )


def test_pytest_runtime_validator_requires_canonical_public_ids() -> None:
    native = FrameworkCollectionValidationContext(
        collected=(
            CollectedTestPoint(
                id="tests/test_api.py::test_value[param]",
                path="tests/test_api.py",
            ),
        )
    )
    feedback = DEFAULT_FRAMEWORK_VALIDATORS.validate_collection(
        native,
        frameworks=("pytest",),
    )
    assert feedback is not None
    assert feedback.issues[0].code == "artifact.runner_public_id"

    canonical = FrameworkCollectionValidationContext(
        collected=(
            CollectedTestPoint(
                id="tests/test_api.py::test_value#0",
                path="tests/test_api.py",
            ),
            CollectedTestPoint(
                id="tests/test_api.py::test_value#1",
                path="tests/test_api.py",
            ),
        )
    )
    assert (
        DEFAULT_FRAMEWORK_VALIDATORS.validate_collection(
            canonical,
            frameworks=("pytest",),
        )
        is None
    )
    assert (
        DEFAULT_FRAMEWORK_VALIDATORS.validate_collection(
            native,
            frameworks=("go test",),
        )
        is None
    )


def test_pytest_runtime_validator_rejects_occurrence_gaps() -> None:
    context = FrameworkCollectionValidationContext(
        collected=(
            CollectedTestPoint(
                id="tests/test_api.py::test_value#1",
                path="tests/test_api.py",
            ),
        )
    )
    feedback = DEFAULT_FRAMEWORK_VALIDATORS.validate_collection(
        context,
        frameworks=("pytest",),
    )
    assert feedback is not None
    assert "non-contiguous" in (feedback.issues[0].actual or "")
