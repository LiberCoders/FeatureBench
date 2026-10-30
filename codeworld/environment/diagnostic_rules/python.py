"""Python, uv, and pytest failure diagnostics."""

from __future__ import annotations

import re

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.diagnostics import (
    FailureDiagnosticContext,
    FailureDiagnosticRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_UV_LOCK_FAILURE = re.compile(
    r"lockfile[^\n]*(?:needs to be updated|is not up-to-date)|"
    r"--locked[^\n]*(?:lockfile|lock file)",
    re.IGNORECASE,
)
_PYTEST_REMOVED_WARNING = re.compile(
    r"pytestremovedin(?P<major>\d+)warning",
    re.IGNORECASE,
)
_PYTEST_MINVERSION_MISMATCH = re.compile(
    r"['\"]?minversion['\"]?\s+requires\s+pytest-(?P<required>[^,\s]+),\s*"
    r"actual\s+pytest-(?P<actual>[^\s'\"]+)",
    re.IGNORECASE,
)
_SHADOWED_PROJECT_IMPORT = re.compile(
    r"(?:importerror|modulenotfounderror)[\s\S]{0,3000}?"
    r"\((?P<path>/[^\n)]*/site-packages/[^\n)]+)\)",
    re.IGNORECASE,
)
_PYTEST_COLLECT_EXECUTED_TESTS = re.compile(
    r"Runner collect returned too few[\s\S]*?received=0[\s\S]*?\d+ passed in ",
    re.IGNORECASE,
)
_PYTEST_MISSING_RUNNER_PLUGIN = re.compile(
    r"error importing plugin\s+['\"](?P<plugin>[A-Za-z_][A-Za-z0-9_.]*)['\"]:\s*"
    r"no module named\s+['\"](?P=plugin)['\"]",
    re.IGNORECASE,
)
_PYTEST_AGGREGATE_COLLECTION = re.compile(
    r"Runner collect returned too few[\s\S]*?received=0[\s\S]*?"
    r"tests/[A-Za-z0-9_./-]+\.py:\s+\d+",
    re.IGNORECASE,
)
_PYTEST_SELECTED_NODE_NOT_FOUND = re.compile(
    r"quickcheck environment health gate failed[\s\S]*?"
    r"error:\s*file or directory not found:\s*['\"][^\r\n]*::[^\r\n]*['\"]",
    re.IGNORECASE,
)
_PYTEST_RESTRICTED_IMPORT_PATH_MISMATCH = re.compile(
    r"quickcheck environment health gate failed[\s\S]*?"
    r"requested test ID was absent from restricted pytest collection[\s\S]*?"
    r"ImportPathMismatchError",
    re.IGNORECASE,
)
_PYTEST_SELECTED_EVIDENCE_UNMAPPED = re.compile(
    r"quickcheck environment health gate failed[\s\S]*?"
    r"requested test produced no unambiguous pytest evidence[\s\S]*?"
    r"\d+ passed in ",
    re.IGNORECASE,
)
_RUNNER_PLUGIN_IMPORT = re.compile(r"error importing plugin[\s\S]*featurebench", re.IGNORECASE)
_RUNNER_HELPER_TRACEBACK = re.compile(
    r'File ["\']/tmp/[^"\']*(?:featurebench|runner|/fb)[^"\']*\.py["\']', re.IGNORECASE
)
_RUNNER_PLUGIN_AST_FAILURE = re.compile(
    r'typeerror:\s*required field ["\'](?:lineno|col_offset|end_lineno|end_col_offset)["\'] '
    r"missing from (?:alias|arg|keyword|stmt|expr|excepthandler|type_ignore)",
    re.IGNORECASE,
)
_PYTEST_PLUGIN_HOOK_SIGNATURE = re.compile(
    r"PluginValidationError:[\s\S]*?hookimpl definition:[\s\S]*?"
    r"Argument\(s\).*?can not be found in the hookspec",
    re.IGNORECASE,
)
_PYTHON_RUNTIME_MATERIALIZATION = re.compile(
    r"creating virtual environment at:|creating virtualenv|"
    r"(?:^|\n)[^\n]*:\s*uv-sync>|(?:^|\n)[^\n]*:\s*venv>|"
    r"/uv\s+sync\s|runtime dependency (?:install|sync|fetch)|"
    r"uv run without --no-sync",
    re.IGNORECASE,
)
_PYTHON_DEPENDENCY_FAILURE = re.compile(
    r"\bno module named\b|\bmodulenotfounderror\b|\bimporterror\b|"
    r"\bdistributionnotfound\b|\bversionconflict\b",
    re.IGNORECASE,
)


def _evidence(detail: str) -> str:
    return detail[-5000:]


def _uv_lock(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _UV_LOCK_FAILURE.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_DEPENDENCY_LOCK,
                path=("dockerfile",),
                expected="dependency installation that preserves the committed repository state",
                actual="uv sync --locked",
                evidence=_evidence(detail),
            ),
        )
    )


def _pytest_version(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    minversion = _PYTEST_MINVERSION_MISMATCH.search(detail)
    if minversion is not None:
        return AgentFeedback(
            issues=(
                issue(
                    FeedbackCode.BUILD_DEPENDENCY_VERSION,
                    path=("dockerfile",),
                    expected=(
                        "the installed pytest distribution reports a version satisfying "
                        f"the repository's minversion {minversion.group('required')}"
                    ),
                    actual=(
                        "the activated environment reports pytest "
                        f"{minversion.group('actual')}, commonly caused by incomplete VCS version "
                        "metadata in a shallow source build"
                    ),
                    evidence=_evidence(detail),
                ),
            )
        )
    match = _PYTEST_REMOVED_WARNING.search(detail)
    if match is None:
        return None
    removal_major = int(match.group("major"))
    incompatible_major = max(1, removal_major - 1)
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_DEPENDENCY_VERSION,
                path=("dockerfile",),
                expected=f"pytest<{incompatible_major}",
                actual=f"pytest>={incompatible_major} selected by an unbounded requirement",
                evidence=_evidence(detail),
            ),
        )
    )


def _shadowed_project(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    match = _SHADOWED_PROJECT_IMPORT.search(detail)
    if match is None:
        return None
    imported_path = match.group("path")
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.BUILD_PROJECT_IMPORT_SHADOWED,
                path=("dockerfile",),
                expected="the tested project import resolves to the pinned checkout under /testbed",
                actual=f"the import resolved to {imported_path}",
                evidence=_evidence(detail),
            ),
        )
    )


def _pytest_collect_executed_tests(
    detail: str, _: FailureDiagnosticContext
) -> AgentFeedback | None:
    if not _PYTEST_COLLECT_EXECUTED_TESTS.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.VALIDATION_PYTEST_COLLECT_EXECUTED,
                path=("runner",),
                expected="pytest --collect-only with exact unique node IDs",
                actual="the collect action ran test bodies and returned zero collected entries",
                evidence=_evidence(detail),
            ),
        )
    )


def _pytest_aggregate_collection(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _PYTEST_AGGREGATE_COLLECTION.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.VALIDATION_COLLECTION_GRANULARITY,
                path=("runner",),
                expected=(
                    "a pytest_collection_finish machine-readable stream containing every exact "
                    "item.nodeid and repository-relative item.path"
                ),
                actual=(
                    "pytest emitted aggregate 'test_file.py: count' presentation lines, commonly "
                    "caused by accumulating -q into -qq, and the Runner parsed zero node IDs"
                ),
                evidence=_evidence(detail),
            ),
        )
    )


def _pytest_selected_node_not_found(
    detail: str, _: FailureDiagnosticContext
) -> AgentFeedback | None:
    if not _PYTEST_SELECTED_NODE_NOT_FOUND.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "pass every exact collected pytest node ID back to pytest as one unchanged "
                    "argument; an argument file contains raw arguments, not shell syntax"
                ),
                actual=(
                    "the Runner added literal shell quotes around pytest node IDs, so pytest "
                    "looked for a quoted path that does not exist"
                ),
                evidence=_evidence(detail),
            ),
        )
    )


def _pytest_restricted_import_path_mismatch(
    detail: str, _: FailureDiagnosticContext
) -> AgentFeedback | None:
    if not _PYTEST_RESTRICTED_IMPORT_PATH_MISMATCH.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "resolve stable IDs through the authoritative collection's framework-visible "
                    "source selectors and execute every requested ID without importing conflicting "
                    "test roots in one pytest collection"
                ),
                actual=(
                    "restricted execution mixed canonical paths from independent pytest roots, "
                    "so equal module names caused ImportPathMismatchError before requested IDs "
                    "could be resolved"
                ),
                evidence=_evidence(detail),
                context={
                    "repair_strategy": (
                        "Preserve the framework-visible selector associated with collect; when "
                        "requested paths span incompatible roots, use a fixed bounded invocation "
                        "per coherent root and merge only machine-reported outcomes."
                    )
                },
            ),
        )
    )


def _pytest_selected_evidence_unmapped(
    detail: str, _: FailureDiagnosticContext
) -> AgentFeedback | None:
    if not _PYTEST_SELECTED_EVIDENCE_UNMAPPED.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "rebuild the canonical pytest public-ID mapping inside the selected run "
                    "invocation and attribute every setup, call, and teardown report back to "
                    "the exact requested ID"
                ),
                actual=(
                    "pytest executed and passed the requested items, but the Runner could not "
                    "associate their phase reports with the requested public IDs"
                ),
                evidence=_evidence(detail),
                context={
                    "repair_strategy": (
                        "Use one shared base-and-occurrence mapping for collection filtering and "
                        "report translation in the same pytest process; do not infer outcomes "
                        "from pytest's aggregate passed count."
                    )
                },
            ),
        )
    )


def _pytest_missing_runner_plugin(
    detail: str, context: FailureDiagnosticContext
) -> AgentFeedback | None:
    match = _PYTEST_MISSING_RUNNER_PLUGIN.search(detail)
    if match is None or context.runner_source is None:
        return None
    plugin = match.group("plugin")
    if plugin not in context.runner_source:
        # The repository may configure its own pytest plugin. That is an image
        # provisioning problem, not evidence that the external Runner is wrong.
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "the Runner uses only framework plugins that it materializes itself or that "
                    "the repository environment already provides"
                ),
                actual=f"the Runner requested its undeclared plugin {plugin!r}",
                evidence=_evidence(detail),
            ),
        )
    )


def _runner_plugin_import(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _RUNNER_PLUGIN_IMPORT.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "load the Runner's temporary pytest plugin by an importable module name "
                    "with its parent directory on PYTHONPATH"
                ),
                actual="the Runner passed a non-importable pytest plugin reference",
                evidence=_evidence(detail),
            ),
        )
    )


def _runner_helper_exception(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not (
        _RUNNER_HELPER_TRACEBACK.search(detail)
        or _RUNNER_PLUGIN_AST_FAILURE.search(detail)
        or _PYTEST_PLUGIN_HOOK_SIGNATURE.search(detail)
    ):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_RUNNER_CONTRACT,
                path=("runner",),
                expected=(
                    "the Runner's generated temporary helper is self-contained and "
                    "uses exact framework hook signatures, and completes every action without "
                    "an exception"
                ),
                actual="the Runner's temporary Python helper raised an exception",
                evidence=_evidence(detail),
            ),
        )
    )


def _runtime_materialization(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _PYTHON_RUNTIME_MATERIALIZATION.search(detail):
        return None
    return AgentFeedback(
        issues=tuple(
            issue(
                FeedbackCode.VALIDATION_RUNTIME_MATERIALIZATION,
                path=(field_name,),
                expected="all Python dependencies materialized in the image before validation",
                actual="the Runner created or synchronized a Python environment at runtime",
                evidence=_evidence(detail),
            )
            for field_name in ("dockerfile", "runner")
        )
    )


def _python_dependency_failure(detail: str, _: FailureDiagnosticContext) -> AgentFeedback | None:
    if not _PYTHON_DEPENDENCY_FAILURE.search(detail):
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.VALIDATION_RUNTIME,
                path=("dockerfile",),
                expected="all required Python packages and compatible versions installed in the image",
                actual="the activated Python environment has a missing or incompatible dependency",
                evidence=_evidence(detail),
            ),
        )
    )


def register(registry: FailureDiagnosticRegistry) -> None:
    registry.register(
        "python.pytest-runner-plugin-import",
        _runner_plugin_import,
        priority=440,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.runner-helper-exception",
        _runner_helper_exception,
        priority=435,
        languages=("python",),
    )
    registry.register(
        "python.pytest-missing-runner-plugin",
        _pytest_missing_runner_plugin,
        priority=430,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.pytest-collect-executed-tests",
        _pytest_collect_executed_tests,
        priority=420,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.pytest-selected-node-not-found",
        _pytest_selected_node_not_found,
        priority=415,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.pytest-selected-evidence-unmapped",
        _pytest_selected_evidence_unmapped,
        priority=416,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.pytest-restricted-import-path-mismatch",
        _pytest_restricted_import_path_mismatch,
        priority=414,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.pytest-aggregate-collection",
        _pytest_aggregate_collection,
        priority=410,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.uv-lock",
        _uv_lock,
        priority=400,
        languages=("python",),
        package_managers=("uv",),
    )
    registry.register(
        "python.pytest-version",
        _pytest_version,
        priority=390,
        languages=("python",),
        test_frameworks=("pytest",),
    )
    registry.register(
        "python.shadowed-project",
        _shadowed_project,
        priority=380,
        languages=("python",),
    )
    registry.register(
        "python.runtime-materialization",
        _runtime_materialization,
        priority=370,
        languages=("python",),
        package_managers=("pip", "uv", "poetry", "pdm", "conda"),
    )
    registry.register(
        "python.dependency-failure",
        _python_dependency_failure,
        priority=100,
        languages=("python",),
        package_managers=("pip", "uv", "poetry", "pdm", "conda"),
    )
