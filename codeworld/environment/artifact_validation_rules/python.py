"""Python and uv artifact consistency validators."""

from __future__ import annotations

import re
import shlex

from codeworld.agents.feedback import AgentFeedback
from codeworld.environment.artifact_validation import (
    ArtifactRepairValidationContext,
    ArtifactValidationContext,
    ArtifactValidationRegistry,
)
from codeworld.environment.feedback import FeedbackCode, issue

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?$")
_PIP_EXECUTABLE = re.compile(r"(?:^|/)pip(?:\d+(?:\.\d+)*)?$")
_PIP_INSTALL_COMMAND = re.compile(
    r"\b(?:(?:python(?:\d+(?:\.\d+)*)?)\s+-m\s+)?pip(?:\d+(?:\.\d+)*)?\s+install\b"
    r"(?P<arguments>.*?)(?=(?:&&|\|\||;|\n|$))",
    re.IGNORECASE,
)
_PYTHON_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?"
    r"(?:(?:===|==|~=|!=|<=|>=|<|>|@).*)?$"
)
_PIP_OPTIONS_WITH_VALUE = frozenset(
    {
        "-c",
        "--constraint",
        "-e",
        "--editable",
        "-f",
        "--find-links",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-r",
        "--requirement",
        "--trusted-host",
    }
)
_PYTHON_RUNTIME_DEPENDENCY_MUTATION = re.compile(
    r"\b(?:uv\s+sync|pip(?:\d+(?:\.\d+)*)?\s+install|"
    r"python(?:\d+(?:\.\d+)*)?\s+-m\s+pip\s+install|"
    r"poetry\s+(?:install|sync|update)|pdm\s+(?:install|sync|update)|"
    r"pipenv\s+(?:install|sync|update)|"
    r"(?:conda|mamba|micromamba)\s+(?:install|create|update|env\s+(?:create|update))|"
    r"tox|nox)\b",
    re.IGNORECASE,
)
_UV_RUN = re.compile(r"\buv\s+run\b|['\"]uv['\"]\s*,\s*['\"]run['\"]", re.IGNORECASE)
_UV_SYNC_NO_LOCK = re.compile(
    r"\buv\s+sync\b(?:(?!&&|\|\||;|\n).)*?(?<!\S)--no-lock(?=\s|$)",
    re.IGNORECASE,
)


def _unsafe_shell_requirement(dockerfile: str) -> str | None:
    logical = dockerfile.replace("\\\r\n", " ").replace("\\\n", " ")
    for line in logical.splitlines():
        if not re.match(r"^\s*RUN\s+", line, re.IGNORECASE):
            continue
        command = re.sub(r"^\s*RUN\s+", "", line, count=1, flags=re.IGNORECASE)
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            continue
        for index, token in enumerate(tokens):
            if token not in {"<", ">"} or index == 0 or index + 1 >= len(tokens):
                continue
            name = tokens[index - 1]
            version = tokens[index + 1]
            prefix = tokens[: index - 1]
            in_pip_install = any(
                _PIP_EXECUTABLE.search(candidate)
                and offset + 1 < len(prefix)
                and prefix[offset + 1] == "install"
                for offset, candidate in enumerate(prefix)
            ) or any(
                candidate == "-m"
                and offset + 2 < len(prefix)
                and prefix[offset + 1 : offset + 3] == ["pip", "install"]
                for offset, candidate in enumerate(prefix)
            )
            if (
                in_pip_install
                and _REQUIREMENT_NAME.fullmatch(name)
                and re.match(r"^=?[A-Za-z0-9*]", version)
            ):
                return f"{name}{token}{version}"
    return None


def _pip_requirement_names(dockerfile: str) -> set[str]:
    logical = dockerfile.replace("\\\r\n", " ").replace("\\\n", " ")
    names: set[str] = set()
    for command in _PIP_INSTALL_COMMAND.finditer(logical):
        try:
            tokens = shlex.split(command.group("arguments"), posix=True)
        except ValueError:
            continue
        skip_value = False
        for token in tokens:
            if skip_value:
                skip_value = False
                continue
            option = token.split("=", 1)[0]
            if option in _PIP_OPTIONS_WITH_VALUE:
                skip_value = "=" not in token
                continue
            if token.startswith("-") or "://" in token or "/" in token:
                continue
            if match := _PYTHON_REQUIREMENT.fullmatch(token):
                names.add(match.group("name").casefold().replace("_", "-"))
    return names


def _pip_local_install_targets(dockerfile: str) -> set[str]:
    logical = dockerfile.replace("\\\r\n", " ").replace("\\\n", " ")
    targets: set[str] = set()
    for command in _PIP_INSTALL_COMMAND.finditer(logical):
        try:
            tokens = shlex.split(command.group("arguments"), posix=True)
        except ValueError:
            continue
        editable_target = False
        for token in tokens:
            if editable_target:
                editable_target = False
                if "://" not in token:
                    targets.add(token.rstrip("/"))
                continue
            option = token.split("=", 1)[0]
            if option in {"-e", "--editable"}:
                if "=" in token:
                    value = token.split("=", 1)[1]
                    if "://" not in value:
                        targets.add(value.rstrip("/"))
                else:
                    editable_target = True
                continue
            if token == "." or token.startswith((".[", "/testbed")):
                targets.add(token.rstrip("/"))
    return targets


def _feedback(code: FeedbackCode, *, expected: str, actual: str) -> AgentFeedback:
    return AgentFeedback(
        issues=(
            issue(
                code,
                path=("dockerfile",),
                expected=expected,
                actual=actual,
            ),
        )
    )


def _uv_project_environment_activation(
    context: ArtifactValidationContext,
) -> AgentFeedback | None:
    if not re.search(r"\buv\s+sync\b", context.dockerfile):
        return None
    if "UV_PROJECT_ENVIRONMENT" in context.dockerfile:
        return None
    if ".venv/bin" in context.activation:
        return None
    return AgentFeedback(
        issues=(
            issue(
                FeedbackCode.ARTIFACT_ACTIVATION_ENVIRONMENT,
                path=("activation",),
                expected="activate /testbed/.venv created by uv sync",
                actual="activation does not reference .venv/bin",
            ),
        )
    )


def _python_artifact_contract(context: ArtifactValidationContext) -> AgentFeedback | None:
    if unsafe := _unsafe_shell_requirement(context.dockerfile):
        return _feedback(
            FeedbackCode.ARTIFACT_SHELL_REQUIREMENT,
            expected="quote the complete Python requirement",
            actual=unsafe,
        )
    for field_name in ("activation", "runner"):
        source = getattr(context, field_name).replace("\\\r\n", " ").replace("\\\n", " ")
        if match := _PYTHON_RUNTIME_DEPENDENCY_MUTATION.search(source):
            return AgentFeedback(
                issues=(
                    issue(
                        FeedbackCode.ARTIFACT_RUNTIME_MUTATION,
                        path=(field_name,),
                        actual=match.group(0),
                    ),
                )
            )
        for line in source.splitlines():
            if _UV_RUN.search(line) and "--no-sync" not in line:
                return AgentFeedback(
                    issues=(
                        issue(
                            FeedbackCode.ARTIFACT_RUNTIME_MUTATION,
                            path=(field_name,),
                            actual="uv run without --no-sync",
                        ),
                    )
                )
    if _UV_SYNC_NO_LOCK.search(context.dockerfile.replace("\\\r\n", " ").replace("\\\n", " ")):
        return _feedback(
            FeedbackCode.BUILD_DEPENDENCY_LOCK,
            expected=(
                "use a uv sync option supported by the managed uv CLI and the repository's lock policy; "
                "use --frozen when a committed lock must be consumed without modification"
            ),
            actual="uv sync --no-lock is unsupported",
        )
    return _uv_project_environment_activation(context)


def _python_repair_contract(
    context: ArtifactRepairValidationContext,
) -> AgentFeedback | None:
    version_issues = tuple(
        item
        for item in context.constraints.issues
        if item.code == FeedbackCode.BUILD_DEPENDENCY_VERSION
    )
    if version_issues:
        target_names = {
            match.group("name").casefold().replace("_", "-")
            for item in version_issues
            if item.expected and (match := _PYTHON_REQUIREMENT.fullmatch(item.expected))
        }
        removed = (
            _pip_requirement_names(context.current.dockerfile)
            - _pip_requirement_names(context.candidate.dockerfile)
            - target_names
        )
        if removed:
            return _feedback(
                FeedbackCode.ARTIFACT_REPAIR_REGRESSION,
                expected="preserve unrelated pip requirements: " + ", ".join(sorted(removed)),
                actual="removed unrelated pip requirements: " + ", ".join(sorted(removed)),
            )
        removed_local_targets = _pip_local_install_targets(
            context.current.dockerfile
        ) - _pip_local_install_targets(context.candidate.dockerfile)
        if removed_local_targets:
            rendered = ", ".join(sorted(removed_local_targets))
            return _feedback(
                FeedbackCode.ARTIFACT_REPAIR_REGRESSION,
                expected="preserve repository-local pip install targets: " + rendered,
                actual="removed repository-local pip install targets: " + rendered,
            )
        normalized = context.candidate.dockerfile.replace('"', "").replace("'", "")
        for item in version_issues:
            # Only a concrete package requirement can be proven statically from the
            # Dockerfile. Natural-language runtime contracts (for example a project
            # whose setuptools-scm version must satisfy pytest minversion) are proven
            # by the next real validation run, not by substring matching prose.
            if (
                item.expected
                and _PYTHON_REQUIREMENT.fullmatch(item.expected)
                and item.expected not in normalized
            ):
                return _feedback(
                    FeedbackCode.ARTIFACT_REPAIR_REGRESSION,
                    expected=f"retain accepted dependency constraint {item.expected}",
                    actual="the constraint is absent from the repaired Dockerfile",
                )
    failure_codes = {item.code for item in context.constraints.issues}
    if FeedbackCode.BUILD_DEPENDENCY_LOCK in failure_codes and re.search(
        r"\buv\s+sync\b[^\n]*\s--locked(?:\s|$)", context.candidate.dockerfile
    ):
        return _feedback(
            FeedbackCode.ARTIFACT_FAILED_FIELD_UNCHANGED,
            expected="remove the rejected uv sync --locked mode",
            actual="uv sync --locked remains in the Dockerfile",
        )
    return None


def register(registry: ArtifactValidationRegistry) -> None:
    registry.register(
        "python.artifact-contract",
        _python_artifact_contract,
        ecosystem_modules=("python",),
    )
    registry.register_repair(
        "python.repair-contract",
        _python_repair_contract,
        ecosystem_modules=("python",),
    )
