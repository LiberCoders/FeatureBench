"""Temporary PyPI configuration, including frozen uv lockfile artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from codeworld.sources.base import (
    SourceAdapter,
    SourceBuildContext,
    SourceBuildFragments,
    SourceFailureKind,
    SourceMatchConfidence,
    harness_run,
    observed_failure_output,
)

_UV_PROJECT_COMMAND = re.compile(
    r"(?:^|[\s;&|/])uv\s+(?:sync|run|lock|export|tree)(?:\s|$)",
    re.IGNORECASE,
)
_PYPI_INDEX_MISS = re.compile(
    r"no matching distribution|could not find a version that satisfies",
    re.IGNORECASE,
)
_PYPI_SEMANTIC_FAILURE = re.compile(
    r"resolutionimpossible|package versions have conflicting dependencies|"
    r"\b[A-Za-z0-9_.+-]+ executable not found\b",
    re.IGNORECASE,
)
_PIP_PROJECT_COMMAND = re.compile(
    r"(?:^|[\s;&|/])(?:python(?:\d+(?:\.\d+)*)?\s+-m\s+pip|pip\d*(?:\.\d+)*)\s+"
    r"(?:install|download|wheel)(?:\s|$)",
    re.IGNORECASE,
)
_UV_PACKAGE_COMMAND = re.compile(
    r"(?:^|[\s;&|/])uv\s+(?:sync|lock|export|pip\s+(?:install|sync|compile))"
    r"(?:\s|$)",
    re.IGNORECASE,
)
_UV_RUN_COMMAND = re.compile(r"(?:^|[\s;&|/])uv\s+run(?:\s|$)", re.IGNORECASE)
_POETRY_EXECUTABLE = re.compile(
    r"(?<![A-Za-z0-9_./'\"-])poetry"
    r"(?=(?:\s+(?:--?[A-Za-z][A-Za-z0-9-]*(?:=[^\s\\]+)?))*"
    r"\s+(?:install|sync|lock|update)(?:\s|\\|$))",
    re.IGNORECASE,
)
_POETRY_COMMAND = re.compile(
    r"(?:^|[\s;&|])(?:poetry|/var/tmp/\.featurebench-pypi-poetry-wrapper)"
    r"(?:\s+(?:--?[A-Za-z][A-Za-z0-9-]*(?:=[^\s\\]+)?))*"
    r"\s+(?:install|sync|lock|update)(?:\s|$)",
    re.IGNORECASE,
)
_VCS_REQUIREMENT = re.compile(
    r"(?:git\+[^\s'\"]*github\.com|git\+(?:https?|ssh|git)|git://|"
    r"ssh://git@|git@github\.com:)",
    re.IGNORECASE,
)
_VCS_FAILURE = re.compile(
    r"\bgit (?:clone|fetch)\b|fatal:.*(?:github\.com|unable to fork|could not read)|"
    r"failed to build ['\"]?git\+",
    re.IGNORECASE | re.DOTALL,
)
_POETRY_APPLIED = "FEATUREBENCH_PYPI_POETRY_SOURCE_APPLIED="
_POETRY_UNAVAILABLE = "FEATUREBENCH_PYPI_POETRY_ADAPTER_UNAVAILABLE"
_MARKER = "/var/tmp/.featurebench-pypi-uv-wrapper"
_POETRY_WRAPPER_PATH = "/var/tmp/.featurebench-pypi-poetry-wrapper"
_OFFICIAL_ARTIFACTS = "https://files.pythonhosted.org/"
_RUN_INSTRUCTION = re.compile(r"^\s*RUN\s+(?!\[)", re.IGNORECASE)

_WRAPPER = """#!/bin/sh
set -u
real_uv="${0}.featurebench-source-real"
directory=$PWD
uses_lock=false
directory_option=false
for argument in "$@"; do
  if [ "$directory_option" = true ]; then
    directory=$argument
    directory_option=false
    continue
  fi
  case "$argument" in
    sync|run|lock|export|tree) uses_lock=true ;;
    --directory|--project) directory_option=true ;;
    --directory=*|--project=*) directory=${argument#*=} ;;
  esac
done
if [ "$uses_lock" = false ]; then
  exec "$real_uv" "$@"
fi
case "$directory" in
  /*) ;;
  *) directory="$PWD/$directory" ;;
esac
while [ "$directory" != / ] && [ ! -f "$directory/uv.lock" ]; do
  directory=${directory%/*}
  [ -n "$directory" ] || directory=/
done
lock_file="$directory/uv.lock"
if [ ! -f "$lock_file" ]; then
  exec "$real_uv" "$@"
fi
backup=$(mktemp /var/tmp/featurebench-uv-lock.XXXXXX)
cp -a "$lock_file" "$backup"
restored=false
restore_lock() {
  if [ "$restored" = false ]; then
    cp -a "$backup" "$lock_file"
    rm -f "$backup"
    restored=true
  fi
}
trap 'restore_lock' EXIT
trap 'restore_lock; exit 129' HUP
trap 'restore_lock; exit 130' INT
trap 'restore_lock; exit 143' TERM
sed -i \
  -e "s|https://pypi.org/simple|${UV_DEFAULT_INDEX}|g" \
  -e "s|https://files.pythonhosted.org/|${FEATUREBENCH_PYPI_ARTIFACT_BASE_URL}|g" \
  "$lock_file"
"$real_uv" "$@"
status=$?
restore_lock
trap - EXIT HUP INT TERM
exit "$status"
"""

_POETRY_WRAPPER = f"""#!/bin/sh
set -u
real_poetry=$(command -v poetry 2>/dev/null || true)
if [ -z "$real_poetry" ]; then
  echo "{_POETRY_UNAVAILABLE}: poetry executable not found" >&2
  exit 127
fi
case "${{PIP_INDEX_URL%/}}" in
  https://pypi.org/simple) exec "$real_poetry" "$@" ;;
esac
directory=$PWD
directory_option=false
for argument in "$@"; do
  if [ "$directory_option" = true ]; then
    directory=$argument
    directory_option=false
    continue
  fi
  case "$argument" in
    -C|-P|--directory|--project) directory_option=true ;;
    -C=*|-P=*|--directory=*|--project=*) directory=${{argument#*=}} ;;
  esac
done
case "$directory" in
  /*) ;;
  *) directory="$PWD/$directory" ;;
esac
if ! directory=$(cd "$directory" 2>/dev/null && pwd -P); then
  echo "{_POETRY_UNAVAILABLE}: project directory is unavailable" >&2
  exec "$real_poetry" "$@"
fi
pyproject="$directory/pyproject.toml"
if [ ! -f "$pyproject" ]; then
  exec "$real_poetry" "$@"
fi
pyproject_backup=$(mktemp /var/tmp/featurebench-pyproject.XXXXXX)
lock_file="$directory/poetry.lock"
lock_backup=$(mktemp /var/tmp/featurebench-poetry-lock.XXXXXX)
cp -a "$pyproject" "$pyproject_backup"
lock_existed=false
if [ -f "$lock_file" ]; then
  cp -a "$lock_file" "$lock_backup"
  lock_existed=true
fi
restored=false
restore_project() {{
  if [ "$restored" = false ]; then
    cp -a "$pyproject_backup" "$pyproject"
    if [ "$lock_existed" = true ]; then
      cp -a "$lock_backup" "$lock_file"
    else
      rm -f "$lock_file"
    fi
    rm -f "$pyproject_backup" "$lock_backup"
    restored=true
  fi
}}
trap 'restore_project' EXIT
trap 'restore_project; exit 129' HUP
trap 'restore_project; exit 130' INT
trap 'restore_project; exit 143' TERM
source_name=featurebench-build-pypi
(
  cd "$directory" || exit 1
  "$real_poetry" source remove "$source_name" >/dev/null 2>&1 || true
  if "$real_poetry" source add --help 2>&1 | grep -q -- '--priority'; then
    "$real_poetry" source add --priority=primary "$source_name" "$PIP_INDEX_URL" >/dev/null
  else
    "$real_poetry" source add --default "$source_name" "$PIP_INDEX_URL" >/dev/null
  fi
)
configured=$?
if [ "$configured" -ne 0 ]; then
  restore_project
  trap - EXIT HUP INT TERM
  echo "{_POETRY_UNAVAILABLE}: Poetry rejected the temporary package source" >&2
  exec "$real_poetry" "$@"
fi
printf '{_POETRY_APPLIED}%s\\n' "$PIP_INDEX_URL" >&2
"$real_poetry" "$@"
status=$?
restore_project
trap - EXIT HUP INT TERM
exit "$status"
"""


def _artifact_base(url: str, *, canonical: bool) -> str:
    if canonical:
        return _OFFICIAL_ARTIFACTS
    normalized = url.rstrip("/")
    if normalized.endswith("/simple"):
        normalized = normalized[: -len("/simple")]
    return normalized.rstrip("/") + "/"


def _rewrite_poetry_runs(source: str) -> str:
    """Wrap Poetry project commands only inside shell-form Dockerfile RUN blocks."""

    lines = source.splitlines(keepends=True)
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _RUN_INSTRUCTION.match(line) is None:
            rendered.append(line)
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end - 1].rstrip("\r\n").rstrip().endswith("\\"):
            end += 1
        block = "".join(lines[index:end])
        rendered.append(_POETRY_EXECUTABLE.sub(_POETRY_WRAPPER_PATH, block))
        index = end
    return "".join(rendered)


def _script_commands(path: str, script: str) -> str:
    def quote(line: str) -> str:
        return "'" + line.replace("'", "'\"'\"'") + "'"

    lines = " \\\n      ".join(quote(line) for line in script.splitlines())
    return f"""printf '%s\\n' \\
      {lines} > {path}; \\
    chmod 0755 {path}"""


def _write_script(path: str, script: str) -> str:
    return harness_run(
        f"""set -eu; \\
    {_script_commands(path, script)}
"""
    )


def _wrapper_setup() -> str:
    return harness_run(f"""set -eu; \\
    real_uv=$(command -v uv 2>/dev/null || true); \\
    marker={_MARKER}; \\
    if [ -n "$real_uv" ] && [ ! -e "${{real_uv}}.featurebench-source-real" ]; then \\
      mv "$real_uv" "${{real_uv}}.featurebench-source-real"; \\
      {_script_commands("$real_uv", _WRAPPER)}; \\
      printf '%s\\n' "$real_uv" > "$marker"; \\
    fi
""")


_POETRY_SETUP = _write_script(_POETRY_WRAPPER_PATH, _POETRY_WRAPPER)


_CLEANUP = harness_run(f"""set -eu; \\
    marker={_MARKER}; \\
    if [ -f "$marker" ]; then \\
      real_uv=$(cat "$marker"); \\
      if [ -e "${{real_uv}}.featurebench-source-real" ]; then \\
        mv -f "${{real_uv}}.featurebench-source-real" "$real_uv"; \\
      fi; \\
      rm -f "$marker"; \\
    fi
""")

_POETRY_CLEANUP = harness_run(f"rm -f {_POETRY_WRAPPER_PATH}")


@dataclass(frozen=True, slots=True)
class PypiSourceAdapter(SourceAdapter):
    """Apply a PyPI mirror without changing the repository or final image."""

    name: str = "pypi"
    priority: int = 150
    services: frozenset[str] = frozenset({"pypi"})

    def build_arguments(self, service: str, *, url: str, canonical: bool) -> Mapping[str, str]:
        if service != "pypi":
            return {}
        return {
            "PIP_INDEX_URL": url,
            "UV_INDEX_URL": url,
            "UV_DEFAULT_INDEX": url,
            "FEATUREBENCH_PYPI_ARTIFACT_BASE_URL": _artifact_base(url, canonical=canonical),
        }

    def download_arguments(
        self,
        service: str,
        *,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        retries: int,
    ) -> Mapping[str, str]:
        if service != "pypi":
            return {}
        return {
            "PIP_DEFAULT_TIMEOUT": str(timeout_seconds),
            "PIP_RETRIES": str(retries),
            "UV_HTTP_CONNECT_TIMEOUT": str(connect_timeout_seconds),
            "UV_HTTP_TIMEOUT": str(timeout_seconds),
            "UV_HTTP_RETRIES": str(retries),
        }

    def fragments(self, source: str, context: SourceBuildContext) -> SourceBuildFragments | None:
        if "pypi" not in context.services:
            return None
        setup = ""
        cleanup = ""
        if _UV_PROJECT_COMMAND.search(source) is not None:
            setup += _wrapper_setup()
            cleanup = _CLEANUP + cleanup
        if _POETRY_WRAPPER_PATH in source:
            setup += _POETRY_SETUP
            cleanup = _POETRY_CLEANUP + cleanup
        if not setup:
            return None
        return SourceBuildFragments(after_from=setup, before_final_user=cleanup)

    def transform(self, source: str, context: SourceBuildContext) -> str:
        if "pypi" not in context.services:
            return source
        return _rewrite_poetry_runs(source)

    def failure_hints(self, service: str) -> tuple[str, ...]:
        if service != "pypi":
            return ()
        return ("pip ", "pip3 ", "uv sync", "uv pip", "poetry ", "pypi", "pythonhosted")

    def match_build_step(
        self,
        service: str,
        *,
        instruction: str,
        output: str,
        endpoint_url: str,
    ) -> SourceMatchConfidence | None:
        if service != "pypi":
            return None
        observed = observed_failure_output(output)
        if _POETRY_UNAVAILABLE in observed:
            return None
        if _VCS_REQUIREMENT.search(instruction) is not None and _VCS_FAILURE.search(observed):
            return None
        if _UV_RUN_COMMAND.search(instruction) is not None:
            if "--no-sync" in instruction.casefold():
                return None
            confidence = (
                SourceMatchConfidence.ENDPOINT
                if endpoint_url.casefold() in observed.casefold()
                else SourceMatchConfidence.OPERATION
            )
            observed_failure = SourceAdapter.classify_build_failure(
                self,
                service,
                detail=observed,
                confidence=confidence,
            )
            if observed_failure is None and _PYPI_INDEX_MISS.search(observed) is None:
                return None
            return confidence
        if endpoint_url.casefold() in observed.casefold():
            return SourceMatchConfidence.ENDPOINT
        if _PIP_PROJECT_COMMAND.search(instruction) is not None:
            return SourceMatchConfidence.OPERATION
        if _UV_PACKAGE_COMMAND.search(instruction) is not None:
            return SourceMatchConfidence.OPERATION
        if _POETRY_COMMAND.search(instruction) is not None:
            applied = f"{_POETRY_APPLIED}{endpoint_url}".casefold()
            return SourceMatchConfidence.ENDPOINT if applied in observed.casefold() else None
        return None

    def classify_build_failure(
        self,
        service: str,
        *,
        detail: str,
        confidence: SourceMatchConfidence,
    ) -> SourceFailureKind | None:
        if service == "pypi" and _PYPI_SEMANTIC_FAILURE.search(detail) is not None:
            return None
        classified = SourceAdapter.classify_build_failure(
            self,
            service,
            detail=detail,
            confidence=confidence,
        )
        if classified is not None:
            return classified
        if service == "pypi" and _PYPI_INDEX_MISS.search(detail) is not None:
            return SourceFailureKind.NOT_FOUND
        return None
