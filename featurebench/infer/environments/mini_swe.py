"""Restricted command environment loaded by mini-swe-agent inside a task container.

The mini-swe-agent controller runs as root because it needs its private Python
environment and model credentials. Commands proposed by the model run as the
unprivileged ``fbagent`` user with a sanitized environment instead.
"""

from __future__ import annotations

import os
import platform
import pwd
import signal
import subprocess
from typing import Any

from minisweagent.environments.local import (
    LocalEnvironment,
    LocalEnvironmentConfig,
)
from minisweagent.utils.serialize import recursive_merge


_TASK_USER = "fbagent"
_PRIVATE_PATH_PREFIXES = (
    "/opt/mini-swe-agent-venv",
    "/opt/featurebench-controller",
)
_BLOCKED_ENV_NAMES = {
    "ALL_PROXY",
    "ANTHROPIC_BASE_URL",
    "API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "AZURE_API_BASE",
    "FB_UPSTREAM_PROXY",
    "GEMINI_API_BASE",
    "GOOGLE_GEMINI_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "MINI_SWE_AGENT_PYTHON",
    "MSWEA_BASE_URL",
    "NO_PROXY",
    "OPENAI_BASE_URL",
    "PYTHONHOME",
    "PYTHONPATH",
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "VIRTUAL_ENV",
}
_BLOCKED_ENV_SUFFIXES = (
    "_API_KEY",
    "_ACCESS_KEY",
    "_AUTH_TOKEN",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


def _is_blocked_env_name(name: str) -> bool:
    normalized = name.upper()
    return normalized in _BLOCKED_ENV_NAMES or normalized.endswith(
        _BLOCKED_ENV_SUFFIXES
    )


def _safe_environment(extra: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in (os.environ | extra).items()
        if not _is_blocked_env_name(key)
    }
    path = env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PATH"] = os.pathsep.join(
        entry
        for entry in path.split(os.pathsep)
        if entry and not entry.startswith(_PRIVATE_PATH_PREFIXES)
    )
    env.update(
        {
            "HOME": f"/home/{_TASK_USER}",
            "LOGNAME": _TASK_USER,
            "SHELL": "/bin/bash",
            "USER": _TASK_USER,
        }
    )
    return env


def _run_as_task_user(
    command: str,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    account = pwd.getpwnam(_TASK_USER)
    process = subprocess.Popen(
        [
            "/usr/bin/setpriv",
            f"--reuid={account.pw_uid}",
            f"--regid={account.pw_gid}",
            "--init-groups",
            "--no-new-privs",
            "/bin/bash",
            "-c",
            command,
        ],
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)


class RestrictedLocalEnvironment(LocalEnvironment):
    """Execute model-proposed commands without controller access or secrets."""

    def __init__(
        self,
        *,
        config_class: type[LocalEnvironmentConfig] = LocalEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_class=config_class, **kwargs)

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        try:
            result = _run_as_task_user(
                command,
                cwd,
                _safe_environment(self.config.env),
                timeout or self.config.timeout,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            }
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(),
            platform.uname()._asdict(),
            _safe_environment(self.config.env),
            kwargs,
        )
