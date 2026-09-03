"""
mini-swe-agent implementation.

This agent runs the external `mini` CLI in the container.
"""

import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY


class MiniSweAgent(BaseAgent):
    """mini-swe-agent for FeatureBench inference."""

    _PROVIDER_API_KEY_ENV: Dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHERAI_API_KEY",
        "togetherai": "TOGETHERAI_API_KEY",
        "together_ai": "TOGETHERAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "azure": "AZURE_API_KEY",
    }
    _PROVIDER_BASE_URL_ENV: Dict[str, List[str]] = {
        "anthropic": ["ANTHROPIC_BASE_URL"],
        "openai": ["OPENAI_BASE_URL"],
        "azure": ["AZURE_API_BASE"],
        "google": ["GOOGLE_GEMINI_BASE_URL"],
    }

    @property
    def name(self) -> str:
        return "mini_swe_agent"

    def _get_model_name(self) -> str:
        model = self._kwargs.get("model")
        model_name = str(model).strip() if model is not None else ""
        if not model_name:
            raise RuntimeError("Model is required for mini_swe_agent")
        return model_name

    def _get_model_provider(self) -> str:
        model_name = self._get_model_name()
        if "/" not in model_name:
            raise RuntimeError(
                "mini_swe_agent expects model in 'provider/model' format, "
                f"got: {model_name!r}"
            )
        provider, _ = model_name.split("/", 1)
        return provider.strip().lower()

    def _get_provider_api_key_env(self, provider: str) -> Optional[str]:
        return self._PROVIDER_API_KEY_ENV.get(provider)

    def _get_provider_base_url_envs(self, provider: str) -> List[str]:
        return self._PROVIDER_BASE_URL_ENV.get(provider, [])

    @property
    def install_script(self) -> str:
        """Installation script for mini-swe-agent CLI."""
        version = self._kwargs.get("version") or self.env_vars.get("MINI_SWE_AGENT_VERSION")
        version_str = str(version).strip() if version is not None else ""
        pkg = "mini-swe-agent" if not version_str else f"mini-swe-agent=={version_str}"
        pkg_escaped = shlex.quote(pkg)

        return f"""#!/bin/bash
set -e

echo "Installing mini-swe-agent..."

apt-get update
apt-get install -y passwd python3 python3-pip python3-venv util-linux

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/pip"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

VENV_DIR="/opt/mini-swe-agent-venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install {pkg_escaped}

"$VENV_DIR/bin/python" -m minisweagent.run.mini --version || true
"$VENV_DIR/bin/python" -m minisweagent.run.mini --help >/dev/null 2>&1 || true

echo "mini-swe-agent installation complete"
"""

    def pre_run_hook(self, container, log_file) -> bool:
        """Create agent logs directory before running."""
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)
        return True

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        """Isolate agent-issued shell commands from controller dependencies."""
        controller_dir = "/opt/featurebench-controller"
        environment_source = (
            Path(__file__).parent.parent / "environments" / "mini_swe.py"
        )
        environment_target = f"{controller_dir}/featurebench_mini_swe_environment.py"

        try:
            exit_code, output = self.cm.exec_command(
                container,
                "set -e\n"
                "getent group fbagent >/dev/null || groupadd --system fbagent\n"
                "id -u fbagent >/dev/null 2>&1 || "
                "useradd --system --gid fbagent --home-dir /home/fbagent "
                "--create-home --shell /bin/bash fbagent\n"
                f"install -d -o root -g root -m 700 {controller_dir} "
                "/agent-logs\n",
                log_file=log_file,
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Failed to create restricted mini-swe-agent user: {output}"
                )

            self.cm.copy_to_container(
                container,
                environment_source,
                environment_target,
            )
            exit_code, output = self.cm.exec_command(
                container,
                "set -e\n"
                f"chown root:root {environment_target}\n"
                f"chmod 600 {environment_target}\n"
                "chown -R fbagent:fbagent /testbed\n"
                "chmod 700 /opt/mini-swe-agent-venv\n"
                f"chmod 700 {controller_dir} /installed-agent /agent-logs\n"
                "setpriv --reuid=fbagent --regid=fbagent --init-groups "
                "--no-new-privs /bin/bash -c "
                "'test -r /testbed && test -w /testbed && "
                "test ! -r /opt/mini-swe-agent-venv/pyvenv.cfg && "
                "test ! -r /installed-agent/setup-env.sh'",
                log_file=log_file,
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Failed to enforce mini-swe-agent filesystem isolation: {output}"
                )
        except Exception as e:
            self.logger.error(f"Failed to prepare mini-swe-agent isolation: {e}")
            # Isolation is a security boundary. Do not let the runner continue
            # with root-level tool execution if setup fails.
            raise

        self.logger.info(
            "mini-swe-agent tool commands will run as restricted user fbagent"
        )
        return True

    def post_run_hook(self, container, log_file) -> bool:
        """Persist artifacts and accept only an explicit agent submission."""
        return self._collect_run_artifacts(container, log_file)

    def _collect_run_artifacts(self, container, log_file: Path) -> bool:
        """Copy run artifacts and validate mini-swe-agent's terminal status."""
        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/mini_swe_agent_output.log",
                log_dir / "mini_swe_agent_output.log",
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect mini-swe-agent output: {e}")

        trajectory_path = log_dir / "traj.json"
        try:
            self.cm.copy_from_container(
                container,
                "/root/.config/mini-swe-agent/last_mini_run.traj.json",
                trajectory_path,
            )
        except Exception as e:
            self.logger.error(f"Failed to collect mini-swe-agent trajectory: {e}")
            return False

        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            exit_status = trajectory.get("info", {}).get("exit_status")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"Failed to read mini-swe-agent exit status: {e}")
            return False

        if not isinstance(exit_status, str) or not exit_status.strip():
            self.logger.error("mini-swe-agent trajectory has no exit_status")
            return False

        self.agent_exit_status = exit_status.strip()
        if self.agent_exit_status != "Submitted":
            self.logger.warning(
                f"mini-swe-agent exited without submission: {self.agent_exit_status}"
            )
            return False

        self.logger.info("mini-swe-agent exit status: Submitted")
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        """Collect mini-swe-agent output on failures (best-effort)."""
        trajectory_path = Path(log_file).parent / "traj.json"
        if not trajectory_path.exists():
            self._collect_run_artifacts(container, log_file)

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run mini-swe-agent."""
        escaped_instruction = shlex.quote(instruction)
        model_name = self._get_model_name()
        cost_limit = self._kwargs.get("cost_limit")
        cost_limit_arg = ""
        if cost_limit is not None:
            cost_limit_arg = f"--cost-limit {shlex.quote(str(cost_limit))} "
        return (
            "set -o pipefail; "
            f'"$MINI_SWE_AGENT_PYTHON" -I - -m {shlex.quote(model_name)} -t {escaped_instruction} -y --exit-immediately '
            f"{cost_limit_arg}"
            "--environment-class "
            "featurebench_mini_swe_environment.RestrictedLocalEnvironment "
            "<<'PY' | tee /agent-logs/mini_swe_agent_output.log\n"
            "import runpy\n"
            "import sys\n"
            "\n"
            "sys.path = [p for p in sys.path if p and not p.startswith('/testbed')]\n"
            "sys.path.insert(0, '/opt/featurebench-controller')\n"
            "sys.argv = ['minisweagent.run.mini', *sys.argv[1:]]\n"
            "runpy.run_module('minisweagent.run.mini', run_name='__main__')\n"
            "PY"
        )

    def get_env_setup_script(self) -> str:
        """Get environment setup script for mini-swe-agent."""
        lines = ["#!/bin/bash", ""]

        provider = self._get_model_provider()
        env_settings: Dict[str, str] = {"MSWEA_CONFIGURED": "true"}
        env_settings["MINI_SWE_AGENT_PYTHON"] = "/opt/mini-swe-agent-venv/bin/python"

        # api key setup
        mswea_api_key = self.env_vars.get("MSWEA_API_KEY")
        if not mswea_api_key:
            raise RuntimeError("MSWEA_API_KEY is required for mini_swe_agent")
        env_settings["MSWEA_API_KEY"] = mswea_api_key
        provider_api_key_env = self._get_provider_api_key_env(provider)
        if provider_api_key_env:
            env_settings[provider_api_key_env] = mswea_api_key

        # base url setup
        mswea_base_url = self.env_vars.get("MSWEA_BASE_URL")
        if mswea_base_url:
            env_settings["MSWEA_BASE_URL"] = mswea_base_url
            for provider_base_key in self._get_provider_base_url_envs(provider):
                env_settings[provider_base_key] = mswea_base_url

        # cost tracking setup
        mswea_cost_tracking = self.env_vars.get("MSWEA_COST_TRACKING")
        if mswea_cost_tracking is not None and str(mswea_cost_tracking).strip():
            env_settings["MSWEA_COST_TRACKING"] = str(mswea_cost_tracking).strip()
        else:
            # Avoid hard failures when LiteLLM has no price mapping for the selected model.
            env_settings["MSWEA_COST_TRACKING"] = "ignore_errors"

        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        # proxy setup
        _no_rewrite_keys = {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        }
        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                if key not in _no_rewrite_keys and (
                    "localhost" in value_str or "127.0.0.1" in value_str
                ):
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        # Handle --runtime-proxy off mode.
        lines.extend(self._get_proxy_unset_lines())

        return "\n".join(lines)
