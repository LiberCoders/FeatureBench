"""
Codex agent implementation.

This agent runs the external `codex` CLI in the container.
"""

import shlex
from pathlib import Path
from typing import Dict
import json

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY


class CodexAgent(BaseAgent):
    """Codex agent for FeatureBench inference."""

    NODE_VERSION = "22"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def install_script(self) -> str:
        """Installation script for Codex CLI."""
        version = self._kwargs.get("version")
        if not version:
            version = self.env_vars.get("CODEX_VERSION")
        if not version or not str(version).strip():
            version = "latest"

        # Best-effort npm package name for the `codex` CLI.
        # If your environment uses a different distribution, adjust here.
        npm_pkg = f"@openai/codex@{version}"

        return f"""#!/bin/bash
set -e

echo "Installing Codex agent..."

if command -v codex >/dev/null 2>&1; then
    echo "codex already available; skipping installation"
    codex --version || true
    exit 0
fi

# Update package manager
apt-get update
apt-get install -y curl ca-certificates tar xz-utils

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/npm"

export npm_config_cache="$CACHE_ROOT/npm"
export NPM_CONFIG_CACHE="$CACHE_ROOT/npm"

NVM_DIR="/opt/featurebench/nvm"
mkdir -p "$NVM_DIR"

# NOTE: Do NOT share NVM's download cache across containers.
# FeatureBench often runs many infer containers concurrently; sharing the tarball
# cache can lead to corrupted archives and checksum/tar extraction failures.
mkdir -p "$NVM_DIR/.cache"

# Install NVM (idempotent)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash

export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if [ -z "$(command -v nvm)" ]; then
    echo "nvm not available after install" >&2
    exit 1
fi

# Install or reuse Node
nvm install "{self.NODE_VERSION}"
nvm use "{self.NODE_VERSION}"

# Verify npm
npm -v

# Install Codex CLI (downloads cached via NPM_CONFIG_CACHE)
npm install -g {npm_pkg}

# Verify installation
command -v codex >/dev/null 2>&1 || (echo "codex not found after install" >&2 && exit 1)
codex --version || true

codex --version || true

echo "Codex installation complete"
"""

    def pre_run_hook(self, container, log_file) -> bool:
        """Create log dir."""

        # Ensure log dir exists.
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """Collect outputs and validate completion."""

        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(container, "/agent-logs/codex_events.jsonl", log_dir / "codex_events.jsonl")
        except Exception:
            pass
        # Validate the run finished cleanly: last JSONL event should be turn.completed with usage.
        try:
            events_path = log_dir / "codex_events.jsonl"
            if not events_path.is_file():
                self.logger.error("codex_events.jsonl not found; treating run as failed")
                return False

            # Simple gemini_cli-style check: read the last non-empty line as JSON.
            with events_path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            last_line = ""
            for raw in reversed(lines):
                s = raw.strip()
                if s:
                    last_line = s
                    break
            if not last_line:
                self.logger.error("codex_events.jsonl is empty; treating run as failed")
                return False

            last_obj = json.loads(last_line)
            if not isinstance(last_obj, dict):
                self.logger.error("codex_events.jsonl last line is not a JSON object; treating run as failed")
                return False

            if last_obj.get("type") != "turn.completed":
                self.logger.error(
                    f"Codex did not finish with turn.completed (last type={last_obj.get('type')}); run failed"
                )
                return False

            if not isinstance(last_obj.get("usage"), dict):
                self.logger.error("Codex turn.completed missing usage; run failed")
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate codex_events.jsonl completion: {e}")
            return False

        return True

    def failure_hook(self, container, log_file: Path) -> None:
        """Collect outputs on failures (best-effort)."""

        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(container, "/agent-logs/codex_events.jsonl", log_dir / "codex_events.jsonl")
        except Exception:
            pass

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Codex."""
#         instruction = """
# please create a hello-world.txt containing your self-introduction under testbed.     
# """
        escaped_instruction = shlex.quote(instruction)

        model = self._kwargs.get("model")
        # Preserve provider-prefixed model names (e.g., "azure/<deployment>").
        # Codex CLI may use the prefix to select the correct provider/endpoints.

        model_arg = f"--model {shlex.quote(str(model))} " if model else ""

        # Ensure codex is available on PATH by loading NVM.
        return (
            "NVM_DIR=${NVM_DIR:-/opt/featurebench/nvm}; "
            "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            # Preserve the exit code when piping JSONL to a file.
            "set -o pipefail; "
            f"codex exec "
            f"--sandbox danger-full-access "
            f"--skip-git-repo-check "
            f"--json "
            f"{model_arg}"
            f"-- {escaped_instruction}"
            f" | tee /agent-logs/codex_events.jsonl"
        ).rstrip()

    def get_env_setup_script(self) -> str:
        """Get environment setup script for Codex."""
        lines = ["#!/bin/bash", ""]

        api_key = self.env_vars.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for codex agent")

        raw_model = self._kwargs.get("model") or ""
        base_url = self.env_vars.get("OPENAI_BASE_URL") or ""
        is_azure = raw_model.startswith("azure/") or (".openai.azure.com" in base_url)

        env_settings: Dict[str, str] = {
            "OPENAI_API_KEY": api_key,
        }

        # Only set the Azure-specific env var when we're actually on the Azure route.
        # Codex CLI reads the key from the env var referenced by `env_key` in ~/.codex/config.toml.
        if is_azure:
            env_settings["AZURE_OPENAI_API_KEY"] = self.env_vars.get("AZURE_OPENAI_API_KEY") or api_key

        # Optional settings
        for key in [
            "OPENAI_BASE_URL",
            "OPENAI_ORG",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT",
            "OPENAI_API_VERSION",
            "CODEX_REASONING_EFFORT",
        ]:
            if self.env_vars.get(key):
                env_settings[key] = self.env_vars[key]

        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                # IMPORTANT: do not rewrite proxy loopback endpoints.
                # Rewriting 127.0.0.1/localhost to the Docker host gateway breaks local proxies.
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
                if key not in _no_rewrite_keys and ("localhost" in value_str or "127.0.0.1" in value_str):
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())

        # If the user is targeting Azure via the Codex CLI, Codex expects a config file
        # at ~/.codex/config.toml describing the provider wiring.
        # See: base_url must include /openai/v1 and env_key must reference an env var.
        reasoning_effort = (
            self.env_vars.get("CODEX_REASONING_EFFORT")
            or self.env_vars.get("MODEL_REASONING_EFFORT")
            or ""
        )
        reasoning_effort = str(reasoning_effort).strip()
        if reasoning_effort:
            # Basic escaping for TOML double-quoted string.
            reasoning_effort = reasoning_effort.replace("\\", "\\\\").replace('"', '\\"')

        # Azure: https://learn.microsoft.com/zh-cn/azure/ai-foundry/openai/how-to/codex?view=foundry-classic&tabs=npm
        if is_azure:
            deployment = raw_model.split("/", 1)[1] if raw_model.startswith("azure/") else raw_model

            # Codex CLI Azure v1 requires /openai/v1 in the base URL.
            base = (base_url or "").rstrip("/")
            if base and not base.endswith("/openai/v1"):
                if base.endswith("/openai"):
                    base = f"{base}/v1"
                else:
                    base = f"{base}/openai/v1"

            # Keep it minimal: model, provider, and Azure provider wiring.
            lines.extend(
                [
                    "",
                    "# Codex CLI config (auto-generated by FeatureBench)",
                    'mkdir -p "$HOME/.codex"',
                    "cat > \"$HOME/.codex/auth.json\" <<'CODEX_AUTH'",
                    json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False),
                    "CODEX_AUTH",
                    "cat > \"$HOME/.codex/config.toml\" <<'CODEX_TOML'",
                    f'model = "{deployment}"',
                    'model_provider = "azure"',
                    *(
                        [f'model_reasoning_effort = "{reasoning_effort}"']
                        if reasoning_effort
                        else []
                    ),
                    "",
                    "[model_providers.azure]",
                    'name = "Azure OpenAI"',
                    f'base_url = "{base}"',
                    'env_key = "AZURE_OPENAI_API_KEY"',
                    'wire_api = "responses"',
                    "CODEX_TOML",
                ]
            )

        # Non-Azure (OpenAI-compatible): write a Codex config following the user's preferred template.
        else:
            model_name = raw_model
            if model_name.startswith("openai/"):
                model_name = model_name.split("/", 1)[1]

            # Follow the provided template strictly.
            provider_id = "featurebench"
            base = (base_url or "").rstrip("/")

            lines.extend(
                [
                    "",
                    "# Codex CLI config (auto-generated by FeatureBench)",
                    'mkdir -p "$HOME/.codex"',
                    "cat > \"$HOME/.codex/auth.json\" <<'CODEX_AUTH'",
                    json.dumps({"OPENAI_API_KEY": api_key}, ensure_ascii=False),
                    "CODEX_AUTH",
                    "cat > \"$HOME/.codex/config.toml\" <<'CODEX_TOML'",
                    f'model_provider = "{provider_id}"',
                    *([f'model = "{model_name}"'] if model_name else []),
                    *([f'model_reasoning_effort = "{reasoning_effort}"'] if reasoning_effort else []),
                    "disable_response_storage = true",
                    "",
                    f"[model_providers.{provider_id}]",
                    f'name = "{provider_id}"',
                    *([f'base_url = "{base}"'] if base else []),
                    'wire_api = "responses"',
                    "CODEX_TOML",
                ]
            )

        # Load NVM (in case codex is installed via npm)
        lines.extend(
            [
                "",
                "# Load NVM",
                'export NVM_DIR="/opt/featurebench/nvm"',
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true',
            ]
        )

        return "\n".join(lines)
