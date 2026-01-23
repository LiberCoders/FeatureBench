"""
Gemini CLI agent implementation.

This agent runs the external `gemini` CLI in the container.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Dict

from acebench.infer.agents.base import BaseAgent
from acebench.infer.container import DOCKER_HOST_GATEWAY


class GeminiCliAgent(BaseAgent):
    """Gemini CLI agent for ACE-Bench inference."""

    NODE_VERSION = "22"

    @property
    def name(self) -> str:
        return "gemini_cli"

    @property
    def install_script(self) -> str:
        """Installation script for Gemini CLI."""
        version = self._kwargs.get("version")
        if not version:
            version = self.env_vars.get("GEMINI_CLI_VERSION")
        if not version or not str(version).strip():
            version = "latest"

        npm_pkg = f"@google/gemini-cli@{version}"

        return f"""#!/bin/bash
set -e

echo "Installing Gemini CLI agent..."

# Update package manager
apt-get update
apt-get install -y curl ca-certificates tar xz-utils

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/npm"

export npm_config_cache="$CACHE_ROOT/npm"
export NPM_CONFIG_CACHE="$CACHE_ROOT/npm"

NVM_DIR="/opt/acebench/nvm"
mkdir -p "$NVM_DIR"

# NOTE: Do NOT share NVM's download cache across containers.
# In ACE-Bench we often run many infer containers concurrently; sharing the
# tarball cache can lead to corrupted archives and checksum/tar extraction
# failures. Keep it container-local for reliability.
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

# Install Gemini CLI (downloads cached via NPM_CONFIG_CACHE)
npm install -g {npm_pkg}

# Verify installation
command -v gemini >/dev/null 2>&1 || (echo "gemini not found after install" >&2 && exit 1)
gemini --version || true

echo "Gemini CLI installation complete"
"""

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Gemini CLI."""
#         instruction = """
# please create a hello-world.txt containing your self-introduction under testbed.
# """
        escaped_instruction = shlex.quote(instruction)

        model = self._kwargs.get("model")
        if model and "/" in model:
            model = model.split("/")[-1]
        model_arg = f"-m {shlex.quote(model)}" if model else ""

        # Ensure gemini is available on PATH by loading NVM.
        return (
            "NVM_DIR=${NVM_DIR:-/opt/acebench/nvm}; "
            "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            f"gemini -p {escaped_instruction} --record-responses /agent-logs/gemini_cli_responses.jsonl --output-format stream-json -y {model_arg} | tee /agent-logs/gemini_cli_stream_output.jsonl".rstrip()
        )

    def pre_run_hook(self, container, log_file) -> bool:
        """
        Create agent logs directory before running.
        """
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """
        Save gemini_cli_stream_output.jsonl
        Check if the last line of gemini_cli_stream_output.jsonl is a success
        If not, return False
        If success, return True
        """
        log_dir = Path(log_file).parent

        gemini_cli_responses_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/gemini_cli_responses.jsonl",
            log_dir / "gemini_cli_responses.jsonl"
        )
        
        if not gemini_cli_responses_copied:
            self.logger.error("Failed to copy gemini_cli_responses.jsonl from container")
            return False

        gemini_cli_stream_output_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/gemini_cli_stream_output.jsonl",
            log_dir / "gemini_cli_stream_output.jsonl"
        )
        
        if not gemini_cli_stream_output_copied:
            self.logger.error("Failed to copy gemini_cli_stream_output.jsonl from container")
            return False

        # check the last line of gemini_cli_stream_output.jsonl as a json object
        # if key "type" is "result" and key "subtype" is "success", then return True
        # otherwise return False
        with open(log_dir / "gemini_cli_stream_output.jsonl", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
            last_line_json = json.loads(last_line)
            if "type" in last_line_json and last_line_json["type"] == "result" and "status" in last_line_json and last_line_json["status"] == "success":
                return True
            else:
                self.logger.error("Last line of gemini_cli_stream_output.jsonl is not a success, agent may not have finished properly")
                return False

    def get_env_setup_script(self) -> str:
        """Get environment setup script for Gemini CLI."""
        lines = ["#!/bin/bash", ""]

        env_settings: Dict[str, str] = {}

        # Primary authentication method: Gemini API key
        if self.env_vars.get("GEMINI_API_KEY"):
            env_settings["GEMINI_API_KEY"] = self.env_vars["GEMINI_API_KEY"]

        # Alternative authentication methods
        for key in [
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
        ]:
            if self.env_vars.get(key):
                env_settings[key] = self.env_vars[key]

        # Optional: allow configuring model via CLI.
        # Normalize provider-style model names like ".../models/gemini-3-pro-preview".
        model = self._kwargs.get("model")
        if model and "/" in model:
            model = model.split("/")[-1]
        if model:
            env_settings["GEMINI_MODEL"] = str(model)

        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                if "localhost" in value_str or "127.0.0.1" in value_str:
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())

        # Proxy bypass for selected gateways (avoid flaky proxy TLS issues).
        # We append to existing NO_PROXY/no_proxy instead of overwriting.
        lines.extend(
            [
                "",
                "# Proxy bypass for selected gateways",
                '# Always bypass proxies for local loopback',
                'ACE_NO_PROXY_LOOPBACK="${ACE_NO_PROXY_LOOPBACK:-localhost,127.0.0.1,::1}"',
                'ACE_NO_PROXY_HOSTS="${ACE_NO_PROXY_HOSTS:-yunwu.ai,api3.wlai.vip,dashscope.aliyuncs.com}"',
                'ACE_NO_PROXY_ALL="${ACE_NO_PROXY_LOOPBACK},${ACE_NO_PROXY_HOSTS}"',
                '_ace_no_proxy_current="${NO_PROXY:-${no_proxy:-}}"',
                'if [ -n "$_ace_no_proxy_current" ]; then',
                '  export NO_PROXY="${_ace_no_proxy_current},${ACE_NO_PROXY_ALL}"',
                "else",
                '  export NO_PROXY="${ACE_NO_PROXY_ALL}"',
                "fi",
                'export no_proxy="$NO_PROXY"',
            ]
        )

        # Load NVM
        lines.extend(
            [
                "",
                "# Load NVM",
                'export NVM_DIR="/opt/acebench/nvm"',
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true',
            ]
        )

        # Workaround: gemini-cli-core may call internal service models (e.g. loop detection)
        # that default to gemini-2.5-flash / gemini-2.5-pro, ignoring the main -m selection.
        # We write a workspace settings file to remap those internal aliases to the chosen
        # model so enterprise gateways / allowlists don't fail.
        lines.extend(
            [
                "",
                "# Configure workspace model overrides (avoid internal gemini-2.5-flash calls)",
                'if [ -n "${GEMINI_MODEL:-}" ]; then',
                '  mkdir -p /root/.gemini',
                "  cat > /root/.gemini/settings.json << 'JSONEOF'",
                "{",
                '  "modelConfigs": {',
                '    "overrides": [',
                "      {",
                '        "match": { "model": "loop-detection" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "loop-detection-double-check" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "classifier" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "prompt-completion" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "summarizer-default" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "summarizer-shell" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "next-speaker-checker" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "llm-edit-fixer" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "edit-corrector" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-search" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-fetch" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "web-fetch-fallback" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-default" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-pro" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-flash" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-2.5-flash-lite" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-3-pro" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      },",
                "      {",
                '        "match": { "model": "chat-compression-3-flash" },',
                '        "modelConfig": { "model": "$GEMINI_MODEL" }',
                "      }",
                "    ]",
                "  }",
                "}",
                "JSONEOF",
                "fi",
            ]
        )

        return "\n".join(lines)

    def failure_hook(self, container, log_file: Path) -> None:
        """Persist gemini-cli detailed error reports on failures.

        gemini-cli writes a JSON report to /tmp (e.g.
        /tmp/gemini-client-error-Turn.run-sendMessageStream-<timestamp>.json).
        Containers are removed on failure, so we copy these reports into the
        attempt output directory for debugging.
        """

        # Prefer the *exact* report path(s) that gemini-cli printed for this run.
        # This avoids ambiguity when /tmp contains many old reports.
        candidates: list[str] = []
        try:
            # Read only the tail of the log to keep this O(1) in file size.
            with open(log_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 64 * 1024), 0)
                tail = f.read().decode("utf-8", errors="ignore")

            # Example line:
            #   Error when talking to Gemini API Full report available at: /tmp/gemini-client-error-....json
            candidates = re.findall(r"(/tmp/gemini-client-error-[^\s]+?\.json)", tail)
            # Dedupe while preserving order
            seen: set[str] = set()
            candidates = [p for p in candidates if not (p in seen or seen.add(p))]
        except Exception:
            candidates = []

        # Fallback: copy a few most recent reports.
        if not candidates:
            exit_code, output = self.cm.exec_command(
                container,
                "ls -1t /tmp/gemini-client-error-*.json 2>/dev/null | head -n 5 || true",
            )
            if exit_code != 0:
                return

            candidates = [line.strip() for line in output.splitlines() if line.strip()]
            if not candidates:
                return

        output_dir = log_file.parent

        # with open(log_file, "a", encoding="utf-8") as f:
        #     f.write("\n[Gemini CLI] Detected error report(s); copying from container:\n")

        # for src in candidates[:5]:
        #     dest = output_dir / Path(src).name
        #     copied = self.cm.copy_from_container(container, src, dest)
        #     with open(log_file, "a", encoding="utf-8") as f:
        #         if copied:
        #             f.write(f"- Copied: {src} -> {dest}\n")
        #         else:
        #             f.write(f"- Failed to copy: {src}\n")
