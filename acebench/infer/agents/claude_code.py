"""
Claude Code agent implementation.
"""

import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from acebench.infer.agents.base import BaseAgent
from acebench.infer.container import DOCKER_HOST_GATEWAY



class ClaudeCodeAgent(BaseAgent):
    """Claude Code agent for ACE-Bench inference."""
    
    # Allowed tools for Claude Code
    ALLOWED_TOOLS = [
        "Bash",
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
        "LS",
        "WebFetch",
        "NotebookEdit", 
        "NotebookRead",
        "TodoRead",
        "TodoWrite",
        "Agent",
    ]
    
    @property
    def name(self) -> str:
        return "claude_code"
    
    @property
    def install_script(self) -> str:
        """Installation script for Claude Code."""
        version = self._kwargs.get("version") or self.env_vars.get("CLAUDE_CODE_VERSION") or "latest"
        
        return f"""#!/bin/bash
set -e

echo "Installing Claude Code agent..."

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
# ACE-Bench often runs many infer containers concurrently; sharing the tarball
# cache can lead to corrupted archives and checksum/tar extraction failures.
mkdir -p "$NVM_DIR/.cache"

NODE_VERSION="22"

# Install NVM (idempotent)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env NVM_DIR="$NVM_DIR" bash

export NVM_DIR
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if [ -z "$(command -v nvm)" ]; then
    echo "nvm not available after install" >&2
    exit 1
fi

# Install or reuse Node
nvm install "$NODE_VERSION"
nvm use "$NODE_VERSION"

# Verify npm
npm -v

# Install Claude Code (downloads cached via NPM_CONFIG_CACHE)
npm install -g @anthropic-ai/claude-code@{version}

# Verify installation
claude --version || echo "Claude Code installed"

echo "Claude Code installation complete"
"""
    
    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Claude Code."""
        full_instruction = instruction.rstrip()

        escaped_instruction = shlex.quote(full_instruction)
        allowed_tools = " ".join(self.ALLOWED_TOOLS)
        
        return (
            f"NVM_DIR=${{NVM_DIR:-/opt/acebench/nvm}}; "
            f"[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            f"claude --verbose "
            f"-p {escaped_instruction} --allowedTools {allowed_tools} "
            # f"-p \"please create a hello-world.txt containing your self-introduction under testbed\" --allowedTools {allowed_tools} "
            f"--output-format stream-json | tee /agent-logs/claude_code_stream_output.jsonl"
        )
    
    def pre_run_hook(self, container, log_file) -> bool:
        """
        Create agent logs directory before running.
        """
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """
        Save claude_code_stream_output.jsonl
        Check if the last line of claude_code_stream_output.jsonl is a success
        If not, return False
        If success, return True
        """
        log_dir = Path(log_file).parent

        claude_code_stream_output_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/claude_code_stream_output.jsonl",
            log_dir / "claude_code_stream_output.jsonl"
        )

        if not claude_code_stream_output_copied:
            self.logger.error("Failed to copy claude_code_stream_output.jsonl from container")
            return False

        # check the last line of claude_code_stream_output.jsonl as a json object
        # if key "type" is "result" and key "subtype" is "success", then return True
        # otherwise return False
        with open(log_dir / "claude_code_stream_output.jsonl", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
        last_line_json = json.loads(last_line)

        is_result_success = (
            last_line_json.get("type") == "result" and last_line_json.get("subtype") == "success"
        )
        if not is_result_success:
            self.logger.error(
                "Last line of claude_code_stream_output.jsonl is not result/success; agent may not have finished properly"
            )
            return False

        # Strengthened success criteria: even if the CLI reports a result/success,
        # treat runs with is_error=true as failures. This avoids counting API
        # errors (e.g., token exhaustion) as successful attempts.
        if "is_error" not in last_line_json:
            self.logger.error(
                "claude_code_stream_output.jsonl last result is missing is_error; treating as failure"
            )
            return False

        is_error = last_line_json.get("is_error")
        if is_error is False:
            return True

        if is_error is True:
            result = last_line_json.get("result")
            if isinstance(result, str):
                result_preview = result.replace("\n", " ")[:300]
            else:
                result_preview = str(result)[:300]
            self.logger.error(
                "Claude Code reported is_error=true; treating as failure. "
                f"Result preview: {result_preview}"
            )
            return False

        self.logger.error(f"Claude Code reported unexpected is_error={is_error!r}; treating as failure")
        return False

    def failure_hook(self, container, log_file) -> None:
        log_dir = Path(log_file).parent

        # Copy stream output if present (can be partial on failures/timeouts).
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/claude_code_stream_output.jsonl",
                log_dir / "claude_code_stream_output.jsonl",
            )
        except Exception:
            pass


    def get_env_setup_script(self) -> str:
        """Get environment setup script for Claude Code."""
        lines = ["#!/bin/bash", ""]
        
        # Required environment variables
        required_vars = {
            "ANTHROPIC_API_KEY": self.env_vars.get("ANTHROPIC_API_KEY", ""),
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
            "DISABLE_TELEMETRY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        
        # Add model if specified (CLI only)
        model = self._kwargs.get("model")
        if model:
            # Remove provider prefix if present
            if "/" in model:
                model = model.split("/")[-1]
            required_vars["ANTHROPIC_MODEL"] = model

            # Force Claude Code internal routing defaults to the selected model.
            # This prevents automatic fallback to smaller families (e.g. haiku/sonnet)
            # when you want everything to run on the specified model.
            required_vars["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
            required_vars["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
            required_vars["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        
        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in required_vars and value:
                required_vars[key] = value
        
        for key, value in required_vars.items():
            if value:
                # Replace localhost/127.0.0.1 with Docker host gateway for bridge mode
                value_str = str(value)
                if 'localhost' in value_str or '127.0.0.1' in value_str:
                    value_str = value_str.replace('localhost', DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace('127.0.0.1', DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())
        
        # Add NVM setup
        lines.extend([
            "",
            "# Load NVM",
            'export NVM_DIR="/opt/acebench/nvm"',
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"'
        ])
        
        return "\n".join(lines)
