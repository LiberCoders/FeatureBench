"""
OpenHands agent implementation.
"""

import json
import shlex
import shutil
from pathlib import Path
from typing import Dict, Optional  # noqa: F401

from acebench.infer.agents.base import BaseAgent
from acebench.infer.container import DOCKER_HOST_GATEWAY


class OpenHandsAgent(BaseAgent):
    """OpenHands agent for ACE-Bench inference."""
    
    @property
    def name(self) -> str:
        return "openhands"
    
    @property
    def install_script(self) -> str:
        """Installation script for OpenHands."""
        # Priority: git_version > version (from kwargs) > OPENHANDS_VERSION (from env_vars)
        version = self._kwargs.get("version")
        git_version = self._kwargs.get("git_version")
        if not version and not git_version:
            version = self.env_vars.get("OPENHANDS_VERSION")
        
        # Determine installation source
        if git_version:
            install_cmd = f"uv pip install git+https://github.com/All-Hands-AI/OpenHands.git@{git_version}"
        elif version:
            install_cmd = f"uv pip install --prerelease=allow openhands-ai=={version}"
        else:
            install_cmd = "uv pip install --prerelease=allow openhands-ai"

        version_label = git_version or version or "latest"
        venv_name = f"openhands-venv-3.13-{version_label}"
        self.venv_name = venv_name
        
        return f"""#!/bin/bash
set -e

echo "Installing OpenHands agent..."

# Update package manager
apt-get update
apt-get install -y curl git build-essential python3-pip python3-venv

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/uv" "$CACHE_ROOT/pip" "$CACHE_ROOT/uv/python-mirror"

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

# If a local uv Python mirror exists and is non-empty, use it. Otherwise, let uv download Python from the default upstream sources.
UV_PYTHON_MIRROR_DIR="$CACHE_ROOT/uv/python-mirror"
if [ -z "${{UV_PYTHON_INSTALL_MIRROR:-}}" ]; then
    if [ -d "$UV_PYTHON_MIRROR_DIR" ] && [ "$(ls -A "$UV_PYTHON_MIRROR_DIR" 2>/dev/null)" ]; then
        export UV_PYTHON_INSTALL_MIRROR="file://$UV_PYTHON_MIRROR_DIR"
        echo "Using local uv python mirror: $UV_PYTHON_INSTALL_MIRROR"
    else
        unset UV_PYTHON_INSTALL_MIRROR
        echo "Local uv python mirror is empty; using upstream python downloads"
    fi
fi

UV_DIR="/opt/acebench/uv"
UV_BIN_PRIMARY="$UV_DIR/bin/uv"
UV_BIN_ALT="$UV_DIR/uv"
UV_BIN="$UV_BIN_PRIMARY"
PY_VERSION="3.13"
VENV_DIR="/opt/openhands-venv"

# Install uv if missing
mkdir -p "$UV_DIR"
if [ ! -x "$UV_BIN_PRIMARY" ] && [ ! -x "$UV_BIN_ALT" ]; then
    export PIPX_HOME="$UV_DIR/pipx"
    export PIPX_BIN_DIR="$UV_DIR/bin"
    PIPX_VENV="$UV_DIR/pipx-venv"
    mkdir -p "$PIPX_HOME" "$PIPX_BIN_DIR"
    # Create an isolated venv for pipx itself so it doesn't import the container's global packaging.
    python3 -m venv "$PIPX_VENV"
    "$PIPX_VENV/bin/python" -m pip install --upgrade pip pipx
    "$PIPX_VENV/bin/python" -m pipx ensurepath --force
    source ~/.bashrc 2>/dev/null || true
    # Install uv into a pipx-managed venv.
    "$PIPX_VENV/bin/python" -m pipx install uv
fi

# Select uv binary
if [ -x "$UV_BIN_PRIMARY" ]; then
    UV_BIN="$UV_BIN_PRIMARY"
elif [ -x "$UV_BIN_ALT" ]; then
    UV_BIN="$UV_BIN_ALT"
else
    echo "uv not found after install" >&2
    exit 1
fi

export PATH="$UV_DIR/bin:$UV_DIR:$PATH"
source "$UV_DIR/env" 2>/dev/null || true

# Configure uv index mirror (TUNA)
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml <<'EOF'
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
EOF

# Install Python via uv (downloads cached via UV_CACHE_DIR)
$UV_BIN python install $PY_VERSION

# Create venv (container-local)
$UV_BIN venv "$VENV_DIR" --python $PY_VERSION

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Skip VSCode extension build
export SKIP_VSCODE_BUILD=true

ADD_DEPS='deprecated typing_extensions numpy>=2.0.0'
{install_cmd}
uv pip install $ADD_DEPS

# venv is already at fixed path used by runner
mkdir -p /opt

echo "OpenHands installation complete"

echo 'export LOG_ALL_EVENTS=true' >> ~/.bashrc
echo 'export LLM_LOG_COMPLETIONS=true' >> ~/.bashrc
echo 'export LLM_LOG_COMPLETIONS_FOLDER=/agent-logs/completions' >> ~/.bashrc
"""
    
    def get_run_command(self, instruction: str) -> str:
        """Get the command to run OpenHands."""
#         instruction = """
# please create a hello-world.txt containing your self-introduction under testbed.   
# """
        escaped_instruction = shlex.quote(instruction)
        
        return (
            f"SANDBOX_VOLUMES=${{PWD}}:/workspace:rw "
            f"/opt/openhands-venv/bin/python -m openhands.core.main "
            f"--task {escaped_instruction}"
        )
    
    def get_env_setup_script(self) -> str:
        """Get environment setup script for OpenHands."""
        lines = ["#!/bin/bash", ""]
        
        # Determine LLM API key
        llm_api_key = self.env_vars.get("LLM_API_KEY")
        if not llm_api_key:
            # Try Anthropic key
            if self.env_vars.get("ANTHROPIC_API_KEY"):
                llm_api_key = self.env_vars["ANTHROPIC_API_KEY"]
            # Try OpenAI key
            elif self.env_vars.get("OPENAI_API_KEY"):
                llm_api_key = self.env_vars["OPENAI_API_KEY"]
        
        # Determine LLM model
        llm_model = self.env_vars.get("LLM_MODEL")
        if not llm_model:
            llm_model = self._kwargs.get("model")
            if not llm_model:
                llm_model = self.env_vars.get("ANTHROPIC_MODEL")
        
        # Required environment variables
        env_settings = {
            "LLM_API_KEY": llm_api_key,
            "LLM_MODEL": llm_model,
            "LLM_BASE_URL": self.env_vars.get("LLM_BASE_URL"),
            "LLM_API_VERSION": self.env_vars.get("LLM_API_VERSION"),
            "LLM_REASONING_EFFORT": self.env_vars.get("LLM_REASONING_EFFORT"),
            # Force native tool calling (OpenHands LLMConfig.native_tool_calling via LLM_ env mapping)
            "LLM_NATIVE_TOOL_CALLING": self.env_vars.get("LLM_NATIVE_TOOL_CALLING"),
            # Disable features not needed for ACE-Bench
            "AGENT_ENABLE_PROMPT_EXTENSIONS": "false",
            "AGENT_ENABLE_BROWSING": "false",
            "ENABLE_BROWSER": "false",
            # Sandbox settings
            "SANDBOX_ENABLE_AUTO_LINT": "true",
            "SKIP_DEPENDENCY_CHECK": "1",
            # Save trajectory
            "SAVE_TRAJECTORY_PATH": "/agent-logs/trajectory.json",
            # Run without creating new user
            "RUN_AS_OPENHANDS": "false",
            # Use local runtime (we're inside the container)
            "RUNTIME": "local",
        }
        
        # Add any OPENHANDS_ prefixed variables
        for key, value in self.env_vars.items():
            if key.startswith("OPENHANDS_"):
                new_key = key.replace("OPENHANDS_", "")
                env_settings[new_key] = value
        
        for key, value in env_settings.items():
            if value:
                # Replace localhost/127.0.0.1 with Docker host gateway for bridge mode
                value_str = str(value)
                if 'localhost' in value_str or '127.0.0.1' in value_str:
                    value_str = value_str.replace('localhost', DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace('127.0.0.1', DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())

        # Proxy bypass for selected gateways (avoid flaky proxy TLS issues).
        # We append to existing NO_PROXY/no_proxy instead of overwriting.
        lines.extend(
            [
                "",
                "# Proxy bypass for selected gateways",
                '# Always bypass proxies for local loopback (OpenHands LocalRuntime uses localhost:<port>)',
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

        return "\n".join(lines)
    
    def pre_run_hook(self, container, log_file) -> bool:
        """
        Create agent logs directory before running.
        And ignore python path /testbed in /opt/openhands-venv/lib/python3.13/site-packages/openhands/core/main.py
        to avoid using codes under /testbed for openhands
        """
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        def _verify_patch(description: str, check_cmd: str) -> None:
            exit_code, output = self.cm.exec_command(
                container,
                check_cmd,
                log_file=log_file,
            )
            if exit_code == 0:
                self.logger.info(f"{description} verification: OK")
            else:
                self.logger.warning(
                    f"{description} verification: FAILED (exit_code={exit_code})\n{output}"
                )

        llm_model = (
            self.env_vars.get("LLM_MODEL")
            or self._kwargs.get("model")
            or ""
        )
        llm_model_lower = str(llm_model).strip().lower()

        is_openhands_0_62_0 = self.venv_name.split('-')[-1] == '0.62.0'
        is_claude_model = 'claude' in llm_model_lower
        is_gemini_model = 'gemini' in llm_model_lower

        # IMPORTANT: Avoid leaking OpenHands' own site-packages into the runtime
        # BUG: For keep same with before experiment
#         exit_code, output = self.cm.exec_command(
#             container,
#             r"""python3 - << 'EOF'
# file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/runtime/impl/local/local_runtime.py'

# with open(file_path, 'r', encoding='utf-8') as f:
#     content = f.read()

# lines = content.splitlines(True)

# target_idx = None
# for i, line in enumerate(lines):
#     s = line.strip()
#     if not s:
#         continue
#     if (
#         ("env['PYTHONPATH']" in line or 'env["PYTHONPATH"]' in line)
#         and "os.pathsep.join" in line
#         and "code_repo_path" in line
#     ):
#         target_idx = i
#         break

# if target_idx is None:
#     print("local_runtime.py: PYTHONPATH injection line not found; aborting")
#     raise SystemExit(2)

# deleted = lines[target_idx].rstrip("\n")
# del lines[target_idx]

# with open(file_path, 'w', encoding='utf-8') as f:
#     f.write("".join(lines))
# EOF""",
#             log_file=log_file,
#         )
#         if exit_code != 0:
#             raise RuntimeError(
#                 f"Failed to patch OpenHands local_runtime.py to fix PYTHONPATH leakage (exit_code={exit_code}).\n{output}"
#             )

        # NOTE: Do not use a multi-line `sed '1i ...'` here.
        # GNU sed treats the second line as a new command; since it starts with
        # `sys.path...` it is parsed as an `s` command, causing:
        # `sed: unterminated 's' command`.
        self.cm.exec_command(
            container,
            r"""python3 - << 'EOF'
file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/core/main.py'

prepend_lines = [
    'import sys\n',
    'sys.path = [p for p in sys.path if not p.startswith("/testbed")]\n',
]

with open(file_path, 'r', encoding='utf-8') as f:
    original = f.read()

prepend_text = ''.join(prepend_lines)
if not original.startswith(prepend_text):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(prepend_text + original)
EOF""",
            log_file=log_file,
        )

        # Patch llm.py to support Opus 4.5 and Sonnet 4 models (remove top_p when temperature is set)
        patch_support_opus_4_5_and_sonnet_4_script = r'''
cp /opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/llm.py \
   /opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/llm.py.bak

python3 - << 'EOF'
file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/llm.py'
search_text = """        # Limit to Opus 4.1 specifically to avoid changing behavior of other Anthropic models
        if ('claude-opus-4-1' in _model_lower) and (
            'temperature' in kwargs and 'top_p' in kwargs
        ):
            kwargs.pop('top_p', None)"""
replace_text = """        # Apply to Opus 4.1, Opus 4.5, and Sonnet 4 models to avoid API errors
        if (
            ('claude-opus-4-1' in _model_lower)
            or ('claude-opus-4-5' in _model_lower)
            or ('claude-sonnet-4' in _model_lower)
        ) and ('temperature' in kwargs and 'top_p' in kwargs):
            kwargs.pop('top_p', None)"""
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
new_content = content.replace(search_text, replace_text)
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
EOF
'''
        if is_openhands_0_62_0 and is_claude_model:
            self.logger.info("OpenHands==0.62.0 detected, applying Opus 4.5 and Sonnet 4 support patch...")
            self.cm.exec_command(
                container,
                patch_support_opus_4_5_and_sonnet_4_script,
                log_file=log_file
            )
            _verify_patch(
                "Opus 4.5 / Sonnet 4 top_p patch",
                r"""python3 - << 'EOF'
file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/llm.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read().lower()

required = [
    'claude-opus-4-5',
    'claude-sonnet-4',
    "kwargs.pop('top_p'",
]

missing = [s for s in required if s not in content]
if missing:
    raise SystemExit(f'missing markers: {missing}')
print('ok')
EOF""",
            )

        # Patch model_features.py to add gemini-3* to FUNCTION_CALLING_PATTERNS
        patch_gemini_3_support_script = r'''
python3 - << 'EOF'
file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/model_features.py'
search_text = """# Pattern tables capturing current behavior. Keep patterns lowercase.
FUNCTION_CALLING_PATTERNS: list[str] = [
    # Anthropic families
    'claude-3-7-sonnet*',
    'claude-3.7-sonnet*',
    'claude-sonnet-3-7-latest',
    'claude-3-5-sonnet*',
    'claude-3.5-sonnet*',  # Accept dot-notation for Sonnet 3.5 as well
    'claude-3.5-haiku*',
    'claude-3-5-haiku*',
    'claude-sonnet-4*',
    'claude-opus-4*',
    # OpenAI families
    'gpt-4o*',
    'gpt-4.1',
    'gpt-5*',
    # o-series (keep exact o1 support per existing list)
    'o1-2024-12-17',
    'o3*',
    'o4-mini*',
    # Google Gemini
    'gemini-2.5-pro*',
    # Others
    'kimi-k2-0711-preview',
    'kimi-k2-instruct',
    'qwen3-coder*',
    'qwen3-coder-480b-a35b-instruct',
    'deepseek-chat',
]"""
replace_text = """# Pattern tables capturing current behavior. Keep patterns lowercase.
FUNCTION_CALLING_PATTERNS: list[str] = [
    # Anthropic families
    'claude-3-7-sonnet*',
    'claude-3.7-sonnet*',
    'claude-sonnet-3-7-latest',
    'claude-3-5-sonnet*',
    'claude-3.5-sonnet*',  # Accept dot-notation for Sonnet 3.5 as well
    'claude-3.5-haiku*',
    'claude-3-5-haiku*',
    'claude-sonnet-4*',
    'claude-opus-4*',
    # OpenAI families
    'gpt-4o*',
    'gpt-4.1',
    'gpt-5*',
    # o-series (keep exact o1 support per existing list)
    'o1-2024-12-17',
    'o3*',
    'o4-mini*',
    # Google Gemini
    'gemini-2.5-pro*',
    'gemini-3*',
    # Others
    'kimi-k2-0711-preview',
    'kimi-k2-instruct',
    'qwen3-coder*',
    'qwen3-coder-480b-a35b-instruct',
    'deepseek-chat',
]"""
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
if search_text in content:
    new_content = content.replace(search_text, replace_text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Patched model_features.py to add gemini-3* support")
else:
    print("Search text not found in model_features.py, skipping patch")
EOF
'''
        if is_openhands_0_62_0 and is_gemini_model:
            self.logger.info("OpenHands==0.62.0 detected, applying Gemini 3 model support patch...")
            self.cm.exec_command(
                container,
                patch_gemini_3_support_script,
                log_file=log_file
            )
            _verify_patch(
                "Gemini 3 FUNCTION_CALLING_PATTERNS patch",
                r"""python3 - << 'EOF'
file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/llm/model_features.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read().lower()
if 'gemini-3*' not in content:
    raise SystemExit('gemini-3* not found in model_features.py')
print('ok')
EOF""",
            )

        # Patch for Gemini 3 Pro thought_signature support.
        # Root cause: OpenHands serializes tool_calls without preserving
        # tool_call.provider_specific_fields (Gemini requires
        # provider_specific_fields.google.thought_signature to be echoed back).
        patch_gemini_3_thought_signature_script = r'''
source /opt/openhands-venv/bin/activate

echo "Patching OpenHands tool_calls serialization to preserve provider_specific_fields (Gemini thought_signature)..."

python3 - << 'EOF'
import re

file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/core/message.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = '# an assistant message calling a tool'
end_marker = '# an observation message with tool response'

start_idx = content.find(start_marker)
if start_idx == -1:
    raise RuntimeError('Could not find start marker in message.py: ' + start_marker)

end_idx = content.find(end_marker, start_idx)
if end_idx == -1:
    raise RuntimeError('Could not find end marker in message.py: ' + end_marker)

start_line_start = content.rfind('\n', 0, start_idx) + 1
end_line_start = content.rfind('\n', 0, end_idx) + 1

indent_match = re.match(r"(\s*)#", content[start_line_start:])
indent = indent_match.group(1) if indent_match else ''

# If already patched, skip.
block_existing = content[start_line_start:end_line_start]
if "provider_specific_fields" in block_existing and "tool_calls_list" in block_existing:
    print('message.py already patched for provider_specific_fields; skipping')
else:
    replacement_block = (
        f"{indent}{start_marker}\n"
        f"{indent}if self.tool_calls is not None:\n"
        f"{indent}    tool_calls_list = []\n"
        f"{indent}    for tool_call in self.tool_calls:\n"
        f"{indent}        tool_call_dict = {{\n"
        f"{indent}            'id': tool_call.id,\n"
        f"{indent}            'type': 'function',\n"
        f"{indent}            'function': {{\n"
        f"{indent}                'name': tool_call.function.name,\n"
        f"{indent}                'arguments': tool_call.function.arguments,\n"
        f"{indent}            }},\n"
        f"{indent}        }}\n"
        f"{indent}        provider_specific_fields = None\n"
        f"{indent}        if hasattr(tool_call, 'provider_specific_fields'):\n"
        f"{indent}            provider_specific_fields = getattr(tool_call, 'provider_specific_fields', None)\n"
        f"{indent}        elif isinstance(tool_call, dict):\n"
        f"{indent}            provider_specific_fields = tool_call.get('provider_specific_fields')\n"
        f"{indent}        if provider_specific_fields:\n"
        f"{indent}            tool_call_dict['provider_specific_fields'] = provider_specific_fields\n"
        f"{indent}        tool_calls_list.append(tool_call_dict)\n"
        f"{indent}    message_dict['tool_calls'] = tool_calls_list\n\n"
    )

    new_content = content[:start_line_start] + replacement_block + content[end_line_start:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Patched message.py: tool_calls now preserve provider_specific_fields')
EOF

echo "Gemini thought_signature patch completed!"
'''
        if is_openhands_0_62_0 and is_gemini_model:
            self.logger.info(
                "OpenHands==0.62.0 detected, patching tool_calls serialization for Gemini thought_signature..."
            )
            self.cm.exec_command(
                container,
                patch_gemini_3_thought_signature_script,
                log_file=log_file,
            )
            _verify_patch(
                "Gemini thought_signature provider_specific_fields preservation patch",
                r"""python3 - << 'EOF'
import re

file_path = '/opt/openhands-venv/lib/python3.13/site-packages/openhands/core/message.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = '# an assistant message calling a tool'
end_marker = '# an observation message with tool response'
start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx + 1)
if start_idx == -1 or end_idx == -1:
    raise SystemExit('markers not found')

block = content[start_idx:end_idx]
required = [
    "tool_call_dict['provider_specific_fields']",
    'provider_specific_fields',
    'tool_calls_list',
]
missing = [s for s in required if s not in block]
if missing:
    raise SystemExit(f'missing markers in serialization block: {missing}')

# Also make sure we did not delete the tool response marker content.
if end_marker not in content:
    raise SystemExit('end marker missing after patch')

print('ok')
EOF""",
            )

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """
        Save llm completions if SAVE_COMPLETIONS is enabled
        Move /agent-logs/trajectory.json to the host (the same directory as log_file)
        Then check if Agent finished successfully.
        The logic to check if Agent finished successfully is to observe the last step of the trajectory whether the action is finish.
        """
        # Determine the destination path for trajectory.json and completions directory
        log_dir = Path(log_file).parent
        
        # Save completions if enabled
        if str(self.env_vars.get("SAVE_COMPLETIONS", "false")).strip().lower() == "true":
            completions_container_path = "/agent-logs/completions"
            completions_host_path = log_dir / "completions"
            # Remove old completions to avoid mixing runs/attempts
            if completions_host_path.exists():
                try:
                    shutil.rmtree(completions_host_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove old completions at {completions_host_path}: {e}")
            # Copy completions directory from container to host
            copied = self.cm.copy_from_container(
                container,
                completions_container_path,
                completions_host_path
            )
            if not copied:
                self.logger.warning("Failed to copy completions directory from container")
            else:
                self.logger.info(f"Copied completions directory to {completions_host_path}")
        
        # Copy trajectory.json from container to host
        trajectory_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/trajectory.json",
            log_dir / "trajectory.json"
        )
        
        if not trajectory_copied:
            self.logger.error("Failed to copy trajectory.json from container")
            return False
        
        trajectory_path = log_dir / "trajectory.json"
        self.logger.info(f"Copied trajectory.json to {trajectory_path}")
        
        # Read and parse trajectory.json to check if agent finished successfully
        try:
            with open(trajectory_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            
            if not trajectory:
                self.logger.error("Trajectory is empty")
                return False
            
            # Check if the last event has action: finish
            last_event = trajectory[-1]
            last_action = last_event.get("action")
            
            if last_action == "finish":
                self.logger.info("Agent finished successfully (action: finish)")
                return True
            else:
                self.logger.warning(
                    f"Agent did not finish properly. Last action: {last_action}"
                )
                
                # Check if the agent reached maximum iteration (treat as success)
                infer_log_path = log_dir / "infer.log"
                if infer_log_path.exists():
                    try:
                        with open(infer_log_path, "r", encoding="utf-8") as f:
                            infer_log_content = f.read()
                        
                        if "RuntimeError: Agent reached maximum iteration." in infer_log_content:
                            self.logger.info(
                                "Agent reached maximum iteration - treating as successful completion"
                            )
                            return True
                        elif "AgentStuckInLoopError: Agent got stuck in a loop" in infer_log_content:
                            self.logger.info(
                                "Agent got stuck in a loop - treating as successful completion"
                            )
                            return True
                    except Exception as e:
                        self.logger.warning(f"Failed to read infer.log: {e}")
                
                self.logger.error("Agent did not complete successfully")
                return False
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse trajectory.json: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error reading trajectory.json: {e}")
            return False
