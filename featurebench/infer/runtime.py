"""
Runtime initialization and completion handlers.
Based on SWE-Bench's run_infer.py implementation.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from docker.models.containers import Container

from featurebench.environment_fixes import apply_environment_fixes
from featurebench.infer.container import ContainerManager
from featurebench.infer.models import TaskInstance


class RuntimeHandler:
    """Handles runtime initialization and completion for inference."""
    
    def __init__(
        self,
        container_manager: ContainerManager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the runtime handler.
        
        Args:
            container_manager: Container manager instance
            logger: Logger instance
        """
        self.cm = container_manager
        self.logger = logger or logging.getLogger(__name__)
    
    def initialize_runtime(
        self,
        container: Container,
        instance: TaskInstance,
        log_file: Path,
        white_box: bool = False,
    ) -> bool:
        """
        Initialize the runtime environment before agent execution.
        
        Based on swe-infer's initialize_runtime implementation:
        - Level 1: Restore and mask the project, apply environment fixes, init git
        - Level 2: Activate env, clean /testbed, init git
        
        Args:
            container: Docker container
            instance: Task instance
            log_file: Log file path
            
        Returns:
            True if initialization successful
        """
        self.logger.info(f"Initializing runtime for {instance.instance_id}")
        
        # Write header to log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("BEGIN Runtime Initialization\n")
            f.write("=" * 60 + "\n\n")
        
        try:
            # Install tmux and asciinema
            self.logger.info("Installing tmux and asciinema...")
            exit_code, _ = self.cm.exec_command(
                container,
                "apt-get update && apt-get install -y tmux asciinema",
                log_file=log_file
            )
            if exit_code != 0:
                self.logger.error("Failed to install tmux and asciinema")
                return False

            # Set instance ID and configure git
            exit_code, _ = self.cm.exec_command(
                container,
                f"echo 'export FB_INSTANCE_ID={instance.instance_id}' >> ~/.bashrc && "
                f"echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && "
                f"echo \"alias git='git --no-pager'\" >> ~/.bashrc && "
                f"git config --global core.pager \"\" && "
                f"git config --global diff.binary false",
                log_file=log_file
            )
            if exit_code != 0:
                self.logger.error("Failed to configure environment")
                return False
            
            # Export USER variable
            exit_code, _ = self.cm.exec_command(
                container,
                "export USER=$(whoami); echo USER=${USER}",
                log_file=log_file
            )
            
            level = instance.level
            self.logger.info(f"Instance level: {level}")
            
            if level == 1:
                return self._initialize_level1(container, instance, log_file, white_box=white_box)
            elif level == 2:
                return self._initialize_level2(container, instance, log_file)
            else:
                self.logger.error(f"Unknown level: {level}")
                return False
                
        except Exception as e:
            self.logger.error(f"Runtime initialization failed: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            return False
        finally:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write("END Runtime Initialization\n")
                f.write("=" * 60 + "\n\n")
    
    def _initialize_level1(
        self,
        container: Container,
        instance: TaskInstance,
        log_file: Path,
        white_box: bool = False,
    ) -> bool:
        """Initialize runtime for Level 1 instances."""
        self.logger.info("Processing Level 1 instance...")
        
        # Step 1: Activate conda and restore project
        self.logger.info("Step 1: Activating conda environment and restoring project")
        exit_code, _ = self.cm.exec_command(
            container,
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/ &&"
            "rm -rf /root/my_repo",
            log_file=log_file
        )
        if exit_code != 0:
            self.logger.error("Failed to activate conda and restore project")
            return False
        
        # Step 2: Apply patch (for masking files)
        self.logger.info("Step 2: Applying patch to mask files")
        patch_content = instance.patch
        
        if patch_content and patch_content.strip():
            # Create temporary patch file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch_content)
                temp_patch_path = Path(f.name)
            
            try:
                # Copy patch to container
                self.cm.copy_to_container(container, temp_patch_path, "/tmp/mask.patch")
                
                # Apply patch
                exit_code, output = self.cm.exec_command(
                    container,
                    "cd /testbed; "
                    "git apply --whitespace=fix /tmp/mask.patch; "
                    "status=$?; "
                    "rm -f -- /tmp/mask.patch; "
                    'exit "$status"',
                    log_file=log_file
                )
                if exit_code != 0:
                    self.logger.error(f"Failed to apply patch: {output}")
                    return False
                else:
                    self.logger.info("Successfully applied patch for masking")
            finally:
                temp_patch_path.unlink(missing_ok=True)
        else:
            self.logger.info("No patch to apply for masking")
        
        # Step 3: Delete F2P test files (skip in white-box mode)
        if white_box:
            self.logger.info("Step 3: White-box enabled; keeping FAIL_TO_PASS test files visible")
        else:
            self.logger.info("Step 3: Deleting F2P test files")
            fail_to_pass = instance.fail_to_pass

            if fail_to_pass:
                f2p_tests = fail_to_pass if isinstance(fail_to_pass, list) else [fail_to_pass]

                for f2p_test in f2p_tests:
                    # Ensure path starts with /testbed/
                    if not f2p_test.startswith('/testbed/'):
                        f2p_test_path = f'/testbed/{f2p_test}'
                    else:
                        f2p_test_path = f2p_test

                    self.logger.info(f"Deleting F2P test file: {f2p_test_path}")
                    exit_code, _ = self.cm.exec_command(
                        container,
                        f"rm -f {f2p_test_path}",
                        log_file=log_file
                    )
                    if exit_code != 0:
                        self.logger.warning(f"Failed to delete F2P test file {f2p_test_path}")
            else:
                self.logger.warning("No FAIL_TO_PASS tests found")
        
        # Step 4: Apply repository-specific environment fixes. This must happen
        # before Git is reinitialized so cleanup is part of the task baseline,
        # not part of the patch later attributed to the agent.
        self.logger.info("Step 4: Applying repository-specific environment fixes")
        repo_settings = instance.get_repo_settings()
        repository = str(
            repo_settings.get("repository") or instance.metadata.get("repo") or ""
        )
        if not apply_environment_fixes(
            repository,
            lambda command: self.cm.exec_command(
                container,
                command,
                log_file=log_file,
            ),
            self.logger,
        ):
            return False

        # Step 5: Remove bytecode copied from the pristine image. These caches
        # may still contain implementations that were removed by mask.patch,
        # so they must not be visible to the agent or included in its baseline.
        self.logger.info("Step 5: Removing stale Python bytecode")
        if not self._clear_python_bytecode(container, log_file):
            return False

        # Step 6: Re-initialize git repository
        self.logger.info("Step 6: Re-initializing git repository")
        return self._init_git_repo(container, log_file)
    
    def _initialize_level2(
        self,
        container: Container,
        instance: TaskInstance,
        log_file: Path
    ) -> bool:
        """Initialize runtime for Level 2 instances."""
        self.logger.info("Processing Level 2 instance...")
        
        # Step 1: Activate conda
        self.logger.info("Step 1: Activating conda environment")
        exit_code, _ = self.cm.exec_command(
            container,
            "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed",
            log_file=log_file
        )
        if exit_code != 0:
            self.logger.error("Failed to activate conda environment")
            return False
        
        # Step 2: Clean /testbed/
        self.logger.info("Step 2: Cleaning /testbed/ directory")
        exit_code, _ = self.cm.exec_command(
            container,
            "rm -rf /testbed/* /testbed/.* 2>/dev/null || true && mkdir -p /testbed &&"
            "rm -rf /root/my_repo",
            log_file=log_file
        )
        
        # Step 3: Create README.md
        exit_code, _ = self.cm.exec_command(
            container,
            'cd /testbed && echo "put all codes in this folder" > README.md',
            log_file=log_file
        )
        
        # Step 4: Initialize git repository
        self.logger.info("Step 3: Initializing git repository")
        return self._init_git_repo(container, log_file)
    
    def _init_git_repo(self, container: Container, log_file: Path) -> bool:
        """Initialize a fresh git repository in /testbed."""
        commands = [
            "cd /testbed && rm -rf .git",
            "cd /testbed && git init",
            'cd /testbed && git config user.email "fb@bench.com" && git config user.name "FeatureBench"',
            'cd /testbed && git add -A && git commit -m "Initial commit for FeatureBench evaluation" --allow-empty',
            "cd /testbed && git rev-parse HEAD"
        ]
        
        for cmd in commands:
            exit_code, output = self.cm.exec_command(container, cmd, log_file=log_file)
            if exit_code != 0:
                self.logger.error(f"Git init failed at: {cmd}")
                return False
        
        self.logger.info(f"Initial commit hash: {output.strip()}")
        return True

    def _clear_python_bytecode(
        self,
        container: Container,
        log_file: Path,
    ) -> bool:
        """Remove Python bytecode that may predate the task's mask patch."""
        command = r"""
        set -e
        cd /testbed || exit 1
        find . -type d -name __pycache__ -prune -exec rm -rf -- {} +
        find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
        """
        exit_code, output = self.cm.exec_command(
            container,
            command,
            log_file=log_file,
        )
        if exit_code != 0:
            self.logger.error(f"Failed to remove stale Python bytecode: {output}")
            return False

        self.logger.info("Successfully removed stale Python bytecode")
        return True

    def clear_package_caches(self, container: Container, log_file: Path) -> bool:
        """Remove package-manager caches before the agent can inspect the container.

        Cache paths backed by Docker mounts are preserved so a task can never
        erase a user-configured shared host-side download cache.
        """
        self.logger.info("Clearing package-manager caches before agent execution")
        command = r"""
        shopt -s nullglob

        cache_paths=(
            /download
            /opt/miniconda3/pkgs
            /opt/conda/pkgs
            /root/.cache/pip
            /root/.cache/uv
            /root/.conda/pkgs
        )
        cache_paths+=(/home/*/.cache/pip /home/*/.cache/uv /home/*/.conda/pkgs)

        is_on_non_root_mount() {
            local target="$1"
            awk -v target="$target" '
                {
                    mountpoint = $5
                    prefix = mountpoint "/"
                    if (mountpoint != "/" && target == mountpoint) {
                        found = 1
                    }
                    if (mountpoint != "/" && index(target, prefix) == 1) {
                        found = 1
                    }
                }
                END { exit(found ? 0 : 1) }
            ' /proc/self/mountinfo
        }

        for cache_path in "${cache_paths[@]}"; do
            [ -e "$cache_path" ] || continue
            cache_path="$(readlink -f -- "$cache_path")"

            if is_on_non_root_mount "$cache_path"; then
                echo "Skipping mounted package cache: $cache_path"
                continue
            fi

            size_kb="$(du -sk "$cache_path" 2>/dev/null | awk '{print $1}')"
            find "$cache_path" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            echo "Cleared package cache: $cache_path (${size_kb:-unknown} KB)"
        done

        temp_paths=(/tmp/pip-* /tmp/uv-*)
        for temp_path in "${temp_paths[@]}"; do
            [ -e "$temp_path" ] || continue
            temp_path="$(readlink -f -- "$temp_path")"

            if is_on_non_root_mount "$temp_path"; then
                echo "Skipping mounted package temporary path: $temp_path"
                continue
            fi

            rm -rf -- "$temp_path"
            echo "Cleared package temporary path: $temp_path"
        done
        """
        exit_code, output = self.cm.exec_command(
            container,
            command,
            log_file=log_file,
        )
        if exit_code != 0:
            self.logger.error(f"Failed to clear package caches: {output}")
            return False

        self.logger.info("Successfully cleared package-manager caches")
        return True
    
    def complete_runtime(
        self,
        container: Container,
        instance: TaskInstance,
        log_file: Path
    ) -> Optional[str]:
        """
        Complete the runtime and extract the git patch.
        
        Based on swe-infer's complete_runtime implementation.
        
        Args:
            container: Docker container
            instance: Task instance
            log_file: Log file path
            
        Returns:
            Git patch string or None if failed
        """
        self.logger.info(f"Completing runtime for {instance.instance_id}")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("BEGIN Runtime Completion\n")
            f.write("=" * 60 + "\n\n")
        
        try:
            # Change to /testbed
            exit_code, _ = self.cm.exec_command(
                container, "cd /testbed", log_file=log_file
            )
            
            # Handle running commands (send interrupt signals)
            if exit_code == -1:
                self.logger.info("Previous command still running, sending interrupt...")
                self.cm.exec_command(container, "kill -INT -1 2>/dev/null || true", log_file=log_file)
            
            # Configure git
            self.cm.exec_command(
                container,
                'git config --global core.pager ""',
                log_file=log_file
            )
            
            # Remove nested git repositories
            exit_code, output = self.cm.exec_command(
                container,
                'cd /testbed && find . -type d -name .git -not -path "./.git"',
                log_file=log_file
            )
            
            if exit_code == 0 and output.strip():
                git_dirs = [p.strip() for p in output.strip().split('\n') if p.strip()]
                for git_dir in git_dirs:
                    self.cm.exec_command(
                        container,
                        f'cd /testbed && rm -rf "{git_dir}"',
                        log_file=log_file
                    )
            
            # Add all files to git staging
            self.cm.exec_command(
                container, "cd /testbed && git add -A", log_file=log_file
            )
            
            # Remove binary files from git staging using Git's own detection.
            binary_remove_cmd = r"""
            cd /testbed || exit 1
            git diff --cached --numstat --no-renames --diff-filter=ACMRTD \
            | awk -F '\t' '$1=="-" || $2=="-" {print $3}' \
            | while IFS= read -r file; do
                git reset HEAD -- "$file" >/dev/null 2>&1 || true
            done
            """
            self.cm.exec_command(container, binary_remove_cmd, log_file=log_file)
            
            # Generate git diff
            git_patch = None
            for attempt in range(5):
                # Get base commit
                exit_code, output = self.cm.exec_command(
                    container,
                    "cd /testbed && git rev-list --max-parents=0 HEAD",
                    log_file=log_file
                )
                
                if exit_code == 0 and output.strip():
                    base_commit = output.strip().split('\n')[0]
                else:
                    base_commit = "HEAD"
                
                # Generate diff
                exit_code, output = self.cm.exec_command(
                    container,
                    f"cd /testbed && git diff --no-color --cached {base_commit}",
                    log_file=log_file
                )
                
                if exit_code == 0:
                    git_patch = output
                    break
                else:
                    self.logger.warning(f"Git diff attempt {attempt + 1} failed")
            
            if git_patch is None:
                self.logger.error("Failed to generate git patch after 5 attempts")
                return None
            
            # Remove binary diffs from patch
            git_patch = self._remove_binary_diffs(git_patch)
            
            self.logger.info(f"Generated patch ({len(git_patch)} chars)")
            return git_patch
            
        except Exception as e:
            self.logger.error(f"Runtime completion failed: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            return None
        finally:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write("END Runtime Completion\n")
                f.write("=" * 60 + "\n\n")
    
    def _remove_binary_diffs(self, patch: str) -> str:
        """
        Remove binary diffs from a patch string.
        
        Args:
            patch: Git patch string
            
        Returns:
            Cleaned patch string
        """
        if not patch:
            return ""

        had_trailing_newline = patch.endswith("\n")
        
        lines = patch.split('\n')
        result_lines = []
        skip_until_next_diff = False
        
        for line in lines:
            # Check for new diff header
            if line.startswith('diff --git'):
                skip_until_next_diff = False
            
            # Check for binary file marker
            if 'Binary files' in line or 'GIT binary patch' in line:
                # Remove the diff header we just added
                while result_lines and not result_lines[-1].startswith('diff --git'):
                    result_lines.pop()
                if result_lines and result_lines[-1].startswith('diff --git'):
                    result_lines.pop()
                skip_until_next_diff = True
                continue
            
            if not skip_until_next_diff:
                result_lines.append(line)
        
        cleaned = "\n".join(result_lines)
        if had_trailing_newline and cleaned and not cleaned.endswith("\n"):
            cleaned += "\n"
        return cleaned
