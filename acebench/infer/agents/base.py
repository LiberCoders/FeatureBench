"""
Base agent class for ACE-Bench inference.
"""

import logging
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from docker.models.containers import Container

from acebench.infer.container import ContainerManager

if TYPE_CHECKING:
    from acebench.infer.models import TaskInstance


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(
        self,
        container_manager: ContainerManager,
        env_vars: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            container_manager: Container manager instance
            env_vars: Environment variables for the agent
            logger: Logger instance
            **kwargs: Additional arguments
        """
        self.cm = container_manager
        self.env_vars = env_vars or {}
        self.logger = logger or logging.getLogger(__name__)
        self._kwargs = kwargs
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        pass
    
    @property
    @abstractmethod
    def install_script(self) -> str:
        """
        Installation script content.
        This script will be executed to install the agent in the container.
        """
        pass
    
    @abstractmethod
    def get_run_command(self, instruction: str) -> str:
        """
        Get the command to run the agent with the given instruction.
        
        Args:
            instruction: Task instruction/prompt
            
        Returns:
            Shell command to run the agent
        """
        pass
    
    def get_env_setup_script(self) -> str:
        """
        Get the environment setup script content.
        
        Returns:
            Script content to set up environment variables
        """
        lines = ["#!/bin/bash", ""]
        for key, value in self.env_vars.items():
            if value:
                # Escape single quotes in value
                escaped_value = value.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")
        return "\n".join(lines)

    def _get_proxy_unset_lines(self) -> List[str]:
        """Return shell lines to unset proxy at runtime when runtime_proxy is off."""
        runtime_proxy = str(self.env_vars.get("ACE_RUNTIME_PROXY", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if runtime_proxy:
            return []
        return [
            "",
            "# Disable proxy for runtime API calls",
            "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy",
        ]
    
    def install(
        self,
        container: Container,
        log_file: Path
    ) -> bool:
        """
        Install the agent in the container.
        
        Args:
            container: Docker container
            log_file: Log file path
            
        Returns:
            True if installation successful
        """
        self.logger.info(f"Installing {self.name} agent...")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"BEGIN Agent Installation: {self.name}\n")
            f.write("=" * 60 + "\n\n")
        
        try:
            # Create agent directory
            self.cm.exec_command(
                container,
                "mkdir -p /installed-agent",
                log_file=log_file
            )
            
            # Write environment setup script
            env_script = self.get_env_setup_script()
            exit_code, _ = self.cm.exec_command(
                container,
                f"cat > /installed-agent/setup-env.sh << 'ENVEOF'\n{env_script}\nENVEOF",
                log_file=log_file
            )
            
            # Write installation script
            install_script = self.install_script
            exit_code, _ = self.cm.exec_command(
                container,
                f"cat > /installed-agent/install-agent.sh << 'INSTALLEOF'\n{install_script}\nINSTALLEOF",
                log_file=log_file
            )
            
            # Make scripts executable
            self.cm.exec_command(
                container,
                "chmod +x /installed-agent/*.sh",
                log_file=log_file
            )
            
            # Source environment and run installation
            # wokdir is set to /installed-agent to avoid conflicts of uv.lock with the main working directory /testbed
            exit_code = self.cm.exec_command_stream(
                container,
                "source /installed-agent/install-agent.sh",
                log_file=log_file,
                workdir="/installed-agent",
                timeout=1800  # 30 minutes timeout for installation
            )
            
            if exit_code != 0:
                self.logger.error(f"Agent installation failed with exit code {exit_code}")
                return False
            
            self.logger.info(f"{self.name} agent installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent installation failed: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            return False
        finally:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"END Agent Installation: {self.name}\n")
                f.write("=" * 60 + "\n\n")
    
    def pre_run_hook(
        self,
        container: Container,
        log_file: Path
    ) -> bool:
        """
        Pre-run hook for agent-specific setup.
        Override this method to add custom pre-run logic.
        
        Args:
            container: Docker container
            log_file: Log file path
            
        Returns:
            True if successful
        """
        return True
    
    def post_run_hook(
        self,
        container: Container,
        log_file: Path
    ) -> bool:
        """
        Post-run hook for agent-specific logging.
        Override this method to add custom post-run logging.
        
        Args:
            container: Docker container
            log_file: Log file path
            
        Returns:
            True if successful
        """
        return True

    def failure_hook(self, container: Container, log_file: Path) -> None:
        """Best-effort hook invoked when agent execution fails.

        Subclasses may override this to persist diagnostics from inside the
        container (e.g., client error reports) before the container is removed.
        """
        return None

    def pre_run_setup(
        self,
        container: Container,
        instance: "TaskInstance",
        log_file: Path
    ) -> bool:
        """
        Pre-run setup interface for custom container processing.
        
        This method is called AFTER agent installation and BEFORE agent execution.
        Override this method to perform custom setup such as:
        - Installing additional dependencies
        - Configuring environment variables
        - Preparing data files
        - Any other custom container processing
        
        Args:
            container: Docker container
            instance: Task instance with metadata
            log_file: Log file path for recording operations
            
        Returns:
            True if setup successful, False otherwise (non-fatal)
            
        Example:
            def pre_run_setup(self, container, instance, log_file):
                # Install additional package
                self.cm.exec_command(
                    container,
                    "pip install some-package",
                    log_file=log_file
                )
                
                # Configure something based on task
                if instance.level == 2:
                    self.cm.exec_command(
                        container,
                        "echo 'special config' > /testbed/config.txt",
                        log_file=log_file
                    )
                
                return True
        """
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"BEGIN Pre-Run Setup: {self.name}\n")
            f.write(f"Instance: {instance.instance_id}\n")
            f.write(f"Level: {instance.level}\n")
            f.write("=" * 60 + "\n\n")
        
        try:
            # Default implementation does nothing
            # Override this method in subclasses for custom processing
            self.logger.debug(f"Pre-run setup for {instance.instance_id} (default: no-op)")
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-run setup failed: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            return False
        finally:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"END Pre-Run Setup: {self.name}\n")
                f.write("=" * 60 + "\n\n")
    
    def run(
        self,
        container: Container,
        instruction: str,
        log_file: Path,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Run the agent with the given instruction.
        
        Args:
            container: Docker container
            instruction: Task instruction
            log_file: Log file path
            timeout: Timeout in seconds
            
        Returns:
            True if agent completed successfully
        """
        self.logger.info(f"Running {self.name} agent...")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"BEGIN Agent Execution: {self.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Instruction:\n{instruction}\n\n")
            f.write("-" * 60 + "\n\n")
        
        try:
            # Run pre-run hook
            success_pre_run = self.pre_run_hook(container, log_file)
            if not success_pre_run:
                raise RuntimeError("Pre-run hook failed")
            
            # Get run command
            run_command = self.get_run_command(instruction)
            
            # Source environment and run agent
            full_command = f"source /installed-agent/setup-env.sh && cd /testbed && {run_command}"
            
            exit_code = self.cm.exec_command_stream(
                container,
                full_command,
                log_file=log_file,
                timeout=timeout
            )
            
            success_run = exit_code == 0
            if not success_run:
                raise RuntimeError(f"Agent execution failed with code {exit_code}")
            
            # Run post-run hook
            success_post_run = self.post_run_hook(container, log_file)
            if not success_post_run:
                raise RuntimeError("Post-run hook failed. Agent execution may not be successful.")
            
            self.logger.info(f"{self.name} agent completed successfully")
            
            # if not raise any exception, return True
            return True
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")

            # Best-effort: collect additional diagnostics before container teardown.
            try:
                self.failure_hook(container, log_file)
            except Exception as hook_error:
                self.logger.warning(f"Failure hook error for {self.name}: {hook_error}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[WARNING] Failure hook error: {hook_error}\n")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            # if raise any exception, return False
            return False
        finally:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"END Agent Execution: {self.name}\n")
                f.write("=" * 60 + "\n\n")

