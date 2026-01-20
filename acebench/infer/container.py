"""
Container management for inference.
Handles Docker container lifecycle and command execution.
"""

import io
import logging
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
from docker.models.containers import Container

import selectors
# Regex pattern to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')

# Docker host gateway IP (docker0 bridge network)
# In bridge mode, containers can access host services via this IP
DOCKER_HOST_GATEWAY = "172.17.0.1"


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ANSI_ESCAPE_PATTERN.sub('', text)


class ContainerManager:
    """Manages Docker containers for inference."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        env_vars: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the container manager.
        
        Args:
            logger: Logger instance
            env_vars: Environment variables to inject into containers
        """
        self.logger = logger or logging.getLogger(__name__)
        self.env_vars = env_vars or {}
        
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError(
                f"Failed to connect to Docker: {e}. "
                "Please ensure Docker is installed and running."
            )
    
    def pull_image(self, image_name: str) -> bool:
        """
        Pull a Docker image if it doesn't exist locally.
        
        Args:
            image_name: Full image name with tag
            
        Returns:
            True if successful
        """
        try:
            # Check if image exists locally
            try:
                self.client.images.get(image_name)
                self.logger.info(f"Image {image_name} already exists locally")
                return True
            except docker.errors.ImageNotFound:
                pass

            # try short-name fallback
            # Extract short name (strip registry/repo prefix)
            # e.g., "xxx.com/myrepo/foo:latest" -> "foo:latest"
            short_name = image_name.split("/")[-1]
            try:
                short_img = self.client.images.get(short_name)
                self.logger.info(
                    f"Image {image_name} not found, but found local image {short_name}. "
                    f"Tagging it as {image_name}"
                )

                # Tag the image: short_name -> image_name
                short_img.tag(image_name)
                return True

            except docker.errors.ImageNotFound:
                # short name also not found: continue pulling
                pass
            
            # Pull the image
            self.logger.info(f"Pulling image {image_name}...")
            self.client.images.pull(image_name)
            self.logger.info(f"Successfully pulled image {image_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to pull image {image_name}: {e}")
            raise
    
    def create_container(
        self,
        image_name: str,
        container_name: Optional[str] = None,
        working_dir: str = "/testbed",
        extra_env: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict]] = None,
        use_host_network: bool = False,
        proxy_port: Optional[int] = None,
        gpu_ids: Optional[str] = None,
        docker_runtime_config: Optional[Dict[str, Any]] = None
    ) -> Container:
        """
        Create and start a Docker container.
        
        Args:
            image_name: Docker image name
            container_name: Optional container name
            working_dir: Working directory inside container
            extra_env: Additional environment variables
            volumes: Volume mounts
            use_host_network: Whether to use host network mode (default: False, if proxy_port is not None, use host network)
            proxy_port: Proxy port to use for inference (default: None)
            gpu_ids: Comma-separated GPU IDs (e.g., "0,1,2,3"), None means all GPUs
            docker_runtime_config: Runtime config from repo_settings (need_gpu, shm_size, env_vars, env_exports, number_once)
            
        Returns:
            Docker container object
        """
        docker_runtime_config = docker_runtime_config or {}
        
        # Merge environment variables
        env = dict(self.env_vars)
        if extra_env:
            env.update(extra_env)
        
        # Add environment variables from docker_runtime_config
        env_vars_from_config = docker_runtime_config.get("env_vars", {})
        if env_vars_from_config:
            env.update(env_vars_from_config)
            self.logger.info(f"Added environment variables from config: {list(env_vars_from_config.keys())}")
        
        # Convert env dict to list format
        # Replace localhost/127.0.0.1 with Docker host gateway IP for bridge mode
        processed_env = {}
        for k, v in env.items():
            if v:
                # Replace localhost references with Docker host gateway
                v_str = str(v)
                if 'localhost' in v_str or '127.0.0.1' in v_str:
                    original = v_str
                    v_str = v_str.replace('localhost', DOCKER_HOST_GATEWAY)
                    v_str = v_str.replace('127.0.0.1', DOCKER_HOST_GATEWAY)
                    if original != v_str:
                        self.logger.info(f"Replaced host reference in {k}: {original} -> {v_str}")
                processed_env[k] = v_str
        
        env_list = [f"{k}={v}" for k, v in processed_env.items()]
        
        # Proxy configuration (uses bridge mode with Docker host gateway)
        if proxy_port is not None:
            self.logger.info(f"Using proxy port {proxy_port} via Docker host gateway ({DOCKER_HOST_GATEWAY})")
            # Keep bridge mode for port isolation (don't set use_host_network = True)
            use_host_network = False
            env_list.append(f"http_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"https_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"HTTP_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"HTTPS_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"no_proxy=localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}")
            env_list.append(f"NO_PROXY=localhost,127.0.0.1,{DOCKER_HOST_GATEWAY}")
            # Also expose the proxy as an explicit "upstream" endpoint for tools that run a local MITM proxy inside the container and need to chain to the host proxy.
            env_list.append(f"ACE_UPSTREAM_PROXY=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
            env_list.append(f"ace_upstream_proxy=http://{DOCKER_HOST_GATEWAY}:{proxy_port}")
        
        # Get GPU-related configs
        need_gpu = bool(docker_runtime_config.get("need_gpu"))
        number_once = docker_runtime_config.get("number_once", 1)
        if not isinstance(number_once, int) or number_once <= 0:
            number_once = 1
        shm_size = docker_runtime_config.get("shm_size")

        # Configure GPU access
        device_requests = None
        if need_gpu:
            if gpu_ids is not None:
                device_requests = [
                    docker.types.DeviceRequest(device_ids=gpu_ids.split(','), capabilities=[["gpu"]])
                ]
                self.logger.info(f"GPU access requested for specific GPUs: {gpu_ids}")
                env_list.append(f"NVIDIA_VISIBLE_DEVICES={gpu_ids}")
            else:
                device_requests = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
                self.logger.info("GPU access requested for all available GPUs")
                env_list.append("NVIDIA_VISIBLE_DEVICES=all")

            env_list.append("NVIDIA_DRIVER_CAPABILITIES=compute,utility")
        else:
            self.logger.info("This task does not require GPU; starting container without GPU access")

        if shm_size:
            self.logger.info(f"Using shared memory size: {shm_size}")
        
        # Remove existing container with the same name if it exists
        if container_name:
            try:
                existing_container = self.client.containers.get(container_name)
                self.logger.info(f"Found existing container with name '{container_name}', removing it...")
                existing_container.stop(timeout=20)
                existing_container.remove(force=True)
                self.logger.info(f"Removed existing container '{container_name}'")
            except docker.errors.NotFound:
                pass  # Container doesn't exist, no need to remove
            except Exception as e:
                self.logger.warning(f"Error removing existing container '{container_name}': {e}")
                # Try force remove
                try:
                    existing_container.remove(force=True)
                except:
                    pass
        
        try:
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir=working_dir,
                environment=env_list,
                volumes=volumes,
                command="tail -f /dev/null",  # Keep container running
                platform="linux/amd64",
                network_mode="host" if use_host_network else "bridge",
                device_requests=device_requests,
                shm_size=shm_size
            )
            
            # Check if GPU is available
            if need_gpu:
                exit_code, output = container.exec_run("nvidia-smi --list-gpus")
                if exit_code == 0:
                    gpu_list = output.decode().strip().split("\n")
                    gpu_count = len(gpu_list)
                    if gpu_count < number_once:
                        raise RuntimeError(f"Container can only access {gpu_count} GPU(s), but {number_once} are required")
                    else:
                        self.logger.info(f"Container can access {gpu_count} GPU(s)")
                else:
                    raise RuntimeError(f"This task needs GPU, but failed to query GPUs: {output.decode()}")
            
            self.logger.info(f"Created container {container.short_id} (network: {'host' if use_host_network else 'bridge'})")
            
            # Apply -ee environment exports (write to .bashrc) if any
            env_exports = docker_runtime_config.get("env_exports", [])
            if env_exports:
                self._apply_env_exports(container, env_exports)
            
            return container
            
        except Exception as e:
            self.logger.error(f"Failed to create container: {e}")
            raise
    
    def _apply_env_exports(self, container: Container, env_exports: List[str]) -> None:
        """
        Apply environment exports to container's .bashrc.
        
        Args:
            container: Docker container
            env_exports: List of export statements (e.g., ['export VAR=value'])
        """
        if not env_exports:
            return
        
        try:
            # Build command to append exports to .bashrc
            export_cmds = [
                'echo "" >> ~/.bashrc',
                'echo "# Custom environment variables from repo_settings" >> ~/.bashrc'
            ]
            for export_stmt in env_exports:
                # Escape single quotes in the export statement
                escaped_stmt = export_stmt.replace("'", "'\\''")
                export_cmds.append(f"echo '{escaped_stmt}' >> ~/.bashrc")
            export_cmds.append('source ~/.bashrc')
            
            full_cmd = " && ".join(export_cmds)
            exit_code, output = self.exec_command(container, full_cmd)
            
            if exit_code == 0:
                self.logger.info(f"Applied {len(env_exports)} environment exports to .bashrc")
            else:
                self.logger.warning(f"Failed to apply env exports: {output}")
        except Exception as e:
            self.logger.warning(f"Error applying env exports: {e}")
    
    def exec_command(
        self,
        container: Container,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[int, str]:
        """
        Execute a command inside a container.
        
        Args:
            container: Docker container
            command: Command to execute
            timeout: Timeout in seconds
            workdir: Working directory for command
            log_file: Optional file to write output to
            
        Returns:
            Tuple of (exit_code, output)
        """
        # Wrap command with bash and conda activation
        full_command = f"source ~/.bashrc && conda activate testbed 2>/dev/null || true && {command}"
        
        if workdir:
            full_command = f"cd {workdir} && {full_command}"
        
        try:
            # Execute command
            exec_result = container.exec_run(
                cmd=["bash", "-c", full_command],
                workdir=workdir,
                demux=True
            )
            
            exit_code = exec_result.exit_code
            stdout = exec_result.output[0] if exec_result.output[0] else b""
            stderr = exec_result.output[1] if exec_result.output[1] else b""
            
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace")
            
            # Log to file if specified
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(output)
                    f.write(f"\n[Exit code: {exit_code}]\n\n")
            
            return exit_code, output
            
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            self.logger.error(error_msg)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(f"ERROR: {error_msg}\n\n")
            return -1, error_msg
    
    def exec_command_stream(
        self,
        container: Container,
        command: str,
        log_file: Path,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        skip_bashrc: bool = False
    ) -> int:
        """
        Execute a command with streaming output to log file (real-time).
        
        Args:
            container: Docker container
            command: Command to execute
            log_file: File to write output to
            timeout: Timeout in seconds
            workdir: Working directory
            skip_bashrc: If True, don't source ~/.bashrc before running command
            
        Returns:
            Exit code
        """
        # Build the command - optionally skip bashrc for installation scripts
        if skip_bashrc:
            # For installation scripts, don't source bashrc as it may cause issues
            # in non-interactive shells
            full_command = command
        else:
            full_command = f"source ~/.bashrc && conda activate testbed 2>/dev/null || true && {command}"
        
        if workdir:
            full_command = f"cd {workdir} && {full_command}"
        
        # Use subprocess.Popen for real-time output capture
        docker_cmd = [
            "docker", "exec",
            container.id,
            "bash", "-c", full_command
        ]
        
        process = None
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"$ {command}\n")
                f.flush()
                
                # Start process with pipe for stdout/stderr combined
                process = subprocess.Popen(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                start_time = time.time()

                # NOTE: Using blocking readline() can prevent timeout from firing if the
                # subprocess produces no output. Use a selector to poll stdout so we can
                # always check timeout.
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)

                def _timed_out() -> bool:
                    return bool(timeout) and (time.time() - start_time) > timeout

                try:
                    # Stream output while enforcing timeout.
                    while True:
                        if _timed_out():
                            process.kill()
                            try:
                                process.wait(timeout=5)
                            except Exception:
                                pass
                            f.write(f"\n[TIMEOUT after {timeout} seconds]\n\n")
                            f.flush()
                            return -1

                        # If process has finished, drain any remaining output and exit.
                        if process.poll() is not None:
                            # Drain whatever is left without blocking indefinitely.
                            while True:
                                events = selector.select(timeout=0)
                                if not events:
                                    break
                                chunk = process.stdout.read()
                                if not chunk:
                                    break
                                f.write(strip_ansi_codes(chunk))
                                f.flush()
                            break

                        # Wait briefly for output to become available.
                        events = selector.select(timeout=0.1)
                        if not events:
                            continue

                        # Read a single line (safe now because selector signaled readability).
                        line = process.stdout.readline()
                        if line:
                            f.write(strip_ansi_codes(line))
                            f.flush()

                finally:
                    try:
                        selector.unregister(process.stdout)
                    except Exception:
                        pass
                    try:
                        selector.close()
                    except Exception:
                        pass
                
                f.write(f"\n[Exit code: {process.returncode}]\n\n")
                f.flush()
                return process.returncode
                
        except Exception as e:
            self.logger.error(f"Stream execution failed: {e}")
            import traceback
            if process and process.poll() is None:
                process.kill()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            return -1
    
    def copy_to_container(
        self,
        container: Container,
        src_path: Path,
        dest_path: str
    ) -> None:
        """
        Copy a file or directory to a container.
        
        Args:
            container: Docker container
            src_path: Source path on host
            dest_path: Destination path in container
        """
        # Ensure destination directory exists
        dest_dir = str(Path(dest_path).parent)
        container.exec_run(["mkdir", "-p", dest_dir])
        
        # Create tar archive
        tar_stream = io.BytesIO()
        dest_name = Path(dest_path).name
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if src_path.is_file():
                tar.add(src_path, arcname=dest_name)
            else:
                # Include the destination directory name in arcname
                # so that files are placed in dest_path, not dest_dir
                for item in src_path.rglob("*"):
                    tar.add(item, arcname=str(Path(dest_name) / item.relative_to(src_path)))
        
        tar_stream.seek(0)
        container.put_archive(dest_dir, tar_stream.read())
    
    def copy_from_container(
        self,
        container: Container,
        src_path: str,
        dest_path: Path
    ) -> bool:
        """
        Copy a file or directory from a container.
        
        Args:
            container: Docker container
            src_path: Source path in container (file or directory)
            dest_path: Destination path on host
            
        Returns:
            True if successful
        """
        try:
            bits, stat = container.get_archive(src_path)
            
            # Extract from tar
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                members = tar.getmembers()
                
                if not members:
                    self.logger.warning(f"No files found in archive from {src_path}")
                    return False
                
                # Check if this is a directory or a single file
                # Docker's get_archive always returns a tar with the basename as root
                root_member = members[0]
                is_directory = root_member.isdir() or len(members) > 1
                
                if is_directory:
                    # For directories, extract all contents
                    # The archive contains the directory itself, we want to extract to dest_path
                    extracted_count = 0
                    for member in members:
                        # Skip the root directory itself (first member)
                        if member == root_member and member.isdir():
                            continue
                        
                        # Calculate target path relative to dest_path
                        # Remove the root directory name from the member path
                        if member.name.startswith(root_member.name + '/'):
                            relative_path = member.name[len(root_member.name) + 1:]
                        elif member.name == root_member.name:
                            continue
                        else:
                            relative_path = member.name
                        
                        if not relative_path:
                            continue
                        
                        target_path = dest_path / relative_path
                        
                        if member.isdir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                with open(target_path, "wb") as f:
                                    f.write(file_obj.read())
                                extracted_count += 1
                    
                    if extracted_count > 0:
                        self.logger.info(f"Extracted {extracted_count} files from {src_path} to {dest_path}")
                        return True
                    else:
                        self.logger.warning(f"No files extracted from {src_path}")
                        return False
                else:
                    # For single file, extract directly to dest_path
                    if root_member.isfile():
                        file_obj = tar.extractfile(root_member)
                        if file_obj:
                            with open(dest_path, "wb") as f:
                                f.write(file_obj.read())
                            self.logger.info(f"Copied file {src_path} to {dest_path}")
                            return True
                    
                    return False
            
        except Exception as e:
            self.logger.warning(f"Failed to copy {src_path} from container: {e}")
            return False

    def stop_container(self, container: Container, force: bool = False) -> None:
        """
        Stop and remove a container.
        
        Args:
            container: Docker container
            force: Force kill if True
        """
        try:
            if force:
                container.kill()
            else:
                container.stop(timeout=10)
            container.remove(force=True)
            self.logger.info(f"Stopped and removed container {container.short_id}")
        except Exception as e:
            self.logger.warning(f"Error stopping container: {e}")
            try:
                container.remove(force=True)
            except:
                pass
    
    def get_container_logs(self, container: Container) -> str:
        """
        Get container logs.
        
        Args:
            container: Docker container
            
        Returns:
            Container logs as string
        """
        try:
            return container.logs().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Failed to get logs: {e}"

