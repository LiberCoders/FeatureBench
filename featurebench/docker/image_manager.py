import logging
import subprocess
import tempfile
import os
import hashlib
import platform
import importlib.util
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import time
import tarfile
import tempfile

from featurebench.utils.repo_manager import RepoManager
from featurebench.utils.logger import (
	create_docker_build_logger, 
	run_subprocess_with_logging, 
	print_build_report,
	run_command_with_streaming_log
)
from featurebench.utils.utils import select_candidate_pool_and_allocate_gpu, release_gpu


class ImageManager:
    """Docker image manager."""
    
    def __init__(
        self, 
        config,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the image manager.
        
        Args:
            config: Config object (env vars, logs_dir, etc.)
            logger: Logger instance (uses default if None)
        """
        self.logs_dir = config.logs_dir
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.env_vars = config.env_vars or {}  # Global env vars from config
        self.gpu_ids = list(config.gpu_ids) if getattr(config, "gpu_ids", None) else None
        
        # Image info: {specs_name: {base_image: str, instance_image: str}}
        self.image_info: Dict[str, Dict[str, str]] = {}
        
        # Cache specs for later Docker build commands: {specs_name: specs}
        self._specs_cache: Dict[str, Dict] = {}
        
        # Track GPUs per container: {container_id: [gpu_id, ...]}
        self._container_gpu_map: Dict[str, List[int]] = {}
    
    def prepare_images(self, repo_manager: RepoManager) -> None:
        """
        Prepare Docker images (base + instance) for all loaded repos.
        
        Args:
            repo_manager: Repo manager
        """
        self.logger.info("Starting Docker image preparation for repos...")
        
        # Check Docker availability
        if not self._check_docker_available():
            raise RuntimeError("Docker is unavailable. Please ensure Docker is installed and running.")
        
        # Get loaded repo info
        for specs_name, repo_info in repo_manager.loaded_repos.items():
            # specs_name like SPECS_LITGPT, repo_info = {local_path: Path, specs: Dict}
            self.logger.info(f"Preparing images for repo {specs_name}...")
            
            try:
                # Get repo local path and specs
                local_path = repo_info['local_path']
                specs = repo_info['specs']
                
                # Cache specs for later use
                self._specs_cache[specs_name] = specs
                
                # Prepare images for this repo
                base_image, instance_image = self._prepare_repo_image(specs_name, local_path, specs)
                
                # Record image info
                self.image_info[specs_name] = {
                    'base_image': base_image,
                    'instance_image': instance_image
                }
                
                self.logger.info(f"Repo {specs_name} images prepared")
                self.logger.info(f"  Base image: {base_image}")
                self.logger.info(f"  Instance image: {instance_image}")
                
            except Exception as e:
                self.logger.error(f"Failed to prepare images for repo {specs_name}: {e}")
                raise RuntimeError(f"❌ Image preparation failed for repo {specs_name}: {e}")
        
        self.logger.info(f"All repo images prepared; processed {len(self.image_info)} repos")
    
    def prepare_images_parallel(self, repo_manager: RepoManager, max_workers: Optional[int] = None) -> None:
        """
        Prepare Docker images in parallel for all loaded repos.
        
        Args:
            repo_manager: Repo manager
            max_workers: Max workers; auto-select if None
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Starting Docker image preparation for repos...")
        self.logger.info("=" * 60)
        
        # Check Docker availability
        if not self._check_docker_available():
            raise RuntimeError("Docker is unavailable. Please ensure Docker is installed and running.")
        
        repo_items = list(repo_manager.loaded_repos.items())
        
        # Set default worker count
        if max_workers is None:
            # Limit parallelism due to Docker build resource usage
            max_workers = min(len(repo_items), os.cpu_count() or 1, 3)  # Max 3 parallel builds
        
        self.logger.info(f"Building {len(repo_items)} repo images with {max_workers} workers")
        
        # Info about failed builds
        failed_repos: List[Tuple[str, str]] = []  # (repo_name, error_message)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit build tasks
            future_to_repo = {}
            for specs_name, repo_info in repo_items:
                # Cache specs for later use
                self._specs_cache[specs_name] = repo_info['specs']
                
                future = executor.submit(
                    self._prepare_repo_image,
                    specs_name,
                    repo_info['local_path'],
                    repo_info['specs'],
                    True
                )
                future_to_repo[future] = specs_name
            
            # Track per-repo status
            completed_count = 0
            
            # Use progress bar for completed tasks
            with tqdm(total=len(repo_items), desc="Building images", unit="repo") as pbar:
                for future in as_completed(future_to_repo):
                    specs_name = future_to_repo[future]
                    completed_count += 1
                    
                    try:
                        base_image, instance_image = future.result()
                        
                        # Record image info
                        self.image_info[specs_name] = {
                            'base_image': base_image,
                            'instance_image': instance_image
                        }
                        
                        # Update status
                        pbar.update(1)
                        pbar.set_postfix_str(f"Latest success: {specs_name}")
                        tqdm.write(f"✅ {specs_name}: Image build completed ({completed_count}/{len(repo_items)})")
                        
                    except Exception as e:
                        failed_repos.append((specs_name, str(e)))
                        
                        # Update status
                        pbar.update(1)
                        pbar.set_postfix_str(f"Latest failure: {specs_name}")
                        tqdm.write(f"❌ {specs_name}: Build failed - {str(e)[:50]}... ({completed_count}/{len(repo_items)})")
        
        # Use unified report logger
        # Convert image info format
        success_info = {}
        for specs_name, image_info in self.image_info.items():
            success_info[specs_name] = {
                'base_image': image_info['base_image'],
                'instance_image': image_info['instance_image']
            }
        print_build_report(
            logger=self.logger,
            total_repos=repo_items,
            failed_repos=failed_repos,
            success_info=success_info,
            operation_name="Docker image build"
        )
        
        # Raise if any repos failed
        if failed_repos:
            failed_names = [name for name, _ in failed_repos]
            raise RuntimeError(f"❌ Failed to prepare images for {len(failed_repos)} repos: {failed_names}")
    
    def _check_docker_available(self) -> bool:
        """Check whether Docker is available."""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.debug("Docker is available")
                return True
            else:
                self.logger.error("Docker command failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"Docker is unavailable: {e}")
            return False
    
    def _prepare_repo_image(self, specs_name: str, local_path: Path, specs: Dict, parallel_mode: bool = False) -> tuple:
        """
        Prepare Docker images for a single repo.

        Args:
            specs_name: Repo spec name (e.g. SPECS_LITGPT)
            local_path: Local repo path
            specs: Repo specs config
            parallel_mode: Whether running in parallel mode (affects logging)

        Returns:
            Tuple of (base_image_name, instance_image_name)
        """
        if parallel_mode:
            tqdm.write(f"🏃 {specs_name}: Starting Docker image preparation...")
        
        # 1. Get or build base image
        base_image = self._get_base_image(specs_name, specs, parallel_mode=parallel_mode)
        
        # 2. Get or build instance image
        instance_image = self._get_instance_image(specs_name, local_path, specs, base_image, parallel_mode=parallel_mode)
        
        return base_image, instance_image
    
    def _get_base_image(self, specs_name: str, specs: Dict, parallel_mode: bool = False) -> str:
        """Get or build the base Docker image."""
        # Read base image type from config
        base_image_type = specs.get("base_image")
        
        # Get Dockerfile template and default config (two constants)
        dockerfile_template, default_specs = self._get_dockerfile_template(base_image_type)
        
        # Fill dockerfile_template with default_specs
        dockerfile_content = dockerfile_template.format(
            ubuntu_version=default_specs["ubuntu_version"],
            conda_version=default_specs["conda_version"],
            conda_arch="aarch64" if platform.machine().lower() in ['aarch64', 'arm64'] else "x86_64",
            python_version=default_specs["python_version"]
        )
        
        # Build image hash suffix from Dockerfile content
        dockerfile_hash = hashlib.md5(dockerfile_content.encode()).hexdigest()[:8]
        image_name = f"featurebench-{base_image_type}-base_{dockerfile_hash}"
        
        # Check if image exists
        rebuild_base = specs.get("rebuild_base_image", False)
        image_exists = self._image_exists(image_name)
        
        if image_exists and not rebuild_base:
            if not parallel_mode:
                self.logger.info(f"Base Docker image already exists: {image_name}")
            else:
                tqdm.write(f"⏭️ {specs_name}: Base image {base_image_type} exists; skipping build")
            return image_name
        
        # If force rebuild and image exists, delete old image first
        if rebuild_base and image_exists:
            if not parallel_mode:
                self.logger.info(f"Removing existing base image: {image_name}")
            else:
                tqdm.write(f"🗑️ {specs_name}: Removing existing base image {base_image_type}")
            self._remove_image(image_name)
        
        # Image missing or force rebuild
        if not parallel_mode:
            if rebuild_base:
                self.logger.info(f"Forcing rebuild of base image: {image_name}")
            else:
                self.logger.info(f"Building base image: {image_name}")
        self._build_base_image(image_name, dockerfile_content, parallel_mode, specs_name, rebuild_base)
        
        return image_name
    
    def _get_instance_image(self, specs_name: str, local_path: Path, specs: Dict, base_image: str, parallel_mode: bool = False) -> str:
        """Get or build the instance image."""
        # Build instance image name
        repository = specs.get('repository')
        commit = specs.get('commit')
        
        # Build unique identifier from repo + commit
        repo_identifier = f"{repository}_{commit}".replace('/', '_').replace('-', '_')
        repo_hash = hashlib.md5(repo_identifier.encode()).hexdigest()[:8]
        instance_image_name = f"featurebench-{specs_name.lower()}-instance_{repo_hash}"
        
        # Check if image exists
        rebuild_instance = specs.get("rebuild_instance_image", False)
        image_exists = self._image_exists(instance_image_name)
        
        if image_exists and not rebuild_instance:
            if not parallel_mode:
                self.logger.info(f"Instance image already exists: {instance_image_name}")
            else:
                tqdm.write(f"⏭️ {specs_name}: Instance image exists; skipping build")
            return instance_image_name
        
        # If force rebuild and image exists, delete old image first
        if rebuild_instance and image_exists:
            if not parallel_mode:
                self.logger.info(f"Removing existing instance image: {instance_image_name}")
            else:
                tqdm.write(f"🗑️ {specs_name}: Removing existing instance image")
            self._remove_image(instance_image_name)
        
        # Image missing or force rebuild
        if not parallel_mode:
            if rebuild_instance:
                self.logger.info(f"Forcing rebuild of instance image: {instance_image_name}")
            else:
                self.logger.info(f"Building instance image: {instance_image_name}")
        self._build_instance_image(instance_image_name, local_path, specs, base_image, parallel_mode, specs_name, rebuild_instance)
        
        return instance_image_name
    
    def _image_exists(self, image_name: str) -> bool:
        """Check whether a Docker image exists."""
        try:
            result = subprocess.run(
                ['docker', 'image', 'inspect', image_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.warning(f"Error checking Docker image {image_name}: {e}")
            return False
    
    def _remove_image(self, image_name: str) -> bool:
        """Remove a Docker image."""
        try:
            result = subprocess.run(
                ['docker', 'rmi', image_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                self.logger.debug(f"Deleted image successfully: {image_name}")
                return True
            else:
                self.logger.warning(f"Failed to delete image {image_name}: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.warning(f"Error deleting Docker image {image_name}: {e}")
            return False
    
    def _build_base_image(self, image_name: str, dockerfile_content: str, parallel_mode: bool = False, specs_name: str = None, is_rebuild: bool = False) -> None:
        """Build the base Docker image."""
        # Create temporary build directory
        with tempfile.TemporaryDirectory(prefix='featurebench_docker_build_') as build_dir:
            dockerfile_path = os.path.join(build_dir, 'Dockerfile')
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build Docker image command
            build_command = [
                'docker', 'build',
                '-t', image_name,
                '-f', dockerfile_path,
                build_dir
            ]
            
            # Create log file
            log_file_path = create_docker_build_logger('base', image_name, logs_dir=self.logs_dir)
            
            # Run build using new logging system
            return_code = run_subprocess_with_logging(
                command=build_command,
                log_file_path=log_file_path,
                logger=self.logger,
                parallel_mode=parallel_mode,
                specs_name=specs_name,
                build_type='base',
                is_rebuild=is_rebuild
            )
            
            if return_code != 0:
                self.logger.error(f"Failed to build base image {image_name}, log: {log_file_path}")
                raise RuntimeError(f"Failed to build base image {image_name}, log: {log_file_path}")
            
            if not parallel_mode:
                self.logger.info(f"Base image {image_name} built successfully, log: {log_file_path}")
            else:
                tqdm.write(f"✅ {specs_name}: Base image build completed")
    
    def _build_instance_image(self, image_name: str, local_path: Path, specs: Dict, base_image: str, parallel_mode: bool = False, specs_name: str = None, is_rebuild: bool = False) -> None:
        """Build the instance Docker image."""
        # Instance image Dockerfile template
        instance_dockerfile_template = f"""FROM {base_image}

# Handle custom instance image build command
{self._get_custom_build_commands(specs)}

# Copy env pre-setup script & repo code into container
COPY ./setup_env.sh /root/
COPY ./project_code/ /testbed/
RUN sed -i -e 's/\r$//' /root/setup_env.sh
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"

# After env setup, create a repo backup my_repo:
RUN cp -r /testbed/ /root/my_repo/

# Set working directory
WORKDIR /testbed/

# Activate testbed environment
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > ~/.bashrc
"""
        
        # Create temporary build directory
        with tempfile.TemporaryDirectory(prefix='featurebench_instance_build_') as build_dir:
            dockerfile_path = os.path.join(build_dir, 'Dockerfile')
            setup_env_path = os.path.join(build_dir, 'setup_env.sh')
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(instance_dockerfile_template)
            
            # Generate and write setup_env.sh
            setup_env_content = self._generate_setup_env_script(specs)
            with open(setup_env_path, 'w') as f:
                f.write(setup_env_content)
            
            # Copy repo code into build directory
            build_project_dir = os.path.join(build_dir, 'project_code')
            shutil.copytree(local_path, build_project_dir)
            
            # Build Docker image command
            build_command = [
                'docker', 'build',
                '-t', image_name,
                '-f', dockerfile_path,
                build_dir  # Use build dir as context
            ]
            
            # Create log file
            log_file_path = create_docker_build_logger('instance', image_name, logs_dir=self.logs_dir)
            
            # Run build using new logging system
            return_code = run_subprocess_with_logging(
                command=build_command,
                log_file_path=log_file_path,
                logger=self.logger,
                parallel_mode=parallel_mode,
                specs_name=specs_name,
                build_type='instance',
                is_rebuild=is_rebuild
            )
            
            if return_code != 0:
                self.logger.error(f"Failed to build instance image {image_name}, log: {log_file_path}")
                raise RuntimeError(f"Failed to build instance image {image_name}, log: {log_file_path}")
            
            if not parallel_mode:
                self.logger.info(f"Instance image {image_name} built successfully, log: {log_file_path}")
            else:
                tqdm.write(f"✅ {specs_name}: Instance image build completed")
    
    def _generate_setup_env_script(self, specs: Dict) -> str:
        script_lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Activate conda environment",
            "source /opt/miniconda3/etc/profile.d/conda.sh",
            "",
            "# Create testbed env (clone from pytorch_base to inherit PyTorch)",
            "conda activate pytorch_base",
            "conda create -n testbed --clone pytorch_base -y",
            "conda activate testbed",
            "",
            "# Avoid hardcoded pip preinstalls; use pip_packages if needed",
            "pip install pytest-timeout",
        ]
        
        # Install pip packages from config
        for package in specs.get("pip_packages", []):
            # Quote to parse versions like tokenizers>=0.19,<0.20
            script_lines.append(f"pip install '{package}'")
        
        # Run pre_install commands (before project install)
        if "pre_install" in specs:
            script_lines.extend([
                "",
                "# Run pre_install commands",
            ])
            for cmd in specs["pre_install"]:
                script_lines.append(cmd)

        # Use repo code copied into container (under /testbed)
        script_lines.extend([
            "",
            "# Repo code copied to /testbed; install directly",
            "cd /testbed",
            "",
            "# Install the project itself",
            f"{specs.get('install')}",
            "",
            "echo 'Environment setup complete'",
        ])
        
        return "\n".join(script_lines)
    
    def _get_dockerfile_template(self, base_image_type: str) -> tuple:
        """Get Dockerfile template and default config."""
        # Resolve dockerfiles directory
        current_file = Path(__file__)
        featurebench_dir = current_file.parent.parent  # Go from docker/ back to featurebench/
        dockerfiles_dir = featurebench_dir / "resources" / "dockerfiles"
        
        # Locate template file
        template_file = dockerfiles_dir / f"{base_image_type}.py"
        
        # Check template file exists
        if not template_file.exists():
            # List available templates
            available_templates = []
            if dockerfiles_dir.exists():
                for file in dockerfiles_dir.glob("*.py"):
                    available_templates.append(file.stem)
            raise FileNotFoundError(
                f"Dockerfile template not found: {template_file}\n"
                f"Available templates: {', '.join(available_templates)}"
            )
        
        # Dynamically import template module and config
        spec = importlib.util.spec_from_file_location(f"dockerfile_{base_image_type}", template_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get template content and default config
        if hasattr(module, '_DOCKERFILE_BASE_PY'):
            dockerfile_template = module._DOCKERFILE_BASE_PY
        else:
            raise AttributeError(f"_DOCKERFILE_BASE_PY not found in template file {template_file}")
        
        if hasattr(module, 'DEFAULT_DOCKER_SPECS'):
            default_specs = module.DEFAULT_DOCKER_SPECS
        else:
            raise AttributeError(f"DEFAULT_DOCKER_SPECS not found in template file {template_file}")
        
        return dockerfile_template, default_specs
    
    def _get_custom_build_commands(self, specs: Dict) -> str:
        """Get custom build command."""
        custom_commands = specs.get("custom_instance_image_build", "")
        
        if not custom_commands:
            return "# No custom build command"
        
        if isinstance(custom_commands, list):
            # List format: each element is a Dockerfile directive
            return "\n".join(custom_commands)
        elif isinstance(custom_commands, str):
            # String format: use as-is
            return custom_commands
        else:
            self.logger.warning("custom_instance_image_build has invalid format; expected string or list")
            return "# Invalid custom build command format"

    def render_instance_dockerfile(self, specs_name: str) -> str:
        """Render the exact Dockerfile used to build the instance image.

        This is useful for provenance: the data pipeline can record the build recipe
        into instance.json.
        """
        if specs_name not in self.image_info:
            raise KeyError(f"Unknown specs_name (image_info not prepared): {specs_name}")
        specs = self._specs_cache.get(specs_name)
        if not isinstance(specs, dict):
            raise KeyError(f"Missing cached specs for: {specs_name}")

        base_image = self.image_info[specs_name]["base_image"]
        return (
            f"""FROM {base_image}

# Handle custom instance image build command
{self._get_custom_build_commands(specs)}

# Copy env pre-setup script & repo code into container
COPY ./setup_env.sh /root/
COPY ./project_code/ /testbed/
RUN sed -i -e 's/\r$//' /root/setup_env.sh
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"

# After env setup, create a repo backup my_repo:
RUN cp -r /testbed/ /root/my_repo/

# Set working directory
WORKDIR /testbed/

# Activate testbed environment
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > ~/.bashrc
"""
        )

    def render_setup_env_script(self, specs_name: str) -> str:
        """Render the setup_env.sh script used during instance-image build."""
        specs = self._specs_cache.get(specs_name)
        if not isinstance(specs, dict):
            raise KeyError(f"Missing cached specs for: {specs_name}")
        return self._generate_setup_env_script(specs)

    def render_base_dockerfile(self, specs_name: str) -> str:
        """Render the Dockerfile content used to build the base image."""
        specs = self._specs_cache.get(specs_name)
        if not isinstance(specs, dict):
            raise KeyError(f"Missing cached specs for: {specs_name}")
        base_image_type = specs.get("base_image")
        dockerfile_template, default_specs = self._get_dockerfile_template(base_image_type)
        return dockerfile_template.format(
            ubuntu_version=default_specs["ubuntu_version"],
            conda_version=default_specs["conda_version"],
            conda_arch="aarch64" if platform.machine().lower() in ["aarch64", "arm64"] else "x86_64",
            python_version=default_specs["python_version"],
        )
    
    def run_container(
        self,
        specs_name: str,
        working_dir: str = "/testbed",
        prepare_env: bool = True
    ) -> str:
        """
        Start a Docker container (background) and optionally prepare the env.
        
        Args:
            specs_name: Repo spec name
            working_dir: Working directory, default /testbed
            prepare_env: Whether to prepare env (conda + repo restore)
            
        Returns:
            Container ID
        """
        instance_image = self.image_info[specs_name]['instance_image']
        specs = self._specs_cache.get(specs_name)
        
        # Build base command
        docker_command = [
            'docker', 'run',
            '-d',  # Run in background
            '-w', working_dir,
        ]
        
        # Add runtime config to docker_command and compute env_setup + GPU IDs
        env_setup_cmd, selected_gpu_ids = self._add_docker_runtime_config(docker_command, specs)
        
        # Add image
        docker_command.append(instance_image)
        
        # Build start command
        if prepare_env:
            # If prepare_env, build full startup command
            conda_activate_cmd = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
            restore_project_cmd = "rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/"
            ready_marker_cmd = "touch /tmp/env_ready"  # Marker file for env readiness
            
            # If env_setup_cmd writes .bashrc, re-activate conda after source
            if env_setup_cmd:
                full_command = f'{conda_activate_cmd} && {restore_project_cmd}{env_setup_cmd} && {conda_activate_cmd} && {ready_marker_cmd} && tail -f /dev/null'
            else:
                full_command = f'{conda_activate_cmd} && {restore_project_cmd} && {ready_marker_cmd} && tail -f /dev/null'
            
            docker_command.extend(['bash', '-c', full_command])
        else:
            # No prepare_env needed; keep container running
            docker_command.extend(['tail', '-f', '/dev/null'])
        
        # Run command
        result = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        container_id = result.stdout.strip()
        
        # Record GPUs used by container (if any)
        if selected_gpu_ids:
            self._container_gpu_map[container_id] = selected_gpu_ids
            self.logger.debug(f"Container {container_id[:12]} using GPUs {selected_gpu_ids}")
        
        # If prepare_env, wait for env readiness to avoid missing files
        if prepare_env:
            wait_interval = 1  # Check every 1 second
            elapsed = 0
            
            # Wait for env-ready marker file
            while True:
                # Check marker file exists
                check_ready_cmd = "test -f /tmp/env_ready && echo 'ready' || echo 'not_ready'"
                check_result = subprocess.run(
                    ['docker', 'exec', container_id, 'bash', '-c', check_ready_cmd],
                    capture_output=True,
                    text=True
                )
                
                if check_result.returncode == 0 and check_result.stdout.strip() == 'ready':
                    # Marker exists; env setup done
                    self.logger.debug(f"Container {container_id[:12]} env ready (elapsed {elapsed:.1f}s)")
                    break
                
                time.sleep(wait_interval)
                elapsed += wait_interval
        
        return container_id
    
    def _add_docker_runtime_config(self, docker_command: List[str], specs: Dict) -> Tuple[str, List[int]]:
        """
        Add runtime config to Docker command (GPU/env/volumes).
        
        Args:
            docker_command: Docker command list (modified in place)
            specs: Repo specs config
            
        Returns:
            (env_setup_cmd, selected_gpu_ids)
            - env_setup_cmd: command appended to startup; empty if none
            - selected_gpu_ids: list of GPU IDs (empty if none)
        """
        selected_gpu_ids: List[int] = []  # Selected GPU IDs
        
        # Add global env vars (from config.env_vars)
        for var_name, var_value in self.env_vars.items():
            if var_value:  # Only add non-empty values
                docker_command.extend(['-e', f'{var_name}={var_value}'])
        
        # Add extra Docker run parameters from config
        docker_specs = specs.get("docker_specs", {})
        run_args = docker_specs.get("run_args", {})
        custom_docker_args = docker_specs.get("custom_docker_args", [])
        env_exports = []  # Env vars to append to .bashrc
        
        if custom_docker_args:
            if isinstance(custom_docker_args, list):
                # Handle list-format params: only -e, -v, -ee supported
                for arg in custom_docker_args:
                    if isinstance(arg, str):
                        if arg.startswith('-e '):
                            # Plain env var parameter, only add to Docker command
                            env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                            if '=' in env_part:
                                docker_command.extend(['-e', env_part])
                            else:
                                self.logger.warning(f"Invalid env var arg format: {arg}")
                        elif arg.startswith('-v '):
                            # Volume mount parameter
                            volume_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                            if volume_part:
                                docker_command.extend(['-v', volume_part])
                            else:
                                self.logger.warning(f"Invalid volume mount arg format: {arg}")
                        elif arg.startswith('-ee '):
                            # Custom env var: write to .bashrc without expansion
                            env_part = arg.split(' ', 1)[1] if ' ' in arg else ''
                            if '=' in env_part:
                                # Keep raw string (e.g., $HOME) for container expansion
                                # Let container expand on source ~/.bashrc
                                env_exports.append(f'export {env_part}')
                            else:
                                self.logger.warning(f"Invalid custom env var arg format: {arg}")
                        else:
                            self.logger.warning(f"Unsupported arg format: {arg}; only -e, -v, -ee are supported")
                    else:
                        self.logger.warning(f"custom_docker_args entries must be strings: {arg}")
            else:
                    self.logger.warning("custom_docker_args has invalid format; expected list or dict")
        
        # Handle GPU support
        if "cuda_visible_devices" in run_args:
            raise ValueError(
                "run_args.cuda_visible_devices is no longer supported; "
                "use run_args.cuda_visible_num plus CLI --gpu-ids"
            )

        cuda_visible_num = run_args.get("cuda_visible_num")
        number_once = run_args.get("number_once", 1)
        if not isinstance(number_once, int) or number_once <= 0:
            number_once = 1

        if cuda_visible_num is not None:
            if not isinstance(cuda_visible_num, int) or cuda_visible_num <= 0:
                raise ValueError(
                    f"Invalid run_args.cuda_visible_num={cuda_visible_num}; expected a positive integer or None"
                )

            if cuda_visible_num < number_once:
                raise ValueError(
                    f"Invalid GPU config: cuda_visible_num ({cuda_visible_num}) must be >= number_once ({number_once})"
                )

            if not self.gpu_ids:
                raise ValueError(
                    "GPU is requested by run_args.cuda_visible_num, but CLI --gpu-ids is not provided"
                )

            if len(self.gpu_ids) < cuda_visible_num:
                raise ValueError(
                    f"GPU config mismatch: provided --gpu-ids has {len(self.gpu_ids)} id(s), "
                    f"which is smaller than cuda_visible_num ({cuda_visible_num})"
                )

            # Semantics:
            # - cuda_visible_num: smart-selected candidate pool size (from CLI --gpu-ids)
            # - number_once: actual GPU count allocated to this container
            requested_gpu_count = number_once
            candidate_gpus, selected_gpu_ids = select_candidate_pool_and_allocate_gpu(
                list(self.gpu_ids),
                self.logger,
                pool_count=cuda_visible_num,
                allocate_count=requested_gpu_count
            )
            if len(candidate_gpus) < cuda_visible_num:
                raise RuntimeError(
                    f"Unable to choose candidate GPU pool: requested {cuda_visible_num}, "
                    f"selected {len(candidate_gpus)} from --gpu-ids {self.gpu_ids}"
                )
            if len(selected_gpu_ids) < requested_gpu_count:
                raise RuntimeError(
                    f"Unable to allocate enough GPUs: requested {requested_gpu_count}, "
                    f"allocated {len(selected_gpu_ids)} from pool {candidate_gpus}"
                )

            selected_str = ",".join(str(gid) for gid in selected_gpu_ids)
            docker_command.extend(['--gpus', f'"device={selected_str}"'])
        
        # Handle shared memory size
        shm_size = run_args.get("shm_size")
        if shm_size:
            docker_command.extend(['--shm-size', shm_size])
        
        # Handle capabilities
        for cap in run_args.get("cap_add", []):
            docker_command.extend(['--cap-add', cap])
        
        # If env vars need .bashrc, build env setup command
        if env_exports:
            env_setup_cmd = " && " + " && ".join([
                'echo "" >> ~/.bashrc',  # Add separator line
                'echo "# Custom environment variables" >> ~/.bashrc'
            ] + [f'echo \'{export_stmt}\' >> ~/.bashrc' for export_stmt in env_exports] + [
                'source ~/.bashrc'  # source ~/.bashrc to load new env vars
            ])
            return (env_setup_cmd, selected_gpu_ids)
        
        return ("", selected_gpu_ids)

    def copy_to_container(
        self, 
        container_id: str, 
        src_path: str, 
        dest_path: str, 
        is_directory: bool = False,
        use_tar: bool = False,
        files_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Copy files or directories into a container.
        
        Args:
            container_id: Container ID
            src_path: Source path on host (use when use_tar=False)
            dest_path: Destination path in container (use when use_tar=False)
            is_directory: If True, copy directory contents (like cp -r src/. dest/)
            use_tar: Use tar for batch transfer (better for many files)
            files_mapping: Mapping {container_path: host_path}, only for use_tar=True
        """
        if use_tar:
            # Use tar to batch transfer multiple files
            if not files_mapping:
                raise ValueError("files_mapping is required when use_tar=True")
            
            # Create temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_tar:
                tar_path = tmp_tar.name
            
            try:
                # Pack all files
                with tarfile.open(tar_path, 'w') as tar:
                    for container_path, host_path in files_mapping.items():
                        # Use container path as arcname (strip leading /)
                        arcname = container_path.lstrip('/')
                        tar.add(host_path, arcname=arcname)
                
                # Copy tar file into container
                docker_command = [
                    'docker', 'cp',
                    tar_path,
                    f'{container_id}:/tmp/batch_transfer.tar'
                ]
                subprocess.run(docker_command, check=True, capture_output=True)
                
                # Extract in container root and delete tar
                extract_cmd = "tar -xf /tmp/batch_transfer.tar -C / && rm /tmp/batch_transfer.tar"
                result = subprocess.run(
                    ['docker', 'exec', container_id, 'bash', '-c', extract_cmd],
                    capture_output=True,
                    text=True,
                    timeout=None
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to extract tar: {result.stderr}")
            
            finally:
                # Cleanup temporary tar file
                import os
                if os.path.exists(tar_path):
                    os.unlink(tar_path)
        
        elif is_directory:
            # For directories, use "/." to copy contents
            # This copies all contents under src_path to dest_path
            src_with_contents = f"{src_path}/."
            docker_command = [
                'docker', 'cp',
                src_with_contents,
                f'{container_id}:{dest_path}/'
            ]
            subprocess.run(docker_command, check=True, capture_output=True)
        
        else:
            # For files, copy directly
            docker_command = [
                'docker', 'cp',
                src_path,
                f'{container_id}:{dest_path}'
            ]
            subprocess.run(docker_command, check=True, capture_output=True)
    
    def copy_from_container(self, container_id: str, src_path: str, dest_path: str) -> None:
        """
        Copy a file from container to host.
        
        Args:
            container_id: Container ID
            src_path: Source path in container
            dest_path: Destination path on host
        """

        # Check if file exists in container first
        check_result = self.exec_in_container(
            container_id=container_id,
            command=f"test -e '{src_path}'",
            timeout=None,       # No timeout
        )
        if check_result.returncode != 0:
            tqdm.write(f"❌ File not found in container: {container_id[:12]}:{src_path}")
            raise FileNotFoundError

        docker_command = [
            'docker', 'cp',
            f'{container_id}:{src_path}',
            dest_path
        ]
		
        try:
            subprocess.run(docker_command, check=True, capture_output=True)
        except Exception as e:
            tqdm.write(f"❌ Container copy failed: {container_id[:12]} {src_path} -> {dest_path}")
            raise e
    
    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: Optional[int] = None,
        log_file_path: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """
        Execute command in container (supports streaming log capture).
        
        Args:
            container_id: Container ID
            command: Command to execute
            timeout: Timeout in seconds
            log_file_path: Optional log file path for streaming output
            
        Returns:
            subprocess.CompletedProcess
        """
        # Activate conda before running (non-interactive startup doesn't load .bashrc)
        full_command = f"source ~/.bashrc && conda activate testbed && {command}"
        
        docker_command = [
            'docker', 'exec',
            container_id,
            'bash', '-c',
            full_command
        ]
        
        # If log path provided, stream output to file
        if log_file_path:
            log_header = {
                'Container ID': container_id,
                'Command': command
            }
            return run_command_with_streaming_log(
                command=docker_command,
                log_file_path=log_file_path,
                timeout=timeout,
                log_header=log_header
            )
        else:
            # Legacy synchronous execution
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result
    
    def reset_container_env(self, container_id: str, specs_name: str, timeout: None = None) -> bool:
        """
        Reset a running container environment to initial state.
        
        This mirrors run_container(prepare_env=True):
        1. Activate conda
        2. Restore project files (/root/my_repo -> /testbed)
        3. Re-apply env var setup (if any)
        
        Args:
            container_id: Container ID
            specs_name: Repo spec name (for env setup command)
            timeout: Timeout in seconds; None for no timeout
            
        Returns:
            bool: Whether reset succeeded
        """
        specs = self._specs_cache.get(specs_name)
        if not specs:
            self.logger.error(f"specs_name not found: {specs_name}")
            return False
        
        # Build env setup command (same as run_container)
        docker_command = []  # Temp command list to compute env_setup_cmd
        env_setup_cmd, _ = self._add_docker_runtime_config(docker_command, specs)
        
        # Build reset command (same as run_container)
        conda_activate_cmd = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        restore_project_cmd = "rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/"
        
        # If env_setup_cmd exists, re-activate conda after source
        if env_setup_cmd:
            reset_command = f'{conda_activate_cmd} && {restore_project_cmd}{env_setup_cmd} && {conda_activate_cmd}'
        else:
            reset_command = f'{conda_activate_cmd} && {restore_project_cmd}'
        
        try:
            # Execute reset command in container
            result = subprocess.run(
                ['docker', 'exec', container_id, 'bash', '-c', reset_command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return True
            
        except subprocess.TimeoutExpired:
            tqdm.write(f"❌ Container {container_id[:12]} env reset timed out ({timeout}s)")
            return False
        except subprocess.CalledProcessError as e:
            tqdm.write(f"❌ Container {container_id[:12]} env reset failed: {e.stderr}")
            return False
        except Exception as e:
            tqdm.write(f"❌ Container {container_id[:12]} env reset error: {e}")
            return False
    
    def stop_container(self, container_id: str, force: bool = False) -> None:
        """
        Stop and remove a container.
        
        Args:
            container_id: Container ID
            force: Force kill without graceful stop (use on timeout)
        """
        # Release GPUs used by container before stop
        gpu_ids = self._container_gpu_map.get(container_id)
        if gpu_ids:
            release_gpu(gpu_ids, self.logger)
            del self._container_gpu_map[container_id]
            self.logger.debug(f"Container {container_id[:12]} released GPUs {gpu_ids}")
        
        if force:
            # Force kill container (no wait)
            subprocess.run(['docker', 'kill', container_id], capture_output=True)
        else:
            # Graceful stop
            subprocess.run(['docker', 'stop', container_id], capture_output=True)
        subprocess.run(['docker', 'rm', container_id], capture_output=True)
