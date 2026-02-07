import subprocess
import shutil
import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
import importlib.util


class RepoManager:
    """Repo manager for loading and managing local repos."""
    
    def __init__(self, repos_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize repo manager.
        
        Args:
            repos_dir: Repo storage directory, defaults to featurebench/resources/repos
            logger: Logger; if None, uses the default logger
        """
        if repos_dir is None:
            # Default to featurebench/resources/repos
            current_file = Path(__file__)
            featurebench_dir = current_file.parent.parent  # Go from utils/ back to featurebench/
            repos_dir = featurebench_dir / "resources" / "repos"
        
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided logger or create default
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Track loaded repos: {specs_name: {local_path: Path, specs: Dict}}
        # specs_name looks like SPECS_LITGPT
        self.loaded_repos: Dict[str, Dict] = {}
    
    def load(self, config):
        """
        Load repos required by config into local self.repos_dir.
        
        Args:
            config: Config instance
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Starting to load repos...")
        self.logger.info("=" * 60)
        
        # 1. Load all repo specs from config
        all_specs = self._load_specs_from_config(config.config_path)
        
        # 2. Determine repos to load (only --debug-repo if set)
        repos_to_load = self._determine_repos_to_load(all_specs, config.debug_repo)
        
        self.logger.info(f"Need to load {len(repos_to_load)} repos: {list(repos_to_load.keys())}")
        
        # 3. Load repos one by one
        for specs_name, specs in repos_to_load.items():
            try:
                self._load_single_repo(specs_name, specs)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load repo {specs_name}: {e}")
    
    def _load_specs_from_config(self, config_path: Path) -> Dict[str, Dict]:
        """
        Dynamically load all SPECS_ variables from config.
        
        Args:
            config_path: Config file path
            
        Returns:
            Dict of all repo specs {SPEC_xxx: specs(dict)}
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        
        # Dynamically import the config module
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract all SPECS_ variables
        all_specs = {}
        for attr_name in dir(config_module):
            if attr_name.startswith('SPECS_'):
                specs = getattr(config_module, attr_name)
                all_specs[attr_name] = specs
        
        self.logger.info(f"Loaded {len(all_specs)} repo specs from config")
        return all_specs
    
    def _determine_repos_to_load(self, all_specs: Dict[str, Dict], debug_repo: Optional[List[str]]) -> Dict[str, Dict]:
        """
        Determine repos to load (consider debug_repo).
        
        Args:
            all_specs: All repo specs, e.g. {SPEC_xxx: specs(dict)}
            debug_repo: Repos specified in debug mode
            
        Returns:
            Dict of repos to load
        """
        if debug_repo:
            # Debug mode: only load specified repos
            repos_to_load = {}
            for specs_name in debug_repo:
                if specs_name in all_specs:
                    repos_to_load[specs_name] = all_specs[specs_name]
                else:
                    raise ValueError(
                        f"Debug repo {specs_name} not found in config. "
                        f"Available repos: {list(all_specs.keys())}"
                    )
            return repos_to_load
        else:
            # Normal mode: load all repos
            return all_specs
    
    def _load_single_repo(self, specs_name: str, specs: Dict):
        """
        Load a single repo locally.
        
        Args:
            specs_name: Repo spec name (e.g., SPECS_LITGPT)
            specs: Repo spec config
        """
        repository = specs.get('repository')
        commit = specs.get('commit', None)
        clone_method = specs.get('clone_method', 'https')  # Default to https
        base_url = specs.get('base_url') or 'https://github.com'    # Default to GitHub
        clone_timeout = specs.get('clone_timeout', 600)
        
        # Compute local path: repos/{repository_name}
        # Example: Lightning-AI/litgpt -> repos/litgpt
        repo_dir_name = repository.split('/')[-1]
        local_path = self.repos_dir / repo_dir_name
        
        # If directory exists, check whether update is needed
        if local_path.exists():
            if self._is_repo_up_to_date(local_path, commit):
                # If commit is None, read latest commit and update specs
                if commit is None:
                    current_commit = self._get_current_commit(local_path)
                    specs['commit'] = current_commit
                else:
                    current_commit = commit
                
                self.logger.info(f"⏭️ Repo {specs_name} is up to date; skip clone (commit: {current_commit[:7]})")
                
                # Record repo info even if clone is skipped
                self.loaded_repos[specs_name] = {
                    'local_path': local_path,
                    'specs': specs
                }
                return
            else:
                self.logger.info(f"Repo {specs_name} needs update; recloning")
                shutil.rmtree(local_path)
        
        # Clone repository
        self._clone_repo(repository, local_path, clone_method, base_url, clone_timeout)
        
        # Checkout specified commit (if any)
        if commit:
            self._checkout_commit(local_path, commit)
            current_commit = commit
        else:
            # If commit not specified, record current HEAD commit in specs
            current_commit = self._get_current_commit(local_path)
            specs['commit'] = current_commit
        
        self.logger.info(f"✅ Repo {specs_name} loaded (commit: {current_commit[:7]})")
        
        # Record loaded repo info
        self.loaded_repos[specs_name] = {
            'local_path': local_path,
            'specs': specs
        }
    
    def _is_repo_up_to_date(self, local_path: Path, commit: Optional[str]) -> bool:
        """
        Check whether local repo matches the specified commit.
        
        Args:
            local_path: Local repo path
            commit: Target commit; if None, treat existing as latest
            
        Returns:
            Whether the repo matches the commit
        """
        # Check for a valid git repo by verifying .git directory
        git_dir = local_path / ".git"
        if not git_dir.exists():
            return False
        
        # Check for HEAD file (valid git repo signal)
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return False
        
        # Check repo has commits (not empty)
        result = subprocess.run(
            ['git', '--git-dir', str(git_dir), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            # No commits means empty repo
            return False
        
        current_commit = result.stdout.strip()
        
        if commit:
            # Check current commit matches target
            return current_commit.startswith(commit)
        else:
            # If no commit specified, any content is treated as latest
            return True
    
    def _get_current_commit(self, local_path: Path) -> Optional[str]:
        """
        Get the current commit hash of the local repo.
        
        Args:
            local_path: Local repo path
            
        Returns:
            Current commit hash; raises if it fails
        """
        git_dir = local_path / ".git"

        result = subprocess.run(
            ['git', '--git-dir', str(git_dir), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(
                f"Failed to get current commit for repo {local_path}, error: {result.stderr.strip()}"
            )
    
    def _clone_repo(self, repository: str, local_path: Path, clone_method: str = 'https', base_url: str = 'https://github.com', timeout: int = 600):
        """
        Clone a remote repo locally.
        
        Args:
            repository: Remote repo address (e.g., Lightning-AI/litgpt)
            local_path: Local clone path
            clone_method: Clone method, 'https' or 'ssh'
            base_url: Base URL, defaults to https://github.com
        """
        # Ensure parent dir exists without creating target dir
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if clone_method == 'ssh':
            domain = base_url.replace('https://', '').replace('http://', '').rstrip('/')
            git_url = f"git@{domain}:{repository}.git"
        else:
            git_url = f"{base_url.rstrip('/')}/{repository}.git"
        
        self.logger.info(f"Trying to clone repo from {git_url}...")
        
        result = subprocess.run(
            ['git', 'clone', git_url, str(local_path)],
            cwd=local_path.parent,  # Run clone in parent directory
            capture_output=False,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            self.logger.info(f"Successfully cloned repo from {git_url}")
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"Failed to clone repo {repository}, error: {error_msg}")
    
    def _checkout_commit(self, local_path: Path, commit: str):
        """
        Checkout to the specified commit.
        
        Args:
            local_path: Local repo path
            commit: Target commit hash
        """
        result = subprocess.run(
            ['git', 'checkout', commit],
            cwd=local_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            self.logger.info(f"Checked out to commit {commit}")
        else:
            error_msg = result.stderr.strip()
            raise RuntimeError(f"Failed to checkout commit {commit}: {error_msg}")
    
    def get_file_git_last_modified_time(self, specs_name: str, file_path: str) -> Optional[float]:
        """
        Get last modified time of a file in Git.
        
        Args:
            specs_name: Repo spec name (e.g., SPECS_LITGPT)
            file_path: File path (container path /testbed/xxx or relative)
            
        Returns:
            Unix timestamp; None if unavailable
        """
        try:
            # Get repo local path
            if specs_name not in self.loaded_repos:
                self.logger.warning(f"Repo {specs_name} is not loaded")
                return None
            
            local_path = self.loaded_repos[specs_name]['local_path']
            
            # Convert container path to relative path (if needed)
            if file_path.startswith('/testbed/'):
                relative_path = file_path.replace('/testbed/', '')
            else:
                relative_path = file_path
            
            # Use git log to get last commit time for file
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ct', relative_path],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to get Git time for {file_path}: {e}")
            return None
    
    def get_file_git_first_commit_time(self, specs_name: str, file_path: str) -> Optional[float]:
        """
        Get first commit time of a file in Git.
        
        Args:
            specs_name: Repo spec name (e.g., SPECS_LITGPT)
            file_path: File path (container path /testbed/xxx or relative)
            
        Returns:
            Unix timestamp; None if unavailable
        """
        try:
            # Get repo local path
            if specs_name not in self.loaded_repos:
                self.logger.warning(f"Repo {specs_name} is not loaded")
                return None
            
            local_path = self.loaded_repos[specs_name]['local_path']
            
            # Convert container path to relative path (if needed)
            if file_path.startswith('/testbed/'):
                relative_path = file_path.replace('/testbed/', '')
            else:
                relative_path = file_path
            
            # Use git log to get first commit time (--reverse to pick earliest)
            result = subprocess.run(
                ['git', 'log', '--reverse', '--format=%ct', relative_path],
                cwd=str(local_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Take first line (earliest commit)
                first_line = result.stdout.strip().split('\n')[0]
                return float(first_line)
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to get first Git commit time for {file_path}: {e}")
            return None
    
    def convert_container_path_to_local(self, specs_name: str, container_path: str) -> str:
        """
        Convert a container path to a local path.
        
        Args:
            specs_name: Repo spec name
            container_path: Container path (e.g., /testbed/test/test_foo.py)
            
        Returns:
            str: Local path
        """
        repo_path = self.loaded_repos[specs_name]['local_path']
        relative_path = Path(container_path.replace('/testbed/', ''))
        local_path = repo_path / relative_path
        return str(local_path)