"""Level2 post-checker - verify masked + shell code against level2 tests."""

import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

from featurebench.utils.logger import create_level2_validator_logger
from featurebench.utils.parser.pytest_parser import PytestParser
from featurebench.utils.command_utils import apply_uv_run_prefix


@dataclass
class Level2ValidationResult:
    """Level2 validation result."""
    success: bool                      # Whether post-check ran normally (entered tests)
    pass_rate: float = 0.0             # Pass rate (passed / (passed + failed + error))
    passed_count: int = 0              # Passed test count
    failed_count: int = 0              # Failed test count
    error_count: int = 0               # Error test count
    xfailed_count: int = 0             # Expected failure count
    xpassed_count: int = 0             # Unexpected pass count
    total_count: int = 0               # Total test count
    error_message: Optional[str] = None  # Error message
    log_file: Optional[str] = None     # Log file path


class Level2Validator:
    """Level2 post-checker for shell code against level2 tests."""
    
    def __init__(
        self,
        config,
        repo_manager,
        image_manager,
        storage_manager,
        data_item,
        level2_dir,
        logger
    ):
        """
        Initialize Level2 validator.
        
        Args:
            config: Config object
            repo_manager: Repo manager
            image_manager: Image manager
            storage_manager: Storage manager
            data_item: Data item
            level2_dir: Level2 case dir (contains test_patch.diff)
            logger: Logger
        """
        self.config = config
        self.repo_manager = repo_manager
        self.image_manager = image_manager
        self.storage_manager = storage_manager
        self.data_item = data_item
        self.level2_dir = Path(level2_dir)
        self.logger = logger
        self.logs_dir = config.logs_dir
    
    def run(self) -> Level2ValidationResult:
        """
        Run Level2 validation.
        
        Returns:
            Level2ValidationResult: Validation result
        """
        try:
            test_file_name = Path(self.data_item.file_path).name
            
            # Run level2 post-check
            result = self._validate_level2_test()
            
            if result.success:
                # Read level2 pass-rate threshold (same as f2p threshold)
                f2p_pass_rate_threshold = self.data_item.specs.get("f2p_pass_rate_threshold")
                is_passed = result.pass_rate <= f2p_pass_rate_threshold
                if is_passed:
                    pass
                    # tqdm.write(f"✅ {self.data_item.repo}: Level2 post-verify passed - {test_file_name}, "
                    #          f"pass rate: {result.pass_rate:.2%}, total: {result.total_count} (threshold: {f2p_pass_rate_threshold:.2%}), log: {result.log_file}")
                else:
                    result.error_message = (
                        f"Level2 post-validation pass rate too high - "
                        f"pass rate: {result.pass_rate:.2%} > threshold: {f2p_pass_rate_threshold:.2%}, "
                        f"log: {result.log_file}, cannot generate Level2 data"
                    )
                    # tqdm.write(f"❌ {self.data_item.repo}: Level2 post-verify pass rate too high - {test_file_name}, "
                    #          f"pass rate: {result.pass_rate:.2%}, total: {result.total_count} (threshold: {f2p_pass_rate_threshold:.2%}), log: {result.log_file}")
            else:
                is_passed = False
                # Error details are logged at the failure site

            return result, is_passed
            
        except Exception as e:
            error_msg = f"Level2 validation failed - {test_file_name}: {e}"
            tqdm.write(f"❌ {self.data_item.repo}: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _validate_level2_test(self) -> Level2ValidationResult:
        """
        Validate a single level2 test.
        
        Returns:
            Level2ValidationResult: Validation result
        """
        container_id = None
        is_timeout = False
        
        # Create log file path
        log_file_path = create_level2_validator_logger(
            specs_name=self.data_item.repo,
            test_file=self.data_item.file_path,
            logs_dir=self.logs_dir
        )
                
        try:
            # 1. Start container
            container_id = self.image_manager.run_container(
                specs_name=self.data_item.repo,
                working_dir="/testbed",
                prepare_env=True
            )
            
            # 2. Read test_patch.diff and copy into container
            test_patch_file = self.level2_dir / "test_patch.diff"
            if not test_patch_file.exists():
                raise FileNotFoundError(f"test_patch.diff does not exist: {test_patch_file}")
            
            # Copy test_patch.diff into /tmp inside container
            self.image_manager.copy_to_container(
                container_id,
                str(test_patch_file),
                "/tmp/test_patch.diff"
            )
            
            # 3. Apply patch inside container
            apply_patch_cmd = "git apply /tmp/test_patch.diff"
            apply_result = self.image_manager.exec_in_container(
                container_id,
                apply_patch_cmd,
                timeout=30
            )
            
            if apply_result.returncode != 0:
                raise RuntimeError(f"Failed to apply test_patch.diff: {apply_result.stderr}")
            
            # 4. Install dummy agent_code to avoid import errors
            install_dummy_cmd = (
                'python -c "import site, os; '
                'd = os.path.join(site.getsitepackages()[0], \'agent_code\'); '
                'os.makedirs(d, exist_ok=True); '
                'open(os.path.join(d, \'__init__.py\'), \'w\').close()"'
            )
            try:
                self.image_manager.exec_in_container(
                    container_id,
                    install_dummy_cmd,
                    timeout=30
                )
            except Exception as e:
                self.logger.warning(f"Failed to install dummy agent_code package: {e}")
            
            # 5. Run test file
            test_cmd = apply_uv_run_prefix(self.data_item.specs.get("test_cmd"), self.data_item.specs)
            timeout_test = self.data_item.specs.get("timeout_run", 300)
            
            # Build full test command
            full_test_cmd = f"{test_cmd} {self.data_item.file_path}"
            
            test_result = self.image_manager.exec_in_container(
                container_id,
                full_test_cmd,
                timeout=timeout_test,
                log_file_path=log_file_path
            )
            
            # 6. Parse test results
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                test_output = f.read()
            
            pytest_result = PytestParser.parse_output(test_output)
            
            # Success if return code is 0 or 1 (tests executed)
            # 0: all passed
            # 1: failures (tests completed)
            # 2: collection/execution interrupted (import/syntax errors)
            # 3: pytest internal error
            # 4: CLI usage error
            # 5: no tests collected
            success = pytest_result.return_code in [0, 1]
            
            if not success:

                # If error is agent_code attribute in test definition, treat as pass
                if pytest_result.return_code == 2:
                    if pytest_result.errors_detail:
                        agent_code_attribute_error = True
                        # All errors must be agent_code attribute errors
                        for error_type, error_message in pytest_result.errors_detail:
                            if error_type == "AttributeError" and "agent_code" in error_message:
                                continue
                            else:
                                agent_code_attribute_error = False
                                break
                        if agent_code_attribute_error:
                            return Level2ValidationResult(
                                success=True,
                                error_message=None,
                                pass_rate=0.0,
                                passed_count=0,
                                failed_count=0,
                                error_count=0,
                                xfailed_count=0,
                                xpassed_count=0,
                                total_count=0,
                                log_file=str(log_file_path)
                            )

                error_messages = {
                    2: "Test collection/execution interrupted (import or syntax error)",
                    3: "Pytest internal error",
                    4: "CLI usage error",
                    5: "No tests collected",
                    -1: "Unable to parse return code"
                }
                error_msg = error_messages.get(
                    pytest_result.return_code,
                    f"Unknown error (return code: {pytest_result.return_code})"
                )
                tqdm.write(
                    f"❌ {self.data_item.repo}: {error_msg} - {self.data_item.file_path}, log: {log_file_path}"
                )
                return Level2ValidationResult(
                    success=False,
                    error_message=error_msg,
                    pass_rate=pytest_result.pass_rate,
                    passed_count=pytest_result.passed,
                    failed_count=pytest_result.failed,
                    error_count=pytest_result.error,
                    xfailed_count=pytest_result.xfailed,
                    xpassed_count=pytest_result.xpassed,
                    total_count=pytest_result.total,
                    log_file=str(log_file_path)
                )
            
            return Level2ValidationResult(
                success=True,
                pass_rate=pytest_result.pass_rate,
                passed_count=pytest_result.passed,
                failed_count=pytest_result.failed,
                error_count=pytest_result.error,
                xfailed_count=pytest_result.xfailed,
                xpassed_count=pytest_result.xpassed,
                total_count=pytest_result.total,
                log_file=str(log_file_path)
            )
            
        except subprocess.TimeoutExpired:
            is_timeout = True
            tqdm.write(f"❌ {self.data_item.repo}: Level2 test timed out - {self.data_item.file_path}")
            error_msg = f"Level2 test timed out, log: {log_file_path}"
            return Level2ValidationResult(
                success=False,
                error_message=error_msg,
                log_file=str(log_file_path)
            )
        
        except Exception as e:
            tqdm.write(f"❌ {self.data_item.repo}: Level2 validation error: {e} - {self.data_item.file_path}")
            error_msg = f"Level2 validation error: {e}, log: {log_file_path}"
            return Level2ValidationResult(
                success=False,
                error_message=error_msg,
                log_file=str(log_file_path)
            )
        
        finally:
            # Cleanup container
            if container_id:
                self.image_manager.stop_container(container_id, force=is_timeout)
