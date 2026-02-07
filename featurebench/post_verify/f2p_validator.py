"""F2P post-checker - verify masked code against f2p tests."""

import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm
import tempfile
import os

from featurebench.utils.logger import create_f2p_validator_logger
from featurebench.utils.parser.pytest_parser import PytestParser
from featurebench.utils.command_utils import apply_uv_run_prefix


@dataclass
class F2PValidationResult:
    """F2P validation result."""
    success: bool                      # Whether f2p post-check ran normally (entered tests)
    pass_rate: float = 0.0             # Pass rate (passed / (passed + failed + error))
    passed_count: int = 0              # Passed test count
    failed_count: int = 0              # Failed test count
    error_count: int = 0               # Error test count
    xfailed_count: int = 0             # Expected failure count
    xpassed_count: int = 0             # Unexpected pass count
    total_count: int = 0               # Total test count
    error_message: Optional[str] = None  # Error message
    log_file: Optional[str] = None     # Log file path


class F2PValidator:
    """F2P post-checker for masked code against f2p tests."""
    
    def __init__(
        self,
        config,
        repo_manager,
        image_manager,
        storage_manager,
        data_item,
        mask_results,
        logger
    ):
        """
        Initialize F2P validator.
        
        Args:
            config: Config object
            repo_manager: Repo manager
            image_manager: Image manager
            storage_manager: Storage manager
            data_item: Data item
            mask_results: Mask results dict
            logger: Logger
        """
        self.config = config
        self.repo_manager = repo_manager
        self.image_manager = image_manager
        self.storage_manager = storage_manager
        self.data_item = data_item
        self.mask_results = mask_results
        self.logger = logger
        self.logs_dir = config.logs_dir
    
    def run(self) -> F2PValidationResult:
        """
        Run F2P validation.
        
        Returns:
            F2PValidationResult: Validation result
        """
        try:
            test_file_name = Path(self.data_item.file_path).name
            
            # Run f2p post-check
            result = self._validate_f2p_test()
            
            if result.success:
                # Read f2p pass-rate threshold from specs
                f2p_pass_rate_threshold = self.data_item.specs.get("f2p_pass_rate_threshold")
                is_passed = result.pass_rate <= f2p_pass_rate_threshold
                if is_passed:
                    pass
                    # tqdm.write(f"✅ {self.data_item.repo}: F2P post-verify passed - {test_file_name}, "
                    #          f"pass rate: {result.pass_rate:.2%}, total: {result.total_count} (threshold: {f2p_pass_rate_threshold:.2%}), log: {result.log_file}")
                else:
                    result.error_message = (
                        f"F2P post-validation pass rate too high - "
                        f"pass rate: {result.pass_rate:.2%} > threshold: {f2p_pass_rate_threshold:.2%}, "
                        f"log: {result.log_file}"
                    )
                    # tqdm.write(f"❌ {self.data_item.repo}: F2P post-verify pass rate too high - {test_file_name}, "
                    #          f"pass rate: {result.pass_rate:.2%}, total: {result.total_count} (threshold: {f2p_pass_rate_threshold:.2%}), log: {result.log_file}")
            else:
                is_passed = False
                # Error details are logged at the failure site

            return result, is_passed
            
        except Exception as e:
            error_msg = f"F2P validation failed - {test_file_name}: {e}"
            tqdm.write(f"❌ {self.data_item.repo}: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _validate_f2p_test(self) -> F2PValidationResult:
        """
        Validate a single f2p test.
        
        Returns:
            F2PValidationResult: Validation result
        """
        container_id = None
        is_timeout = False
        
        # Create log file path
        log_file_path = create_f2p_validator_logger(
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
            
            # 2. Copy all mask_results files into container
            for container_path, mask_result in self.mask_results.items():
                if not mask_result.success:
                    continue
                
                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(mask_result.masked_code)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Copy into container
                    self.image_manager.copy_to_container(
                        container_id,
                        tmp_file_path,
                        container_path
                    )
                finally:
                    # Delete temp file
                    os.unlink(tmp_file_path)
            
            # 3. Run f2p tests
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
            
            # 4. Parse test results
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
                return F2PValidationResult(
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
            
            return F2PValidationResult(
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
            tqdm.write(f"❌ {self.data_item.repo}: F2P test timed out - {self.data_item.file_path}")
            error_msg = f"F2P test timed out, log: {log_file_path}"
            return F2PValidationResult(
                success=False,
                error_message=error_msg,
                log_file=str(log_file_path)
            )
        
        except Exception as e:
            tqdm.write(f"❌ {self.data_item.repo}: F2P validation error: {e} - {self.data_item.file_path}")
            error_msg = f"F2P validation error: {e}, log: {log_file_path}"
            return F2PValidationResult(
                success=False,
                error_message=error_msg,
                log_file=str(log_file_path)
            )
        
        finally:
            # Cleanup container
            if container_id:
                self.image_manager.stop_container(container_id, force=is_timeout)
