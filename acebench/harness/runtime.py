"""
Runtime handlers for ACE-Bench evaluation.
Contains Level 1 and Level 2 evaluation logic.
"""

import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from docker.models.containers import Container

from acebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    LOG_TEST_OUTPUT,
    UTF8,
)
from acebench.harness.container import copy_to_container, exec_run_with_timeout
from acebench.harness.utils import (
    EvaluationError,
    build_test_command,
    get_test_config_from_repo_settings,
    parse_repo_settings,
)
from acebench.harness.test_parsers import MAP_REPO_TO_TEST_CMD
from acebench.utils.command_utils import apply_uv_run_prefix


def _normalize_patch_content(patch_content: str) -> str:
    # Some LLM outputs omit the final newline; `git apply` may report
    # `error: corrupt patch at line ...` when the patch file ends abruptly.
    if patch_content and not patch_content.endswith("\n"):
        return patch_content + "\n"
    return patch_content

def run_instance_level1(
    instance: pd.Series,
    pred: dict,
    container: Container,
    logger: logging.Logger,
    log_dir: Path,
    timeout: int | None = None,
    white: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation for Level 1 instance.

    Level 1 workflow:
    1. Activate conda environment
    2. Restore project from /root/my_repo/
    3. Apply patch (for masking) and directly delete F2P test files
    4. Reinitialize git
    5. Apply agent's patch
    6. Delete any generated F2P files, restore F2P test, and run tests

    Args:
        instance: Instance data (from HuggingFace)
        pred: Prediction data with patch
        container: Docker container
        logger: Logger instance
        log_dir: Directory to save test outputs
        timeout: Test timeout in seconds

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo_name = instance['repo']
    logger.info(f"Starting Level 1 evaluation for {instance_id}")

    results = {
        "instance_id": instance_id,
        "level": 1,
        "patch_applied": False,
        "f2p_success": False,
        "p2p_success": False,
        "error": None,
    }

    try:
        # Step 1: Activate conda and restore project
        logger.info("Step 1: Activating conda environment and restoring project")
        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(container, cmd, timeout=600)
        logger.info(f"Restore project exit code: {exit_code}")
        logger.info(f"Output: {output.decode(UTF8, errors='replace')}")

        if exit_code != 0:
            raise EvaluationError(f"Failed to restore project: {output.decode(UTF8, errors='replace')}")

        # Step 2: Apply patch (for masking) and test_patch (for deleting F2P test)
        logger.info("Step 2: Applying patch to mask files")
        patch_content = instance.get('patch', '')

        if patch_content and patch_content.strip():
            patch_content = _normalize_patch_content(patch_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(patch_content)
                temp_patch_path = f.name

            try:
                patch_path_container = "/tmp/mask_patch.diff"
                copy_to_container(container, temp_patch_path, patch_path_container)
                logger.info(f"Copied mask patch to container: {patch_path_container}")

                apply_cmd = f"cd /testbed && git apply --whitespace=fix {patch_path_container}"
                exit_code, output = exec_run_with_timeout(container, apply_cmd, timeout=120)

                patch_output = output.decode(UTF8, errors='replace')
                logger.info(f"Mask patch apply exit code: {exit_code}")
                logger.info(f"Mask patch output: {patch_output}")

                if exit_code != 0:
                    logger.warning(f"Failed to apply mask patch: {patch_output}")
                else:
                    logger.info("Successfully applied mask patch")
            finally:
                os.unlink(temp_patch_path)
        else:
            logger.info("No patch to apply for masking")

        # Step 2b: Prepare test_patch for later use and delete F2P test files (unless white-box is enabled)
        if white:
            logger.info("Step 2b: White-box enabled; keeping FAIL_TO_PASS test files visible")
        else:
            logger.info("Step 2b: Preparing test_patch and deleting F2P test files")

        test_patch_content = instance.get('test_patch', '')
        test_patch_path_container = None

        if test_patch_content and test_patch_content.strip():
            test_patch_content = _normalize_patch_content(test_patch_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(test_patch_content)
                temp_test_patch_path = f.name

            try:
                test_patch_path_container = "/tmp/test_patch.diff"
                copy_to_container(container, temp_test_patch_path, test_patch_path_container)
                logger.info(f"Saved test_patch to container: {test_patch_path_container} (for later reverse apply)")
            finally:
                os.unlink(temp_test_patch_path)

        # Delete F2P test files (skip in white-box mode)
        if not white:
            fail_to_pass = instance.get('FAIL_TO_PASS', [])
            if fail_to_pass:
                f2p_tests = fail_to_pass if isinstance(fail_to_pass, list) else [fail_to_pass]

                for f2p_test in f2p_tests:
                    if not f2p_test.startswith('/testbed/'):
                        f2p_test_path = f'/testbed/{f2p_test}'
                    else:
                        f2p_test_path = f2p_test

                    logger.info(f"Deleting F2P test file: {f2p_test_path}")
                    delete_cmd = f"rm -f {f2p_test_path}"
                    exit_code, output = exec_run_with_timeout(container, delete_cmd, timeout=60)

                    if exit_code != 0:
                        logger.warning(
                            f"Failed to delete F2P test file {f2p_test_path}: {output.decode(UTF8, errors='replace')}"
                        )
                    else:
                        logger.info(f"Successfully deleted F2P test file: {f2p_test_path}")
            else:
                logger.warning("No FAIL_TO_PASS tests found in instance, skipping test file deletion")

        # Step 3: Reinitialize git repository
        logger.info("Step 3: Reinitializing git repository")
        git_cmds = [
            "cd /testbed && rm -rf .git",
            "cd /testbed && git init",
            'cd /testbed && git config user.email "ace@bench.com"',
            'cd /testbed && git config user.name "ACE Bench"',
            'cd /testbed && git add -A',
            'cd /testbed && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty',
        ]

        for cmd in git_cmds:
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)
            if exit_code != 0:
                logger.warning(f"Git command failed: {cmd}")
                logger.warning(f"Output: {output.decode(UTF8, errors='replace')}")

        # Step 4: Apply agent's patch
        logger.info("Step 4: Applying agent patch")
        patch_content = pred[KEY_PREDICTION]

        if not patch_content or patch_content.strip() == "":
            logger.warning("Empty patch provided")
            results["error"] = "Empty patch"
            return results

        patch_content = _normalize_patch_content(patch_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False, encoding=UTF8) as f:
            f.write(patch_content)
            temp_patch_path = f.name

        try:
            patch_path_container = "/tmp/agent_patch.diff"
            copy_to_container(container, temp_patch_path, patch_path_container)
            logger.info(f"Copied patch file to container: {patch_path_container}")
        finally:
            os.unlink(temp_patch_path)

        apply_cmd = f"cd /testbed && git apply --whitespace=fix --verbose {patch_path_container}"
        exit_code, output = exec_run_with_timeout(container, apply_cmd, timeout=120)

        patch_output = output.decode(UTF8, errors='replace')
        logger.info(f"Patch apply exit code: {exit_code}")
        logger.info(f"Patch output: {patch_output}")

        if exit_code != 0:
            logger.error(f"{APPLY_PATCH_FAIL}")
            results["error"] = f"Patch apply failed: {patch_output}"
            return results

        logger.info(f"{APPLY_PATCH_PASS}")
        results["patch_applied"] = True

        # Step 5: Prepare for testing - delete generated F2P files and restore F2P test
        logger.info("Step 5: Preparing for testing - cleaning up and restoring F2P test")

        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if not fail_to_pass:
            logger.error("No FAIL_TO_PASS tests found in instance")
            results["error"] = "No FAIL_TO_PASS tests"
            return results

        f2p_test_path = fail_to_pass[0] if isinstance(fail_to_pass, list) else fail_to_pass
        if not f2p_test_path.startswith('/testbed/'):
            f2p_test_path = f"/testbed/{f2p_test_path}"

        logger.info(f"F2P test path: {f2p_test_path}")

        # Delete F2P test file if exists
        delete_cmd = f"rm -f {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(container, delete_cmd, timeout=60)
        logger.info(f"Delete F2P file exit code: {exit_code}")

        # Restore F2P test file using reverse apply test_patch
        if test_patch_path_container:
            logger.info("Reverse applying test_patch to restore F2P test file")
            reverse_cmd = f"cd /testbed && git apply --reverse --whitespace=fix {test_patch_path_container}"
            exit_code, output = exec_run_with_timeout(container, reverse_cmd, timeout=120)

            reverse_output = output.decode(UTF8, errors='replace')
            logger.info(f"Reverse apply exit code: {exit_code}")
            logger.info(f"Reverse apply output: {reverse_output}")

            if exit_code != 0:
                logger.warning(f"Failed to reverse apply test_patch: {reverse_output}")
            else:
                logger.info("Successfully restored F2P test file using test_patch")
        else:
            logger.warning("No test_patch available to restore F2P test file")

        # Step 6: Run tests
        logger.info("Step 6: Running tests")

        # Get test configuration from repo_settings
        repo_settings = parse_repo_settings(instance)
        test_config = get_test_config_from_repo_settings(repo_settings)

        # Build test command with timeout_one if specified
        # the MAP_REPO_TO_TEST_CMD is preferred over the test_config["test_cmd"]
        base_test_cmd = MAP_REPO_TO_TEST_CMD.get(repo_name, None)

        if base_test_cmd is None:
            base_test_cmd = test_config["test_cmd"]
        timeout_one = test_config["timeout_one"]
        test_cmd = build_test_command(base_test_cmd, timeout_one)
        test_cmd = apply_uv_run_prefix(test_cmd, test_config)

        # Use timeout_run from repo_settings if not overridden
        effective_timeout = timeout if timeout else test_config.get("timeout_run", 1800)

        logger.info(f"Test command: {test_cmd}")
        logger.info(f"Test timeout: {effective_timeout}s")

        pass_to_pass = instance.get('PASS_TO_PASS', [])
        p2p_tests = pass_to_pass if isinstance(pass_to_pass, list) else []

        # Run F2P test
        logger.info(f"Running F2P test: {f2p_test_path}")
        test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(
            container, test_cmd_full, timeout=effective_timeout
        )

        f2p_output = output.decode(UTF8, errors='replace')
        logger.info(f"F2P test exit code: {exit_code}")
        logger.info(f"F2P output (truncated): {f2p_output[:500]}")

        results["f2p_success"] = (exit_code == 0)

        # Save F2P test output
        test_output_file = log_dir / LOG_TEST_OUTPUT
        with open(test_output_file, "w", encoding=UTF8) as f:
            f.write(f2p_output)
        logger.info(f"Saved F2P test output to {test_output_file}")

        # Run P2P tests
        if p2p_tests:
            logger.info(f"Running {len(p2p_tests)} P2P tests")
            p2p_results = []
            for p2p_test in p2p_tests:
                p2p_test_path = p2p_test
                if not p2p_test_path.startswith('/testbed/'):
                    p2p_test_path = f"/testbed/{p2p_test_path}"

                logger.info(f"Running P2P test: {p2p_test_path}")

                test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {p2p_test_path}"
                exit_code, output = exec_run_with_timeout(
                    container, test_cmd_full, timeout=effective_timeout
                )

                p2p_output = output.decode(UTF8, errors='replace')
                logger.info(f"P2P test {p2p_test_path} exit code: {exit_code}")

                p2p_results.append(exit_code == 0)

                # Save P2P test output
                test_file_name = os.path.basename(p2p_test_path).replace('.py', '')
                p2p_output_file = log_dir / f"test_output_p2p_{test_file_name}.txt"
                with open(p2p_output_file, "w", encoding=UTF8) as f:
                    f.write(p2p_output)
                logger.info(f"Saved P2P test output to {p2p_output_file}")

            results["p2p_success"] = all(p2p_results)
        else:
            results["p2p_success"] = True

        return results

    except Exception as e:
        logger.error(f"Error in Level 1 evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results


def run_instance_level2(
    instance: pd.Series,
    pred: dict,
    container: Container,
    logger: logging.Logger,
    log_dir: Path,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for Level 2 instance.

    Level 2 workflow:
    1. Activate conda environment
    2. Clean /testbed/ and initialize git
    3. Apply agent's patch
    4. Install agent's implementation (pip install .)
    5. Restore original project from /root/my_repo/
    6. Copy masked test files
    7. Run F2P tests only

    Args:
        instance: Instance data
        pred: Prediction data with patch
        container: Docker container
        logger: Logger instance
        log_dir: Directory to save test outputs
        timeout: Test timeout in seconds

    Returns:
        Dictionary with evaluation results
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo_name = instance['repo']
    logger.info(f"Starting Level 2 evaluation for {instance_id}")

    results = {
        "instance_id": instance_id,
        "level": 2,
        "patch_applied": False,
        "install_success": False,
        "f2p_success": False,
        "error": None,
    }

    try:
        # Step 1: Activate conda environment
        logger.info("Step 1: Activating conda environment")
        cmd = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)

        if exit_code != 0:
            raise EvaluationError(f"Failed to activate conda: {output.decode(UTF8, errors='replace')}")

        # Step 2: Clean /testbed/ and initialize git
        logger.info("Step 2: Cleaning /testbed/ and initializing git")
        init_cmds = [
            "rm -rf /testbed/* /testbed/.*  2>/dev/null || true",
            "mkdir -p /testbed",
            "cd /testbed && git init",
            'cd /testbed && git config user.email "ace@bench.com"',
            'cd /testbed && git config user.name "ACE Bench"',
            'cd /testbed && echo "put all codes in this folder" > README.md',
            'cd /testbed && git add -A',
            'cd /testbed && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty',
        ]

        for cmd in init_cmds:
            exit_code, output = exec_run_with_timeout(container, cmd, timeout=60)
            if exit_code != 0 and "rm -rf" not in cmd:
                logger.warning(f"Init command failed: {cmd}")
                logger.warning(f"Output: {output.decode(UTF8, errors='replace')}")

        # Step 3: Apply agent's patch
        logger.info("Step 3: Applying agent patch")
        patch_content = pred[KEY_PREDICTION]

        if not patch_content or patch_content.strip() == "":
            logger.warning("Empty patch provided")
            results["error"] = "Empty patch"
            return results

        patch_content = _normalize_patch_content(patch_content)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False, encoding=UTF8) as f:
            f.write(patch_content)
            temp_patch_path = f.name

        try:
            patch_path_container = "/tmp/agent_patch.diff"
            copy_to_container(container, temp_patch_path, patch_path_container)
            logger.info(f"Copied patch file to container: {patch_path_container}")
        finally:
            os.unlink(temp_patch_path)

        apply_cmd = f"cd /testbed && git apply --whitespace=fix --verbose {patch_path_container}"
        exit_code, output = exec_run_with_timeout(container, apply_cmd, timeout=120)

        patch_output = output.decode(UTF8, errors='replace')
        logger.info(f"Patch apply exit code: {exit_code}")
        logger.info(f"Patch output: {patch_output}")

        if exit_code != 0:
            logger.error(f"{APPLY_PATCH_FAIL}")
            results["error"] = f"Patch apply failed: {patch_output}"
            return results

        logger.info(f"{APPLY_PATCH_PASS}")
        results["patch_applied"] = True

        # Step 4: Install agent's implementation
        logger.info("Step 4: Installing agent's implementation")
        install_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "cd /testbed && "
            "pip install ."
        )
        exit_code, output = exec_run_with_timeout(
            container, install_cmd, timeout=600
        )

        install_output = output.decode(UTF8, errors='replace')
        logger.info(f"Install exit code: {exit_code}")
        logger.info(f"Install output (truncated): {install_output[:500]}")

        if exit_code != 0:
            logger.warning(f"Installation failed, but continuing: {install_output}")
        else:
            results["install_success"] = True

        # Step 5: Restore original project
        logger.info("Step 5: Restoring original project")
        restore_cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_with_timeout(
            container, restore_cmd, timeout=600
        )

        if exit_code != 0:
            raise EvaluationError(f"Failed to restore project: {output.decode(UTF8, errors='replace')}")

        # Step 6: Apply test_patch to modify test files
        logger.info("Step 6: Applying test_patch to modify test files")
        test_patch_content = instance.get('test_patch', '')

        if test_patch_content and test_patch_content.strip():
            test_patch_content = _normalize_patch_content(test_patch_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding=UTF8) as f:
                f.write(test_patch_content)
                temp_test_patch_path = f.name

            try:
                test_patch_path_container = "/tmp/test_patch.diff"
                copy_to_container(container, temp_test_patch_path, test_patch_path_container)
                logger.info(f"Copied test_patch to container: {test_patch_path_container}")

                apply_cmd = f"cd /testbed && git apply --whitespace=fix {test_patch_path_container}"
                exit_code, output = exec_run_with_timeout(container, apply_cmd, timeout=120)

                test_patch_output = output.decode(UTF8, errors='replace')
                logger.info(f"Test patch apply exit code: {exit_code}")
                logger.info(f"Test patch output: {test_patch_output}")

                if exit_code != 0:
                    logger.warning(f"Failed to apply test_patch: {test_patch_output}")
                else:
                    logger.info("Successfully applied test_patch")
            finally:
                os.unlink(temp_test_patch_path)
        else:
            logger.info("No test_patch to apply")

        # Step 7: Run F2P test
        logger.info("Step 7: Running F2P tests")

        # Get test configuration from repo_settings
        repo_settings = parse_repo_settings(instance)
        test_config = get_test_config_from_repo_settings(repo_settings)

        # Build test command with timeout_one if specified
        # the MAP_REPO_TO_TEST_CMD is preferred over the test_config["test_cmd"]
        base_test_cmd = MAP_REPO_TO_TEST_CMD.get(repo_name, None)

        if base_test_cmd is None:
            base_test_cmd = test_config["test_cmd"]
        timeout_one = test_config["timeout_one"]
        test_cmd = build_test_command(base_test_cmd, timeout_one)
        test_cmd = apply_uv_run_prefix(test_cmd, test_config)

        # Use timeout_run from repo_settings if not overridden
        effective_timeout = timeout if timeout else test_config.get("timeout_run", 1800)

        logger.info(f"Test command: {test_cmd}")
        logger.info(f"Test timeout: {effective_timeout}s")

        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if not fail_to_pass:
            logger.error("No FAIL_TO_PASS tests found in instance")
            results["error"] = "No FAIL_TO_PASS tests"
            return results

        f2p_test = fail_to_pass[0] if isinstance(fail_to_pass, list) else fail_to_pass
        f2p_test_path = f2p_test
        if not f2p_test_path.startswith('/testbed/'):
            f2p_test_path = f"/testbed/{f2p_test_path}"

        logger.info(f"Running F2P test: {f2p_test_path}")
        test_cmd_full = f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && {test_cmd} {f2p_test_path}"
        exit_code, output = exec_run_with_timeout(
            container, test_cmd_full, timeout=effective_timeout
        )

        f2p_output = output.decode(UTF8, errors='replace')
        logger.info(f"F2P test exit code: {exit_code}")
        logger.info(f"F2P output (truncated): {f2p_output[:500]}")

        results["f2p_success"] = (exit_code == 0)

        # Save F2P test output
        test_output_file = log_dir / LOG_TEST_OUTPUT
        with open(test_output_file, "w", encoding=UTF8) as f:
            f.write(f2p_output)
        logger.info(f"Saved F2P test output to {test_output_file}")

        return results

    except Exception as e:
        logger.error(f"Error in Level 2 evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        return results

