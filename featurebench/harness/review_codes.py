"""
Review codes module for FeatureBench evaluation.

This module provides functionality to save the agent-generated code for review.
"""

import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import docker
import pandas as pd

from featurebench.harness.container import copy_to_container, exec_run_with_timeout
from featurebench.utils.identifiers import normalize_instance_id

UTF8 = "utf-8"


def extract_changed_files_from_patch(patch_content: str) -> list[str]:
    """
    Extract list of changed files from a git patch.

    Args:
        patch_content: Git patch content

    Returns:
        List of file paths that were changed
    """
    changed_files = []
    lines = patch_content.split("\n")

    for line in lines:
        # Match lines like: diff --git a/path/to/file.py b/path/to/file.py
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)", line)
            if match:
                # Use the 'b/' path (after change)
                file_path = match.group(2)
                changed_files.append("/testbed/" + file_path)

    return changed_files


def copy_from_container(
    container: docker.models.containers.Container, src_path: str, dst_path: Path
) -> None:
    """
    Copy directory from container to host.

    Args:
        container: Docker container
        src_path: Source path in container
        dst_path: Destination path on host
    """
    bits, _ = container.get_archive(src_path)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in bits:
            tmp_file.write(chunk)
        tmp_path = Path(tmp_file.name)

    try:
        with tarfile.open(tmp_path, "r") as tar:
            members = tar.getmembers()
            expected_root = dst_path.name
            if not members:
                raise ValueError(f"container archive for {src_path} is empty")
            for member in members:
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != expected_root:
                    raise ValueError(f"container archive for {src_path} has an unexpected root")

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=dst_path.parent) as extraction_dir:
                extraction_root = Path(extraction_dir)
                tar.extractall(extraction_root, members=members, filter="data")
                extracted_path = extraction_root / expected_root
                if not extracted_path.exists():
                    raise ValueError(f"container archive for {src_path} has no root entry")

                if dst_path.is_symlink() or dst_path.is_file():
                    dst_path.unlink()
                elif dst_path.exists():
                    shutil.rmtree(dst_path)
                extracted_path.replace(dst_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _copy_text_to_container(
    container: docker.models.containers.Container,
    content: str,
    destination: str,
) -> None:
    """Copy text to a container through the shared tar-based file helper."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=Path(destination).suffix,
        delete=False,
        encoding=UTF8,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    try:
        copy_to_container(container, temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def exec_run_simple(
    container: docker.models.containers.Container, cmd: str, timeout: int = 300
) -> tuple[int, str]:
    """
    Execute command in container and return exit code and output.

    Args:
        container: Docker container
        cmd: Command to execute
        timeout: Timeout in seconds

    Returns:
        Tuple of (exit_code, output_str)
    """
    exit_code, output = exec_run_with_timeout(
        container,
        cmd,
        timeout=timeout,
        user="root",
        workdir="/",
    )
    return exit_code, output.decode(UTF8, errors="replace")


def save_review_codes_level1(
    instance: pd.Series,
    patch_content: str,
    log_dir: Path,
    docker_image: str,
    logger: logging.Logger,
    labels: dict[str, str] | None = None,
) -> bool:
    """
    Save review codes for Level 1 instance.

    Level 1 workflow:
    1. Create container and restore project
    2. Apply instance patch (for masking)
    3. Apply the agent patch unchanged
    4. Remove Git metadata
    5. Copy /testbed/ to the review directory
    6. Save changed_files.txt

    Args:
        instance: Instance data
        patch_content: Agent-generated patch
        log_dir: Log directory for this instance
        docker_image: Docker image name
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("Saving review codes for Level 1")
    logger.info("=" * 60)

    # Create review directory
    review_dir = log_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    client = docker.from_env()
    container = None

    try:
        # Create container
        instance_id = normalize_instance_id(
            instance.get("instance_id"),
            field="dataset field 'instance_id'",
        )
        container_name = f"fb_review_{instance_id}_{os.getpid()}"
        logger.info(f"Creating review container: {container_name}")

        container = client.containers.run(
            docker_image,
            command="/bin/bash -c 'sleep infinity'",
            name=container_name,
            detach=True,
            remove=False,
            labels=labels,
        )

        # Step 1: Restore project
        logger.info("Step 1: Restoring project")
        cmd = (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate testbed && "
            "rm -rf /testbed/* && "
            "cp -r /root/my_repo/* /testbed/"
        )
        exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
        if exit_code != 0:
            raise RuntimeError(f"failed to restore project: {output}")

        # Step 2: Apply instance patch (masking)
        logger.info("Step 2: Applying instance patch for masking")
        instance_patch = instance.get("patch", "")
        if instance_patch and instance_patch.strip():
            _copy_text_to_container(container, instance_patch, "/tmp/mask_patch.diff")
            cmd = "cd /testbed && git apply --whitespace=fix /tmp/mask_patch.diff"
            exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
            if exit_code != 0:
                raise RuntimeError(f"failed to apply mask patch: {output}")

        # Step 3: Apply the exact agent patch without language-specific rewrites.
        logger.info("Step 3: Applying agent patch")
        if patch_content and patch_content.strip():
            # Initialize git
            git_init_cmds = [
                "cd /testbed && rm -rf .git",
                "cd /testbed && git init",
                'cd /testbed && git config user.email "fb@bench.com"',
                'cd /testbed && git config user.name "FeatureBench"',
                "cd /testbed && git add -A",
                'cd /testbed && git commit -m "Initial" --allow-empty',
            ]
            for cmd in git_init_cmds:
                exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
                if exit_code != 0:
                    raise RuntimeError(f"failed to initialize review repository: {output}")

            _copy_text_to_container(container, patch_content, "/tmp/agent_patch.diff")

            # Apply patch
            cmd = "cd /testbed && git apply --whitespace=fix /tmp/agent_patch.diff"
            exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
            if exit_code != 0:
                raise RuntimeError(f"failed to apply agent patch: {output}")

        # Step 4: Delete .git
        logger.info("Step 4: Deleting .git directory")
        exec_run_simple(container, "/bin/bash -c 'cd /testbed && rm -rf .git'")

        # Step 5: Copy /testbed/ to review directory
        logger.info("Step 5: Copying /testbed/ to review directory")
        testbed_dst = review_dir / "testbed"
        copy_from_container(container, "/testbed", testbed_dst)
        logger.info(f"Saved testbed to {testbed_dst}")

        # Step 6: Save changed_files.txt
        logger.info("Step 6: Saving changed_files.txt")
        changed_files = extract_changed_files_from_patch(patch_content)
        changed_files_path = review_dir / "changed_files.txt"
        with changed_files_path.open("w", encoding=UTF8) as f:
            for file_path in changed_files:
                f.write(f"{file_path}\n")
        logger.info(f"Saved changed files list to {changed_files_path}")

        logger.info("Review codes saved successfully for Level 1")
        return True

    except Exception as e:
        logger.error(f"Error saving review codes: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
                container.remove()
            except Exception as e:
                logger.warning(f"Failed to cleanup review container: {e}")


def save_review_codes_level2(
    instance: pd.Series,
    patch_content: str,
    log_dir: Path,
    docker_image: str,
    logger: logging.Logger,
    labels: dict[str, str] | None = None,
) -> bool:
    """
    Save review codes for Level 2 instance.

    Level 2 workflow:
    1. Create container and clean /testbed/
    2. Create README.md
    3. Initialize git
    4. Apply agent patch
    5. Delete .git
    6. Copy /testbed/ to review directory
    7. Save changed_files.txt with message

    Args:
        instance: Instance data
        patch_content: Agent-generated patch
        log_dir: Log directory for this instance
        docker_image: Docker image name
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("Saving review codes for Level 2")
    logger.info("=" * 60)

    # Create review directory
    review_dir = log_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    client = docker.from_env()
    container = None

    try:
        # Create container
        instance_id = normalize_instance_id(
            instance.get("instance_id"),
            field="dataset field 'instance_id'",
        )
        container_name = f"fb_review_{instance_id}_{os.getpid()}"
        logger.info(f"Creating review container: {container_name}")

        container = client.containers.run(
            docker_image,
            command="/bin/bash -c 'sleep infinity'",
            name=container_name,
            detach=True,
            remove=False,
            labels=labels,
        )

        # Step 1: Clean /testbed/ and create README.md
        logger.info("Step 1: Cleaning /testbed/ and creating README.md")
        init_cmds = [
            "rm -rf /testbed/* /testbed/.*  2>/dev/null || true",
            "mkdir -p /testbed",
            'cd /testbed && echo "put all codes in this folder" > README.md',
        ]
        for cmd in init_cmds:
            exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
            if exit_code != 0:
                raise RuntimeError(f"failed to prepare review workspace: {output}")

        # Step 2: Initialize git
        logger.info("Step 2: Initializing git")
        git_init_cmds = [
            "cd /testbed && git init",
            'cd /testbed && git config user.email "fb@bench.com"',
            'cd /testbed && git config user.name "FeatureBench"',
            "cd /testbed && git add -A",
            'cd /testbed && git commit -m "Initial" --allow-empty',
        ]
        for cmd in git_init_cmds:
            exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
            if exit_code != 0:
                raise RuntimeError(f"failed to initialize review repository: {output}")

        # Step 3: Apply agent patch
        logger.info("Step 3: Applying agent patch")
        if patch_content and patch_content.strip():
            _copy_text_to_container(container, patch_content, "/tmp/agent_patch.diff")

            # Apply patch
            cmd = "cd /testbed && git apply --whitespace=fix /tmp/agent_patch.diff"
            exit_code, output = exec_run_simple(container, f"/bin/bash -c '{cmd}'")
            if exit_code != 0:
                raise RuntimeError(f"failed to apply agent patch: {output}")

        # Step 4: Delete .git
        logger.info("Step 4: Deleting .git directory")
        exec_run_simple(container, "/bin/bash -c 'cd /testbed && rm -rf .git'")

        # Step 5: Copy /testbed/ to review directory
        logger.info("Step 5: Copying /testbed/ to review directory")
        testbed_dst = review_dir / "testbed"
        copy_from_container(container, "/testbed", testbed_dst)
        logger.info(f"Saved testbed to {testbed_dst}")

        # Step 6: Save changed_files.txt
        logger.info("Step 6: Saving changed_files.txt")
        changed_files_path = review_dir / "changed_files.txt"
        with changed_files_path.open("w", encoding=UTF8) as f:
            f.write("all files under /testbed/ was generated by agent\n")
        logger.info(f"Saved changed files message to {changed_files_path}")

        logger.info("Review codes saved successfully for Level 2")
        return True

    except Exception as e:
        logger.error(f"Error saving review codes: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
                container.remove()
            except Exception as e:
                logger.warning(f"Failed to cleanup review container: {e}")
