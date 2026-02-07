"""
Test log parsers for FeatureBench evaluation harness.

This module provides parsers that extract test results from log files.
Parsers return a dict mapping test names to their status values.
Parses: PASSED, FAILED, ERROR, XFAIL, SKIPPED.
"""

import re

from featurebench.harness.constants import TestStatus


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework.

    Args:
        log (str): log content
    Returns:
        dict[str, str]: test case to test status mapping
    """
    test_status_map = {}

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    # Find the "short test summary info" section
    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1  # Start from the next line
            break

    # If we didn't find the summary section, return empty result
    if summary_start_idx == -1:
        return test_status_map

    # Parse lines from the summary section until we hit a separator line or end
    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        # Stop if we hit a separator line (starts with ===)
        if line.startswith('==='):
            break

        # Skip empty lines
        if not line:
            continue

        # Match PASSED
        if line.startswith(TestStatus.PASSED.value + ' '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.PASSED.value

        # Match FAILED
        elif line.startswith(TestStatus.FAILED.value + ' '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.FAILED.value

        # Match ERROR
        elif line.startswith(TestStatus.ERROR.value + ' '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.ERROR.value

        # Match XFAIL (expected failure)
        elif line.startswith(TestStatus.XFAIL.value + ' '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                test_status_map[test_name] = TestStatus.XFAIL.value

        # Match SKIPPED
        elif line.startswith(TestStatus.SKIPPED.value + ' '):
            # Format: SKIPPED [1] test_example.py:15: reason
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_path = match.group(1).strip()
                test_status_map[test_path] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_pytest_with_subfailed(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework, handling SUBFAILED duplicates.
    
    When the same test name appears multiple times in SUBFAILED entries,
    adds [1], [2], etc. suffixes to keep them distinct in the result dict.
    Other status types (PASSED, FAILED, ERROR, etc.) do not get numbered suffixes.
    
    Args:
        log (str): log content
    Returns:
        dict[str, str]: test case to test status mapping
    """
    test_status_map = {}
    subfailed_counts = {}  # Track SUBFAILED occurrences per test name

    if not log or not log.strip():
        return test_status_map

    lines = log.split('\n')

    # Find the "short test summary info" section
    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1  # Start from the next line
            break

    # If we didn't find the summary section, return empty result
    if summary_start_idx == -1:
        return test_status_map

    # Parse lines from the summary section until we hit a separator line or end
    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        # Stop if we hit a separator line (starts with ===)
        if line.startswith('==='):
            break

        # Skip empty lines
        if not line:
            continue

        test_name = None
        status = None
        is_subfailed = False

        # Match PASSED
        if line.startswith(TestStatus.PASSED.value + ' '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.PASSED.value

        # Match FAILED
        elif line.startswith(TestStatus.FAILED.value + ' '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value

        # Match SUBFAILED (subtest failures, treat as FAILED)
        elif line.startswith('SUBFAILED'):
            match = re.match(r'SUBFAILED(?:\(<subtest>\))?\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.FAILED.value
                is_subfailed = True

        # Match ERROR
        elif line.startswith(TestStatus.ERROR.value + ' '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.ERROR.value

        # Match XFAIL (expected failure)
        elif line.startswith(TestStatus.XFAIL.value + ' '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.XFAIL.value

        # Match SKIPPED
        elif line.startswith(TestStatus.SKIPPED.value + ' '):
            # Format: SKIPPED [1] test_example.py:15: reason
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_name = match.group(1).strip()
                status = TestStatus.SKIPPED.value

        # If we found a test name and status, record it
        if test_name and status:
            # Only add numbered suffix for SUBFAILED entries
            if is_subfailed:
                if test_name in subfailed_counts:
                    subfailed_counts[test_name] += 1
                else:
                    subfailed_counts[test_name] = 1
                # Add suffix [1], [2], etc. for SUBFAILED
                unique_test_name = f"{test_name}[{subfailed_counts[test_name]}]"
            else:
                # For non-SUBFAILED, use test name as-is
                unique_test_name = test_name
            
            test_status_map[unique_test_name] = status

    return test_status_map


def parse_log_pytest_verbose(log: str) -> dict[str, str]:
    """
    Parser for verbose test logs generated with PyTest framework.
    
    Parses pytest verbose output format where test results appear as:
    tests/test_file.py::test_name STATUS [percentage] [percentage]
    
    Args:
        log (str): log content
    Returns:
        dict[str, str]: test case to test status mapping
    """
    test_status_map = {}
    
    if not log or not log.strip():
        return test_status_map
    
    lines = log.split('\n')
    
    # Define the status values to look for
    statuses = [
        TestStatus.PASSED.value,
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
        TestStatus.XFAIL.value,
        TestStatus.SKIPPED.value
    ]
    
    for line in lines:
        line_stripped = line.strip()
        
        # Stop if we hit a separator line (starts with ===)
        # This indicates the end of test execution results and start of detailed output
        if line_stripped.startswith('===') and  ('test session starts' not in line_stripped):
            break
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Check if line contains any of the status keywords
        for status in statuses:
            # Look for status keyword surrounded by spaces or at end of relevant part
            if f' {status} ' in line_stripped or f' {status}\t' in line_stripped:
                # Extract test name (everything before the status)
                # Format: test_path::test_name STATUS [percentage] [percentage]
                parts = line_stripped.split(status, 1)
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    # Only record if it looks like a valid test path (contains ::)
                    # This filters out lines like "tests/test_file.py:8 test_name - message"
                    if '::' in test_name:
                        test_status_map[test_name] = status
                        break  # Found a status, move to next line
    
    return test_status_map


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_pytest,
    "fastapi/fastapi": parse_log_pytest,
    "huggingface/accelerate": parse_log_pytest,
    "huggingface/transformers": parse_log_pytest,
    "huggingface/trl": parse_log_pytest,
    "Lightning-AI/pytorch-lightning": parse_log_pytest,
    "linkedin/Liger-Kernel": parse_log_pytest,
    "matplotlib/matplotlib": parse_log_pytest,
    "mesonbuild/meson": parse_log_pytest_with_subfailed,
    "mlflow/mlflow": parse_log_pytest,
    "mwaskom/seaborn": parse_log_pytest,
    "Netflix/metaflow": parse_log_pytest,
    "optuna/optuna": parse_log_pytest,
    "pandas-dev/pandas": parse_log_pytest,
    "pydantic/pydantic": parse_log_pytest_verbose,
    "pydata/xarray": parse_log_pytest,
    "pypa/hatch": parse_log_pytest,
    "pypa/packaging": parse_log_pytest,
    "pypa/setuptools": parse_log_pytest,
    "pytest-dev/pytest": parse_log_pytest,
    "python/mypy": parse_log_pytest,
    "scikit-learn/scikit-learn": parse_log_pytest,
    "sphinx-doc/sphinx": parse_log_pytest,
    "sympy/sympy": parse_log_pytest,
}

MAP_REPO_TO_TEST_CMD = {
    "pydantic/pydantic": "pytest -rA -v --color=no",
}
