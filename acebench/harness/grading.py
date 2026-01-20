"""
Grading module for ACE-Bench evaluation harness.

This module provides functions to compute test results and metrics.
Supports two evaluation modes:
- PASS_AND_FAIL (default): Test passes if status is PASSED or XFAIL; fails if FAILED or ERROR
- FAIL_ONLY: Only ERROR and FAILED count as failure; everything else silently passes
"""

from typing import Any

from acebench.harness.constants import (
    EvalType,
    TestStatus,
)


# MARK: Utility functions
def test_passed(case: str, status_map: dict[str, str]) -> bool:
    """
    Check if a test case passed (PASS_AND_FAIL mode).

    A test is considered passed if:
    - It exists in the status map AND
    - Its status is PASSED or XFAIL

    Args:
        case: Test case name
        status_map: Dict mapping test names to status values

    Returns:
        bool: True if test passed
    """
    return case in status_map and status_map[case] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
    ]


def test_failed(case: str, status_map: dict[str, str]) -> bool:
    """
    Check if a test case failed (PASS_AND_FAIL mode).

    A test is considered failed if:
    - It's not in the status map OR
    - Its status is FAILED or ERROR

    Args:
        case: Test case name
        status_map: Dict mapping test names to status values

    Returns:
        bool: True if test failed
    """
    return case not in status_map or status_map[case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


# MARK: Grading functions
def get_eval_report(
    eval_status_map: dict[str, str],
    expected_tests: list[str],
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, Any]:
    """
    Compute evaluation report based on test results.

    Args:
        eval_status_map: Dict mapping test names to status values (from parser)
        expected_tests: List of expected test names to check
        eval_type: Evaluation mode (PASS_AND_FAIL or FAIL_ONLY)

    Returns:
        dict: Report containing:
            - total: Total number of tests
            - success: Number of successful tests
            - failure: Number of failed tests
            - pass_rate: Pass rate (success / total)
            - success_tests: List of successful test names
            - failure_tests: List of failed test names
    """

    def check_pass_and_fail(
        test_case: str,
        status_map: dict[str, str],
        success: list[str],
        failed: list[str],
    ) -> None:
        """
        Check test case in PASS_AND_FAIL mode.
        - Test passes if status is PASSED or XFAIL.
        - Test fails if status is FAILED, ERROR, or not in map.
        """
        if test_passed(test_case, status_map):
            success.append(test_case)
        elif test_failed(test_case, status_map):
            failed.append(test_case)

    def check_fail_only(
        test_case: str,
        status_map: dict[str, str],
        success: list[str],
        failed: list[str],
    ) -> None:
        """
        Check test case in FAIL_ONLY mode.
        - Test fails only if explicitly marked as FAILED or ERROR.
        - Everything else (PASSED, XFAIL, SKIPPED, or not in map) silently passes.
        """
        if test_case in status_map and status_map[test_case] in [
            TestStatus.FAILED.value,
            TestStatus.ERROR.value,
        ]:
            failed.append(test_case)
        else:
            success.append(test_case)

    # Select check function based on eval type
    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    success_tests: list[str] = []
    failure_tests: list[str] = []

    for test_case in expected_tests:
        check_test_case(test_case, eval_status_map, success_tests, failure_tests)

    total = len(expected_tests)
    success = len(success_tests)
    failure = len(failure_tests)
    pass_rate = round(success / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "failure": failure,
        "pass_rate": pass_rate,
        "success_tests": success_tests,
        "failure_tests": failure_tests,
    }
