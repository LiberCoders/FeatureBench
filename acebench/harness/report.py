"""
Report generation for ACE-Bench evaluation.
"""

import json
from pathlib import Path
from typing import Any

from acebench.harness.constants import (
    EvalType,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    UTF8,
)
from acebench.harness.grading import get_eval_report
from acebench.harness.test_parsers import MAP_REPO_TO_PARSER, parse_log_pytest


def parse_test_outputs(
    log_dir: Path,
    repo_name: str,
    level: int,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """
    Parse test output files and return parsed results (test_name -> status mapping).

    Args:
        log_dir: Directory containing test output files
        repo_name: Repository name for parser selection
        level: Evaluation level (1 or 2)

    Returns:
        Tuple of (f2p_status_map, p2p_status_map_list)
    """
    parser_fn = MAP_REPO_TO_PARSER.get(repo_name, parse_log_pytest)

    # Parse F2P test output
    f2p_output_file = log_dir / LOG_TEST_OUTPUT
    f2p_status_map: dict[str, str] = {}
    if f2p_output_file.exists():
        with open(f2p_output_file, "r", encoding=UTF8, errors='replace') as f:
            f2p_output = f.read()
        f2p_status_map = parser_fn(f2p_output)

    # Parse P2P test outputs (Level 1 only)
    p2p_status_map_list: list[dict[str, str]] = []
    if level == 1:
        for p2p_file in log_dir.glob("test_output_p2p_*.txt"):
            with open(p2p_file, "r", encoding=UTF8, errors='replace') as f:
                p2p_output = f.read()
            p2p_status_map_list.append(parser_fn(p2p_output))

    return f2p_status_map, p2p_status_map_list


def build_test_status(
    f2p_status_map: dict[str, str],
    p2p_status_map_list: list[dict[str, str]],
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> tuple[list, list, list, list]:
    """
    Build test status from parsed results using grading module.

    Args:
        f2p_status_map: F2P test status map (test_name -> status)
        p2p_status_map_list: List of P2P test status maps
        eval_type: Evaluation mode (PASS_AND_FAIL or FAIL_ONLY)

    Returns:
        Tuple of (f2p_success, f2p_failure, p2p_success, p2p_failure)
    """
    # Grade F2P tests
    f2p_tests = list(f2p_status_map.keys())
    f2p_report = get_eval_report(f2p_status_map, f2p_tests, eval_type)
    f2p_success = f2p_report["success_tests"]
    f2p_failure = f2p_report["failure_tests"]

    # Grade P2P tests
    p2p_success = []
    p2p_failure = []
    for p2p_status_map in p2p_status_map_list:
        p2p_tests = list(p2p_status_map.keys())
        p2p_report = get_eval_report(p2p_status_map, p2p_tests, eval_type)
        p2p_success.extend(p2p_report["success_tests"])
        p2p_failure.extend(p2p_report["failure_tests"])

    return f2p_success, f2p_failure, p2p_success, p2p_failure


def generate_instance_report(
    instance_id: str,
    n_attempt: int,
    patch_content: str | None,
    patch_applied: bool,
    f2p_success_list: list,
    f2p_failure_list: list,
    p2p_success_list: list,
    p2p_failure_list: list,
    eval_results: dict,
) -> dict:
    """
    Generate report for a single instance.

    Args:
        instance_id: Instance ID
        n_attempt: Attempt number
        patch_content: Patch content
        patch_applied: Whether patch was applied
        f2p_success_list: List of successful F2P tests
        f2p_failure_list: List of failed F2P tests
        p2p_success_list: List of successful P2P tests
        p2p_failure_list: List of failed P2P tests
        eval_results: Raw evaluation results

    Returns:
        Report dictionary
    """
    patch_is_none = patch_content is None
    patch_exists = bool(patch_content and patch_content.strip())

    # Determine if resolved
    resolved = eval_results.get("f2p_success", False) and eval_results.get("p2p_success", True)

    # Calculate F2P pass rate
    f2p_total = len(f2p_success_list) + len(f2p_failure_list)
    f2p_pass_rate = round(len(f2p_success_list) / f2p_total, 4) if f2p_total > 0 else 0.0

    report = {
        instance_id: {
            "n_attempt": n_attempt,
            "patch_is_None": patch_is_none,
            "patch_exists": patch_exists,
            "patch_successfully_applied": patch_applied,
            "resolved": resolved,
            "pass_rate": f2p_pass_rate,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": f2p_success_list,
                    "failure": f2p_failure_list
                },
                "PASS_TO_PASS": {
                    "success": p2p_success_list,
                    "failure": p2p_failure_list
                }
            }
        }
    }

    return report


def generate_error_report(
    instance_id: str,
    n_attempt: int,
    patch_content: str | None,
    error: str,
    traceback_str: str,
) -> dict:
    """
    Generate error report for a failed instance.

    Args:
        instance_id: Instance ID
        n_attempt: Attempt number
        patch_content: Patch content
        error: Error message
        traceback_str: Traceback string

    Returns:
        Error report dictionary
    """
    return {
        instance_id: {
            "n_attempt": n_attempt,
            "patch_is_None": patch_content is None,
            "patch_exists": bool(patch_content and patch_content.strip()),
            "patch_successfully_applied": False,
            "resolved": False,
            "pass_rate": 0.0,
            "error": error,
            "traceback": traceback_str,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": [],
                    "failure": []
                },
                "PASS_TO_PASS": {
                    "success": [],
                    "failure": []
                }
            }
        }
    }


def save_report(report: dict, log_dir: Path) -> None:
    """
    Save report to file.

    Args:
        report: Report dictionary
        log_dir: Directory to save report
    """
    report_path = log_dir / LOG_REPORT
    with open(report_path, "w", encoding=UTF8) as f:
        json.dump(report, f, indent=4)


def _infer_log_path(output_dir: Path, instance_id: str, n_attempt: int) -> str:
    return str(output_dir / "run_outputs" / instance_id / f"attempt-{n_attempt}" / "infer.log")


def generate_summary_report(results: list[dict], output_dir: Path) -> dict:
    """
    Generate summary report from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Summary report dictionary grouped by attempt
    """
    # Group results by n_attempt
    results_by_attempt = {}
    for r in results:
        n_attempt = r.get('n_attempt', 1)
        if n_attempt not in results_by_attempt:
            results_by_attempt[n_attempt] = []
        results_by_attempt[n_attempt].append(r)

    # Generate summary report grouped by attempt
    summary_report = {}

    for n_attempt in sorted(results_by_attempt.keys()):
        attempt_results = results_by_attempt[n_attempt]

        # Calculate statistics
        total_instances = len(attempt_results)
        completed_instances = sum(1 for r in attempt_results if r.get('completed', False))
        resolved_instances = sum(1 for r in attempt_results if r.get('resolved', False))
        unresolved_instances = completed_instances - resolved_instances
        error_instances = sum(1 for r in attempt_results if r.get('error'))

        # Collect infer.log paths
        completed_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if r.get("completed", False) and r.get("instance_id")
        ]
        submitted_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if r.get("instance_id")
        ]
        resolved_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if r.get("resolved", False) and r.get("instance_id")
        ]
        unresolved_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if r.get("completed", False) and not r.get("resolved", False) and r.get("instance_id")
        ]
        incomplete_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if not r.get("completed", False) and r.get("instance_id")
        ]
        error_ids = [
            _infer_log_path(output_dir, r["instance_id"], r.get("n_attempt", 1))
            for r in attempt_results
            if r.get("error") and r.get("instance_id")
        ]

        # Split "not applied patch" into empty-patch vs other reasons.
        # We treat "empty patch" as: patch_applied == False AND instance_report.patch_exists == False
        not_applied_patch_empty_ids: list[str] = []
        not_applied_patch_other_ids: list[str] = []
        for r in attempt_results:
            if r.get('patch_applied', False):
                continue

            instance_id = r.get('instance_id')
            if not instance_id:
                # Shouldn't happen; keep conservative classification.
                continue

            patch_exists = None
            if 'report' in r and isinstance(r['report'], dict) and instance_id in r['report']:
                instance_report = r['report'][instance_id]
                if isinstance(instance_report, dict):
                    patch_exists = instance_report.get('patch_exists')

            infer_log = _infer_log_path(output_dir, instance_id, r.get("n_attempt", 1))
            if patch_exists is False:
                not_applied_patch_empty_ids.append(infer_log)
            else:
                not_applied_patch_other_ids.append(infer_log)

        not_applied_patch_empty_instances = len(not_applied_patch_empty_ids)
        not_applied_patch_other_instances = len(not_applied_patch_other_ids)

        # Calculate rates
        resolved_rate = round(resolved_instances / total_instances, 4) if total_instances > 0 else 0.0

        # Calculate average F2P pass rate
        pass_rates = []
        for result in attempt_results:
            if 'report' in result:
                for instance_id, instance_report in result['report'].items():
                    pass_rates.append(instance_report.get('pass_rate', 0.0))
        average_f2p_pass_rate = round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0.0

        # Store summary
        attempt_key = f"attempt_{n_attempt}"
        summary_report[attempt_key] = {
            "n_attempt": n_attempt,
            "total_instances": total_instances,
            "submitted_instances": len(submitted_ids),
            "completed_instances": completed_instances,
            "resolved_instances": resolved_instances,
            "unresolved_instances": unresolved_instances,
            "not_applied_patch_empty_instances": not_applied_patch_empty_instances,
            "not_applied_patch_other_instances": not_applied_patch_other_instances,
            "error_instances": error_instances,
            "resolved_rate": resolved_rate,
            "pass_rate": average_f2p_pass_rate,
            "submitted_ids": submitted_ids,
            "completed_ids": completed_ids,
            "incomplete_ids": incomplete_ids,
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "not_applied_patch_empty_ids": not_applied_patch_empty_ids,
            "not_applied_patch_other_ids": not_applied_patch_other_ids,
            "error_ids": error_ids
        }

    return summary_report

