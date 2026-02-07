#!/usr/bin/env python3
"""
Aggregate eval_outputs and generate a CSV report.

Usage:
    python featurebench/scripts/cal_eval_outputs.py --path <eval_outputs_dir> --attempt-mode <attempt_mode>
    <attempt_mode> can be 'best', 'worst', or a number (e.g., 1, 2, 3). Default: 'best'.

Notes:
    When <attempt_mode> is a number k and --merge is enabled, attempts 1..k are merged:
      - pass_rate: mean over the first k attempts
      - resolved: pass@k style (any success in the first k attempts)
      - prompt_tokens/completion_tokens: sum over the first k attempts, then averaged

Examples:
    python featurebench/scripts/cal_eval_outputs.py --path runs/2025-12-06__17-14-12/eval_outputs --attempt-mode best
    python featurebench/scripts/cal_eval_outputs.py --path runs/2025-12-06__17-14-12/eval_outputs --attempt-mode 3 --merge
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def merge_attempts_metrics(task_id: str, attempts: List[Tuple[int, Dict]], k: int) -> Optional[Dict]:
    """Merge task metrics over attempts 1..k.

    Rules:
      - pass_rate: mean(pass_rate_i)
      - resolved: any(resolved_i)
      - total: taken from the first available attempt (fallback to max seen)
      - num_passed/num_failed: derived from mean pass_rate and total to keep consistency
    """
    if k <= 0:
        return None

    selected = [task_data for attempt_num, task_data in attempts if attempt_num <= k]
    if not selected:
        return None

    pass_rates = [float(t.get('pass_rate', 0.0)) for t in selected]
    mean_pass_rate = sum(pass_rates) / len(pass_rates)
    resolved_any = any(bool(t.get('resolved', False)) for t in selected)

    totals = [int(t.get('total', 0)) for t in selected if t.get('total') is not None]
    total = totals[0] if totals else 0
    if total == 0 and totals:
        total = max(totals)

    if total > 0:
        num_passed = int(round(mean_pass_rate * total))
        num_passed = max(0, min(total, num_passed))
        num_failed = total - num_passed
    else:
        num_passed = 0
        num_failed = 0

    return {
        'task_id': task_id,
        'resolved': resolved_any,
        'pass_rate': mean_pass_rate,
        'num_passed': num_passed,
        'num_failed': num_failed,
        'total': total,
    }


def _as_int(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_openhands_tokens(trajectory_path: Path) -> Tuple[int, int]:
    try:
        with trajectory_path.open('r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0, 0

    candidates = []
    if isinstance(data, list):
        candidates = list(reversed(data))
    elif isinstance(data, dict):
        candidates = [data]

    for item in candidates:
        if not isinstance(item, dict):
            continue
        metrics = item.get('llm_metrics')
        if not isinstance(metrics, dict):
            continue
        usage = metrics.get('accumulated_token_usage')
        if not isinstance(usage, dict):
            usage = metrics
        prompt = _as_int(usage.get('prompt_tokens'))
        completion = _as_int(usage.get('completion_tokens'))
        if prompt or completion:
            return prompt, completion
    return 0, 0


def _extract_codex_tokens(jsonl_path: Path) -> Tuple[int, int]:
    last_usage = None
    try:
        with jsonl_path.open('r', encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                usage = obj.get('usage')
                if isinstance(usage, dict) and (
                    'input_tokens' in usage or 'output_tokens' in usage
                ):
                    last_usage = usage
                elif 'input_tokens' in obj or 'output_tokens' in obj:
                    last_usage = obj
    except OSError:
        return 0, 0

    if not last_usage:
        return 0, 0
    prompt = _as_int(last_usage.get('input_tokens'))
    completion = _as_int(last_usage.get('output_tokens'))
    return prompt, completion


def _extract_gemini_cli_tokens(jsonl_path: Path) -> Tuple[int, int]:
    last_usage = None
    try:
        with jsonl_path.open('r', encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                stats = obj.get('stats')
                if isinstance(stats, dict) and (
                    'input_tokens' in stats or 'output_tokens' in stats
                ):
                    last_usage = stats
                elif 'input_tokens' in obj or 'output_tokens' in obj:
                    last_usage = obj
                else:
                    usage = obj.get('usage')
                    if isinstance(usage, dict) and (
                        'input_tokens' in usage or 'output_tokens' in usage
                    ):
                        last_usage = usage
    except OSError:
        return 0, 0

    if not last_usage:
        return 0, 0
    prompt = _as_int(last_usage.get('input_tokens'))
    completion = _as_int(last_usage.get('output_tokens'))
    return prompt, completion


def _normalize_claude_code_usage(raw) -> List[Dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        if any(
            key in raw
            for key in (
                'inputTokens',
                'outputTokens',
                'cacheReadInputTokens',
                'cacheCreationInputTokens',
            )
        ):
            return [raw]
        return [item for item in raw.values() if isinstance(item, dict)]
    return []


def _extract_claude_code_tokens(jsonl_path: Path) -> Tuple[int, int]:
    last_usage_items: Optional[List[Dict]] = None
    try:
        with jsonl_path.open('r', encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                model_usage = obj.get('modelUsage')
                if not model_usage and isinstance(obj.get('result'), dict):
                    model_usage = obj['result'].get('modelUsage')
                if model_usage is not None:
                    usage_items = _normalize_claude_code_usage(model_usage)
                    if usage_items:
                        last_usage_items = usage_items
    except OSError:
        return 0, 0

    if not last_usage_items:
        return 0, 0

    prompt = 0
    completion = 0
    for usage in last_usage_items:
        prompt += _as_int(usage.get('inputTokens'))
        prompt += _as_int(usage.get('cacheReadInputTokens'))
        prompt += _as_int(usage.get('cacheCreationInputTokens'))
        completion += _as_int(usage.get('outputTokens'))
    return prompt, completion


def _extract_tokens_for_attempt(attempt_dir: Path) -> Tuple[int, int]:
    trajectory_path = attempt_dir / 'trajectory.json'
    if trajectory_path.exists():
        return _extract_openhands_tokens(trajectory_path)
    gemini_stream_path = attempt_dir / 'gemini_cli_stream_output.jsonl'
    if gemini_stream_path.exists():
        return _extract_gemini_cli_tokens(gemini_stream_path)
    claude_code_stream_path = attempt_dir / 'claude_code_stream_output.jsonl'
    if claude_code_stream_path.exists():
        return _extract_claude_code_tokens(claude_code_stream_path)
    jsonl_path = attempt_dir / 'codex_events.jsonl'
    if jsonl_path.exists():
        return _extract_codex_tokens(jsonl_path)
    return 0, 0


def _aggregate_tokens_for_attempts(
    task_dir: Path,
    attempt_nums: List[int],
    average: bool = False
) -> Tuple[float, float]:
    total_prompt = 0.0
    total_completion = 0.0
    counted = 0
    for attempt_num in attempt_nums:
        attempt_dir = task_dir / f'attempt-{attempt_num}'
        if not attempt_dir.exists():
            continue
        prompt, completion = _extract_tokens_for_attempt(attempt_dir)
        total_prompt += prompt
        total_completion += completion
        counted += 1
    if average and counted > 0:
        return round(total_prompt / counted, 2), round(total_completion / counted, 2)
    return total_prompt, total_completion


def parse_task_id(task_id: str) -> Tuple[str, str]:
    """
    Parse the task_id, extract the repo and level
    
    Args:
        task_id: The task ID, like "astropy__astropy.b0db0daa.test_basic_rgb.067e927c.lv1"
    
    Returns:
        (repo, level) tuple
    """
    parts = task_id.split('.')
    level = parts[-1]  # The last part, like "lv1" or "lv2"
    
    # repo is the first part, replace __ with /
    repo_part = parts[0]
    repo = repo_part.replace('__', '/')
    
    return repo, level


def read_report_json(report_path: Path) -> Optional[Dict]:
    """
    Read the report.json file
    
    Args:
        report_path: The path to the report.json file
    
    Returns:
        The parsed JSON data, if the file does not exist or parsing fails, return None
    """
    if not report_path.exists():
        return None
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Cannot read {report_path}: {e}")
        return None


def extract_task_data(report_data: Dict, task_id: str) -> Optional[Dict]:
    """
    Extract the task information from the report.json data
    
    Args:
        report_data: The complete data of report.json
        task_id: The task ID
    
    Returns:
        The dictionary containing the task data, if the task does not exist, return None
    """
    if task_id not in report_data:
        return None
    
    task_info = report_data[task_id]
    
    # Extract the FAIL_TO_PASS information
    fail_to_pass = task_info.get('tests_status', {}).get('FAIL_TO_PASS', {})
    success = fail_to_pass.get('success', [])
    failure = fail_to_pass.get('failure', [])
    
    num_passed = len(success)
    num_failed = len(failure)
    total = num_passed + num_failed
    
    return {
        'task_id': task_id,
        'resolved': task_info.get('resolved', False),
        'pass_rate': task_info.get('pass_rate', 0.0),
        'num_passed': num_passed,
        'num_failed': num_failed,
        'total': total,
    }


def collect_all_attempts(eval_outputs_dir: Path, task_id: str) -> List[Tuple[int, Dict]]:
    """
    Collect all the attempt results for a given task
    
    Args:
        eval_outputs_dir: The path to the eval_outputs directory
        task_id: The task ID
    
    Returns:
        [(attempt_num, task_data), ...] list
    """
    task_dir = eval_outputs_dir / task_id
    if not task_dir.exists():
        return []
    
    attempts = []
    
    # Iterate over all the attempt-n directories
    for attempt_dir in task_dir.iterdir():
        if attempt_dir.is_dir() and attempt_dir.name.startswith('attempt-'):
            try:
                attempt_num = int(attempt_dir.name.split('-')[1])
                report_path = attempt_dir / 'report.json'
                
                report_data = read_report_json(report_path)
                if report_data:
                    task_data = extract_task_data(report_data, task_id)
                    if task_data:
                        attempts.append((attempt_num, task_data))
            except (ValueError, IndexError):
                continue
    
    return attempts


def select_attempt_by_mode(
    attempts: List[Tuple[int, Dict]],
    attempt_mode: str
) -> Optional[Tuple[int, Dict]]:
    """
    Select the appropriate attempt result based on the attempt_mode
    
    Args:
        attempts: The list of all the attempt results
        attempt_mode: 'n' (number), 'best', or 'worst'
    
    Returns:
        The selected task data, if none is found, return None
    """
    if not attempts:
        return None
    
    if attempt_mode == 'best':
        # Select the attempt with the highest pass_rate
        return max(attempts, key=lambda x: x[1]['pass_rate'])
    elif attempt_mode == 'worst':
        # Select the attempt with the lowest pass_rate
        return min(attempts, key=lambda x: x[1]['pass_rate'])
    else:
        # attempt_mode should be a string of numbers
        try:
            target_attempt = int(attempt_mode)
            for attempt_num, task_data in attempts:
                if attempt_num == target_attempt:
                    return attempt_num, task_data
            return None
        except ValueError:
            return None


def process_eval_outputs(eval_outputs_dir: Path, attempt_mode: str, merge: bool) -> List[Dict]:
    """
    Process the eval_outputs directory, collect all the task data
    
    Args:
        eval_outputs_dir: The path to the eval_outputs directory
        attempt_mode: The attempt mode ('n', 'best', 'worst')
    
    Returns:
        The list of task data
    """
    all_tasks = []
    
    # Iterate over all the task directories
    run_outputs_dir = eval_outputs_dir.parent / 'run_outputs'

    for task_dir in eval_outputs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_id = task_dir.name
        
        prompt_tokens = 0
        completion_tokens = 0

        if attempt_mode in ('best', 'worst'):
            # Collect all the attempts, then select best/worst
            attempts = collect_all_attempts(eval_outputs_dir, task_id)
            selected = select_attempt_by_mode(attempts, attempt_mode)
            if selected:
                attempt_num, task_data = selected
                if run_outputs_dir.exists():
                    prompt_tokens, completion_tokens = _aggregate_tokens_for_attempts(
                        run_outputs_dir / task_id, [attempt_num], average=False
                    )
            else:
                task_data = None
        else:
            if not is_int_string(attempt_mode):
                print(f"Warning: Invalid attempt mode: {attempt_mode}")
                task_data = None
            else:
                attempt_num = int(attempt_mode)
                if merge:
                    attempts = collect_all_attempts(eval_outputs_dir, task_id)
                    task_data = merge_attempts_metrics(task_id, attempts, attempt_num)
                    if run_outputs_dir.exists():
                        attempt_nums = [num for num, _ in attempts if num <= attempt_num]
                        prompt_tokens, completion_tokens = _aggregate_tokens_for_attempts(
                            run_outputs_dir / task_id, attempt_nums, average=True
                        )
                else:
                    # Only read the specified attempt-n
                    report_path = task_dir / f'attempt-{attempt_num}' / 'report.json'
                    report_data = read_report_json(report_path)
                    if report_data:
                        task_data = extract_task_data(report_data, task_id)
                        if run_outputs_dir.exists():
                            prompt_tokens, completion_tokens = _aggregate_tokens_for_attempts(
                                run_outputs_dir / task_id, [attempt_num], average=False
                            )
                    else:
                        task_data = None
        
        if task_data:
            # Parse the repo and level
            repo, level = parse_task_id(task_id)
            task_data['repo'] = repo
            task_data['level'] = level
            task_data['prompt_tokens'] = prompt_tokens
            task_data['completion_tokens'] = completion_tokens
            all_tasks.append(task_data)
    
    return all_tasks


def sort_tasks(tasks: List[Dict]) -> List[Dict]:
    """
    Sort the tasks: level1 first, level2 second, and sort each level by task_id lexicographically
    
    Args:
        tasks: The list of task data
    
    Returns:
        The sorted list of tasks
    """
    # Separate level1 and level2
    level1_tasks = [t for t in tasks if t['level'] == 'lv1']
    level2_tasks = [t for t in tasks if t['level'] == 'lv2']
    
    # Sort separately
    level1_tasks.sort(key=lambda x: x['task_id'])
    level2_tasks.sort(key=lambda x: x['task_id'])
    
    # Merge: level1 first, level2 second
    return level1_tasks + level2_tasks


def write_csv(tasks: List[Dict], output_path: Path):
    """
    Write the task data to a CSV file
    
    Args:
        tasks: The list of task data
        output_path: The path to the output CSV file
    """
    fieldnames = [
        'task_id', 'repo', 'level', 'is_resolved',
        'pass_rate', 'total', 'num_passed', 'num_failed',
        'prompt_tokens', 'completion_tokens'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for task in tasks:
            writer.writerow({
                'task_id': task['task_id'],
                'repo': task['repo'],
                'level': task['level'],
                'is_resolved': 1 if task['resolved'] else 0,
                'pass_rate': task['pass_rate'],
                'total': task['total'],
                'num_passed': task['num_passed'],
                'num_failed': task['num_failed'],
                'prompt_tokens': task.get('prompt_tokens', 0),
                'completion_tokens': task.get('completion_tokens', 0),
            })


def main():
    parser = argparse.ArgumentParser(
        description='Calculate the evaluation outputs and generate a CSV report'
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='The path to the eval_outputs directory (e.g.: runs/2025-12-06__17-14-12/eval_outputs)'
    )
    parser.add_argument(
        '--attempt-mode',
        type=str,
        required=False,
        default='best',
        help='The attempt mode: n (number, like 1,2,3), best (highest pass_rate), or worst (lowest pass_rate)'
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='When --attempt-mode is a number k, merge metrics over attempts 1..k (pass_rate=avg, resolved=pass@k).'
    )
    
    args = parser.parse_args()
    
    # Parse the path
    eval_outputs_dir = Path(args.path)
    if not eval_outputs_dir.exists():
        print(f"Error: Path does not exist: {eval_outputs_dir}")
        return
    
    if not eval_outputs_dir.is_dir():
        print(f"Error: Path is not a directory: {eval_outputs_dir}")
        return
    
    # Process data
    print(f"Processing: {eval_outputs_dir}")
    tasks = process_eval_outputs(eval_outputs_dir, args.attempt_mode, args.merge)
    print(f"Found {len(tasks)} tasks")

    if is_int_string(args.attempt_mode) and not args.merge and len(tasks) == 0:
        print(
            "Hint: No tasks found for this attempt number. "
            "If you want pass@k style aggregation over attempts 1..k, add --merge."
        )
    
    # Sort
    sorted_tasks = sort_tasks(tasks)
    
    # Generate the output file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = eval_outputs_dir.parent
    output_filename = f'cal_eval_outputs_{timestamp}.csv'
    output_path = output_dir / output_filename
    
    # Write to CSV
    write_csv(sorted_tasks, output_path)
    print(f"Results saved to: {output_path}")
    
    # Print the statistics
    level1_count = sum(1 for t in sorted_tasks if t['level'] == 'lv1')
    level2_count = sum(1 for t in sorted_tasks if t['level'] == 'lv2')
    resolved_count = sum(1 for t in sorted_tasks if t['resolved'])
    
    print(f"\nStatistics:")
    print(f"  Level 1 tasks: {level1_count}")
    print(f"  Level 2 tasks: {level2_count}")
    print(f"  Resolved tasks: {resolved_count}")
    print(f"  Total tasks: {len(sorted_tasks)}")


if __name__ == '__main__':
    main()
