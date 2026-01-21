"""Interactive CLI for filtering Level1/Level2 case directories by pass-rate thresholds.

Usage example:
    python -m acebench.scripts.post_filter_cases \
        --input-dir outputs/2025-11-23/21-52-40 \
        --lv1-threshold 0.3 \
        --lv2-threshold 0.3 \
        --lines 100 \
        --test-count 10 \
        --commit-time 2022-05-01T00:00:00

Flags:
--input-dir (required, accepts multiple paths): run directories containing metadata_outputs/data_status.json and cases/.
--output-dir: destination root (defaults to <cases parent>/cases_filter, required when multiple --input-dir values are provided).
--lv1-threshold / --lv2-threshold: relaxed pass-rate limits in [0, 1].
--no-lv1 / --no-lv2: when set, ignore that level entirely.
--verified-list: path to a text file containing allowed case identifiers (one per line).
--lines / --test-count / --test-count-run: minimum numeric filters.
--commit-time / --upd-time: ISO timestamps enforcing earliest commit/update times.

The script lists all cases whose stored lv1_post_rate / lv2_post_rate are within the supplied thresholds,
then asks which ones to copy into the destination directory.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from acebench.utils.utils import (
    CaseRecord,
    CaseSummary,
    MetadataFilters,
    build_candidates,
    compute_case_summary,
    format_average,
    parse_user_datetime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively filter and copy case directories after adjusting pass-rate thresholds."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories (e.g., outputs/2025-11-10/20-00-00).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination root for copied cases. Defaults to <cases parent>/cases_filter.",
    )
    parser.add_argument(
        "--verified-list",
        type=Path,
        default=None,
        help=(
            "Only consider cases listed in this text file (one case id per line, e.g. "
            "'astropy__astropy.b0db0daa.test_basic_rgb.067e927c.lv1')."
        ),
    )
    parser.add_argument(
        "--no-lv1",
        action="store_true",
        help="Do not include Level1 (lv1) cases in candidates or summary.",
    )
    parser.add_argument(
        "--no-lv2",
        action="store_true",
        help="Do not include Level2 (lv2) cases in candidates or summary.",
    )
    parser.add_argument(
        "--lv1-threshold",
        type=float,
        help="New maximum acceptable Level1/F2P pass rate (0-1).",
    )
    parser.add_argument(
        "--lv2-threshold",
        type=float,
        help="New maximum acceptable Level2 pass rate (0-1).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        help="Minimum deleted lines required for a case to qualify.",
    )
    parser.add_argument(
        "--commit-time",
        type=str,
        help="Earliest allowed first_commit timestamp (ISO 8601).",
    )
    parser.add_argument(
        "--upd-time",
        type=str,
        help="Earliest allowed last_modified timestamp (ISO 8601).",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        help="Minimum discovery-phase test_count required.",
    )
    parser.add_argument(
        "--test-count-run",
        type=int,
        help="Minimum execution-phase test_count_run required.",
    )
    args = parser.parse_args()

    if args.lv1_threshold is None and args.lv2_threshold is None:
        parser.error("Provide at least one of --lv1-threshold or --lv2-threshold.")

    for name, value in (("lv1-threshold", args.lv1_threshold), ("lv2-threshold", args.lv2_threshold)):
        if value is None:
            continue
        if not (0 <= value <= 1):
            parser.error(f"{name} must be within [0, 1], got {value}.")

    try:
        args.commit_time = parse_user_datetime("--commit-time", args.commit_time)
        args.upd_time = parse_user_datetime("--upd-time", args.upd_time)
    except ValueError as exc:
        parser.error(str(exc))

    for name in ("lines", "test_count", "test_count_run"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative, got {value}.")

    if len(args.input_dir) > 1 and args.output_dir is None:
        parser.error("--output-dir is required when multiple --input-dir values are provided.")

    if args.verified_list is not None and not args.verified_list.exists():
        parser.error(f"Cannot locate --verified-list at {args.verified_list}")

    return args


def load_verified_case_ids(path: Path) -> Set[str]:
    """Load allowed case identifiers from a text file.

    Expected format (one per line):
        <repo_commit_dir>.<case_dir_name>
    Example:
        astropy__astropy.b0db0daa.test_basic_rgb.067e927c.lv1
    """
    allowed: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            allowed.add(line)
    return allowed


def case_id_from_record(record: CaseRecord) -> str:
    return f"{record.source_dir.parent.name}.{record.source_dir.name}"


def resolve_data_status_path(input_dir: Path) -> Path:
    path = input_dir / "metadata_outputs" / "data_status.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot locate data_status.json at {path}")
    return path


def resolve_destination_root(input_dir: Path, explicit_output: Optional[Path]) -> Path:
    if explicit_output:
        return explicit_output
    cases_root = input_dir / "cases"
    return cases_root.parent / "cases_filter"


def print_summary(
    records: Sequence[CaseRecord], data_status_paths: Sequence[Path], dest_root: Path
) -> None:
    print("")
    sources = ", ".join(str(path) for path in data_status_paths)
    print(f"Loaded {len(records)} eligible case(s) from {sources}")
    print(f"Destination root: {dest_root}")
    print("")

    summary: CaseSummary = compute_case_summary(records)
    overall = summary.overall_metrics

    print("Overall summary:")
    print(f"  Total candidates: {summary.total_candidates}")
    print(f"  Level1 candidates: {summary.lv1_candidates}")
    print(f"  Level2 candidates: {summary.lv2_candidates}")
    print(
        f"  Avg deleted lines: {format_average(overall['deleted_sum'], overall['deleted_samples'])}"
    )
    print(
        f"  Avg mask files: {format_average(overall['mask_file_sum'], overall['mask_file_samples'])}"
    )
    print(
        f"  Avg mask objects: {format_average(overall['mask_object_sum'], overall['mask_object_samples'])}"
    )
    print(
        f"  Avg test count: {format_average(overall['test_count_sum'], overall['test_count_samples'])}"
    )
    print(
        f"  Avg test count run: {format_average(overall['test_count_run_sum'], overall['test_count_run_samples'])}"
    )

    total_counts = compute_total_test_point_summary(records)
    print(
        "  Avg total test count (f2p+p2p): "
        f"{format_average(total_counts['overall']['total_test_count_sum'], total_counts['overall']['total_test_count_samples'])}"
    )
    print(
        "  Avg total test count run (f2p+p2p): "
        f"{format_average(total_counts['overall']['total_test_count_run_sum'], total_counts['overall']['total_test_count_run_samples'])}"
    )

    if not summary.repo_stats:
        print("")
        return

    print("")
    print("Per-repo statistics:")
    for repo in sorted(summary.repo_stats.keys()):
        stats = summary.repo_stats[repo]
        extra = total_counts['repo_stats'].get(
            repo,
            {
                'total_test_count_sum': 0.0,
                'total_test_count_samples': 0,
                'total_test_count_run_sum': 0.0,
                'total_test_count_run_samples': 0,
            },
        )
        print("-")
        print(f"{repo}:")
        print(
            f"  Cases: total={int(stats['total'])}, lv1={int(stats['lv1'])}, lv2={int(stats['lv2'])}"
        )
        print(
            f"  Avg deleted lines: {format_average(stats['deleted_sum'], stats['deleted_samples'])}"
        )
        print(
            f"  Avg mask files: {format_average(stats['mask_file_sum'], stats['mask_file_samples'])}"
        )
        print(
            f"  Avg mask objects: {format_average(stats['mask_object_sum'], stats['mask_object_samples'])}"
        )
        print(
            f"  Avg test count: {format_average(stats['test_count_sum'], stats['test_count_samples'])}"
        )
        print(
            f"  Avg test count run: {format_average(stats['test_count_run_sum'], stats['test_count_run_samples'])}"
        )
        print(
            "  Avg total test count (f2p+p2p): "
            f"{format_average(extra['total_test_count_sum'], extra['total_test_count_samples'])}"
        )
        print(
            "  Avg total test count run (f2p+p2p): "
            f"{format_average(extra['total_test_count_run_sum'], extra['total_test_count_run_samples'])}"
        )
    print("-")


def resolve_run_root(case_dir: Path) -> Optional[Path]:
    cases_root = find_cases_root(case_dir)
    if not cases_root:
        return None
    return cases_root.parent


def load_files_status_by_specs(run_root: Path) -> Dict[str, Any]:
    path = run_root / "metadata_outputs" / "files_status.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data
    return {}


def build_test_point_index(files_status_entry: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    index: Dict[str, Dict[str, Optional[float]]] = {}
    for bucket in ("p2p_files", "f2p_files"):
        items = files_status_entry.get(bucket) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            index[file_path] = {
                "test_count": item.get("test_count"),
                "test_count_run": item.get("test_count_run"),
            }
    return index


def normalize_container_test_path(path_str: str) -> str:
    value = (path_str or "").strip()
    if not value:
        return value
    if value.startswith("/testbed/"):
        return value
    return "/testbed/" + value.lstrip("/")


def resolve_case_dir_for_instance(record: CaseRecord) -> Path:
    """Return a case directory that contains instance.json.

    Some runs store case_dirs in data_status.json pointing to a different root
    (e.g., outputs/...) while the actual case artifacts live under the provided
    --input-dir (e.g., tmp/...). When _run_dir is present, prefer mapping the
    case path relative to the 'cases' directory into that run.
    """
    run_dir = getattr(record, "_run_dir", None)
    if run_dir is not None:
        cases_root = find_cases_root(record.source_dir)
        if cases_root:
            try:
                relative = record.source_dir.relative_to(cases_root)
                candidate = Path(run_dir) / "cases" / relative
                if (candidate / "instance.json").exists():
                    return candidate
            except Exception:
                pass
        candidate = Path(run_dir) / "cases" / record.source_dir.parent.name / record.source_dir.name
        if (candidate / "instance.json").exists():
            return candidate

    return record.source_dir


def compute_total_test_point_summary(records: Sequence[CaseRecord]) -> Dict[str, Any]:
    """Compute total test points across FAIL_TO_PASS + PASS_TO_PASS per case.

    Totals are derived from each case's instance.json test list, joined with
    per-file test_count/test_count_run stored in metadata_outputs/files_status.json.

    Missing file entries cause the per-case total to be treated as unavailable.
    """

    overall = {
        "total_test_count_sum": 0.0,
        "total_test_count_samples": 0,
        "total_test_count_run_sum": 0.0,
        "total_test_count_run_samples": 0,
    }
    repo_stats: Dict[str, Dict[str, float]] = {}

    files_status_cache: Dict[Path, Dict[str, Any]] = {}
    index_cache: Dict[Tuple[Path, str], Dict[str, Dict[str, Optional[float]]]] = {}

    for record in records:
        run_root = getattr(record, "_run_dir", None) or resolve_run_root(record.source_dir)
        if run_root is None:
            continue

        if run_root not in files_status_cache:
            files_status_cache[run_root] = load_files_status_by_specs(run_root)

        cache_key = (run_root, record.repo)
        if cache_key not in index_cache:
            entry = files_status_cache[run_root].get(record.repo) or {}
            if isinstance(entry, dict):
                index_cache[cache_key] = build_test_point_index(entry)
            else:
                index_cache[cache_key] = {}

        test_index = index_cache[cache_key]

        case_dir = resolve_case_dir_for_instance(record)
        instance_path = case_dir / "instance.json"
        if not instance_path.exists():
            continue
        try:
            with open(instance_path, "r", encoding="utf-8") as fh:
                instance = json.load(fh)
        except Exception:
            continue

        f2p_tests = instance.get("FAIL_TO_PASS") or []
        p2p_tests = instance.get("PASS_TO_PASS") or []
        if not isinstance(f2p_tests, list) or not isinstance(p2p_tests, list):
            continue

        combined: Set[str] = set()
        for item in [*f2p_tests, *p2p_tests]:
            if isinstance(item, str) and item.strip():
                combined.add(normalize_container_test_path(item))

        if not combined:
            continue

        total_test_count: Optional[float] = 0.0
        total_test_count_run: Optional[float] = 0.0

        for test_path in combined:
            info = test_index.get(test_path)
            if not info:
                total_test_count = None
                total_test_count_run = None
                break

            tc = info.get("test_count")
            tcr = info.get("test_count_run")

            if tc is None:
                total_test_count = None
            elif total_test_count is not None:
                total_test_count += float(tc)

            if tcr is None:
                total_test_count_run = None
            elif total_test_count_run is not None:
                total_test_count_run += float(tcr)

        stats = repo_stats.setdefault(
            record.repo,
            {
                "total_test_count_sum": 0.0,
                "total_test_count_samples": 0,
                "total_test_count_run_sum": 0.0,
                "total_test_count_run_samples": 0,
            },
        )

        if total_test_count is not None:
            overall["total_test_count_sum"] += total_test_count
            overall["total_test_count_samples"] += 1
            stats["total_test_count_sum"] += total_test_count
            stats["total_test_count_samples"] += 1

        if total_test_count_run is not None:
            overall["total_test_count_run_sum"] += total_test_count_run
            overall["total_test_count_run_samples"] += 1
            stats["total_test_count_run_sum"] += total_test_count_run
            stats["total_test_count_run_samples"] += 1

    return {"overall": overall, "repo_stats": repo_stats}


def prompt_for_selection(records: Sequence[CaseRecord]) -> List[CaseRecord]:
    if not records:
        return []
    while True:
        user_input = input(
            "Select cases to copy (comma-separated indexes, 'all', or 'none') [all]: "
        ).strip()
        if not user_input or user_input.lower() == "all":
            return list(records)
        if user_input.lower() == "none":
            return []
        try:
            indexes = {int(item) for item in user_input.split(",")}
        except ValueError:
            print("Invalid input. Use numbers like 1,3,5 or keywords 'all'/'none'.")
            continue
        selected = []
        for idx in indexes:
            if 1 <= idx <= len(records):
                selected.append(records[idx - 1])
            else:
                print(f"Index {idx} is out of range. There are {len(records)} items.")
                selected = None
                break
        if selected is None:
            continue
        return selected


def find_cases_root(case_dir: Path) -> Optional[Path]:
    for parent in case_dir.parents:
        if parent.name == "cases":
            return parent
    return None


def materialize_case(
    record: CaseRecord,
    dest_root: Path,
) -> Tuple[Optional[Path], Optional[str]]:
    if not record.source_dir.exists():
        return None, f"Source directory missing: {record.source_dir}"

    cases_root = find_cases_root(record.source_dir)
    if not cases_root:
        return None, f"Cannot locate 'cases' root for {record.source_dir}"

    relative = record.source_dir.relative_to(cases_root)
    destination = dest_root / relative

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    shutil.copytree(record.source_dir, destination)
    return destination, None


def copy_selected_cases(
    records: Sequence[CaseRecord],
    dest_root: Path,
) -> None:
    successes = 0
    errors: List[str] = []
    for record in records:
        # Copy current record and report result for transparency
        destination, error_message = materialize_case(record, dest_root)
        if error_message:
            errors.append(error_message)
            continue
        successes += 1
        print(f"Copied: {record.source_dir} -> {destination}")
    if errors:
        print("\nWarnings:")
        for msg in errors:
            print(f"  - {msg}")
    print(f"\nProcessed {len(records)} record(s). Successful operations: {successes}.")


def main() -> None:
    args = parse_args()
    input_dirs: List[Path] = args.input_dir
    explicit_output = args.output_dir

    base_input_dir = input_dirs[0]
    dest_root = resolve_destination_root(base_input_dir, explicit_output)

    metadata_filters = MetadataFilters(
        min_deleted_lines=args.lines,
        min_commit_time=args.commit_time,
        min_update_time=args.upd_time,
        min_test_count=args.test_count,
        min_test_count_run=args.test_count_run,
    )

    data_status_paths: List[Path] = []
    candidates: List[CaseRecord] = []

    for run_dir in input_dirs:
        data_status_path = resolve_data_status_path(run_dir)
        data_status_paths.append(data_status_path)

        with open(data_status_path, "r", encoding="utf-8") as fh:
            data: Dict[str, Dict[str, Dict]] = json.load(fh)

        run_candidates = build_candidates(
            data=data,
            lv1_threshold=None if args.no_lv1 else args.lv1_threshold,
            lv2_threshold=None if args.no_lv2 else args.lv2_threshold,
            metadata_filters=metadata_filters,
        )
        for record in run_candidates:
            setattr(record, "_run_dir", run_dir)
        candidates.extend(run_candidates)

    if args.verified_list is not None:
        allowed_case_ids = load_verified_case_ids(args.verified_list)
        candidates = [
            record for record in candidates if case_id_from_record(record) in allowed_case_ids
        ]

    if not candidates:
        print("No cases matched the provided filters.")
        return

    print_summary(candidates, data_status_paths, dest_root)
    selected = prompt_for_selection(candidates)

    if not selected:
        print("No cases selected. Nothing to do.")
        return

    copy_selected_cases(
        records=selected,
        dest_root=dest_root,
    )


if __name__ == "__main__":
    main()
