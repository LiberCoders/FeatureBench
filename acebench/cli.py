"""Unified CLI entrypoint for core ACE-Bench workflows.

Commands:
  - ace infer ...   -> acebench.infer.run_infer
  - ace eval ...    -> acebench.harness.run_evaluation
  - ace pull ...    -> acebench.scripts.pull_images
"""

from __future__ import annotations

import sys
from typing import Callable, Sequence


def _print_help(stream=None) -> None:
    out = stream if stream is not None else sys.stdout
    out.write(
        "ACE-Bench CLI\n"
        "\n"
        "Usage:\n"
        "  ace infer [INFER_ARGS...]\n"
        "  ace eval [EVAL_ARGS...]\n"
        "  ace pull [PULL_ARGS...]\n"
        "\n"
        "Core commands:\n"
        "  infer  Run inference (supports all existing run_infer args)\n"
        "  eval   Run harness evaluation (supports all existing harness args)\n"
        "  pull   Pre-pull images (supports --mode, and other pull_images args)\n"
        "\n"
        "Examples:\n"
        "  ace infer --agent mini_swe_agent --model openai/gpt-4o --split lite\n"
        "  ace eval --predictions-path runs/xxx/output.jsonl --split lite\n"
        "  ace pull --mode full\n"
        "\n"
        "Use command-specific help:\n"
        "  ace infer --help\n"
        "  ace eval --help\n"
        "  ace pull --help\n"
    )


def _run_with_patched_argv(
    argv0: str,
    args: Sequence[str],
    fn: Callable[[], int | None],
) -> int:
    original_argv = sys.argv
    sys.argv = [argv0, *args]
    try:
        result = fn()
        if result is None:
            return 0
        return int(result)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = original_argv


def _dispatch_infer(args: Sequence[str]) -> int:
    from acebench.infer.run_infer import main as infer_main

    return _run_with_patched_argv("ace infer", args, infer_main)


def _dispatch_eval(args: Sequence[str]) -> int:
    from acebench.harness.run_evaluation import main as harness_main

    return _run_with_patched_argv("ace eval", args, harness_main)


def _dispatch_pull(args: Sequence[str]) -> int:
    from acebench.scripts.pull_images import main as pull_main

    try:
        return int(pull_main(list(args)))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if not args:
        _print_help()
        return 0

    command, command_args = args[0], args[1:]
    if command in {"-h", "--help"}:
        _print_help()
        return 0

    if command == "infer":
        return _dispatch_infer(command_args)
    if command == "eval":
        return _dispatch_eval(command_args)
    if command == "pull":
        return _dispatch_pull(command_args)

    print(f"Unknown command: {command}", file=sys.stderr)
    _print_help(stream=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
