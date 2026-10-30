"""Unified CLI for data production, inference, and evaluation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import TextIO


def _print_help(stream: TextIO | None = None) -> None:
    out = stream if stream is not None else sys.stdout
    out.write(
        "FeatureBench CLI\n"
        "\n"
        "Usage:\n"
        "  fb discover [SOURCE] [OPTIONS]\n"
        "  fb build SOURCE [OPTIONS]\n"
        "  fb generate ENVIRONMENTS --adapter featurebench [OPTIONS]\n"
        "  fb configure RUN_DIRECTORY\n"
        "  fb --resume RUN_DIRECTORY [--select POLICY]\n"
        "  fb infer [INFER_ARGS...]\n"
        "  fb eval [EVAL_ARGS...]\n"
        "  fb pull [PULL_ARGS...]\n"
        "\n"
        "Data factory:\n"
        "  discover   Find candidate GitHub repositories\n"
        "  build      Build and validate reproducible environments\n"
        "  generate   Generate tasks through a registered Adapter\n"
        "  configure  Edit an inactive run before resuming it\n"
        "\n"
        "Benchmark:\n"
        "  infer  Run inference (supports all existing run_infer args)\n"
        "  eval   Run harness evaluation (supports all existing harness args)\n"
        "  pull   Pre-pull images (supports --mode, and other pull_images args)\n"
        "\n"
        "Examples:\n"
        "  fb discover\n"
        "  fb build owner/project\n"
        "  fb generate outputs/build-.../environments.jsonl --adapter featurebench\n"
        "  fb --resume outputs/build-... --select non-success\n"
        "  fb infer --data-version v1.1 --agent mini_swe_agent --model openai/gpt-4o --split fast\n"
        "  fb eval --predictions-path runs/xxx/output.jsonl --split fast\n"
        "  fb pull --mode fast\n"
        "\n"
        "Use command-specific help:\n"
        "  fb discover --help\n"
        "  fb build --help\n"
        "  fb generate --help\n"
        "  fb infer --help\n"
        "  fb eval --help\n"
        "  fb pull --help\n"
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
    except KeyboardInterrupt:
        return 130
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
    from featurebench.infer.run_infer import main as infer_main

    return _run_with_patched_argv("fb infer", args, infer_main)


def _dispatch_eval(args: Sequence[str]) -> int:
    from featurebench.harness.run_evaluation import main as harness_main

    return _run_with_patched_argv("fb eval", args, harness_main)


def _dispatch_pull(args: Sequence[str]) -> int:
    from featurebench.scripts.pull_images import main as pull_main

    try:
        return int(pull_main(list(args)))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


def _dispatch_factory(args: Sequence[str]) -> int:
    from codeworld.cli import main as factory_main

    return factory_main(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if not args:
        _print_help()
        return 0

    command, command_args = args[0], args[1:]
    if command in {"-h", "--help"}:
        _print_help()
        return 0

    if command in {"discover", "build", "generate", "configure", "--resume", "--select"}:
        return _dispatch_factory(args)
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
