"""Stage-level application services composed by the CLI."""

from codeworld.application.build import (
    count_build_targets,
    parse_build_removal_targets,
    remove_build_environments,
    run_build,
)
from codeworld.application.discover import run_discover
from codeworld.application.generate import run_generate

__all__ = [
    "count_build_targets",
    "parse_build_removal_targets",
    "remove_build_environments",
    "run_build",
    "run_discover",
    "run_generate",
]
