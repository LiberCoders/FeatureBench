from pathlib import Path

import pytest

from codeworld.config import DiscoverySettings
from codeworld.repositories.targets import (
    normalize_repository_target,
    read_repository_targets,
)


def test_repository_targets_accept_identities_and_github_urls(tmp_path: Path) -> None:
    source = tmp_path / "repositories.txt"
    source.write_text(
        "# selected repositories\n"
        "pydantic/pydantic-ai\n"
        "https://github.com/astral-sh/uv.git\n"
        "PYDANTIC/PYDANTIC-AI\n"
        "\n",
        encoding="utf-8",
    )

    assert read_repository_targets(source) == (
        "pydantic/pydantic-ai",
        "astral-sh/uv",
    )
    assert normalize_repository_target("https://www.github.com/owner/repo/") == "owner/repo"


def test_repository_targets_report_the_invalid_line(tmp_path: Path) -> None:
    source = tmp_path / "repositories.txt"
    source.write_text("owner/repo\nhttps://example.com/no/repo\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"repositories\.txt:2"):
        read_repository_targets(source)


def test_repository_targets_reject_an_empty_list(tmp_path: Path) -> None:
    source = tmp_path / "repositories.txt"
    source.write_text("# no repositories\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no repositories"):
        read_repository_targets(source)


def test_discovery_yaml_accepts_txt_input_and_rejects_ambiguous_targets() -> None:
    settings = DiscoverySettings(repositories_file=Path("inputs/repositories.txt"))
    assert settings.repositories_file == Path("inputs/repositories.txt")

    with pytest.raises(ValueError, match="cannot be used together"):
        DiscoverySettings(
            repositories_file=Path("inputs/repositories.txt"),
            target_repositories=("owner/repo",),
        )
