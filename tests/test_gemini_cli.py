from __future__ import annotations

from pathlib import Path

from featurebench.infer.agents.gemini_cli import GeminiCliAgent


class FakeContainerManager:
    def __init__(self, fallback: str = "") -> None:
        self.fallback = fallback
        self.copied: list[tuple[str, Path]] = []

    def exec_command(self, _container: object, _command: str) -> tuple[int, str]:
        return 0, self.fallback

    def copy_from_container(
        self,
        _container: object,
        source: str,
        destination: Path,
    ) -> bool:
        self.copied.append((source, destination))
        return "failed" not in source


def _agent(manager: FakeContainerManager) -> GeminiCliAgent:
    return GeminiCliAgent(manager)  # type: ignore[arg-type]


def test_failure_hook_copies_exact_unique_reports_from_the_log(tmp_path: Path) -> None:
    manager = FakeContainerManager(fallback="/tmp/gemini-client-error-stale.json\n")
    log_file = tmp_path / "infer.log"
    log_file.write_text(
        "Full report available at: /tmp/gemini-client-error-first.json\n"
        "Full report available at: /tmp/gemini-client-error-first.json\n"
        "Full report available at: /tmp/gemini-client-error-failed.json\n",
        encoding="utf-8",
    )

    _agent(manager).failure_hook(object(), log_file)

    assert manager.copied == [
        (
            "/tmp/gemini-client-error-first.json",
            tmp_path / "gemini-client-error-first.json",
        ),
        (
            "/tmp/gemini-client-error-failed.json",
            tmp_path / "gemini-client-error-failed.json",
        ),
    ]
    log = log_file.read_text(encoding="utf-8")
    assert "Copied: /tmp/gemini-client-error-first.json" in log
    assert "Failed to copy: /tmp/gemini-client-error-failed.json" in log
    assert "stale" not in log


def test_failure_hook_falls_back_to_reports_listed_by_the_container(tmp_path: Path) -> None:
    manager = FakeContainerManager(
        fallback=("/tmp/gemini-client-error-newest.json\n/tmp/gemini-client-error-older.json\n")
    )
    log_file = tmp_path / "infer.log"
    log_file.write_text("Gemini failed without a report path\n", encoding="utf-8")

    _agent(manager).failure_hook(object(), log_file)

    assert [source for source, _destination in manager.copied] == [
        "/tmp/gemini-client-error-newest.json",
        "/tmp/gemini-client-error-older.json",
    ]
