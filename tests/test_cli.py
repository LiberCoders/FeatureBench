from __future__ import annotations

import codeworld.cli
from featurebench.cli import main


def test_root_help_exposes_factory_and_benchmark_commands(capsys) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "fb discover" in output
    assert "fb generate" in output
    assert "fb infer" in output
    assert "fb eval" in output
    assert "fb data" not in output


def test_factory_commands_dispatch_without_a_legacy_data_wrapper(monkeypatch) -> None:
    observed: list[str] = []

    def fake_main(arguments) -> int:
        observed.extend(arguments)
        return 17

    monkeypatch.setattr(codeworld.cli, "main", fake_main)

    assert main(["generate", "environments.jsonl", "--adapter", "featurebench"]) == 17
    assert observed == ["generate", "environments.jsonl", "--adapter", "featurebench"]


def test_removed_data_command_is_rejected(capsys) -> None:
    assert main(["data"]) == 2
    assert "Unknown command: data" in capsys.readouterr().err
