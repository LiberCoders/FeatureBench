from __future__ import annotations

import json

import pytest

from featurebench.harness.report import save_report, save_summary_report


def test_reports_are_saved_atomically(tmp_path) -> None:
    report = {"task.lv1": {"resolved": True}}

    save_report(report, tmp_path)
    summary_path = save_summary_report({"attempt_1": {"total_instances": 1}}, tmp_path)

    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "attempt_1": {"total_instances": 1}
    }
    assert not summary_path.with_suffix(".json.tmp").exists()


def test_failed_report_serialization_preserves_existing_report(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        save_report({"invalid": object()}, tmp_path)

    assert report_path.read_text(encoding="utf-8") == '{"preserved": true}\n'
    assert not report_path.with_suffix(".json.tmp").exists()
