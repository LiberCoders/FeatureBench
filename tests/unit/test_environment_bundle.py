from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeworld.environment.bundle import read_environment_bundle, write_environment_bundle
from codeworld.environment.jsonl import (
    EnvironmentJsonlAppender,
    capture_environment_snapshot,
    iter_environments,
    write_environment_snapshot,
)
from codeworld.environment.models import EnvironmentRecord
from codeworld.environment.runner_contract import CollectedTestPoint
from codeworld.errors import ContractError
from codeworld.models import RunnerLimits, RuntimeResources, RuntimeSpec


def environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        environment_id="environment-1",
        repository="https://example.com/project.git",
        commit="a" * 40,
        language="go",
        gpu_requirement="none",
        image="example/project@sha256:" + "b" * 64,
        dockerfile=(
            "FROM golang:1.24-bookworm\n"
            "RUN git clone https://example.com/project.git /testbed\n"
            "WORKDIR /testbed\n"
        ),
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\n# Go needs no environment manager here.\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# A real generated runner would translate `go test -json` to Runner V1.\n"
                "request=''\nresponse=''\n"
                "while (($#)); do shift; done\n"
                'echo "$request$response" >/dev/null\n'
            ),
            limits=RunnerLimits(
                collect_timeout_seconds=90,
                baseline_total_timeout_seconds=600,
            ),
            resources=RuntimeResources(
                network="none",
                cpus=4,
                memory="8g",
                gpus=None,
            ),
        ),
        verified_test_points=(
            CollectedTestPoint(id="TestFeature", path="feature_test.go"),
            CollectedTestPoint(id="TestOther", path="other_test.go"),
        ),
    )


def test_environment_bundle_has_external_executable_control_artifacts(tmp_path: Path) -> None:
    expected = environment()
    root = write_environment_bundle(tmp_path / "environment", expected)
    assert {path.name for path in root.iterdir()} == {
        "Dockerfile",
        "activate",
        "environment.json",
        "runner",
    }
    assert (root / "activate").stat().st_mode & 0o111
    assert (root / "runner").stat().st_mode & 0o111
    assert "format" not in json.loads((root / "environment.json").read_text(encoding="utf-8"))
    assert read_environment_bundle(root) == expected


def test_environment_bundle_detects_tampered_artifacts(tmp_path: Path) -> None:
    root = write_environment_bundle(tmp_path / "environment", environment())
    (root / "runner").write_text("#!/usr/bin/env bash\nmodified\n", encoding="utf-8")
    with pytest.raises(ContractError, match="digest mismatch"):
        read_environment_bundle(root)


def test_environment_jsonl_appender_is_the_appendable_machine_interface(tmp_path: Path) -> None:
    output = tmp_path / "environments.jsonl"
    first = environment()
    second = first.model_copy(
        update={
            "environment_id": "environment-2",
            "commit": "c" * 40,
            "image": "example/project@sha256:" + "d" * 64,
        }
    )
    with EnvironmentJsonlAppender(output) as writer:
        assert writer.append(first)
        assert writer.append(second)
        assert not writer.append(first)
        conflicting = first.model_copy(update={"image": "sha256:" + "e" * 64})
        with pytest.raises(ValueError, match="different content"):
            writer.append(conflicting)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert tuple(iter_environments(output)) == (first, second)


def test_environment_snapshot_ignores_only_an_incomplete_trailing_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "environments.jsonl"
    expected = environment()
    source.write_bytes(expected.model_dump_json().encode() + b'\n{"environment_id":"partial"')

    snapshot = capture_environment_snapshot(source)

    assert snapshot.records == (expected,)
    assert snapshot.source_size_bytes == source.stat().st_size
    assert snapshot.ignored_tail_bytes > 0


def test_environment_snapshot_is_a_durable_canonical_jsonl(tmp_path: Path) -> None:
    first = environment()
    second = first.model_copy(update={"environment_id": "environment-2"})
    destination = tmp_path / "snapshot.jsonl"

    assert write_environment_snapshot(destination, (first, second)) == 2
    assert capture_environment_snapshot(destination).records == (first, second)


def test_one_jsonl_line_rehydrates_every_environment_artifact_in_isolation(
    tmp_path: Path,
) -> None:
    expected = environment()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    jsonl = isolated / "one-environment.jsonl"
    jsonl.write_text(expected.model_dump_json() + "\n", encoding="utf-8")

    (rehydrated,) = tuple(iter_environments(jsonl))
    bundle = write_environment_bundle(isolated / "audit", rehydrated)

    assert read_environment_bundle(bundle) == expected
    assert (bundle / "Dockerfile").read_text(encoding="utf-8") == expected.dockerfile
    assert (bundle / "activate").read_text(encoding="utf-8") == expected.runtime.activation
    assert (bundle / "runner").read_text(encoding="utf-8") == expected.runtime.runner
    assert not any("featurebench-sources-" in path.name for path in bundle.iterdir())
    payload = expected.model_dump(mode="json")
    assert "runner_protocol" not in payload
    assert payload["runtime"]["resources"]["memory"] == "8g"
    assert payload["verified_test_points"] == [
        {"id": "TestFeature", "path": "feature_test.go"},
        {"id": "TestOther", "path": "other_test.go"},
    ]
    assert "tests" not in payload
    assert "baseline" not in payload
