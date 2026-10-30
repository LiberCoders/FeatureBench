from __future__ import annotations

import io
import logging
import tarfile

import pandas as pd
import pytest

from featurebench.harness import review_codes
from featurebench.harness.review_codes import _copy_text_to_container, copy_from_container


class ArchiveContainer:
    def __init__(self, archive: bytes = b"") -> None:
        self.archive = archive
        self.uploads: list[tuple[str, bytes]] = []
        self.cleaned_up = False

    def get_archive(self, source: str):
        return iter([self.archive]), {}

    def put_archive(self, destination: str, data) -> None:
        payload = data.read() if hasattr(data, "read") else data
        self.uploads.append((destination, payload))

    def stop(self, timeout: int) -> None:
        self.cleaned_up = True

    def remove(self) -> None:
        self.cleaned_up = True


def _make_archive(*, unsafe_link: bool = False) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        root = tarfile.TarInfo("testbed")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        if unsafe_link:
            link = tarfile.TarInfo("testbed/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            archive.addfile(link)
        else:
            content = b"review content\n"
            member = tarfile.TarInfo("testbed/result.txt")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def _read_uploaded_file(payload: bytes, name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r") as archive:
        extracted = archive.extractfile(archive.getmember(name))
        assert extracted is not None
        return extracted.read()


def test_copy_text_to_container_uses_a_valid_file_archive() -> None:
    container = ArchiveContainer()

    _copy_text_to_container(container, "patch content\n", "/tmp/agent_patch.diff")

    assert len(container.uploads) == 1
    destination, payload = container.uploads[0]
    assert destination == "/tmp"
    assert _read_uploaded_file(payload, "agent_patch.diff") == b"patch content\n"


def test_copy_from_container_replaces_review_tree_safely(tmp_path) -> None:
    destination = tmp_path / "review" / "testbed"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("stale", encoding="utf-8")
    container = ArchiveContainer(_make_archive())

    copy_from_container(container, "/testbed", destination)

    assert not (destination / "stale.txt").exists()
    assert (destination / "result.txt").read_text(encoding="utf-8") == "review content\n"


def test_copy_from_container_rejects_links_outside_review_tree(tmp_path) -> None:
    destination = tmp_path / "review" / "testbed"
    container = ArchiveContainer(_make_archive(unsafe_link=True))

    with pytest.raises(tarfile.FilterError):
        copy_from_container(container, "/testbed", destination)

    assert not (tmp_path / "escape").exists()
    assert not destination.exists()


def test_level1_review_export_copies_both_patches_and_reports_success(
    monkeypatch, tmp_path
) -> None:
    container = ArchiveContainer()

    class Containers:
        def run(self, *args, **kwargs):
            return container

    class Client:
        containers = Containers()

    monkeypatch.setattr(review_codes.docker, "from_env", lambda: Client())
    monkeypatch.setattr(review_codes, "exec_run_simple", lambda *args, **kwargs: (0, ""))
    monkeypatch.setattr(review_codes, "copy_from_container", lambda *args, **kwargs: None)

    agent_patch = "diff --git a/solution.js b/solution.js\n+const answer = 42;\n"
    success = review_codes.save_review_codes_level1(
        pd.Series(
            {
                "instance_id": "owner__repo.task.lv1",
                "patch": "diff --git a/source.py b/source.py\n",
            }
        ),
        agent_patch,
        tmp_path,
        "example/image",
        logging.getLogger("test-review-export"),
    )

    assert success is True
    assert len(container.uploads) == 2
    assert _read_uploaded_file(container.uploads[1][1], "agent_patch.diff") == agent_patch.encode()
    assert container.cleaned_up is True


def test_exec_run_simple_delegates_to_bounded_execution(monkeypatch) -> None:
    captured = {}

    def fake_exec(container, command, **kwargs):
        captured.update(command=command, **kwargs)
        return 0, b"output"

    monkeypatch.setattr(review_codes, "exec_run_with_timeout", fake_exec)

    exit_code, output = review_codes.exec_run_simple(object(), "echo ok", timeout=17)

    assert (exit_code, output) == (0, "output")
    assert captured == {
        "command": "echo ok",
        "timeout": 17,
        "user": "root",
        "workdir": "/",
    }
