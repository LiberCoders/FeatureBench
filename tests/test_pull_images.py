from __future__ import annotations

from featurebench.scripts import pull_images
from featurebench.scripts.pull_images import ImagePullSpec


class _Proc:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_make_image_pull_spec_adds_official_tag_for_prefix_override(monkeypatch) -> None:
    monkeypatch.setenv("FEATUREBENCH_IMAGE_PREFIX", "docker.1ms.run")

    spec = pull_images._make_image_pull_spec(
        "docker.io/libercoders/featurebench-specs_trl-instance_725be9aa"
    )

    assert spec == ImagePullSpec(
        image="docker.1ms.run/libercoders/featurebench-specs_trl-instance_725be9aa",
        official_tag="docker.io/libercoders/featurebench-specs_trl-instance_725be9aa",
    )


def test_make_image_pull_spec_skips_official_tag_without_prefix_override(monkeypatch) -> None:
    monkeypatch.delenv("FEATUREBENCH_IMAGE_PREFIX", raising=False)

    spec = pull_images._make_image_pull_spec(
        "libercoders/featurebench-specs_trl-instance_725be9aa"
    )

    assert spec == ImagePullSpec(
        image="docker.io/libercoders/featurebench-specs_trl-instance_725be9aa",
        official_tag=None,
    )


def test_docker_pull_tags_official_image(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _Proc(stdout="ok")

    monkeypatch.setattr(pull_images.subprocess, "run", fake_run)

    res = pull_images._docker_pull(
        ImagePullSpec(
            image="docker.1ms.run/libercoders/featurebench-specs_trl-instance_725be9aa",
            official_tag="docker.io/libercoders/featurebench-specs_trl-instance_725be9aa",
        ),
        capture_output=True,
    )

    assert res.ok is True
    assert res.tagged_official is True
    assert calls == [
        ["docker", "pull", "docker.1ms.run/libercoders/featurebench-specs_trl-instance_725be9aa"],
        [
            "docker",
            "tag",
            "docker.1ms.run/libercoders/featurebench-specs_trl-instance_725be9aa",
            "docker.io/libercoders/featurebench-specs_trl-instance_725be9aa",
        ],
    ]

