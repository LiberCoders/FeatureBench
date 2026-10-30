from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeworld.config import (
    SourceEndpointSettings,
    SourceMetricsSettings,
    SourceServiceSettings,
    SourceSettings,
)
from codeworld.errors import DockerBuildError, DockerBuildStep, InfrastructureError
from codeworld.sources import (
    SourceAttemptError,
    SourceAttemptMetric,
    SourceChoice,
    SourceManager,
)


def _settings(tmp_path: Path, *, enabled: bool = True) -> SourceSettings:
    return SourceSettings(
        enabled=enabled,
        cooldown_seconds=60,
        maximum_cooldown_seconds=600,
        state_path=tmp_path / "state.json",
        metrics=SourceMetricsSettings(path=tmp_path / "metrics.jsonl"),
        services={
            "github_git": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(name="mirror", url="https://mirror.example"),
                    SourceEndpointSettings(
                        name="official", url="https://github.com", canonical=True
                    ),
                ),
            ),
            "docker_io": SourceServiceSettings(
                endpoints=(
                    SourceEndpointSettings(name="mirror", url="https://docker.example/cache"),
                    SourceEndpointSettings(
                        name="official", url="https://docker.io", canonical=True
                    ),
                ),
            ),
        },
    )


def _build_failure(
    instruction: str,
    output: str,
    *,
    timed_out: bool = False,
) -> DockerBuildError:
    return DockerBuildError(
        exit_code=None if timed_out else 1,
        timeout_seconds=60 if timed_out else None,
        stderr=output,
        failed_step=DockerBuildStep(
            step_id="8",
            stage="stage-1 3/6",
            instruction=instruction,
            output=output,
        ),
    )


@pytest.mark.asyncio
async def test_source_failure_rotates_and_persists_metrics(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path), run_id="run")
    attempted: list[str] = []

    async def operation(choice: SourceChoice) -> str:
        endpoint = choice.endpoint
        attempted.append(endpoint.name)
        if endpoint.name == "mirror":
            raise SourceAttemptError("connection reset")
        return "ok"

    assert await manager.execute("github_git", operation) == "ok"
    assert attempted == ["mirror", "official"]
    assert [choice.endpoint.name for choice in await manager.ordered("github_git")] == [
        "official",
        "mirror",
    ]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["endpoints"]["github_git/mirror"]["consecutive_failures"] == 1
    metrics = [
        SourceAttemptMetric.model_validate_json(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(item.endpoint, item.status) for item in metrics] == [
        ("mirror", "failed"),
        ("official", "succeeded"),
    ]


@pytest.mark.asyncio
async def test_source_attempt_limit_reserves_canonical_as_final_fallback(
    tmp_path: Path,
) -> None:
    endpoints = (
        *(
            SourceEndpointSettings(name=f"mirror-{index}", url=f"https://mirror-{index}.example")
            for index in range(1, 5)
        ),
        SourceEndpointSettings(
            name="official",
            url="https://github.com",
            canonical=True,
        ),
    )
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "github_git": SourceServiceSettings(
                    endpoints=endpoints,
                    maximum_attempts=3,
                )
            }
        }
    )
    manager = SourceManager(settings)
    attempted: list[str] = []

    async def fail(choice: SourceChoice) -> None:
        attempted.append(choice.endpoint.name)
        raise SourceAttemptError("unavailable")

    with pytest.raises(InfrastructureError, match="all sources failed"):
        await manager.execute("github_git", fail)

    assert attempted == ["mirror-1", "mirror-2", "official"]


@pytest.mark.asyncio
async def test_build_plan_translates_one_download_policy_to_native_arguments(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "pypi": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="official",
                            url="https://pypi.org/simple",
                            canonical=True,
                        ),
                    )
                )
            }
        }
    )

    plan = await SourceManager(settings).build_plan()

    assert plan.arguments["CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS"] == "30"
    assert plan.arguments["CW_DOWNLOAD_TIMEOUT_SECONDS"] == "300"
    assert plan.arguments["CW_DOWNLOAD_RETRIES"] == "5"
    assert plan.arguments["PIP_DEFAULT_TIMEOUT"] == "300"
    assert plan.arguments["PIP_RETRIES"] == "5"
    assert plan.arguments["UV_HTTP_CONNECT_TIMEOUT"] == "30"
    assert plan.arguments["UV_HTTP_TIMEOUT"] == "300"
    assert plan.arguments["UV_HTTP_RETRIES"] == "5"
    assert "pypi" in plan.managed_choices({"PIP_RETRIES": "2"})
    assert "pypi" not in plan.managed_choices({"PIP_INDEX_URL": "https://custom.example"})


@pytest.mark.asyncio
async def test_disabled_sources_use_only_canonical_and_transform_safely(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path, enabled=False))
    github = (await manager.ordered("github_git"))[0]
    docker = (await manager.ordered("docker_io"))[0]
    assert github.endpoint.name == "official"
    assert manager.resolve_url(github, "https://github.com/owner/repo.git") == (
        "https://github.com/owner/repo.git"
    )
    request = manager.adapters.image_request("python:3.12")
    assert request is not None
    assert manager.adapters.image_reference(request, endpoint_url=docker.endpoint.url) == (
        "docker.io/library/python:3.12"
    )


@pytest.mark.asyncio
async def test_image_adapter_rotates_sources_and_restores_canonical_tag(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path), run_id="image")
    pulls: list[str] = []
    tags: list[tuple[str, str]] = []

    async def pull(reference: str) -> None:
        pulls.append(reference)
        if reference.startswith("docker.example/"):
            raise SourceAttemptError("connection reset")

    async def tag(source: str, target: str) -> None:
        tags.append((source, target))

    assert await manager.pull_image("python:3.12", pull=pull, tag=tag)
    assert pulls == [
        "docker.example/cache/library/python:3.12",
        "docker.io/library/python:3.12",
    ]
    assert tags == [("docker.io/library/python:3.12", "python:3.12")]


def test_mirror_credentials_are_never_forwarded_to_third_party() -> None:
    with pytest.raises(ValueError, match="credentials may only"):
        SourceServiceSettings(
            endpoints=(
                SourceEndpointSettings(
                    name="proxy",
                    url="https://proxy.example",
                    forward_credentials=True,
                ),
                SourceEndpointSettings(
                    name="official", url="https://api.github.com", canonical=True
                ),
            ),
        )


def test_apt_certificate_bootstrap_failure_is_classified_for_failover(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "ubuntu_apt": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://mirror.example/ubuntu",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="http://archive.ubuntu.com/ubuntu",
                            canonical=True,
                        ),
                    ),
                )
            }
        }
    )
    manager = SourceManager(settings)
    choices = {
        "ubuntu_apt": SourceChoice(
            service="ubuntu_apt",
            endpoint=settings.services["ubuntu_apt"].endpoints[0],
        )
    }
    output = (
        "Failed to fetch https://mirror.example/ubuntu/dists/jammy/InRelease "
        "Certificate verification failed: The certificate is NOT trusted. "
        "Could not handshake."
    )
    assert manager.build_failure_service(_build_failure("RUN apt-get update", output), choices) == (
        "ubuntu_apt"
    )


def test_git_low_speed_failure_selects_github_fallback(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="github_git",
        endpoint=SourceEndpointSettings(name="mirror", url="https://mirror.example"),
    )
    output = (
        "fatal: unable to access 'https://mirror.example/https://github.com/org/repo.git/': "
        "Operation too slow. Less than 1024 bytes/sec transferred the last 30 seconds"
    )
    assert (
        manager.build_failure_service(
            _build_failure("RUN git clone https://github.com/org/repo.git", output),
            {"github_git": choice},
        )
        == "github_git"
    )


def test_conda_connection_failed_selects_the_next_channel(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        "conda": SourceChoice(
            service="conda",
            endpoint=SourceEndpointSettings(
                name="tuna",
                url="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
            ),
        )
    }
    failure = _build_failure(
        'RUN conda install --channel "$CONDA_CHANNEL_URL" python=3.13',
        "CondaHTTPError: HTTP 000 CONNECTION FAILED for url "
        "<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/"
        "linux-64/repodata.json>",
    )

    assert manager.build_failure_service(failure, choices) == "conda"


@pytest.mark.parametrize(
    ("service", "endpoint", "instruction", "output"),
    (
        (
            "playwright",
            "https://cdn.npmmirror.com/binaries/playwright",
            "RUN python -m playwright install chromium",
            "Download failed: https://cdn.npmmirror.com/binaries/playwright/builds/"
            "cft/151.0.7922.34/linux64/chrome-linux64.zip",
        ),
        (
            "rustup",
            "https://rsproxy.cn",
            "RUN curl -sSf https://sh.rustup.rs | sh -s -- -y",
            "curl: (92) HTTP/2 stream 1 was not closed cleanly: PROTOCOL_ERROR "
            "while downloading https://rsproxy.cn/dist/channel-rust-stable.toml",
        ),
    ),
)
def test_binary_distribution_failures_select_their_independent_source(
    tmp_path: Path,
    service: str,
    endpoint: str,
    instruction: str,
    output: str,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        service: SourceChoice(
            service=service,
            endpoint=SourceEndpointSettings(name="mirror", url=endpoint),
        )
    }

    assert manager.build_failure_service(_build_failure(instruction, output), choices) == service


@pytest.mark.parametrize(
    "output",
    (
        "curl: (22) The requested URL returned error: 502",
        "gzip: stdin: not in gzip format\ntar: Child returned status 1",
    ),
)
def test_uv_release_failure_selects_uv_fallback_not_nested_github_proxy(
    tmp_path: Path,
    output: str,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        "github_git": SourceChoice(
            service="github_git",
            endpoint=SourceEndpointSettings(name="ghfast", url="https://ghfast.top"),
        ),
        "uv_dist": SourceChoice(
            service="uv_dist",
            endpoint=SourceEndpointSettings(
                name="ghfast",
                url="https://ghfast.top/https://github.com",
            ),
        ),
    }
    failure = _build_failure(
        'RUN uv_url="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/'
        'releases/download/${UV_VERSION}/uv-${uv_target}.tar.gz"; curl "$uv_url"',
        output,
    )

    assert manager.build_failure_service(failure, choices) == "uv_dist"


def test_semantic_dependency_resolution_failure_does_not_rotate_sources(
    tmp_path: Path,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        "github_git": SourceChoice(
            service="github_git",
            endpoint=SourceEndpointSettings(name="mirror", url="https://mirror.example"),
        )
    }
    output = (
        "ARG GITHUB_PROXY_PREFIX=https://mirror.example/\n"
        "Could not resolve dependencies for project local:test-jpms:jar:1.0-SNAPSHOT: "
        "Could not find artifact local:gson:jar:1.0-SNAPSHOT"
    )
    assert manager.build_failure_service(_build_failure("RUN mvn package", output), choices) is None


@pytest.mark.parametrize(
    "output",
    (
        (
            "Downloading pytest_timeout-2.4.0-py3-none-any.whl\n"
            "ERROR: Cannot install local project because these package versions have "
            "conflicting dependencies.\n"
            "ERROR: ResolutionImpossible"
        ),
        (
            "Earlier download attempt: Read timed out.\n"
            "Error: pg_config executable not found.\n"
            "ERROR: Failed to build psycopg2"
        ),
    ),
)
def test_pypi_semantic_failure_takes_precedence_over_source_markers(
    tmp_path: Path,
    output: str,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "pypi": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://pypi.example/simple",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://pypi.org/simple",
                            canonical=True,
                        ),
                    )
                )
            }
        }
    )
    manager = SourceManager(settings)
    choices = {
        "pypi": SourceChoice(
            service="pypi",
            endpoint=settings.services["pypi"].endpoints[0],
        )
    }

    assert (
        manager.build_failure_service(
            _build_failure("RUN python -m pip install -e .", output),
            choices,
        )
        is None
    )


def test_deepest_cargo_cause_prevents_pypi_source_rotation(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "pypi": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://pypi.example/simple",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://pypi.org/simple",
                            canonical=True,
                        ),
                    )
                ),
                "cargo": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://cargo.example/index",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://index.crates.io",
                            canonical=True,
                        ),
                    )
                ),
            }
        }
    )
    choices = {
        service: SourceChoice(service=service, endpoint=config.endpoints[0])
        for service, config in settings.services.items()
    }
    output = """#22 690.3 error: failed to download replaced source registry `crates-io`
#22 690.3 Caused by:
#22 690.3   failed to parse manifest at `/root/.cargo/registry/src/base64ct/Cargo.toml`
#22 690.3 Caused by:
#22 690.3   feature `edition2024` is required
#22 690.3
#22 690.3   The package requires edition2024, which is not stabilized in Cargo 1.75.0.
#22 690.3 error: cargo metadata failed with code 101
#22 690.3 ERROR: Failed building wheel for tokenizers
"""

    assert (
        SourceManager(settings).build_failure_service(
            _build_failure("RUN python -m pip install -e '.[dev]'", output),
            choices,
        )
        is None
    )


def test_deepest_network_cause_still_rotates_the_owning_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "cargo": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://cargo.example/index",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://index.crates.io",
                            canonical=True,
                        ),
                    )
                )
            }
        }
    )
    choice = SourceChoice(service="cargo", endpoint=settings.services["cargo"].endpoints[0])
    output = """error: failed to download crate metadata
Caused by:
  failed to fetch `https://cargo.example/index/config.json`
Caused by:
  ReadTimeoutError: HTTPSConnectionPool read timed out
"""

    assert (
        SourceManager(settings).build_failure_service(
            _build_failure("RUN cargo build", output),
            {"cargo": choice},
        )
        == "cargo"
    )


def test_rendered_git_timeout_message_does_not_mask_dirty_checkout(
    tmp_path: Path,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="github_git",
        endpoint=SourceEndpointSettings(name="mirror", url="https://mirror.example"),
    )
    command = (
        "RUN git clone https://github.com/org/repo.git /testbed; "
        "echo 'Git command timed out after 300s'; "
        'test -z "$(git status --porcelain)"'
    )
    output = (
        f"#8 [stage-1 3/6] {command}\n"
        "#8 2.1 + git status --porcelain\n"
        "#8 2.1 + test -z M docs/logo.png\n"
        f'#8 ERROR: process "/bin/sh -c {command}" did not complete successfully: exit code: 1'
    )

    assert (
        manager.build_failure_service(
            _build_failure(command, output),
            {"github_git": choice},
        )
        is None
    )


def test_package_index_miss_is_attributed_only_to_the_failed_step_source(
    tmp_path: Path,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        service: SourceChoice(
            service=service,
            endpoint=SourceEndpointSettings(name="mirror", url=url),
        )
        for service, url in {
            "github_git": "https://ghfast.example",
            "pypi": "https://pypi.example/simple",
            "uv_dist": "https://uv.example/github.com",
            "miniforge": "https://miniforge.example/releases",
            "conda": "https://conda.example/conda-forge",
        }.items()
    }
    failure = DockerBuildError(
        exit_code=1,
        stderr=(
            "earlier build output mentioned https://ghfast.example, "
            "https://uv.example/github.com, https://miniforge.example/releases and "
            "https://conda.example/conda-forge"
        ),
        failed_step=DockerBuildStep(
            step_id="227",
            stage="stage-1 9/10",
            instruction="RUN python3 -m pip install 'browserforge>=1.2.5'",
            output=(
                "#227 ERROR: Could not find a version that satisfies the requirement "
                "browserforge>=1.2.5\n"
                "#227 ERROR: No matching distribution found for browserforge>=1.2.5"
            ),
        ),
    )

    assert manager.build_failure_service(failure, choices) == "pypi"

    choices["pypi"] = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="official",
            url="https://pypi.org/simple",
            canonical=True,
        ),
    )
    assert manager.build_failure_service(failure, choices) is None


def test_global_build_timeout_is_not_assumed_to_be_a_source_failure(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path))
    pypi = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="mirror",
            url="https://pypi.example/simple",
        ),
    )
    choices = {"pypi": pypi}

    assert (
        manager.build_failure_service(
            _build_failure(
                "RUN python -m pip install -e .",
                "#8 downloading wheels",
                timed_out=True,
            ),
            choices,
        )
        is None
    )
    assert (
        manager.build_failure_service(
            _build_failure(
                "RUN python -m pytest",
                "#8 tests/test_slow.py::test_training",
                timed_out=True,
            ),
            choices,
        )
        is None
    )


@pytest.mark.parametrize(
    ("instruction", "output"),
    (
        (
            "RUN uv run --no-sync prisma generate --schema schema.prisma",
            "Installing Prisma CLI",
        ),
        ("RUN uv run --no-sync pytest", "tests/test_slow.py"),
        ("RUN uv run pytest", "tests/test_slow.py"),
    ),
)
def test_uv_run_timeout_without_dependency_activity_is_not_a_pypi_failure(
    tmp_path: Path,
    instruction: str,
    output: str,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="mirror",
            url="https://pypi.example/simple",
        ),
    )

    assert (
        manager.build_failure_service(
            _build_failure(instruction, output, timed_out=True),
            {"pypi": choice},
        )
        is None
    )


def test_uv_run_global_timeout_after_dependency_sync_is_not_a_pypi_failure(
    tmp_path: Path,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="mirror",
            url="https://pypi.example/simple",
        ),
    )

    assert (
        manager.build_failure_service(
            _build_failure(
                "RUN uv run pytest",
                "Resolved 120 packages in 3.2s",
                timed_out=True,
            ),
            {"pypi": choice},
        )
        is None
    )


def test_uv_run_explicit_dependency_download_failure_is_a_pypi_failure(
    tmp_path: Path,
) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="mirror",
            url="https://pypi.example/simple",
        ),
    )

    assert (
        manager.build_failure_service(
            _build_failure(
                "RUN uv run pytest",
                "Failed to download wheel from https://pypi.example/simple: ReadTimeoutError",
            ),
            {"pypi": choice},
        )
        == "pypi"
    )


def test_uv_run_endpoint_specific_generic_download_failure_is_pypi(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path))
    choice = SourceChoice(
        service="pypi",
        endpoint=SourceEndpointSettings(
            name="mirror",
            url="https://pypi.example/simple",
        ),
    )

    assert (
        manager.build_failure_service(
            _build_failure(
                "RUN uv run pytest",
                "Download failed: https://pypi.example/simple/packages/example.whl",
            ),
            {"pypi": choice},
        )
        == "pypi"
    )


def test_pip_vcs_transport_failure_is_owned_by_git_not_pypi(tmp_path: Path) -> None:
    manager = SourceManager(_settings(tmp_path))
    choices = {
        "pypi": SourceChoice(
            service="pypi",
            endpoint=SourceEndpointSettings(
                name="mirror",
                url="https://pypi.example/simple",
            ),
        ),
        "github_git": SourceChoice(
            service="github_git",
            endpoint=SourceEndpointSettings(
                name="mirror",
                url="https://ghfast.top",
            ),
        ),
    }
    failure = _build_failure(
        "RUN python -m pip install git+${GITHUB_PROXY_PREFIX}https://github.com/org/dependency.git",
        "Cloning git+https://github.com/org/dependency.git\n"
        "fatal: unable to connect to github.com: Connection timed out",
    )

    assert manager.build_failure_service(failure, choices) == "github_git"


@pytest.mark.asyncio
async def test_conda_source_applies_to_plain_conda_commands(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "services": {
                "conda": SourceServiceSettings(
                    endpoints=(
                        SourceEndpointSettings(
                            name="mirror",
                            url="https://mirror.example/anaconda/cloud/conda-forge",
                        ),
                        SourceEndpointSettings(
                            name="official",
                            url="https://conda.anaconda.org/conda-forge",
                            canonical=True,
                        ),
                    )
                )
            }
        }
    )
    manager = SourceManager(settings)
    plan = await manager.build_plan()
    rendered = manager.materialize_build_sources(
        "FROM managed-python\nRUN conda install -n base -y numpy scipy\n",
        plan,
    )

    assert plan.arguments["CONDA_CHANNEL_URL"].startswith("https://mirror.example/")
    assert plan.arguments["FEATUREBENCH_CONDA_CHANNEL_ROOT"] == (
        "https://mirror.example/anaconda/cloud"
    )
    assert plan.arguments["CONDARC"] == "/var/tmp/featurebench-condarc.yaml"
    assert "ARG CONDA_REMOTE_READ_TIMEOUT_SECS" in rendered
    assert "printf '%s\\n'" in rendered
    assert "printf '%s\n'" not in rendered
    assert '"  conda-forge: $FEATUREBENCH_CONDA_CHANNEL_ROOT"' in rendered
    assert 'config="$CONDARC"' in rendered
