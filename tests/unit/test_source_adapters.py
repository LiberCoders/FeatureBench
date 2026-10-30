from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import codeworld.sources.pypi as pypi_source
from codeworld.sources import (
    SourceAdapter,
    SourceAdapterRegistry,
    SourceBuildContext,
    SourceBuildFragments,
    SourceFailureKind,
    SourceMatchConfidence,
    default_source_adapters,
    harness_run,
)
from codeworld.sources.git import github_transport_config_arguments


def context(*services: str, arguments: tuple[str, ...]) -> SourceBuildContext:
    return SourceBuildContext(
        services=frozenset(services),
        build_argument_names=arguments,
    )


def test_harness_run_places_buildkit_options_before_provenance() -> None:
    assert harness_run(
        "printf ok",
        options=("--mount=type=cache,target=/cache", "--network=host"),
    ) == (
        "RUN --mount=type=cache,target=/cache --network=host "
        "FEATUREBENCH_STEP_ORIGIN=harness; printf ok\n"
    )


def test_harness_run_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="Invalid Dockerfile RUN options"):
        harness_run("printf ok", options=("--network=host RUN false",))


def test_harness_run_rejects_raw_newlines_inside_one_run_instruction() -> None:
    with pytest.raises(ValueError, match="Docker line continuations"):
        harness_run("printf '%s\n' value")


def test_source_fragments_require_explicit_run_provenance() -> None:
    with pytest.raises(ValueError, match=r"harness_run\(\)"):
        SourceBuildFragments(after_from="RUN printf unmarked\n")


def test_apt_adapter_persists_system_configuration_in_the_image() -> None:
    source = "FROM ubuntu:24.04\nRUN apt-get update\nUSER app\n"
    rendered = default_source_adapters().render(
        source,
        context(
            "ubuntu_apt",
            "debian_apt",
            arguments=("UBUNTU_APT_MIRROR", "DEBIAN_APT_MIRROR"),
        ),
    )
    assert "ARG UBUNTU_APT_MIRROR\n" in rendered
    assert "ENV UBUNTU_APT_MIRROR" not in rendered
    assert "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" in rendered
    assert ".featurebench-apt-sources" not in rendered
    assert rendered.index("sed -i") < rendered.index("apt-get update")
    assert rendered.index("sed -i") < rendered.index("USER app")


def test_managed_base_apt_configuration_covers_indirect_descendant_calls() -> None:
    source = (
        "FROM ubuntu:24.04 AS featurebench_managed_base\n"
        "RUN apt-get update && apt-get install -y python3\n"
        "FROM featurebench_managed_base\n"
        "RUN python3 -m playwright install --with-deps chromium\n"
    )
    rendered = default_source_adapters().render(
        source,
        context(
            "ubuntu_apt",
            "ubuntu_ports",
            "debian_apt",
            "debian_security",
            arguments=(
                "UBUNTU_APT_MIRROR",
                "UBUNTU_PORTS_APT_MIRROR",
                "DEBIAN_APT_MIRROR",
                "DEBIAN_SECURITY_APT_MIRROR",
            ),
        ),
    )
    managed_base, descendant = rendered.split("FROM featurebench_managed_base")
    assert managed_base.count("sed -i") == 1
    assert "playwright" not in managed_base
    assert "sed -i" not in descendant
    assert "playwright install --with-deps chromium" in descendant
    assert ".featurebench-apt-sources" not in rendered


def test_git_adapter_covers_indirect_package_manager_git_fetches() -> None:
    source = (
        "FROM ruby:3.3-slim-bookworm\n"
        "RUN bundle install\n"
        "RUN git clone https://github.com/example/dependency.git /dependency\n"
    )
    rendered = default_source_adapters().render(
        source,
        context("github_git", arguments=("GITHUB_PROXY_PREFIX",)),
    )
    assert "ARG GITHUB_PROXY_PREFIX\n" in rendered
    assert '"url.${target}.insteadOf" "https://github.com/"' in rendered
    assert '"url.${target}.insteadOf" "git@github.com:"' in rendered
    assert rendered.index("git config --global") < rendered.index("bundle install")
    assert rendered.rindex('rm -f "$config"') > rendered.index("git clone")


def test_git_adapter_is_safe_before_a_managed_recipe_installs_git() -> None:
    source = (
        "FROM ubuntu:24.04 AS featurebench_managed_base\n"
        "ARG TOOL_URL=https://github.com/example/tool/releases/latest/download/tool\n"
        "RUN apt-get update && apt-get install -y git\n"
        "FROM featurebench_managed_base\n"
        "RUN git clone https://github.com/example/project.git /testbed\n"
    )
    rendered = default_source_adapters().render(
        source,
        context("github_git", arguments=("GITHUB_PROXY_PREFIX",)),
    )
    stages = rendered.split("FROM featurebench_managed_base")
    assert len(stages) == 2
    assert "if command -v git >/dev/null 2>&1; then" in stages[0]
    assert stages[0].index("command -v git") < stages[0].index("apt-get install -y git")
    assert "command -v git" in stages[1]
    assert stages[1].index("git config --global") < stages[1].index("git clone")


def test_git_adapter_redirects_public_literal_assets_but_not_opaque_urls() -> None:
    source = (
        "FROM ubuntu:24.04\n"
        "RUN curl -fL https://github.com/org/tool/releases/download/v1/tool -o /tool\n"
        "RUN wget https://raw.githubusercontent.com/org/repo/main/config.json\n"
        "RUN curl -fL 'https://github.com/org/private/archive?token=opaque'\n"
    )
    rendered = default_source_adapters().render(
        source,
        context("github_git", arguments=("GITHUB_PROXY_PREFIX",)),
    )
    assert "${GITHUB_PROXY_PREFIX}https://github.com/org/tool/releases/download/v1/tool" in rendered
    assert (
        "${GITHUB_PROXY_PREFIX}https://raw.githubusercontent.com/org/repo/main/config.json"
        in rendered
    )
    assert "'https://github.com/org/private/archive?token=opaque'" in rendered
    assert "${GITHUB_PROXY_PREFIX}https://github.com/org/private" not in rendered


def test_git_adapter_normalizes_public_git_transport_forms_before_proxying() -> None:
    source = (
        "FROM python:3.13\n"
        "RUN pip install git+git://github.com/org/vcs.git\n"
        "RUN git clone git@github.com:org/scp.git /scp\n"
        "RUN git clone ssh://git@github.com/org/ssh.git /ssh\n"
        "RUN git clone git://github.com/org/git.git /git\n"
    )

    rendered = default_source_adapters().render(
        source,
        context("github_git", arguments=("GITHUB_PROXY_PREFIX",)),
    )

    assert "git+${GITHUB_PROXY_PREFIX}https://github.com/org/vcs.git" in rendered
    for repository in ("scp", "ssh", "git"):
        assert f"${{GITHUB_PROXY_PREFIX}}https://github.com/org/{repository}.git" in rendered
    assert "git@github.com:org/" not in rendered
    assert "git://github.com/org/" not in rendered
    assert "ssh://git@github.com/org/" not in rendered


def test_git_command_rewrites_cover_public_transports_and_reserve_https_for_proxies() -> None:
    proxy = github_transport_config_arguments(
        "https://proxy.example/https://github.com/",
        redirect_https=True,
    )
    canonical = github_transport_config_arguments(
        "https://github.com/",
        redirect_https=False,
    )

    assert "url.https://proxy.example/https://github.com/.insteadOf=https://github.com/" in proxy
    assert "url.https://github.com/.insteadOf=https://github.com/" not in canonical
    for transport in (
        "git@github.com:",
        "ssh://git@github.com/",
        "git://github.com/",
    ):
        assert f"url.https://github.com/.insteadOf={transport}" in canonical


def test_maven_adapter_materializes_settings_and_removes_them() -> None:
    source = "FROM eclipse-temurin:17-jdk\nRUN mvn -q dependency:go-offline\n"
    rendered = default_source_adapters().render(
        source,
        context("maven", arguments=("MAVEN_MIRROR",)),
    )
    assert "ARG MAVEN_MIRROR\n" in rendered
    assert "<id>featurebench-build-mirror</id>" in rendered
    assert "<url>${MAVEN_MIRROR}</url>" in rendered
    assert "<mirrorOf>*</mirrorOf>" in rendered
    assert "ENV MAVEN_MIRROR" not in rendered
    assert rendered.index("featurebench-build-mirror") < rendered.index(
        "mvn -q dependency:go-offline"
    )
    assert "_remote.repositories" in rendered
    assert "maven-metadata-central.xml" in rendered
    assert rendered.index("_remote.repositories") > rendered.index("mvn -q dependency:go-offline")
    assert rendered.rindex('rm -f "$settings"') > rendered.index("mvn -q dependency:go-offline")


def test_gradle_adapter_consumes_the_maven_service_and_restores_its_init_script() -> None:
    source = "FROM eclipse-temurin:21-jdk\nRUN ./gradlew --no-daemon build\n"
    registry = default_source_adapters()
    adapters = registry.for_service("maven")
    arguments = {
        key: value
        for adapter in adapters
        for key, value in adapter.build_arguments(
            "maven",
            url="https://maven.aliyun.com/repository/public",
            canonical=False,
        ).items()
    }
    assert arguments == {
        "MAVEN_MIRROR": "https://maven.aliyun.com/repository/public",
        "GRADLE_MIRROR": "https://maven.aliyun.com/repository/public",
    }

    rendered = registry.render(
        source,
        context("maven", arguments=tuple(arguments)),
    )
    assert "ARG GRADLE_MIRROR\n" in rendered
    assert "/root/.gradle/init.d/featurebench-mirror.gradle" in rendered
    assert 'System.getenv("GRADLE_MIRROR")' in rendered
    assert rendered.index("featurebench-mirror.gradle") < rendered.index("./gradlew")
    assert rendered.rindex('rm -f "$init"') > rendered.index("./gradlew")


def test_rubygems_adapter_materializes_configs_and_removes_them() -> None:
    source = "FROM ruby:3.3-slim-bookworm\nRUN gem install bundler && bundle install\n"
    rendered = default_source_adapters().render(
        source,
        context("rubygems", arguments=("RUBYGEMS_MIRROR",)),
    )
    assert "ARG RUBYGEMS_MIRROR\n" in rendered
    assert ":sources:" in rendered
    assert "BUNDLE_MIRROR__HTTPS://RUBYGEMS__ORG/" in rendered
    assert rendered.index(":sources:") < rendered.index("gem install bundler")
    assert rendered.rindex('rm -f "$bundle_config"') > rendered.index("bundle install")
    assert "ENV RUBYGEMS_MIRROR" not in rendered


def test_cargo_adapter_materializes_sparse_registry_and_removes_it() -> None:
    source = (
        "FROM ubuntu:24.04\n"
        "RUN apt-get update && apt-get install -y cargo rustc\n"
        "RUN uv sync --frozen\n"
    )
    rendered = default_source_adapters().render(
        source,
        context("cargo", arguments=("CARGO_REGISTRIES_CRATES_IO_INDEX",)),
    )
    assert "ARG CARGO_REGISTRIES_CRATES_IO_INDEX\n" in rendered
    assert "printf '%s\\n'" in rendered
    assert "[source.crates-io]" in rendered
    assert 'replace-with = "featurebench-build-mirror"' in rendered
    assert 'printf \'registry = "%s"\\n\' "$CARGO_REGISTRIES_CRATES_IO_INDEX"' in rendered
    assert rendered.index("[source.crates-io]") < rendered.index("uv sync --frozen")
    assert rendered.rindex('rm -f "$config"') > rendered.index("uv sync --frozen")
    assert "ENV CARGO_REGISTRIES_CRATES_IO_INDEX" not in rendered


def test_playwright_adapter_uses_a_build_only_browser_download_host() -> None:
    registry = default_source_adapters()
    adapter = next(item for item in registry.adapters if item.name == "playwright")
    arguments = adapter.build_arguments(
        "playwright",
        url="https://cdn.npmmirror.com/binaries/playwright/",
        canonical=False,
    )

    assert arguments == {
        "PLAYWRIGHT_DOWNLOAD_HOST": "https://cdn.npmmirror.com/binaries/playwright"
    }
    assert adapter.download_arguments(
        "playwright",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {"PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT": "300000"}
    rendered = registry.render(
        "FROM python:3.12\nRUN python -m playwright install chromium\n",
        context("playwright", arguments=tuple(arguments)),
    )
    assert "ARG PLAYWRIGHT_DOWNLOAD_HOST\n" in rendered
    assert "ENV PLAYWRIGHT_DOWNLOAD_HOST" not in rendered
    assert "playwright install chromium" in rendered
    assert "cdn.playwright.dev" in adapter.failure_hints("playwright")


def test_rustup_adapter_keeps_toolchain_and_self_update_hosts_coherent() -> None:
    registry = default_source_adapters()
    adapter = next(item for item in registry.adapters if item.name == "rustup")
    arguments = adapter.build_arguments(
        "rustup",
        url="https://rsproxy.cn/",
        canonical=False,
    )

    assert arguments == {
        "RUSTUP_DIST_SERVER": "https://rsproxy.cn",
        "RUSTUP_INIT_URL": "https://rsproxy.cn/rustup-init.sh",
        "RUSTUP_UPDATE_ROOT": "https://rsproxy.cn/rustup",
    }
    assert adapter.download_arguments(
        "rustup",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {
        "RUSTUP_DOWNLOAD_TIMEOUT": "300",
        "RUSTUP_MAX_RETRIES": "5",
    }
    rendered = registry.render(
        "FROM ubuntu:24.04\nRUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal\n",
        context("rustup", arguments=tuple(arguments)),
    )
    assert "ARG RUSTUP_DIST_SERVER\n" in rendered
    assert "ARG RUSTUP_INIT_URL\n" in rendered
    assert "ARG RUSTUP_UPDATE_ROOT\n" in rendered
    assert "ENV RUSTUP_DIST_SERVER" not in rendered
    assert "curl -sSf ${RUSTUP_INIT_URL} | sh" in rendered
    assert "https://sh.rustup.rs" not in rendered
    assert "sh.rustup.rs" in adapter.failure_hints("rustup")


def test_download_policy_uses_native_python_cargo_and_node_controls() -> None:
    registry = default_source_adapters()
    pypi = next(item for item in registry.adapters if item.name == "pypi")
    cargo = next(item for item in registry.adapters if item.name == "cargo")
    conda = next(item for item in registry.adapters if item.name == "conda")
    node = next(item for item in registry.adapters if item.name == "node")

    assert pypi.download_arguments(
        "pypi",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {
        "PIP_DEFAULT_TIMEOUT": "300",
        "PIP_RETRIES": "5",
        "UV_HTTP_CONNECT_TIMEOUT": "30",
        "UV_HTTP_RETRIES": "5",
        "UV_HTTP_TIMEOUT": "300",
    }
    assert cargo.download_arguments(
        "cargo",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {"CARGO_HTTP_TIMEOUT": "300", "CARGO_NET_RETRY": "5"}
    assert conda.download_arguments(
        "conda",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {
        "CONDA_REMOTE_CONNECT_TIMEOUT_SECS": "30",
        "CONDA_REMOTE_READ_TIMEOUT_SECS": "300",
        "CONDA_REMOTE_MAX_RETRIES": "5",
    }
    assert node.download_arguments(
        "npm",
        connect_timeout_seconds=30,
        timeout_seconds=300,
        retries=5,
    ) == {"NPM_CONFIG_FETCH_RETRIES": "5", "NPM_CONFIG_FETCH_TIMEOUT": "300000"}


def test_http2_protocol_error_is_a_retryable_source_failure() -> None:
    adapter = next(item for item in default_source_adapters().adapters if item.name == "rustup")

    assert (
        adapter.classify_build_failure(
            "rustup",
            detail="curl: (92) HTTP/2 stream 1 was not closed cleanly: PROTOCOL_ERROR",
            confidence=SourceMatchConfidence.OPERATION,
        )
        is SourceFailureKind.AVAILABILITY
    )


@pytest.mark.parametrize(
    "detail",
    (
        "Downloading pytest_timeout-2.4.0-py3-none-any.whl",
        "ARG PIP_DEFAULT_TIMEOUT=300",
    ),
)
def test_timeout_words_without_a_timeout_failure_do_not_rotate_sources(detail: str) -> None:
    adapter = next(item for item in default_source_adapters().adapters if item.name == "pypi")

    assert (
        adapter.classify_build_failure(
            "pypi",
            detail=detail,
            confidence=SourceMatchConfidence.OPERATION,
        )
        is None
    )


def test_explicit_timeout_error_remains_retryable() -> None:
    adapter = next(item for item in default_source_adapters().adapters if item.name == "pypi")

    assert (
        adapter.classify_build_failure(
            "pypi",
            detail="ReadTimeoutError: HTTPSConnectionPool read timed out",
            confidence=SourceMatchConfidence.OPERATION,
        )
        is SourceFailureKind.AVAILABILITY
    )


def test_composer_binary_source_is_a_build_argument_not_an_image_setting() -> None:
    adapter = next(
        item for item in default_source_adapters().adapters if "composer_binary" in item.services
    )
    arguments = adapter.build_arguments(
        "composer_binary",
        url="https://mirrors.aliyun.com/composer/composer.phar",
        canonical=False,
    )
    assert arguments == {"COMPOSER_BINARY_URL": "https://mirrors.aliyun.com/composer/composer.phar"}
    assert "composer.phar" in adapter.failure_hints("composer_binary")


def test_uv_distribution_source_redirects_only_the_installer_artifact() -> None:
    adapter = next(
        item for item in default_source_adapters().adapters if "uv_dist" in item.services
    )
    arguments = adapter.build_arguments(
        "uv_dist",
        url="https://ghfast.top/https://github.com",
        canonical=False,
    )
    assert arguments == {"UV_INSTALLER_GITHUB_BASE_URL": "https://ghfast.top/https://github.com"}
    assert "astral-sh/uv/releases/download" in adapter.failure_hints("uv_dist")


def test_huggingface_adapter_uses_supported_endpoint_and_public_url_forms() -> None:
    registry = default_source_adapters()
    adapter = next(item for item in registry.adapters if item.name == "huggingface")
    arguments = adapter.build_arguments(
        "huggingface",
        url="https://hf-mirror.com",
        canonical=False,
    )
    assert arguments == {"HF_ENDPOINT": "https://hf-mirror.com"}

    source = (
        "FROM python:3.12\n"
        'RUN python -c "from transformers import AutoModel; '
        "AutoModel.from_pretrained('gpt2')\"\n"
        "RUN wget https://huggingface.co/gpt2/resolve/main/config.json\n"
        "RUN curl -fL 'https://huggingface.co/private/model?token=opaque'\n"
    )
    rendered = registry.render(
        source,
        context("huggingface", arguments=tuple(arguments)),
    )
    assert "ARG HF_ENDPOINT\n" in rendered
    assert "${HF_ENDPOINT}/gpt2/resolve/main/config.json" in rendered
    assert "'https://huggingface.co/private/model?token=opaque'" in rendered
    assert "${HF_ENDPOINT}/private/model" not in rendered


def test_pypi_adapter_temporarily_redirects_frozen_uv_lock_artifacts() -> None:
    adapter = next(item for item in default_source_adapters().adapters if item.name == "pypi")
    arguments = adapter.build_arguments(
        "pypi",
        url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        canonical=False,
    )
    assert arguments == {
        "PIP_INDEX_URL": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        "UV_INDEX_URL": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        "UV_DEFAULT_INDEX": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        "FEATUREBENCH_PYPI_ARTIFACT_BASE_URL": ("https://mirrors.tuna.tsinghua.edu.cn/pypi/web/"),
    }
    rendered = default_source_adapters().render(
        "FROM managed-python\nWORKDIR /testbed\nRUN uv sync --frozen\n",
        context("pypi", arguments=tuple(arguments)),
    )
    assert "ARG FEATUREBENCH_PYPI_ARTIFACT_BASE_URL\n" in rendered
    assert "featurebench-source-real" in rendered
    assert "https://files.pythonhosted.org/" in rendered
    assert "${FEATUREBENCH_PYPI_ARTIFACT_BASE_URL}" in rendered
    assert "\n+      " not in rendered
    assert "      'esac' \\" in rendered
    assert "      'done' \\" in rendered
    assert rendered.index("featurebench-source-real") < rendered.index("uv sync --frozen")
    assert rendered.rindex('mv -f "${real_uv}.featurebench-source-real"') > rendered.index(
        "uv sync --frozen"
    )
    assert "ENV FEATUREBENCH_PYPI_ARTIFACT_BASE_URL" not in rendered


def test_pypi_adapter_wraps_poetry_resolution_and_removes_the_helper() -> None:
    rendered = default_source_adapters().render(
        "FROM managed-python\n"
        "RUN python -m pip install poetry==1.7.1\n"
        "RUN poetry install --no-interaction\n",
        context(
            "pypi",
            arguments=(
                "PIP_INDEX_URL",
                "UV_INDEX_URL",
                "UV_DEFAULT_INDEX",
                "FEATUREBENCH_PYPI_ARTIFACT_BASE_URL",
            ),
        ),
    )

    wrapper = "/var/tmp/.featurebench-pypi-poetry-wrapper"
    assert f"RUN {wrapper} install --no-interaction" in rendered
    assert "source add --priority=primary" in rendered
    assert rendered.index(f"> {wrapper}") < rendered.index("python -m pip install poetry")
    assert rendered.rindex(f"rm -f {wrapper}") > rendered.index(f"RUN {wrapper} install")


def test_pypi_adapter_does_not_rewrite_poetry_text_outside_an_executed_command() -> None:
    source = "FROM managed-python\nLABEL example='poetry install'\nRUN echo 'poetry install'\n"

    rendered = default_source_adapters().render(
        source,
        context("pypi", arguments=("PIP_INDEX_URL",)),
    )

    assert rendered == (
        "FROM managed-python\n"
        "# FEATUREBENCH_STEP_ORIGIN=harness:begin\n"
        "ARG PIP_INDEX_URL\n"
        "# FEATUREBENCH_STEP_ORIGIN=harness:end\n" + source.split("\n", 1)[1]
    )
    assert "/var/tmp/.featurebench-pypi-poetry-wrapper" not in rendered


def test_ecosystem_adapters_own_their_failure_semantics() -> None:
    registry = default_source_adapters()
    pypi = next(item for item in registry.for_service("pypi") if item.name == "pypi")
    github = next(item for item in registry.for_service("github_git") if item.name == "git")
    index_miss = "ERROR: No matching distribution found for browserforge>=1.2.5"

    assert (
        pypi.classify_build_failure(
            "pypi",
            detail=index_miss,
            confidence=SourceMatchConfidence.OPERATION,
        )
        is SourceFailureKind.NOT_FOUND
    )
    assert (
        github.classify_build_failure(
            "github_git",
            detail=index_miss,
            confidence=SourceMatchConfidence.OPERATION,
        )
        is None
    )
    assert (
        github.classify_build_failure(
            "github_git",
            detail="fatal: Could not resolve host github.example",
            confidence=SourceMatchConfidence.OPERATION,
        )
        is SourceFailureKind.AVAILABILITY
    )


def test_pypi_adapter_keeps_official_artifact_host_for_canonical_source() -> None:
    adapter = next(item for item in default_source_adapters().adapters if item.name == "pypi")
    assert (
        adapter.build_arguments(
            "pypi",
            url="https://pypi.org/simple",
            canonical=True,
        )["FEATUREBENCH_PYPI_ARTIFACT_BASE_URL"]
        == "https://files.pythonhosted.org/"
    )


def test_poetry_source_is_temporary_and_restores_project_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    lock = project / "poetry.lock"
    original_project = '[tool.poetry]\nname = "fixture"\nversion = "0.1.0"\n'
    original_lock = "fixture-lock\n"
    pyproject.write_text(original_project, encoding="utf-8")
    lock.write_text(original_lock, encoding="utf-8")

    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls.log"
    fake_poetry = binary / "poetry"
    fake_poetry.write_text(
        r"""#!/bin/sh
printf '%s\n' "$*" >> "$POETRY_CALLS"
case "$*" in
  "source add --help") printf '%s\n' '--priority' ;;
  source\ remove\ *) exit 1 ;;
  source\ add\ --priority=primary\ *)
    printf '%s\n' '[[tool.poetry.source]]' 'name = "featurebench-build-pypi"' \
      'url = "https://mirror.example/simple"' 'priority = "primary"' >> pyproject.toml
    ;;
  install*)
    grep -q 'https://mirror.example/simple' pyproject.toml
    printf '%s\n' 'temporary mutation' >> pyproject.toml
    printf '%s\n' 'temporary lock mutation' >> poetry.lock
    ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_poetry.chmod(0o755)
    wrapper = tmp_path / "poetry-wrapper"
    wrapper.write_text(
        pypi_source._POETRY_WRAPPER,  # pyright: ignore[reportPrivateUsage]
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{binary}{os.pathsep}{environment['PATH']}",
            "PIP_INDEX_URL": "https://mirror.example/simple",
            "POETRY_CALLS": str(calls),
        }
    )

    result = subprocess.run(
        (str(wrapper), "install", "--no-interaction"),
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "FEATUREBENCH_PYPI_POETRY_SOURCE_APPLIED=https://mirror.example/simple" in result.stderr
    assert pyproject.read_text(encoding="utf-8") == original_project
    assert lock.read_text(encoding="utf-8") == original_lock
    assert "source add --priority=primary featurebench-build-pypi " in calls.read_text(
        encoding="utf-8"
    )


def test_pytorch_cpu_adapter_keeps_pypi_as_index_and_supplies_mirror_find_links() -> None:
    adapter = next(
        item for item in default_source_adapters().adapters if item.name == "pytorch_cpu"
    )
    arguments = adapter.build_arguments(
        "pytorch_cpu",
        url="https://mirrors.aliyun.com/pytorch-wheels/cpu",
        canonical=False,
    )
    assert arguments == {"TORCH_CPU_FIND_LINKS": "https://mirrors.aliyun.com/pytorch-wheels/cpu"}

    rendered = default_source_adapters().render(
        "FROM python:3.12\n"
        "RUN python -m pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cpu\n",
        context(
            "pypi",
            "pytorch_cpu",
            arguments=("PIP_INDEX_URL", *arguments),
        ),
    )

    assert "ARG PIP_INDEX_URL\n" in rendered
    assert "ARG TORCH_CPU_FIND_LINKS\n" in rendered
    assert "https://download.pytorch.org/whl/cpu" not in rendered
    assert (
        "RUN python -m pip install torch torchvision --find-links=${TORCH_CPU_FIND_LINKS}\n"
    ) in rendered


def test_pytorch_cpu_adapter_does_not_redirect_cuda_or_inactive_sources() -> None:
    registry = default_source_adapters()
    cuda = (
        "FROM python:3.12\n"
        "RUN pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
    )
    assert registry.render(
        cuda,
        context("pytorch_cpu", arguments=("TORCH_CPU_FIND_LINKS",)),
    ).endswith("RUN pip install torch --index-url https://download.pytorch.org/whl/cu121\n")

    cpu = (
        "FROM python:3.12\nRUN pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    )
    assert "https://download.pytorch.org/whl/cpu" in registry.render(
        cpu,
        context("pypi", arguments=("PIP_INDEX_URL",)),
    )


@pytest.mark.parametrize(
    "source_option",
    (
        "--extra-index-url=https://download.pytorch.org/whl/cpu/",
        "-i 'https://download.pytorch.org/whl/cpu'",
        "--find-links https://download.pytorch.org/whl/cpu/torch_stable.html",
        "-f=https://download.pytorch.org/whl/torch_stable.html",
    ),
)
def test_pytorch_cpu_adapter_normalizes_supported_official_option_forms(
    source_option: str,
) -> None:
    rendered = default_source_adapters().render(
        f"FROM python:3.12\nRUN pip install torch {source_option}\n",
        context("pytorch_cpu", arguments=("TORCH_CPU_FIND_LINKS",)),
    )
    assert "download.pytorch.org" not in rendered
    assert rendered.count("--find-links=${TORCH_CPU_FIND_LINKS}") == 1


def test_pypi_uv_wrapper_restores_lockfile_after_real_command(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    lock = project / "uv.lock"
    original = (
        'source = { registry = "https://pypi.org/simple" }\n'
        'url = "https://files.pythonhosted.org/packages/example.whl"\n'
    )
    lock.write_text(original, encoding="utf-8")
    wrapper = tmp_path / "uv"
    wrapper.write_text(
        pypi_source._WRAPPER,  # pyright: ignore[reportPrivateUsage]
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    real = tmp_path / "uv.featurebench-source-real"
    real.write_text(
        "#!/bin/sh\ncat uv.lock > observed-lock\nexit 0\n",
        encoding="utf-8",
    )
    real.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "UV_DEFAULT_INDEX": "https://mirrors.example/pypi/simple",
            "FEATUREBENCH_PYPI_ARTIFACT_BASE_URL": "https://mirrors.example/pypi/",
        }
    )
    result = subprocess.run(
        (str(wrapper), "sync", "--frozen"),
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert lock.read_text(encoding="utf-8") == original
    assert (project / "observed-lock").read_text(encoding="utf-8") == (
        'source = { registry = "https://mirrors.example/pypi/simple" }\n'
        'url = "https://mirrors.example/pypi/packages/example.whl"\n'
    )


def test_adapters_apply_and_clean_up_only_in_stages_that_use_the_ecosystem() -> None:
    source = (
        "FROM scratch AS empty\n"
        "COPY fixture /fixture\n"
        "FROM eclipse-temurin:17-jdk\n"
        "RUN mvn -q test\n"
    )
    rendered = default_source_adapters().render(
        source,
        context("maven", arguments=("MAVEN_MIRROR",)),
    )
    empty_stage, maven_stage = rendered.split("FROM eclipse-temurin:17-jdk")
    assert "featurebench-build-mirror" not in empty_stage
    assert "featurebench-build-mirror" in maven_stage
    assert maven_stage.index("featurebench-build-mirror") < maven_stage.index("mvn -q test")
    assert maven_stage.rindex('rm -f "$generated"') > maven_stage.index("mvn -q test")


def test_unregistered_ecosystem_requires_only_one_new_adapter() -> None:
    @dataclass(frozen=True, slots=True)
    class GradleAdapter(SourceAdapter):
        name: str = "gradle"
        priority: int = 150
        services: frozenset[str] = frozenset({"gradle"})

        def fragments(
            self, source: str, context: SourceBuildContext
        ) -> SourceBuildFragments | None:
            if "gradle" not in context.services:
                return None
            return SourceBuildFragments(
                after_from=harness_run("install-temporary-gradle-source"),
                before_final_user=harness_run("remove-temporary-gradle-source"),
            )

    registry = SourceAdapterRegistry()
    registry.register(GradleAdapter())
    rendered = registry.render(
        "FROM example\nRUN ./gradlew test\n",
        context("gradle", arguments=("GRADLE_MIRROR",)),
    )
    assert "ARG GRADLE_MIRROR" in rendered
    assert "install-temporary-gradle-source" in rendered
    assert "RUN FEATUREBENCH_STEP_ORIGIN=harness; remove-temporary-gradle-source\n" in rendered
    assert rendered.endswith("# FEATUREBENCH_STEP_ORIGIN=harness:end\n")
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(GradleAdapter())


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "FROM example",
            "FROM example\n# FEATUREBENCH_STEP_ORIGIN=harness:begin\n",
        ),
        (
            "FROM example\nRUN ./gradlew test",
            "RUN ./gradlew test\n# FEATUREBENCH_STEP_ORIGIN=harness:begin\n",
        ),
        (
            "FROM example\nRUN ./gradlew test\nUSER nobody",
            "# FEATUREBENCH_STEP_ORIGIN=harness:end\nUSER nobody",
        ),
    ),
)
def test_source_fragments_preserve_instruction_boundaries_without_final_newline(
    source: str,
    expected: str,
) -> None:
    @dataclass(frozen=True, slots=True)
    class Adapter(SourceAdapter):
        name: str = "fixture"
        priority: int = 1
        services: frozenset[str] = frozenset({"fixture"})

        def fragments(
            self, source: str, context: SourceBuildContext
        ) -> SourceBuildFragments | None:
            del source
            if "fixture" not in context.services:
                return None
            return SourceBuildFragments(
                after_from=harness_run("install-fixture"),
                before_final_user=harness_run("remove-fixture"),
            )

    registry = SourceAdapterRegistry()
    registry.register(Adapter())

    rendered = registry.render(source, context("fixture", arguments=("FIXTURE_URL",)))

    assert expected in rendered
    assert "example# CODEWORLD" not in rendered
    assert "test# CODEWORLD" not in rendered


def test_multiple_ecosystems_can_share_one_logical_source_service() -> None:
    @dataclass(frozen=True, slots=True)
    class ConsumerAdapter(SourceAdapter):
        name: str
        priority: int
        services: frozenset[str] = frozenset({"jvm_repository"})

    registry = SourceAdapterRegistry()
    registry.register(ConsumerAdapter(name="maven", priority=20))
    registry.register(ConsumerAdapter(name="gradle", priority=10))
    assert [adapter.name for adapter in registry.for_service("jvm_repository")] == [
        "gradle",
        "maven",
    ]
