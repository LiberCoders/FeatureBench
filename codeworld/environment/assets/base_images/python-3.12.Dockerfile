# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04
ARG PYTHON_VERSION=3.12
ARG MINIFORGE_DIST_URL=https://github.com/conda-forge/miniforge/releases/latest/download
ARG CONDA_CHANNEL_URL=https://conda.anaconda.org/conda-forge
ARG UV_INSTALLER_GITHUB_BASE_URL=https://github.com
ARG UV_VERSION=0.11.32
ARG UV_CONCURRENT_DOWNLOADS=8
ARG UV_CONCURRENT_INSTALLS=8
ARG CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS=30
ARG CW_DOWNLOAD_RETRIES=5
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=cache,id=env-base-apt-cache-python312,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-python312,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates cmake coreutils curl file findutils gfortran git git-lfs \
      jq libbz2-dev libffi-dev liblapack-dev liblzma-dev libncursesw5-dev libopenblas-dev \
      libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev ninja-build patch \
      pkg-config procps rsync sed tar unzip wget xz-utils zlib1g-dev
RUN --mount=type=cache,id=env-base-miniforge-installers,target=/var/cache/miniforge,sharing=locked \
    set -eux; \
    case "$(uname -m)" in x86_64) arch=x86_64 ;; aarch64|arm64) arch=aarch64 ;; *) exit 1 ;; esac; \
    installer="/var/cache/miniforge/Miniforge3-Linux-${arch}.sh"; \
    url="${MINIFORGE_DIST_URL}/Miniforge3-Linux-${arch}.sh"; \
    download() { tmp="${installer}.download.$$"; rm -f "$tmp"; curl -fsSL --retry "$CW_DOWNLOAD_RETRIES" --retry-all-errors --retry-delay 2 --connect-timeout "$CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS" "$url" -o "$tmp"; mv -f "$tmp" "$installer"; }; \
    test -s "$installer" || download; rm -rf "$CONDA_DIR"; \
    if ! bash "$installer" -b -p "$CONDA_DIR"; then rm -f "$installer"; download; rm -rf "$CONDA_DIR"; bash "$installer" -b -p "$CONDA_DIR"; fi
RUN --mount=type=cache,id=env-base-conda-pkgs,target=/var/cache/conda/pkgs,sharing=locked \
    set -eux; \
    CONDA_PKGS_DIRS=/var/cache/conda/pkgs conda install -y --override-channels \
      --channel "$CONDA_CHANNEL_URL" "python=${PYTHON_VERSION}" pip setuptools wheel; \
    for executable in python python3 python3.12 pip pip3 conda; do \
      ln -sf "$CONDA_DIR/bin/$executable" "/usr/local/bin/$executable"; \
    done
RUN --mount=type=cache,id=env-base-uv-archives,target=/var/cache/uv-archives,sharing=locked \
    set -eux; \
    case "$(uname -m)" in x86_64|amd64) uv_target=x86_64-unknown-linux-gnu ;; aarch64|arm64) uv_target=aarch64-unknown-linux-gnu ;; *) exit 1 ;; esac; \
    uv_cache_dir="/var/cache/uv-archives/${UV_VERSION}"; \
    mkdir -p "$uv_cache_dir"; \
    uv_archive="$uv_cache_dir/uv-${uv_target}.tar.gz"; \
    uv_url="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_target}.tar.gz"; \
    download_uv() { \
      uv_download="${uv_archive}.download.$$"; \
      rm -f "$uv_download"; \
      if ! curl --http1.1 --retry "$CW_DOWNLOAD_RETRIES" --retry-all-errors --connect-timeout "$CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS" -fsSL "$uv_url" -o "$uv_download" \
        || ! tar -tzf "$uv_download" >/dev/null; then \
        rm -f "$uv_download"; \
        return 1; \
      fi; \
      mv -f "$uv_download" "$uv_archive"; \
    }; \
    if [ ! -s "$uv_archive" ] || ! tar -tzf "$uv_archive" >/dev/null; then \
      rm -f "$uv_archive"; \
      download_uv; \
    fi; \
    uv_tmp="$(mktemp -d)"; trap 'rm -rf "$uv_tmp"' EXIT; \
    tar -xzf "$uv_archive" -C "$uv_tmp"; \
    test -x "$uv_tmp/uv-${uv_target}/uv"; \
    test -x "$uv_tmp/uv-${uv_target}/uvx"; \
    install -m 0755 "$uv_tmp/uv-${uv_target}/uv" /usr/local/bin/uv; \
    install -m 0755 "$uv_tmp/uv-${uv_target}/uvx" /usr/local/bin/uvx
WORKDIR /workspace