# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,id=env-base-apt-cache-cuda124,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-cuda124,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates cmake coreutils curl git git-lfs jq ninja-build \
      pkg-config procps python3 python3-pip python3-venv unzip
WORKDIR /workspace
