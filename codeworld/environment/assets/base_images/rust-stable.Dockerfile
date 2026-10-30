# syntax=docker/dockerfile:1.7
FROM rust:1-bookworm
RUN --mount=type=cache,id=env-base-apt-cache-rust,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-rust,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates clang cmake coreutils curl git git-lfs jq libssl-dev \
      pkg-config procps python3
WORKDIR /workspace
