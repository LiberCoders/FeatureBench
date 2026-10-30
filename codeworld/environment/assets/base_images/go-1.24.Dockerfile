# syntax=docker/dockerfile:1.7
FROM golang:1.24-bookworm
RUN --mount=type=cache,id=env-base-apt-cache-go124,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-go124,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq pkg-config procps python3
WORKDIR /workspace
