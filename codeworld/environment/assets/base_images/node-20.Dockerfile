# syntax=docker/dockerfile:1.7
FROM node:20-bookworm-slim
RUN --mount=type=cache,id=env-base-apt-cache-node20,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-node20,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq procps python3 unzip \
    && corepack enable
WORKDIR /workspace
