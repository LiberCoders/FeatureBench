# syntax=docker/dockerfile:1.7
FROM ruby:3.3-slim-bookworm
RUN --mount=type=cache,id=env-base-apt-cache-ruby33,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-ruby33,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq pkg-config procps python3
WORKDIR /workspace
