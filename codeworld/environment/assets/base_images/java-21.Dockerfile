# syntax=docker/dockerfile:1.7
FROM eclipse-temurin:21-jdk-jammy
RUN --mount=type=cache,id=env-base-apt-cache-java21,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-java21,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq maven procps python3 unzip
WORKDIR /workspace
