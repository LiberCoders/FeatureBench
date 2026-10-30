# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,id=env-base-apt-cache-noble,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-noble,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates cmake coreutils curl file findutils git git-lfs \
      jq make ninja-build patch pkg-config procps python3 rsync sed tar unzip wget xz-utils
WORKDIR /workspace
