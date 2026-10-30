# syntax=docker/dockerfile:1.7
FROM mcr.microsoft.com/dotnet/sdk:8.0
RUN --mount=type=cache,id=env-base-apt-cache-dotnet8,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-dotnet8,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq procps python3 unzip
WORKDIR /workspace
