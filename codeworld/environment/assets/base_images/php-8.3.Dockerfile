# syntax=docker/dockerfile:1.7
FROM php:8.3-cli-bookworm
RUN --mount=type=cache,id=env-base-apt-cache-php83,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=env-base-apt-lists-php83,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates coreutils curl git git-lfs jq libzip-dev procps unzip
ARG COMPOSER_BINARY_URL=https://getcomposer.org/download/latest-stable/composer.phar
ARG CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS=30
ARG CW_DOWNLOAD_RETRIES=5
RUN curl -fsSL --retry "$CW_DOWNLOAD_RETRIES" --retry-all-errors \
      --connect-timeout "$CW_DOWNLOAD_CONNECT_TIMEOUT_SECONDS" \
      "$COMPOSER_BINARY_URL" -o /usr/local/bin/composer && \
    chmod 0755 /usr/local/bin/composer && \
    composer --version --no-ansi
WORKDIR /workspace