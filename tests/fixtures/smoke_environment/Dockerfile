FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY python_repo/ /testbed/
WORKDIR /testbed
RUN python -m pip install --no-cache-dir -e '.[test]' \
    && git init \
    && git config user.email featurebench@localhost \
    && git config user.name FeatureBench \
    && git add -A \
    && git -c commit.gpgsign=false commit -m baseline
