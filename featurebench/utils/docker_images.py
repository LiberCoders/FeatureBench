"""Helpers for normalizing FeatureBench Docker image references."""

from __future__ import annotations

import os


DEFAULT_IMAGE_PREFIX = "docker.io"
DEFAULT_IMAGE_NAMESPACE = "libercoders"
IMAGE_PREFIX_ENV_VAR = "FEATUREBENCH_IMAGE_PREFIX"


def _has_explicit_registry(image_name: str) -> bool:
    """Return True when the first path component is a registry host."""
    first_component = image_name.split("/", 1)[0]
    return "." in first_component or ":" in first_component or first_component == "localhost"


def get_image_prefix() -> str:
    """Return the configured prefix used for Docker Hub images."""
    prefix = os.environ.get(IMAGE_PREFIX_ENV_VAR, DEFAULT_IMAGE_PREFIX)
    prefix = prefix.strip().lower().strip("/")
    return prefix or DEFAULT_IMAGE_PREFIX


def normalize_image_name(raw: str) -> str:
    """Normalize an image reference and apply the optional prefix override."""
    image_name = (raw or "").strip().lower()
    if not image_name:
        return ""

    if not _has_explicit_registry(image_name):
        image_name = f"{DEFAULT_IMAGE_PREFIX}/{image_name}"

    if image_name.startswith(f"{DEFAULT_IMAGE_PREFIX}/"):
        image_suffix = image_name[len(DEFAULT_IMAGE_PREFIX) + 1 :]
        return f"{get_image_prefix()}/{image_suffix}"

    return image_name


def canonical_docker_hub_image_name(raw: str) -> str | None:
    """Return the Docker Hub reference for images that originate from Docker Hub.

    This intentionally ignores ``FEATUREBENCH_IMAGE_PREFIX`` so callers that pull
    from a mirror can also tag the local image with its official Docker Hub name.
    """
    image_name = (raw or "").strip().lower()
    if not image_name:
        return None

    if not _has_explicit_registry(image_name):
        return f"{DEFAULT_IMAGE_PREFIX}/{image_name}"

    if image_name.startswith(f"{DEFAULT_IMAGE_PREFIX}/"):
        return image_name

    return None


def build_legacy_instance_image_name(raw: str) -> str:
    """Build the legacy instance image reference and apply the prefix override."""
    image_name = (raw or "").strip().lower()
    if not image_name:
        return ""
    return normalize_image_name(f"{DEFAULT_IMAGE_NAMESPACE}/{image_name}")
