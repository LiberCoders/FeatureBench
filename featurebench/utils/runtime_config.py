"""Parse the runtime subset of the published FeatureBench dataset schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast


class DockerRuntimeConfig(TypedDict):
    """Validated Docker options shared by inference and evaluation."""

    need_gpu: bool
    shm_size: str | None
    number_once: int
    env_vars: dict[str, str]
    env_exports: list[str]


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def parse_docker_runtime_config(
    repo_settings: Mapping[str, object],
) -> DockerRuntimeConfig:
    docker_specs = _mapping(repo_settings.get("docker_specs"))

    run_args = _mapping(docker_specs.get("run_args"))

    cuda_visible_devices = run_args.get("cuda_visible_devices")
    need_gpu = (
        cuda_visible_devices > 0
        if isinstance(cuda_visible_devices, int)
        else bool(cuda_visible_devices)
    )

    number_once = run_args.get("number_once", 1)
    if not isinstance(number_once, int) or number_once <= 0:
        number_once = 1

    env_vars: dict[str, str] = {}
    env_exports: list[str] = []
    custom_docker_args = docker_specs.get("custom_docker_args")
    if isinstance(custom_docker_args, list):
        for argument in cast(list[object], custom_docker_args):
            if not isinstance(argument, str):
                continue
            prefix = "-ee " if argument.startswith("-ee ") else "-e "
            if not argument.startswith(prefix):
                continue
            assignment = argument.removeprefix(prefix)
            key, separator, value = assignment.partition("=")
            key = key.strip()
            if not separator or not key or key.lower() in {"http_proxy", "https_proxy"}:
                continue
            if prefix == "-ee ":
                env_exports.append(f"export {key}={value}")
            else:
                env_vars[key] = value

    raw_shm_size = run_args.get("shm_size")
    shm_size = raw_shm_size.strip() if isinstance(raw_shm_size, str) else None
    return {
        "need_gpu": need_gpu,
        "shm_size": shm_size or None,
        "number_once": number_once,
        "env_vars": env_vars,
        "env_exports": env_exports,
    }
