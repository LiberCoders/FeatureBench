"""Strict project configuration with a single current schema."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import Field, ValidationInfo, field_validator, model_validator

from codeworld.errors import ConfigurationError
from codeworld.models import StrictModel
from codeworld.repositories.targets import normalize_repository_target


class AdapterSettingsCatalog(Protocol):
    """Configuration-facing Adapter catalog supplied only by Adapter stages."""

    @property
    def types(self) -> tuple[str, ...]: ...

    def validate_settings(
        self, configurations: Mapping[str, object]
    ) -> dict[str, dict[str, object]]: ...


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses silent duplicate-key replacement."""


def _construct_yaml_object(loader: _UniqueKeyLoader, node: yaml.Node, *, deep: bool) -> Any:
    # PyYAML exposes this dynamically; its type stub intentionally returns Unknown.
    return loader.construct_object(node, deep=deep)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    pairs = cast(list[tuple[yaml.Node, yaml.Node]], node.value)
    for key_node, value_node in pairs:
        key = _construct_yaml_object(loader, key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = _construct_yaml_object(loader, value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class DockerSettings(StrictModel):
    binary: str = "docker"
    network: str = "none"
    build_network: str = "default"
    pull: bool = True
    command_timeout_seconds: float = Field(default=300, gt=0)
    pull_timeout_seconds: float = Field(default=900, gt=0)
    build_timeout_seconds: float = Field(default=3600, gt=0)
    max_output_chars: int = Field(default=100_000, gt=0)
    cpus: float | None = Field(default=None, gt=0)
    memory: str | None = None
    gpus: str | None = None

    @field_validator("memory", "gpus")
    @classmethod
    def normalized_optional_docker_values(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("optional Docker resource values may not be empty")
        return normalized


class SourceEndpointSettings(StrictModel):
    """One ordered endpoint for a logical network service."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    url: str = Field(min_length=1)
    canonical: bool = False
    forward_credentials: bool = False

    @field_validator("url")
    @classmethod
    def safe_public_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "source endpoint must be an HTTP(S) URL without credentials, query, or fragment"
            )
        if any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~:/%+-=@"
            for character in normalized
        ):
            raise ValueError("source endpoint contains characters unsafe for build-time injection")
        return normalized


class SourceServiceSettings(StrictModel):
    """An ordered candidate-to-canonical endpoint chain with no application logic."""

    endpoints: tuple[SourceEndpointSettings, ...] = Field(min_length=1)
    maximum_attempts: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def unique_endpoints_and_canonical_fallback(self) -> SourceServiceSettings:
        names = [endpoint.name for endpoint in self.endpoints]
        urls = [endpoint.url for endpoint in self.endpoints]
        if len(names) != len(set(names)):
            raise ValueError("source endpoint names must be unique within one service")
        if len(urls) != len(set(urls)):
            raise ValueError("source endpoint URLs must be unique within one service")
        canonical = [endpoint for endpoint in self.endpoints if endpoint.canonical]
        if len(canonical) != 1:
            raise ValueError("every source service requires exactly one canonical endpoint")
        if not self.endpoints[-1].canonical:
            raise ValueError("the canonical endpoint must be the final fallback")
        if any(endpoint.forward_credentials for endpoint in self.endpoints[:-1]):
            raise ValueError("credentials may only be forwarded to the canonical endpoint")
        return self


class SourceMetricsSettings(StrictModel):
    enabled: bool = True
    path: Path = Path(".featurebench/source-metrics.jsonl")


class SourceDownloadSettings(StrictModel):
    """One bounded download policy translated to each tool's native controls."""

    connect_timeout_seconds: int = Field(default=30, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)
    retries: int = Field(default=5, ge=0, le=20)


class SourceSettings(StrictModel):
    """Cross-phase source policy; endpoint definitions live in one independent YAML file."""

    enabled: bool = False
    failure_threshold: int = Field(default=1, ge=1)
    cooldown_seconds: float = Field(default=300, gt=0)
    maximum_cooldown_seconds: float = Field(default=3600, gt=0)
    download: SourceDownloadSettings = SourceDownloadSettings()
    state_path: Path = Path(".featurebench/source-state.json")
    metrics: SourceMetricsSettings = SourceMetricsSettings()
    services: dict[str, SourceServiceSettings] = Field(default_factory=dict)

    @field_validator("services")
    @classmethod
    def safe_service_names(
        cls, services: dict[str, SourceServiceSettings]
    ) -> dict[str, SourceServiceSettings]:
        for name in services:
            if not name or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name
            ):
                raise ValueError("source service names must use lowercase snake_case")
        return services

    @model_validator(mode="after")
    def valid_cooldown_range(self) -> SourceSettings:
        if self.cooldown_seconds > self.maximum_cooldown_seconds:
            raise ValueError("source cooldown_seconds must not exceed maximum_cooldown_seconds")
        return self


class AgentCompactionSettings(StrictModel):
    """Context-window management, not a limit on total Agent work or usage."""

    target_tokens: int = Field(default=240_000, ge=1)
    max_part_tokens: int = Field(default=50_000, ge=1)
    keep_head_chars: int = Field(default=2_000, ge=0)
    keep_tail_chars: int = Field(default=2_000, ge=0)
    keep_tool_pairs: int = Field(default=6, ge=0)
    min_clear_tokens: int = Field(default=4_000, ge=0)
    keep_messages: int = Field(default=20, ge=0)


class AgentPricingSettings(StrictModel):
    """Optional relay billing rates; the built-in catalog estimate is recorded separately."""

    input_per_million_usd: float | None = Field(default=None, ge=0)
    output_per_million_usd: float | None = Field(default=None, ge=0)
    cache_read_per_million_usd: float | None = Field(default=None, ge=0)
    cache_write_per_million_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def input_and_output_are_paired(self) -> AgentPricingSettings:
        if (self.input_per_million_usd is None) != (self.output_per_million_usd is None):
            raise ValueError(
                "agent.pricing input_per_million_usd and output_per_million_usd "
                "must be configured together"
            )
        return self


class AgentTrajectorySettings(StrictModel):
    enabled: bool = False
    directory: Path = Path(".featurebench/agent-trajectories")
    retention: Literal["failures", "all"] = "all"


class AgentMetricsSettings(StrictModel):
    enabled: bool = True
    path: Path = Path(".featurebench/agent-metrics.jsonl")


class AgentProviderSettings(StrictModel):
    """Transport security controls for model-provider requests only."""

    tls_verify: bool = True


class AgentExecutionSettings(StrictModel):
    """Provider-independent controls shared by one owner of Agent runs."""

    provider: AgentProviderSettings = AgentProviderSettings()
    tool_retries: int = Field(default=10, ge=0)
    output_retries: int = Field(default=10, ge=0)
    transient_request_retries: int = Field(default=10, ge=0)
    request_timeout_seconds: float = Field(default=600, gt=0)
    tool_timeout_seconds: float = Field(default=180, gt=0)
    compaction: AgentCompactionSettings = AgentCompactionSettings()
    metrics: AgentMetricsSettings = AgentMetricsSettings()
    pricing: AgentPricingSettings = AgentPricingSettings()
    trajectory: AgentTrajectorySettings = AgentTrajectorySettings()


class AgentFactorySettings(AgentExecutionSettings):
    """Complete settings consumed by the shared host-side Agent factory."""

    model: str | None = None


class AgentSettings(AgentFactorySettings):
    """Environment Build-owned Agent settings."""

    request_timeout_seconds: float = Field(default=180, gt=0)
    trajectory: AgentTrajectorySettings = AgentTrajectorySettings(retention="failures")
    analysis_run_timeout_seconds: float = Field(default=600, gt=0)
    artifact_run_timeout_seconds: float = Field(default=600, gt=0)
    runner_review_run_timeout_seconds: float = Field(default=300, gt=0)

    @field_validator("trajectory", mode="before")
    @classmethod
    def default_to_failed_repository_retention(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        configured = cast(dict[str, object], value)
        if "retention" in configured:
            return configured
        return {**configured, "retention": "failures"}


class DiscoverySettings(StrictModel):
    github_api_url: str = "https://api.github.com"
    github_api_version: str = "2022-11-28"
    github_user_agent: str = "FeatureBench/0.3.0"
    github_token_env: str = "GITHUB_TOKEN"
    languages: tuple[str, ...] = (
        "Python",
        "JavaScript",
        "TypeScript",
        "Go",
        "Rust",
        "Java",
        "Kotlin",
        "Ruby",
        "PHP",
        "C#",
        "C",
        "C++",
        "Shell",
        "Swift",
        "Scala",
        "Dart",
        "Elixir",
        "Clojure",
        "Lua",
        "R",
    )
    licenses: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    repositories_file: Path | None = None
    target_repositories: tuple[str, ...] = ()
    created_after: datetime = datetime(2008, 1, 1, tzinfo=UTC)
    created_before: datetime | None = None
    pushed_after: datetime | None = None
    pushed_before: datetime | None = None
    minimum_stars: int = Field(default=100, ge=0)
    maximum_stars: int | None = Field(default=None, ge=0)
    maximum_repositories: int | None = Field(default=100, ge=1)
    exclude_forks: bool = True
    exclude_archived: bool = True
    public_only: bool = True
    sort: Literal["updated", "stars"] = "updated"
    order: Literal["asc", "desc"] = "desc"
    requests_per_page: int = Field(default=100, ge=1, le=100)
    partition_result_limit: int = Field(default=900, ge=1, le=1000)
    partition_max_span_days: int = Field(default=365, ge=1)
    partition_concurrency: int = Field(default=4, ge=1, le=32)
    max_concurrent_requests: int = Field(default=8, ge=1, le=64)
    request_timeout_seconds: float = Field(default=30, gt=0)
    request_attempts: int = Field(default=5, ge=1, le=20)
    retry_base_seconds: float = Field(default=1, gt=0)
    retry_max_seconds: float = Field(default=60, gt=0)
    incomplete_results_retries: int = Field(default=3, ge=0, le=10)
    partition_consistency_retries: int = Field(default=2, ge=0, le=10)
    rate_limit_fallback_seconds: float = Field(default=60, gt=0)

    @field_validator("created_after", "created_before", "pushed_after", "pushed_before")
    @classmethod
    def utc_second_precision(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).replace(microsecond=0)

    @field_validator("languages", "licenses", "keywords")
    @classmethod
    def normalized_unique_values(
        cls, values: tuple[str, ...], info: ValidationInfo
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(normalized.lower() if info.field_name == "licenses" else normalized)
        return tuple(result)

    @field_validator("target_repositories")
    @classmethod
    def normalized_targets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = normalize_repository_target(value)
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        return tuple(result)

    @field_validator("repositories_file")
    @classmethod
    def txt_repository_file(cls, value: Path | None) -> Path | None:
        if value is not None and value.suffix.lower() != ".txt":
            raise ValueError("repositories_file must be a TXT file")
        return value

    @model_validator(mode="after")
    def valid_ranges(self) -> DiscoverySettings:
        if self.repositories_file is not None and self.target_repositories:
            raise ValueError("repositories_file and target_repositories cannot be used together")
        if self.created_before is not None and self.created_after > self.created_before:
            raise ValueError("created_after must not be later than created_before")
        if (
            self.pushed_after is not None
            and self.pushed_before is not None
            and self.pushed_after > self.pushed_before
        ):
            raise ValueError("pushed_after must not be later than pushed_before")
        if self.maximum_stars is not None and self.minimum_stars > self.maximum_stars:
            raise ValueError("minimum_stars must not exceed maximum_stars")
        if self.retry_base_seconds > self.retry_max_seconds:
            raise ValueError("retry_base_seconds must not exceed retry_max_seconds")
        return self


class RepositorySettings(StrictModel):
    git_binary: str = "git"
    command_timeout_seconds: float = Field(default=1200, gt=0)
    network_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_seconds: float = Field(default=2, gt=0)
    low_speed_bytes_per_second: int = Field(default=1024, ge=1)
    low_speed_timeout_seconds: int = Field(default=30, ge=1)
    partial_clone: bool = True
    update_submodules: bool = True
    submodule_jobs: int = Field(default=4, ge=1, le=32)
    fetch_lfs: bool = True
    stale_checkout_seconds: float = Field(default=86_400, gt=0)


class BaseImageSettings(StrictModel):
    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    languages: tuple[str, ...] = Field(min_length=1)
    runtime_family: str = Field(min_length=1)
    runtime_version: str | None = None
    capabilities: tuple[str, ...] = ()
    gpu_support: bool = False
    recipe: str | None = None
    description: str = ""

    @field_validator("name", "image")
    @classmethod
    def no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("base image name and reference must not have surrounding whitespace")
        return value

    @field_validator("languages")
    @classmethod
    def normalized_languages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if not normalized or any(not value for value in normalized):
            raise ValueError("base image languages must be non-empty labels")
        return normalized

    @field_validator("runtime_family")
    @classmethod
    def normalized_runtime_family(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("base image runtime_family must not be empty")
        return normalized

    @field_validator("runtime_version")
    @classmethod
    def normalized_runtime_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("base image runtime_version must be null or non-empty")
        return normalized

    @field_validator("capabilities")
    @classmethod
    def normalized_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if any(not value for value in normalized):
            raise ValueError("base image capabilities may not contain empty values")
        return normalized

    @field_validator("recipe")
    @classmethod
    def safe_recipe_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-." for character in normalized
        ):
            raise ValueError("base image recipe must be a lowercase asset name")
        return normalized


class EnvironmentAuditSettings(StrictModel):
    enabled: bool = False
    directory: Path = Path(".featurebench/audit/environments")


class EnvironmentSettings(StrictModel):
    workdir: Literal["/testbed"] = "/testbed"
    planner_image: str = "ubuntu:24.04"
    allow_gpu_repositories: bool = False
    clone_command_timeout_seconds: int = Field(default=300, ge=1)
    clone_low_speed_bytes_per_second: int = Field(default=1024, ge=1)
    clone_low_speed_timeout_seconds: int = Field(default=30, ge=1)
    collect_timeout_seconds: float = Field(default=120, gt=0)
    min_verified_test_points: int = Field(default=10, ge=1)
    baseline_group_max_test_points: int = Field(default=100, ge=1)
    baseline_total_timeout_seconds: float = Field(default=3600, gt=0)
    total_timeout_seconds: float = Field(default=7_200, gt=0)
    quickcheck_test_points: int = Field(default=10, ge=1, le=1000)
    quickcheck_min_pass_rate: float = Field(default=0.5, gt=0, le=1)
    collect_stability_runs: int = Field(default=2, ge=2, le=5)
    max_build_attempts: int = Field(default=10, ge=1, le=10)
    cleanup_intermediate_images: bool = True
    audit: EnvironmentAuditSettings = EnvironmentAuditSettings()
    base_images: tuple[BaseImageSettings, ...] = Field(
        default_factory=lambda: (
            BaseImageSettings(
                name="generic-ubuntu",
                image="ubuntu:24.04",
                languages=("generic",),
                runtime_family="generic",
            ),
        )
    )

    @field_validator("planner_image")
    @classmethod
    def valid_planner_image(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(
                "planner_image must be a non-empty reference without surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def unique_base_images(self) -> EnvironmentSettings:
        names = [entry.name for entry in self.base_images]
        if len(names) != len(set(names)):
            raise ValueError("environment base image names must be unique")
        if not self.base_images:
            raise ValueError("environment must configure at least one base image")
        if self.allow_gpu_repositories and not any(entry.gpu_support for entry in self.base_images):
            raise ValueError("allow_gpu_repositories requires at least one GPU-capable base image")
        return self


class FactorySettings(StrictModel):
    """The one current YAML configuration schema."""

    workspace: Path = Path(".featurebench")
    concurrency: int = Field(default=4, ge=1)
    sources: SourceSettings = SourceSettings()
    docker: DockerSettings = DockerSettings()
    agent: AgentSettings = AgentSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    repository: RepositorySettings = RepositorySettings()
    environment: EnvironmentSettings = EnvironmentSettings()
    adapters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    refresh_per_second: float = Field(default=8, gt=0, le=60)
    jsonl_fsync_interval: int = Field(default=1, ge=1, le=1000)
    env_file: Path = Path(".env")

    @model_validator(mode="after")
    def gpu_resources_are_explicit(self) -> FactorySettings:
        if self.environment.allow_gpu_repositories and self.docker.gpus is None:
            raise ValueError(
                "environment.allow_gpu_repositories requires docker.gpus (for example, 'all')"
            )
        return self

    @classmethod
    def _from_mapping(
        cls,
        document: Mapping[str, object],
        *,
        adapter_catalog: AdapterSettingsCatalog | None = None,
    ) -> FactorySettings:
        raw: dict[str, Any] = dict(document)
        if "adapters" in raw:
            raise ValueError(
                "the adapters category is internal; place each Adapter's settings under "
                "its registered top-level type"
            )
        if adapter_catalog is None:
            raw["adapters"] = {}
        else:
            adapter_values = {
                adapter_type: raw.pop(adapter_type)
                for adapter_type in adapter_catalog.types
                if adapter_type in raw
            }
            raw["adapters"] = adapter_catalog.validate_settings(adapter_values)
        return cls.model_validate(raw)

    @classmethod
    def from_mapping(
        cls,
        document: Mapping[str, object],
        *,
        adapter_catalog: AdapterSettingsCatalog | None = None,
    ) -> FactorySettings:
        """Validate one already-composed external configuration document."""
        try:
            return cls._from_mapping(document, adapter_catalog=adapter_catalog)
        except ValueError as exc:
            raise ConfigurationError(f"invalid configuration: {exc}") from exc

    @classmethod
    def from_files(
        cls,
        paths: tuple[str | Path, ...],
        *,
        adapter_catalog: AdapterSettingsCatalog | None = None,
    ) -> FactorySettings:
        yaml_files = tuple(Path(path) for path in paths)
        if not yaml_files:
            raise ConfigurationError("at least one YAML configuration file is required")
        raw: dict[str, Any] = {}
        owners: dict[str, Path] = {}
        for yaml_file in yaml_files:
            if not yaml_file.is_file() or yaml_file.suffix.lower() not in (".yaml", ".yml"):
                raise ConfigurationError(f"configuration path must be a YAML file: {yaml_file}")
            try:
                document: Any = yaml.load(
                    yaml_file.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
                )
            except OSError as exc:
                raise ConfigurationError(f"cannot read configuration {yaml_file}: {exc}") from exc
            except yaml.YAMLError as exc:
                raise ConfigurationError(f"invalid YAML in {yaml_file}: {exc}") from exc
            if document is None:
                document = {}
            if not isinstance(document, dict):
                raise ConfigurationError(f"configuration root must be an object: {yaml_file}")
            document_mapping = cast(dict[Any, Any], document)
            for key, value in document_mapping.items():
                if not isinstance(key, str):
                    raise ConfigurationError(f"configuration keys must be strings: {yaml_file}")
                if key in raw:
                    raise ConfigurationError(
                        f"configuration category {key!r} appears in both "
                        f"{owners[key]} and {yaml_file}"
                    )
                raw[key] = value
                owners[key] = yaml_file
        try:
            return cls._from_mapping(raw, adapter_catalog=adapter_catalog)
        except ValueError as exc:
            rendered = ", ".join(str(path) for path in yaml_files)
            raise ConfigurationError(f"invalid configuration files [{rendered}]: {exc}") from exc

    def initialize_workspace(self) -> Path:
        root = self.workspace.resolve()
        for child in (root, root / "runs", root / "staging"):
            child.mkdir(parents=True, exist_ok=True)
        return root

    def load_host_environment(self) -> bool:
        """Load optional host-only credentials without overriding explicit process values."""
        return load_dotenv(dotenv_path=self.env_file, override=False)
