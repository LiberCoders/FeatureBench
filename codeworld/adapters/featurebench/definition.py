"""FeatureBench adapter component assembly."""

from codeworld.adapters.base import AdapterDefinition
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.adapters.featurebench.generator import FeatureBenchGenerator
from codeworld.adapters.featurebench.models import FeatureBenchPayload
from codeworld.adapters.featurebench.quality_gate import FeatureBenchQualityGate
from codeworld.adapters.featurebench.resources import FeatureBenchRunResources
from codeworld.adapters.featurebench.runtime import FeatureBenchTaskRuntime
from codeworld.progress import AdapterPhase


def featurebench_definition(
    settings: FeatureBenchSettings | None = None,
    *,
    resources: FeatureBenchRunResources | None = None,
) -> AdapterDefinition:
    """Compose one FeatureBench Generator and Adapter-owned quality gate."""
    resolved_settings = settings or FeatureBenchSettings()
    resolved_resources = resources or FeatureBenchRunResources.from_settings(resolved_settings)
    runtime = FeatureBenchTaskRuntime()
    generator = FeatureBenchGenerator(
        resolved_settings,
        runtime=runtime,
        resources=resolved_resources,
    )
    return AdapterDefinition(
        type="featurebench",
        payload_model=FeatureBenchPayload,
        generator=generator,
        quality_gate=FeatureBenchQualityGate(runtime),
        quality_gate_concurrency=resolved_settings.seed_concurrency,
        maximum_tasks=resolved_settings.maximum_tasks_per_environment,
        display_name="FeatureBench",
        progress_phases=(
            AdapterPhase(
                id="inspect_repository",
                label="Indexing",
                unit="repository",
            ),
            AdapterPhase(
                id="analyze_tests",
                label="Tracing",
                unit="test files",
            ),
            AdapterPhase(
                id="create_tasks",
                label="Generating",
                unit="seeds",
            ),
        ),
        lifecycle=generator,
    )
