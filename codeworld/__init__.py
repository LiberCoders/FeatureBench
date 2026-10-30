"""Generic environment and task-generation engine embedded in FeatureBench."""

from codeworld.adapters.base import (
    AdapterDefinition,
    AdapterGenerator,
    AdapterLifecycle,
    GeneratorContext,
    QualityGateResult,
    TaskQualityGate,
)
from codeworld.adapters.catalog import (
    AdapterCatalog,
    AdapterInvocation,
    AdapterRegistration,
    default_adapter_catalog,
)
from codeworld.models import TaskRecord

__all__ = [
    "AdapterCatalog",
    "AdapterDefinition",
    "AdapterGenerator",
    "AdapterInvocation",
    "AdapterLifecycle",
    "AdapterRegistration",
    "GeneratorContext",
    "QualityGateResult",
    "TaskQualityGate",
    "TaskRecord",
    "default_adapter_catalog",
]
