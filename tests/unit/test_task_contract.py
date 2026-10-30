from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from codeworld.adapters.base import (
    AdapterDefinition,
    AdapterGenerator,
    GeneratorContext,
    QualityGateResult,
    TaskQualityGate,
)
from codeworld.adapters.catalog import default_adapter_catalog
from codeworld.adapters.featurebench.config import FeatureBenchSettings
from codeworld.config import FactorySettings
from codeworld.environment.models import EnvironmentRecord
from codeworld.models import RuntimeSpec, TaskRecord
from codeworld.tasks.jsonl import read_tasks, write_tasks


class ExamplePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["example"]
    prompt: str
    value: int


class ExampleGenerator:
    async def generate(
        self,
        environment: EnvironmentRecord,
        context: GeneratorContext,
    ) -> AsyncGenerator[TaskRecord, None]:
        del environment, context
        yield task()


class ExampleQualityGate:
    async def check(
        self,
        task: TaskRecord,
        payload: BaseModel,
        context: GeneratorContext,
    ) -> QualityGateResult:
        del task, payload, context
        return QualityGateResult(passed=True)


def definition() -> AdapterDefinition:
    return AdapterDefinition(
        type="example",
        payload_model=ExamplePayload,
        generator=ExampleGenerator(),
        quality_gate=ExampleQualityGate(),
    )


def task(task_id: str = "task-1") -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        environment_id="environment-1",
        image="example/image@sha256:" + "a" * 64,
        runtime=RuntimeSpec(
            activation="#!/usr/bin/env bash\n# no project-specific activation required\ncd /testbed\n",
            runner=(
                "#!/usr/bin/env bash\n"
                "# Example language-neutral Runner V1 artifact used by the task contract test.\n"
                "echo '{}'\n"
            ),
        ),
        adapter={"type": "example", "prompt": "Fix the implementation.", "value": 3},
    )


def test_task_has_only_the_universal_fields() -> None:
    payload = task().model_dump()
    assert set(payload) == {"task_id", "environment_id", "image", "runtime", "adapter"}
    assert set(payload["runtime"]) == {
        "workdir",
        "activation",
        "runner",
        "limits",
        "resources",
    }
    with pytest.raises(ValueError):
        TaskRecord.model_validate({**payload, "metadata": {}})
    with pytest.raises(ValueError, match="prompt"):
        TaskRecord.model_validate({**payload, "prompt": "must belong to the Adapter"})
    with pytest.raises(ValueError, match="environment_id"):
        TaskRecord.model_validate(
            {key: value for key, value in payload.items() if key != "environment_id"}
        )
    with pytest.raises(ValueError, match="surrounding whitespace"):
        TaskRecord.model_validate({**payload, "environment_id": " environment-1"})


def test_definition_validates_adapter_owned_payload() -> None:
    adapter = definition()
    payload = adapter.validate_task(task())
    assert adapter.type == "example"
    assert isinstance(adapter.generator, ExampleGenerator)
    assert isinstance(adapter.quality_gate, ExampleQualityGate)
    assert isinstance(adapter.generator, AdapterGenerator)
    assert isinstance(adapter.quality_gate, TaskQualityGate)
    assert payload == ExamplePayload(type="example", prompt="Fix the implementation.", value=3)


def test_default_catalog_loads_registration_hook_from_adapter_package() -> None:
    settings = FactorySettings()
    catalog = default_adapter_catalog()
    registration = catalog.get("featurebench")
    definition = registration.build(settings.adapters.get("featurebench"))

    assert registration.settings_model is FeatureBenchSettings
    assert definition.type == "featurebench"


def test_jsonl_is_atomic_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    output = tmp_path / "tasks.jsonl"
    assert write_tasks(output, [task()]) == 1
    assert read_tasks(output) == [task()]
    assert json.loads(output.read_text().strip())["adapter"]["value"] == 3

    with pytest.raises(ValueError, match="duplicate task_id"):
        write_tasks(output, [task(), task()])
    assert read_tasks(output) == [task()]
