"""Reusable Python dynamic tracing for FeatureBench classification."""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import uuid4

from codeworld.adapters.featurebench.models import (
    TraceBundle,
    TraceEdge,
    TraceImport,
    TraceRecord,
)
from codeworld.environment.frameworks import pytest as pytest_contract
from codeworld.errors import ContractError
from codeworld.runtime.base import ContainerSessionProtocol

_TRACE_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "tracing"
_PYTEST_ID_HELPER_SOURCE = Path(pytest_contract.__file__).read_bytes()


@dataclass(frozen=True, slots=True)
class TraceStatistics:
    duration_seconds: float
    report_bytes: int
    test_ids: int
    records: int
    edges: int
    imports: int
    process_count: int


def merge_trace_bundles(test_file: str, bundles: Iterable[TraceBundle]) -> TraceBundle:
    """Merge independently collected point traces into one deterministic file trace."""
    materialized = tuple(bundles)
    record_groups: dict[tuple[str, str, int], list[TraceRecord]] = {}
    edge_groups: dict[tuple[str, str], list[TraceEdge]] = {}
    import_groups: dict[tuple[str, int, str, str, str, int | None], list[TraceImport]] = {}
    for bundle in materialized:
        for record in bundle.records:
            record_groups.setdefault((record.path, record.qualname, record.line), []).append(record)
        for edge in bundle.edges:
            edge_groups.setdefault((edge.caller, edge.callee), []).append(edge)
        for item in bundle.imports:
            key = (
                item.importer_path,
                item.importer_line,
                item.imported_name,
                item.defined_path,
                item.defined_name,
                item.defined_line,
            )
            import_groups.setdefault(key, []).append(item)
    passed_test_ids = tuple(
        sorted({test_id for bundle in materialized for test_id in bundle.passed_test_ids})
    )
    records = tuple(
        TraceRecord(
            path=path,
            qualname=qualname,
            line=line,
            calls=sum(item.calls for item in group),
            callers=tuple(sorted({caller for item in group for caller in item.callers})),
            test_ids=tuple(sorted({test_id for item in group for test_id in item.test_ids})),
            process_ids=tuple(sorted({pid for item in group for pid in item.process_ids})),
        )
        for (path, qualname, line), group in sorted(record_groups.items())
    )
    edges = tuple(
        TraceEdge(
            caller=caller,
            callee=callee,
            calls=sum(item.calls for item in group),
            test_ids=tuple(sorted({test_id for item in group for test_id in item.test_ids})),
            process_ids=tuple(sorted({pid for item in group for pid in item.process_ids})),
            thread_ids=tuple(sorted({tid for item in group for tid in item.thread_ids})),
            provenance=tuple(sorted({value for item in group for value in item.provenance})),
        )
        for (caller, callee), group in sorted(edge_groups.items())
    )
    imports = tuple(
        TraceImport(
            importer_path=key[0],
            importer_line=key[1],
            imported_name=key[2],
            defined_path=key[3],
            defined_name=key[4],
            defined_line=key[5],
            test_ids=tuple(sorted({test_id for item in group for test_id in item.test_ids})),
        )
        for key, group in sorted(
            import_groups.items(),
            key=lambda item: (*item[0][:-1], item[0][-1] if item[0][-1] is not None else -1),
        )
    )
    return TraceBundle(
        test_file=test_file,
        records=records,
        edges=edges,
        imports=imports,
        passed_test_ids=passed_test_ids,
    )


def collection_trace_from_bundles(bundles: Iterable[TraceBundle]) -> TraceBundle:
    """Extract unattributed pytest collection/import evidence from ordinary execution traces."""
    # Filter each physical trace before merging. The same repository object can run once
    # during pytest bootstrap without attribution and later inside a test point; merging
    # first would attach the later test ID and erase the earlier bootstrap contract.
    unattributed = (
        TraceBundle(
            test_file=bundle.test_file,
            records=tuple(item for item in bundle.records if not item.test_ids),
            edges=tuple(item for item in bundle.edges if not item.test_ids),
            imports=tuple(item for item in bundle.imports if not item.test_ids),
        )
        for bundle in bundles
    )
    return merge_trace_bundles("<collection>", unattributed)


def decode_trace_test_id_ranges(value: object, test_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Decode canonical half-open ranges from the trace payload's shared ID dictionary."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("trace test_id_ranges must be an array")
    decoded: list[str] = []
    previous_stop = -1
    ranges = cast(list[object] | tuple[object, ...], value)
    for raw_range_value in ranges:
        if not isinstance(raw_range_value, (list, tuple)):
            raise ValueError("each trace test ID range must contain two integer indexes")
        raw_range = cast(list[object] | tuple[object, ...], raw_range_value)
        if len(raw_range) != 2 or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw_range
        ):
            raise ValueError("each trace test ID range must contain two integer indexes")
        start = cast(int, raw_range[0])
        stop = cast(int, raw_range[1])
        if start < 0 or stop <= start or stop > len(test_ids):
            raise ValueError(f"invalid trace test ID range: {[start, stop]!r}")
        if start <= previous_stop:
            raise ValueError("trace test ID ranges must be sorted, disjoint, and non-adjacent")
        decoded.extend(test_ids[start:stop])
        previous_stop = stop
    return tuple(decoded)


def _restore_item_test_ids(item: object, test_ids: tuple[str, ...]) -> object:
    if not isinstance(item, dict):
        return item
    restored: dict[str, object] = dict(cast(dict[str, object], item))
    if "test_ids" in restored:
        raise ValueError("trace items must reference the payload-level test ID dictionary")
    restored["test_ids"] = decode_trace_test_id_ranges(
        restored.pop("test_id_ranges", None), test_ids
    )
    return restored


class PythonTracer:
    """Trace only environment-verified pytest node IDs and reuse their call graphs."""

    def __init__(
        self, session: ContainerSessionProtocol, *, maximum_report_bytes: int = 200_000_000
    ) -> None:
        self.session = session
        self.maximum_report_bytes = maximum_report_bytes
        self._cache: dict[tuple[str, tuple[str, ...]], TraceBundle] = {}
        self.statistics: dict[tuple[str, tuple[str, ...]], TraceStatistics] = {}

    @staticmethod
    def trace_command(test_command: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if test_command and PurePosixPath(test_command[0]).name in ("pytest", "py.test"):
            return ("python",), test_command[1:]
        if len(test_command) >= 3 and test_command[:3] in (
            ("python", "-m", "pytest"),
            ("python3", "-m", "pytest"),
        ):
            return (test_command[0],), test_command[3:]
        if len(test_command) >= 3 and test_command[:2] == ("uv", "run"):
            pytest_index = next(
                (
                    index
                    for index, value in enumerate(test_command[2:], start=2)
                    if PurePosixPath(value).name in ("pytest", "py.test")
                ),
                None,
            )
            if pytest_index is not None:
                return (*test_command[:pytest_index], "python"), test_command[pytest_index + 1 :]
        raise ContractError(
            "FeatureBench test_command must invoke pytest directly, with python -m pytest, "
            "or with uv run pytest"
        )

    @classmethod
    def _pytest_invocation(
        cls,
        test_command: tuple[str, ...],
        test_file: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        launcher, configured_args = cls.trace_command(test_command)
        pytest_args = list(configured_args)
        # Public environment IDs are owned by the Runner and need not be native
        # pytest node IDs. Collect the bounded physical file; the trace plugin
        # resolves the requested public IDs against its collected items.
        pytest_args.append(test_file)
        return launcher, tuple(pytest_args)

    async def trace(
        self,
        *,
        test_command: tuple[str, ...],
        test_file: str,
        test_ids: tuple[str, ...],
        timeout_seconds: float,
    ) -> TraceBundle:
        if not test_ids:
            raise ContractError("FeatureBench tracing requires verified test IDs")
        if len(test_ids) != len(set(test_ids)):
            raise ContractError("FeatureBench tracing requires unique verified test IDs")
        key = (test_file, test_ids)
        if key in self._cache:
            return self._cache[key]
        started = time.monotonic()
        nonce = uuid4().hex
        runtime_directory = f"/tmp/featurebench-trace-runtime-{nonce}"
        shard_directory = f"/tmp/featurebench-trace-shards-{nonce}"
        output_path = f"/tmp/featurebench-trace-{nonce}.json"
        arguments_path = f"/tmp/featurebench-trace-arguments-{nonce}.json"
        launcher, pytest_args = self._pytest_invocation(
            test_command,
            test_file,
        )
        invocation = {
            "pytest_args": pytest_args,
            "requested_test_ids": test_ids,
        }
        try:
            await self.session.exec(
                ("mkdir", "-p", runtime_directory, shard_directory), workdir="/", check=True
            )
            await self.session.exec(
                ("tee", arguments_path),
                workdir="/",
                input_bytes=json.dumps(invocation).encode("utf-8"),
                check=True,
            )
            for asset in sorted(_TRACE_ASSET_DIRECTORY.glob("*.py.txt")):
                destination = f"{runtime_directory}/{asset.name.removesuffix('.txt')}"
                await self.session.exec(
                    ("tee", destination),
                    workdir="/",
                    input_bytes=asset.read_bytes(),
                    check=True,
                )
            await self.session.exec(
                ("tee", f"{runtime_directory}/featurebench_pytest_ids.py"),
                workdir="/",
                input_bytes=_PYTEST_ID_HELPER_SOURCE,
                check=True,
            )
            result = await self.session.exec(
                (
                    *launcher,
                    f"{runtime_directory}/runner.py",
                    output_path,
                    shard_directory,
                    arguments_path,
                ),
                activated=True,
                timeout_seconds=timeout_seconds,
                check=False,
            )
            if result.exit_code != 0:
                label = test_file or "test collection"
                raise ContractError(
                    f"dynamic trace failed for {shlex.quote(label)}:\n"
                    f"{result.combined_output[-5000:]}"
                )
            raw = await self.session.read_absolute_file(
                output_path, max_bytes=self.maximum_report_bytes
            )
            payload: Any = json.loads(raw)
            if not isinstance(payload, dict):
                raise ContractError("trace payload must be a JSON object")
            typed_payload = cast(dict[str, object], payload)
            test_ids_value = typed_payload.get("test_ids")
            if not isinstance(test_ids_value, (list, tuple)):
                raise ContractError("trace test_ids must be an array of strings")
            raw_test_ids = cast(list[object] | tuple[object, ...], test_ids_value)
            if not all(isinstance(item, str) for item in raw_test_ids):
                raise ContractError("trace test_ids must be an array of strings")
            reported_test_ids = tuple(cast(list[str] | tuple[str, ...], raw_test_ids))
            if (
                len(reported_test_ids) != len(set(reported_test_ids))
                or tuple(sorted(reported_test_ids)) != reported_test_ids
            ):
                raise ContractError("trace test_ids must be unique and sorted")
            unexpected_test_ids = set(reported_test_ids) - set(test_ids)
            if unexpected_test_ids:
                raise ContractError(
                    "trace executed test IDs outside the environment-verified selection: "
                    f"{sorted(unexpected_test_ids)[:20]!r}"
                )
            passed_test_ids_value = typed_payload.get("passed_test_ids")
            if not isinstance(passed_test_ids_value, (list, tuple)):
                raise ContractError("trace passed_test_ids must be an array of strings")
            passed_values = cast(list[object] | tuple[object, ...], passed_test_ids_value)
            if not all(isinstance(item, str) for item in passed_values):
                raise ContractError("trace passed_test_ids must be an array of strings")
            passed_test_ids = cast(tuple[str, ...], tuple(passed_values))
            if (
                len(passed_test_ids) != len(set(passed_test_ids))
                or tuple(sorted(passed_test_ids)) != passed_test_ids
            ):
                raise ContractError("trace passed_test_ids must be unique and sorted")
            unexpected_passed_test_ids = set(passed_test_ids) - set(test_ids)
            if unexpected_passed_test_ids:
                raise ContractError(
                    "trace passed test IDs outside the environment-verified selection: "
                    f"{sorted(unexpected_passed_test_ids)[:20]!r}"
                )
            records_value = typed_payload.get("records", ())
            edges_value = typed_payload.get("edges", ())
            imports_value = typed_payload.get("imports", ())
            if not all(
                isinstance(value, (list, tuple))
                for value in (records_value, edges_value, imports_value)
            ):
                raise ContractError("trace records, edges, and imports must be arrays")
            records = tuple(
                TraceRecord.model_validate(_restore_item_test_ids(item, reported_test_ids))
                for item in cast(list[object] | tuple[object, ...], records_value)
            )
            edges = tuple(
                TraceEdge.model_validate(_restore_item_test_ids(item, reported_test_ids))
                for item in cast(list[object] | tuple[object, ...], edges_value)
            )
            imports = tuple(
                TraceImport.model_validate(_restore_item_test_ids(item, reported_test_ids))
                for item in cast(list[object] | tuple[object, ...], imports_value)
            )
            process_count_value = typed_payload.get("process_count", 0)
            if (
                not isinstance(process_count_value, int)
                or isinstance(process_count_value, bool)
                or process_count_value < 0
            ):
                raise ContractError("trace process_count must be a non-negative integer")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"invalid trace report for {test_file or 'collection'}: {exc}"
            ) from exc
        finally:
            await self.session.exec(
                (
                    "rm",
                    "-rf",
                    runtime_directory,
                    shard_directory,
                    output_path,
                    arguments_path,
                ),
                workdir="/",
                check=False,
            )
        safe = tuple(
            record
            for record in records
            if not PurePosixPath(record.path).is_absolute()
            and ".." not in PurePosixPath(record.path).parts
        )
        safe_ids = {record.object_id for record in safe}
        safe_edges = tuple(
            edge for edge in edges if edge.caller in safe_ids and edge.callee in safe_ids
        )
        safe_imports = tuple(
            item
            for item in imports
            if not PurePosixPath(item.importer_path).is_absolute()
            and ".." not in PurePosixPath(item.importer_path).parts
            and not PurePosixPath(item.defined_path).is_absolute()
            and ".." not in PurePosixPath(item.defined_path).parts
        )
        bundle = TraceBundle(
            test_file=test_file or "<collection>",
            records=safe,
            edges=safe_edges,
            imports=safe_imports,
            passed_test_ids=passed_test_ids,
        )
        self._cache[key] = bundle
        self.statistics[key] = TraceStatistics(
            duration_seconds=time.monotonic() - started,
            report_bytes=len(raw),
            test_ids=len(reported_test_ids),
            records=len(safe),
            edges=len(safe_edges),
            imports=len(safe_imports),
            process_count=process_count_value,
        )
        return bundle
