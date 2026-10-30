"""Bounded Python source indexing shared by generation and unit tests."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import PurePosixPath
from typing import cast
from uuid import uuid4

from codeworld.adapters.featurebench.models import (
    DefinitionTimeReference,
    ImportBinding,
    RepositoryIndexData,
    SourceObject,
    TraceRecord,
)
from codeworld.errors import ContractError
from codeworld.runtime.base import ContainerSessionProtocol

INDEXED_SOURCE_SUFFIXES = frozenset({".py", ".pyi", ".pyx", ".pxd"})

_SNAPSHOT_SCRIPT = r"""
import json
import os
import pathlib
import subprocess
import sys

output = pathlib.Path(sys.argv[1])
max_files = int(sys.argv[2])
max_file_bytes = int(sys.argv[3])
max_total_bytes = int(sys.argv[4])
root = pathlib.Path.cwd().resolve()
suffixes = set(sys.argv[5:])
files = {}
total = 0
tracked = subprocess.run(
    ["git", "ls-files", "-z", "--cached", "--recurse-submodules"],
    cwd=root,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if tracked.returncode != 0:
    sys.stderr.buffer.write(tracked.stderr)
    raise SystemExit(tracked.returncode)
for raw_relative in tracked.stdout.split(b"\0"):
    if not raw_relative:
        continue
    relative = pathlib.Path(os.fsdecode(raw_relative))
    path = root / relative
    if len(files) >= max_files:
        break
    if not path.is_file() or path.suffix not in suffixes:
        continue
    try:
        size = path.stat().st_size
    except OSError:
        continue
    if size > max_file_bytes or total + size > max_total_bytes:
        continue
    try:
        files[str(relative)] = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        continue
    total += size
output.write_text(json.dumps({"files": files, "total_bytes": total}), encoding="utf-8")
"""

_TEST_PARTS = {"test", "tests", "testing"}
_ROOT_PREFIXES = {"src", "lib", "python", "source"}


def is_test_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        any(part.lower() in _TEST_PARTS for part in candidate.parts[:-1])
        or candidate.name.startswith("test_")
        or candidate.name.endswith("_test.py")
        or candidate.name == "conftest.py"
    )


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def _definition_signature(
    source: str, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
) -> str:
    lines = source.splitlines()
    body_line = node.body[0].lineno if node.body else node.lineno + 1
    return "\n".join(lines[node.lineno - 1 : max(node.lineno, body_line - 1)]).strip()


def _literal_string_sequence(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return ()
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return ()
        values.append(item.value)
    return tuple(values)


def _module_names(path: str) -> tuple[str, ...]:
    candidate = PurePosixPath(path)
    parts = list(candidate.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    names: list[str] = []
    for start in range(len(parts)):
        if start and parts[start - 1] not in _ROOT_PREFIXES:
            continue
        name = ".".join(parts[start:])
        if name:
            names.append(name)
    if parts:
        names.append(".".join(parts))
    return tuple(dict.fromkeys(names))


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.source = source
        self.scope: list[tuple[str, bool]] = []
        self.definitions: list[SourceObject] = []

    def _qualname(self, name: str) -> str:
        parts: list[str] = []
        for owner, is_function in self.scope:
            parts.append(owner)
            if is_function:
                parts.append("<locals>")
        parts.append(name)
        return ".".join(parts)

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if node.end_lineno is None:
            return
        qualname = self._qualname(node.name)
        decorators = tuple(_safe_unparse(item) for item in node.decorator_list)
        start_line = min((item.lineno for item in node.decorator_list), default=node.lineno)
        kind = (
            "class"
            if isinstance(node, ast.ClassDef)
            else "async_function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "function"
        )
        parent_qualname = None
        if self.scope:
            parent_parts: list[str] = []
            for owner, is_function in self.scope:
                parent_parts.append(owner)
                if is_function:
                    parent_parts.append("<locals>")
            parent_qualname = ".".join(parent_parts)
        bases = (
            tuple(_safe_unparse(base) for base in node.bases)
            if isinstance(node, ast.ClassDef)
            else ()
        )
        self.definitions.append(
            SourceObject(
                object_id=f"{self.path}::{qualname}::{node.lineno}",
                path=self.path,
                qualname=qualname,
                name=node.name,
                kind=kind,
                line=node.lineno,
                start_line=start_line,
                end_line=node.end_lineno,
                parent_qualname=parent_qualname,
                signature=_definition_signature(self.source, node),
                docstring=ast.get_docstring(node, clean=False) or "",
                decorators=decorators,
                bases=bases,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self.scope.append((node.name, True))
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)
        self.scope.append((node.name, True))
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node)
        self.scope.append((node.name, False))
        self.generic_visit(node)
        self.scope.pop()


class _RuntimeReferenceVisitor(ast.NodeVisitor):
    """Attribute definition-time name references to the declaration owning them."""

    def __init__(self, definitions_by_line: Mapping[int, SourceObject]) -> None:
        self.definitions_by_line = definitions_by_line
        self.scope: list[str] = []
        self.owner_object_id: str | None = None
        self.references: set[tuple[str | None, str | None, str]] = set()

    @property
    def parent_qualname(self) -> str | None:
        return ".".join(self.scope) or None

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.references.add((self.owner_object_id, self.parent_qualname, node.id))

    def _with_owner(self, owner_object_id: str | None, callback: Callable[[], None]) -> None:
        previous = self.owner_object_id
        self.owner_object_id = owner_object_id
        try:
            callback()
        finally:
            self.owner_object_id = previous

    def _visit_function_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if node.args.vararg is not None:
            arguments = (*arguments, node.args.vararg)
        if node.args.kwarg is not None:
            arguments = (*arguments, node.args.kwarg)
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", ()):  # Python 3.12+
            self.visit(type_parameter)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        owner = self.definitions_by_line.get(node.lineno)
        self._with_owner(
            owner.object_id if owner is not None else self.owner_object_id,
            lambda: self._visit_function_header(node),
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        owner = self.definitions_by_line.get(node.lineno)
        self._with_owner(
            owner.object_id if owner is not None else self.owner_object_id,
            lambda: self._visit_function_header(node),
        )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        owner = self.definitions_by_line.get(node.lineno)

        def visit_class() -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)
            for type_parameter in getattr(node, "type_params", ()):  # Python 3.12+
                self.visit(type_parameter)
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()

        self._with_owner(
            owner.object_id if owner is not None else self.owner_object_id,
            visit_class,
        )


def index_sources(sources: Mapping[str, str]) -> RepositoryIndexData:
    """Build a compact, deterministic index from a bounded repository snapshot."""
    definitions: list[SourceObject] = []
    imports: list[ImportBinding] = []
    exports: dict[str, tuple[str, ...]] = {}
    module_to_path: dict[str, str] = {}
    python_sources = {path: source for path, source in sources.items() if path.endswith(".py")}
    for path in python_sources:
        for module in _module_names(path):
            module_to_path.setdefault(module, path)

    parsed: dict[str, ast.Module] = {}
    for path, source in sorted(python_sources.items()):
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError):
            continue
        parsed[path] = tree
        visitor = _DefinitionVisitor(path, source)
        visitor.visit(tree)
        definitions.extend(visitor.definitions)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = tuple((alias.name, alias.asname) for alias in node.names)
                module = node.module or "" if isinstance(node, ast.ImportFrom) else ""
                level = node.level if isinstance(node, ast.ImportFrom) else 0
                imports.append(
                    ImportBinding(
                        importer_path=path,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        module=module,
                        level=level,
                        names=names,
                    )
                )
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    isinstance(target, ast.Name) and target.id == "__all__" for target in targets
                ):
                    values = _literal_string_sequence(node.value) if node.value is not None else ()
                    if values:
                        exports[path] = values

    resolved_imports: list[ImportBinding] = []
    for binding in imports:
        importer_modules = _module_names(binding.importer_path)
        importer_module = importer_modules[0] if importer_modules else ""
        package = importer_module.rsplit(".", 1)[0] if "." in importer_module else ""
        if PurePosixPath(binding.importer_path).name == "__init__.py":
            package = importer_module
        if binding.level:
            parts = package.split(".") if package else []
            keep = max(0, len(parts) - binding.level + 1)
            prefix = ".".join(parts[:keep])
            absolute = ".".join(part for part in (prefix, binding.module) if part)
        else:
            absolute = binding.module
        resolved_imports.append(
            binding.model_copy(update={"resolved_path": module_to_path.get(absolute)})
        )

    extension_references: dict[str, tuple[str, ...]] = {}
    identifiers = re.compile(r"\b[A-Za-z_]\w*\b")
    for path, source in sorted(sources.items()):
        if PurePosixPath(path).suffix not in {".pyi", ".pyx", ".pxd"}:
            continue
        extension_references[path] = tuple(sorted(set(identifiers.findall(source))))
    definitions_by_scope: dict[tuple[str, str | None, str], list[SourceObject]] = defaultdict(list)
    for item in definitions:
        definitions_by_scope[(item.path, item.parent_qualname, item.name)].append(item)
    definition_time_references: set[tuple[str, str | None]] = set()
    for path, tree in parsed.items():
        definitions_by_line = {item.line: item for item in definitions if item.path == path}
        visitor = _RuntimeReferenceVisitor(definitions_by_line)
        for statement in tree.body:
            visitor.visit(statement)
        for owner_object_id, parent_qualname, name in visitor.references:
            matches = definitions_by_scope.get((path, parent_qualname, name), ())
            if not matches and parent_qualname is not None:
                matches = definitions_by_scope.get((path, None, name), ())
            definition_time_references.update((item.object_id, owner_object_id) for item in matches)
    return RepositoryIndexData(
        definitions=tuple(definitions),
        imports=tuple(resolved_imports),
        exports=exports,
        extension_references=extension_references,
        definition_time_references=tuple(
            DefinitionTimeReference(
                referenced_object_id=referenced_object_id,
                owner_object_id=owner_object_id,
            )
            for referenced_object_id, owner_object_id in sorted(
                definition_time_references,
                key=lambda item: (item[0], item[1] or ""),
            )
        ),
    )


class RepositoryIndex:
    """Queryable static index with runtime trace identity normalization."""

    def __init__(self, data: RepositoryIndexData, sources: Mapping[str, str]) -> None:
        self.data = data
        self.sources = dict(sources)
        self.by_id = {item.object_id: item for item in data.definitions}
        self.by_runtime_key: dict[tuple[str, str, int], SourceObject] = {}
        self.by_qualname: dict[tuple[str, str], SourceObject] = {}
        binding_groups: dict[tuple[str, str], list[SourceObject]] = defaultdict(list)
        runtime_locations: dict[tuple[str, int], list[SourceObject]] = defaultdict(list)
        for item in data.definitions:
            self.by_runtime_key[(item.path, item.qualname, item.line)] = item
            self.by_runtime_key[(item.path, item.qualname, item.start_line)] = item
            self.by_qualname[(item.path, item.qualname)] = item
            binding_groups[(item.path, item.qualname)].append(item)
            for line in {item.line, item.start_line}:
                runtime_locations[(item.path, line)].append(item)
        self.by_runtime_location = {key: tuple(values) for key, values in runtime_locations.items()}
        self.binding_groups = {
            key: tuple(sorted(items, key=lambda item: (item.start_line, item.object_id)))
            for key, items in binding_groups.items()
        }
        self._protected_ids_cache: dict[frozenset[str], frozenset[str]] = {}

    @classmethod
    def from_sources(cls, sources: Mapping[str, str]) -> RepositoryIndex:
        return cls(index_sources(sources), sources)

    @classmethod
    async def build(
        cls,
        session: ContainerSessionProtocol,
        *,
        max_files: int,
        max_file_bytes: int,
        max_total_bytes: int,
        timeout_seconds: float,
    ) -> RepositoryIndex:
        script = f"/tmp/featurebench-featurebench-index-{uuid4().hex}.py"
        output = f"/tmp/featurebench-featurebench-index-{uuid4().hex}.json"
        try:
            await session.exec(
                ("tee", script), workdir="/", input_bytes=_SNAPSHOT_SCRIPT.encode(), check=True
            )
            result = await session.exec(
                (
                    "python",
                    script,
                    output,
                    str(max_files),
                    str(max_file_bytes),
                    str(max_total_bytes),
                    *sorted(INDEXED_SOURCE_SUFFIXES),
                ),
                activated=True,
                timeout_seconds=timeout_seconds,
                check=False,
            )
            if result.exit_code != 0:
                raise ContractError(
                    f"FeatureBench source indexing failed: {result.combined_output[-5000:]}"
                )
            raw = await session.read_absolute_file(
                output, max_bytes=max_total_bytes * 2 + 1_000_000
            )
            payload: object = json.loads(raw)
            files_value = (
                cast(dict[str, object], payload).get("files") if isinstance(payload, dict) else None
            )
            if not isinstance(files_value, dict) or not files_value:
                raise ContractError("FeatureBench source index contains no Python sources")
            raw_files = cast(dict[object, object], files_value)
            if not all(
                isinstance(path, str) and isinstance(source, str)
                for path, source in raw_files.items()
            ):
                raise ContractError("FeatureBench source index contains non-string file entries")
            sources = {cast(str, path): cast(str, source) for path, source in raw_files.items()}
            return cls.from_sources(sources)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid FeatureBench source index: {exc}") from exc
        finally:
            await session.exec(("rm", "-f", script, output), workdir="/", check=False)

    def _resolve_runtime_identity(self, path: str, qualname: str, line: int) -> SourceObject | None:
        exact = self.by_runtime_key.get((path, qualname, line)) or self.by_qualname.get(
            (path, qualname)
        )
        if exact is not None:
            return exact
        # Some framework wrappers execute a class method through a code object whose
        # co_qualname has lost only the owning class. Reconcile that representation
        # only when the exact definition location leaves one suffix-compatible AST
        # declaration; never guess from a name alone.
        suffix = f".{qualname}"
        candidates = tuple(
            item
            for item in self.by_runtime_location.get((path, line), ())
            if item.qualname.endswith(suffix)
        )
        return candidates[0] if len(candidates) == 1 else None

    def resolve_trace(self, record: TraceRecord) -> SourceObject | None:
        return self._resolve_runtime_identity(record.path, record.qualname, record.line)

    def normalize_runtime_id(self, runtime_id: str) -> str | None:
        try:
            path, qualname, raw_line = runtime_id.rsplit("::", 2)
            line = int(raw_line)
        except (ValueError, TypeError):
            return None
        item = self._resolve_runtime_identity(path, qualname, line)
        return item.object_id if item is not None else None

    def traced_objects(self, records: Iterable[TraceRecord]) -> tuple[SourceObject, ...]:
        merged: dict[str, SourceObject] = {}
        callers: dict[str, set[str]] = defaultdict(set)
        calls: dict[str, int] = defaultdict(int)
        for record in records:
            item = self.resolve_trace(record)
            if item is None:
                continue
            calls[item.object_id] += record.calls
            for caller in record.callers:
                normalized = self.normalize_runtime_id(caller)
                callers[item.object_id].add(normalized or caller)
            merged[item.object_id] = item
        return tuple(
            merged[object_id].model_copy(
                update={"calls": calls[object_id], "callers": tuple(sorted(callers[object_id]))}
            )
            for object_id in sorted(merged)
        )

    def ancestors(self, item: SourceObject) -> tuple[SourceObject, ...]:
        result: list[SourceObject] = []
        parent = item.parent_qualname
        while parent:
            owner = self.by_qualname.get((item.path, parent))
            if owner is None:
                break
            result.append(owner)
            parent = owner.parent_qualname
        return tuple(reversed(result))

    def protected_ids(self, critical_magic_methods: frozenset[str]) -> frozenset[str]:
        cached = self._protected_ids_cache.get(critical_magic_methods)
        if cached is not None:
            return cached
        protected = {
            item.referenced_object_id
            for item in self.data.definition_time_references
            if item.owner_object_id is None
        }
        for item in self.data.definitions:
            for extension_path, names in self.data.extension_references.items():
                if item.name not in names:
                    continue
                same_module = str(PurePosixPath(extension_path).with_suffix(".py")) == item.path
                extension_source = self.sources.get(extension_path, "")
                imported_name = (
                    re.search(rf"\bimport\s+[^\n]*\b{re.escape(item.name)}\b", extension_source)
                    is not None
                )
                if same_module or imported_name:
                    protected.add(item.object_id)
                    break
            # Python data-model methods participate in class construction and
            # protocol conformance even when a particular test never calls them.
            # Removing an abstract-method implementation, for example, can make
            # the class uninstantiable before execution reaches any traced line.
            is_data_model_method = (
                len(item.name) > 4 and item.name.startswith("__") and item.name.endswith("__")
            )
            if item.name in critical_magic_methods or is_data_model_method:
                protected.add(item.object_id)
        result = frozenset(protected)
        self._protected_ids_cache[critical_magic_methods] = result
        return result

    @staticmethod
    def _is_abstract_declaration(item: SourceObject) -> bool:
        return any(
            decorator.split("(", 1)[0].strip().rsplit(".", 1)[-1]
            in {"abstractmethod", "abstractproperty"}
            for decorator in item.decorators
        )

    def structural_test_impact_ids(self, deleted_object_ids: Iterable[str]) -> frozenset[str]:
        """Expand deletions by class-level effects invisible to body tracing."""
        impacts = set(deleted_object_ids)
        abstract_contract_names = {
            item.name for item in self.data.definitions if self._is_abstract_declaration(item)
        }
        for object_id in tuple(impacts):
            item = self.by_id.get(object_id)
            if (
                item is None
                or item.parent_qualname is None
                or item.name not in abstract_contract_names
                or self._is_abstract_declaration(item)
            ):
                continue
            owner = item.parent_qualname
            impacts.update(
                candidate.object_id
                for candidate in self.data.definitions
                if candidate.path == item.path
                and (candidate.qualname == owner or candidate.parent_qualname == owner)
            )
        return frozenset(impacts)

    def definition_time_blockers(
        self, deleted_object_ids: Iterable[str]
    ) -> dict[str, tuple[str | None, ...]]:
        """Return surviving owners that require declarations in the deletion set."""
        deleted = frozenset(deleted_object_ids)
        blockers: dict[str, set[str | None]] = defaultdict(set)
        for reference in self.data.definition_time_references:
            if reference.referenced_object_id not in deleted:
                continue
            if reference.owner_object_id is None or reference.owner_object_id not in deleted:
                blockers[reference.referenced_object_id].add(reference.owner_object_id)
        return {
            object_id: tuple(sorted(owners, key=lambda owner: (owner is not None, owner or "")))
            for object_id, owners in sorted(blockers.items())
        }
