"""
sitecustomize.py - subprocess tracing injector

This module auto-initializes `DependencyTracer` at subprocess entry and forces
`quiet=True` to avoid stdout messages like "Initializing tracer...". This keeps
stdout clean for tools that require pristine output (e.g., Meson checks for
distutils).

## How Python auto-loads sitecustomize

- CPython imports the built-in `site` module on startup; it extends `sys.path`,
    registers encodings, and performs other initialization.
- `site` tries to import `sitecustomize` and `usercustomize` in order. If either
    is found in any package/directory, its top-level code runs immediately.
- We prepend `acebench/resources/scripts/dynamic_script` to `PYTHONPATH`, so the
    subprocess interpreter sees this file during initialization and runs it.
- The `_init_subprocess_tracer()` function at the end is invoked via this
    mechanism, so no explicit import is needed. All CLI/subprocesses get a full
    tracer attached.

## Workflow

1. **Parent process env**: `DynamicTracer.trace_pytest_execution()` writes a set
     of environment variables before running pytest:
     - `ACE_TRACE_SUBPROCESS=1`: mark subprocesses for tracing;
     - `ACE_TRACE_REPO_ROOT`: repo root (the real path for `/testbed`);
     - `ACE_TRACE_OUTPUT_FILE`: absolute path to the trace output file;
     - `ACE_TRACE_DEBUG=1` (optional): enable debug output;
     - `PYTHONPATH`: prepend dynamic_script directory so subprocesses can load
         `sitecustomize.py` and its dependencies.

2. **sitecustomize injection**: on subprocess startup, the interpreter loads
     `sitecustomize.py` via `PYTHONPATH`. After detecting the env vars, the script:
     - instantiates a full `DependencyTracer`;
     - clears cached modules and installs patches (import/Autograd/joblib);
     - enables `sys.setprofile` and `threading.setprofile` to record the call graph.

3. **Subprocess teardown**: in an `atexit` hook:
     - disable profiling and uninstall patches;
     - normalize call records and rebuild the dependency graph;
     - serialize `objects`, `imports`, and stats;
     - merge data back into the main JSON via `_merge_trace_output()`.

4. **File-level merge**: the merge runs under an exclusive file lock to avoid
     concurrent writes. It also:
     - deduplicates imports by `(importer_file, line, imported_name)`;
     - merges object nodes by `file::name::line`;
     - updates `stats` (added import/object counts, call counts, edge counts,
         trace_runs, etc.).

5. **Parent process finalize**: after pytest returns, the parent still performs
     its own archival logic; the final JSON includes both parent and subprocess data.
"""

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import threading


_SITECUSTOMIZE_LOADED = True
if os.environ.get('ACE_TRACE_DEBUG') == '1':
    print(f"[sitecustomize] loaded, PID={os.getpid()}")


def _merge_trace_output(output_path: Path, new_results: Dict[str, Any], debug: bool) -> None:
    """Merge subprocess trace results under a file lock."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    import fcntl

    with open(output_path, 'r+', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        f.seek(0)
        raw_content = f.read()
        if raw_content:
            try:
                existing_data = json.loads(raw_content)
            except json.JSONDecodeError:
                existing_data = {'imports': [], 'objects': {}, 'stats': {}}
        else:
            existing_data = {'imports': [], 'objects': {}, 'stats': {}}

        existing_imports = existing_data.get('imports') or []
        existing_keys = {
            f"{imp.get('importer_file')}:{imp.get('importer_line')}:{imp.get('imported_name')}"
            for imp in existing_imports
        }

        new_imports_count = 0
        for imp in new_results.get('imports', []) or []:
            key = f"{imp.get('importer_file')}:{imp.get('importer_line')}:{imp.get('imported_name')}"
            if key not in existing_keys:
                existing_imports.append(imp)
                existing_keys.add(key)
                new_imports_count += 1

        existing_data['imports'] = existing_imports

        existing_objects = existing_data.get('objects') or {}
        new_objects = new_results.get('objects', {}) or {}
        new_object_count = 0
        for obj_id, obj_payload in new_objects.items():
            if obj_id not in existing_objects:
                new_object_count += 1
            existing_objects[obj_id] = obj_payload
        existing_data['objects'] = existing_objects

        stats = existing_data.setdefault('stats', {})
        stats.setdefault('subprocess_imports_added', 0)
        stats.setdefault('subprocess_objects_added', 0)
        stats.setdefault('subprocess_trace_runs', 0)

        stats['subprocess_imports_added'] += new_imports_count
        stats['subprocess_objects_added'] += new_object_count
        stats['subprocess_trace_runs'] += 1
        stats['total_imports'] = len(existing_imports)
        stats['total_objects'] = len(existing_objects)
        stats['total_edges'] = sum(len(obj.get('edges', [])) for obj in existing_objects.values())
        stats['total_calls'] = stats.get('total_calls', 0) + new_results.get('stats', {}).get('total_calls', 0)
        if 'repo_root' in new_results.get('stats', {}):
            stats['repo_root'] = new_results['stats']['repo_root']

        f.seek(0)
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
        f.truncate()
        f.flush()

        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    if debug:
        print(
            f"[subprocess trace] merge complete: new imports={new_imports_count}, "
            f"new objects={new_object_count}, file={output_path}"
        )


def _init_subprocess_tracer():
    """Start full dynamic tracing in the subprocess."""
    if os.environ.get('ACE_TRACE_SUBPROCESS') != '1':
        return

    repo_root = os.environ.get('ACE_TRACE_REPO_ROOT')
    output_file = os.environ.get('ACE_TRACE_OUTPUT_FILE')
    debug = os.environ.get('ACE_TRACE_DEBUG') == '1'

    if not repo_root or not output_file:
        return

    try:
        from dynamic_script import DependencyTracer
    except Exception as exc:
        if debug:
            print(f"[subprocess trace] failed to import DependencyTracer: {exc}")
        return

    tracer = DependencyTracer(repo_root, debug=debug, quiet=True)

    try:
        tracer._clear_repo_modules()
    except Exception:
        if debug:
            print("[subprocess trace] failed to clear module cache, continuing")

    tracer._hook_patches()

    sys.setprofile(tracer._trace_calls)
    threading.setprofile(tracer._trace_calls)

    if debug:
        print(
            f"[subprocess trace] full tracing started (PID={os.getpid()}), "
            f"repo_root={repo_root}, output={output_file}"
        )

    def _finalize_subprocess_trace():
        try:
            sys.setprofile(None)
            threading.setprofile(None)
        except Exception:
            pass

        try:
            tracer._unhook_patches()
        except Exception:
            if debug:
                print("[subprocess trace] failed to unhook patches")

        try:
            tracer._normalize_call_records()
            tracer._build_dependency_graph()
        except Exception as exc:
            if debug:
                print(f"[subprocess trace] normalize/build dependency graph failed: {exc}")

        try:
            serialized_objects = tracer._serialize_objects()
            serialized_imports = tracer._serialize_imports()
        except Exception as exc:
            if debug:
                print(f"[subprocess trace] serialization failed: {exc}")
            serialized_objects = {}
            serialized_imports = []

        stats = {
            'total_calls': len(tracer.call_records),
            'total_imports': len(tracer.import_records),
            'total_objects': len(tracer.objects),
            'total_edges': sum(len(obj.edges) for obj in tracer.objects.values()),
            'repo_root': str(tracer.repo_root),
        }

        _merge_trace_output(Path(output_file), {
            'imports': serialized_imports,
            'objects': serialized_objects,
            'stats': stats,
        }, debug)

        if debug:
            print(f"[subprocess trace] trace results written to {output_file}")

    atexit.register(_finalize_subprocess_trace)


_init_subprocess_tracer()
