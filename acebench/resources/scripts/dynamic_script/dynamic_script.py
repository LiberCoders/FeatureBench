#!/usr/bin/env python3
"""
Dynamic dependency tracer - run pytest and trace all function/class calls.
"""
import sys
import os
import json
import threading
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import argparse
import pytest
import traceback

from autograd_backward_patch import AutogradBackwardPatcher
from import_hook_patch import ImportHookPatcher
from joblib_patch import JoblibBackendPatcher

@dataclass
class CallRecord:
    caller_file: str     # Caller file path, e.g. /home/toy_example/test_simple.py
    caller_name: str     # Caller function name, e.g. <module> or test_addition
    caller_line: int     # Caller line number, e.g. 42
    callee_file: str     # Callee file path
    callee_name: str     # Callee function name
    callee_line: int     # Callee line number
    call_stack_snapshot: tuple = None  # Full call stack snapshot at call time

@dataclass
class ImportRecord:
    """Import record - record one import operation."""
    defined_file: Optional[str]   # Actual definition file, e.g. '/testbed/file2.py'
    defined_line: Optional[int]   # Definition line number
    defined_name: str             # Real definition name, e.g. 'RealClass' or 'real_func'
    importer_file: Optional[str]  # File that performs the import
    importer_line: Optional[int]  # Line number of the import statement
    imported_name: str            # Name used in import, e.g. 'ExportedClass' (may be aliased)

@dataclass
class ObjInfo:
    id: str             # Unique identifier (filename::name::lineno)
    name: str           # Function/class name
    file: str           # File path
    line: int           # Definition line
    edges: set          # Set of dependent object ids
    call_count: int = 0 # Call count
    is_pass2pass: bool = False  # Whether this is a pass2pass object

class DependencyTracer:
    
    def __init__(self, repo_root: Optional[str] = None, debug: bool = False, quiet: bool = False):
        self.call_records: List[CallRecord] = []  # All call relationships
        self.import_records: List[ImportRecord] = []  # All import operations
        self.objects: Dict[str, ObjInfo] = {}     # Object dependency graph
        self.trace_lock = threading.Lock()  # Thread lock
        self.repo_root = Path(repo_root or os.getcwd()).resolve()   # Project root
        self.debug = debug  # Debug mode
        self.quiet = quiet  # Suppress normal output
        self._thread_local = threading.local()  # Per-thread call stack
        self.call_records_set: set = set()  # Dedup set
        self.import_records_set: set = set()  # Import dedup set
        self._patches: List[Any] = self._create_default_patches()   # Patches
        self._print(f"Tracer initialized, repo root: {self.repo_root}")
        if debug:
            self._debug("Debug mode enabled")
    
    def _print(self, *args, **kwargs) -> None:
        if not self.quiet:
            print(*args, **kwargs)

    def _debug(self, *args, **kwargs) -> None:
        if self.debug and not self.quiet:
            print(*args, **kwargs)

    def _get_call_stack(self) -> List[tuple]:
        stack = getattr(self._thread_local, 'stack', None)
        if stack is None:
            stack = []
            self._thread_local.stack = stack
        return stack

    def _create_default_patches(self) -> List[Any]:
        return [
            AutogradBackwardPatcher(self, CallRecord),
            ImportHookPatcher(self, ImportRecord),
            JoblibBackendPatcher(self)
        ]

    def _hook_patches(self) -> None:
        for patch in self._patches:
            hook = getattr(patch, "hook", None)
            if hook:
                hook()

    def _unhook_patches(self) -> None:
        for patch in self._patches:
            unhook = getattr(patch, "unhook", None)
            if unhook:
                unhook()

    def _normalize_third_party_call(
        self,
        current_file: str,
        current_name: str,
        frame: Any,
    ) -> Optional[tuple]:
        try:
            # Handle Triton JIT calls
            if current_file and "triton/runtime/jit.py" in current_file:
                # Try to recover the repo jit function; candidate_fn is a placeholder
                candidate_fn: Optional[Any] = None

                self_obj = frame.f_locals.get("self")
                if self_obj is not None and hasattr(self_obj, "fn"):
                    candidate = getattr(self_obj, "fn")
                    if callable(candidate) and hasattr(candidate, "__code__"):
                        candidate_fn = candidate

                if candidate_fn is None:
                    fn = frame.f_locals.get("fn")
                    if callable(fn) and hasattr(fn, "__code__"):
                        candidate_fn = fn

                if candidate_fn is not None and hasattr(candidate_fn, "__code__"):
                    fn_code = candidate_fn.__code__
                    fn_file = fn_code.co_filename
                    if self._is_repo_file(fn_file):
                        return (
                            fn_file,
                            candidate_fn.__name__,
                            fn_code.co_firstlineno,
                            True,
                        )
        except Exception:
            return None

        return None

    def trace_pytest_execution(
        self,
        test_file: str,                           # Test file
        output_file: str = 'pytest_trace.json',   # Output file path
        pytest_args: Optional[List[str]] = None   # Extra pytest args
    ):
        exit_code = 0
        self._print(f"Start tracing test file: {test_file}")
        original_pythonpath = os.environ.get('PYTHONPATH')

        try:
            # Prepare pytest args - run full test file
            args = [test_file, '-v', '-s']
            
            # Add extra pytest args
            if pytest_args:
                args.extend(pytest_args)
            
            self._print(f"Run pytest, args: {args}")
            
            # Set env vars to enable subprocess tracing
            os.environ['ACE_TRACE_SUBPROCESS'] = '1'
            os.environ['ACE_TRACE_REPO_ROOT'] = str(self.repo_root)
            os.environ['ACE_TRACE_OUTPUT_FILE'] = str(Path(output_file).resolve())
            if self.debug:
                os.environ['ACE_TRACE_DEBUG'] = '1'
            
            # Add dynamic_script to PYTHONPATH so subprocess can find sitecustomize.py
            dynamic_script_dir = Path(__file__).parent.resolve()
            repo_pythonpath = str(self.repo_root)
            current_pythonpath = original_pythonpath or ''

            pythonpath_entries = [str(dynamic_script_dir), repo_pythonpath]
            if current_pythonpath:
                pythonpath_entries.append(current_pythonpath)

            os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath_entries)
            
            self._print("Subprocess tracing env vars set:")
            self._print(f"  PYTHONPATH={os.environ['PYTHONPATH']}")
            self._print(f"  ACE_TRACE_SUBPROCESS={os.environ['ACE_TRACE_SUBPROCESS']}")
            self._print(f"  ACE_TRACE_REPO_ROOT={os.environ['ACE_TRACE_REPO_ROOT']}")
            self._print(f"  ACE_TRACE_OUTPUT_FILE={os.environ['ACE_TRACE_OUTPUT_FILE']}")
            
            # Clear repo module cache before installing import hook
            self._clear_repo_modules()

            # Hook all patches (before tracing)
            self._hook_patches()
            
            # Start tracing - use setprofile instead of settrace for performance
            sys.setprofile(self._trace_calls)
            threading.setprofile(self._trace_calls)
            self._print("Start function call tracing (using setprofile)...")
            
            # Execute pytest
            exit_code = pytest.main(args)
            self._print(f"Pytest completed, exit code: {exit_code}")
            
        except Exception as e:
            self._print(f"Error running pytest: {e}")
            traceback.print_exc()
            if exit_code == 0:
                exit_code = 1

        finally:
            # Stop tracing
            sys.setprofile(None)
            threading.setprofile(None)
            self._print("Stop function call tracing")
            
            # Restore all patches (after tracing)
            self._unhook_patches()
            
            # Clean subprocess tracing env vars
            os.environ.pop('ACE_TRACE_SUBPROCESS', None)
            os.environ.pop('ACE_TRACE_REPO_ROOT', None)
            os.environ.pop('ACE_TRACE_OUTPUT_FILE', None)
            os.environ.pop('ACE_TRACE_DEBUG', None)
            if original_pythonpath is None:
                os.environ.pop('PYTHONPATH', None)
            else:
                os.environ['PYTHONPATH'] = original_pythonpath
            self._print("Subprocess tracing env vars cleared")
            
            # Normalize CallRecord line numbers
            self._print("Normalizing call record line numbers...")
            self._normalize_call_records()
            
            # Build dependency graph
            self._build_dependency_graph()

            # Process and save results
            self._print("Processing trace results...")
            results = {
                'stats': {
                    'total_calls': len(self.call_records),
                    'total_imports': len(self.import_records),
                    'total_objects': len(self.objects),
                    'total_edges': sum(len(obj.edges) for obj in self.objects.values()),
                    'repo_root': str(self.repo_root)
                }
            }
            
            # If saving to file, add serialized objects and imports
            if output_file:
                results['objects'] = self._serialize_objects()
                results['imports'] = self._serialize_imports()
            
            self._print(f"Saving results to file: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self._print("Tracing complete!")
            self._print(f"Result file: {output_file}")
            self._print(f"Total call count: {len(self.call_records)}")
            self._print(f"Total import count: {len(self.import_records)}")
            self._print(f"Unique object count: {len(self.objects)}")

        return exit_code
            
    def _clear_repo_modules(self) -> None:
        """Clear repo modules in sys.modules to avoid cache bypassing hooks."""
        try:
            repo_root = str(self.repo_root)
            to_delete = []
            for name, module in list(sys.modules.items()):
                module_file = getattr(module, '__file__', None)
                if not module_file:
                    continue
                try:
                    module_path = Path(module_file).resolve()
                except Exception:
                    continue
                if str(module_path).startswith(repo_root):
                    to_delete.append(name)

            for name in to_delete:
                sys.modules.pop(name, None)

            if self.debug:
                self._debug(f"Removed {len(to_delete)} cached repo modules, ready to re-import")

        except Exception as exc:
            if self.debug:
                self._debug(f"Failed to clear repo module cache: {exc}")

    def _trace_calls(
        self,
        frame,  # Current frame
        event,  # Event type ('call', 'return')
        arg     # Usually unused
    ):
        """
        Trace function calls.
        Core strategy:
        1. call event: record repo functions only, but keep full call stack
        2. return event: maintain call stack correctness
        3. use dedup set to avoid duplicate call relations
        """
        try:
            code = frame.f_code
            current_file = code.co_filename
            current_name = code.co_name
            current_line = frame.f_lineno
            
            if event == 'call':
                # Check if current call is within repo
                current_is_repo = self._is_repo_file(current_file)

                # Apply patch to current call
                normalized = self._normalize_third_party_call(
                    current_file,
                    current_name,
                    frame,
                )
                if normalized:
                    norm_file, norm_name, norm_line, force_repo_flag = normalized
                else:
                    norm_file, norm_name, norm_line = current_file, current_name, current_line
                    force_repo_flag = None
                # Determine if the call should be treated as in-repo
                effective_is_repo = current_is_repo if force_repo_flag is None else force_repo_flag
                
                # If calling forward from repo, re-hook to capture delayed autograd.Function subclasses
                if current_name == 'forward' and effective_is_repo:
                    self._hook_patches()
                
                # Get caller info (from top of call stack)
                caller_file = None
                caller_name = None
                caller_line = None
                caller_is_repo = False
                call_stack = self._get_call_stack()
                if call_stack:
                    caller_file, caller_name, caller_line = call_stack[-1]
                    caller_is_repo = self._is_repo_file(caller_file)
                
                # Push current call to the stack (regardless of repo)
                call_stack.append((current_file, current_name, current_line))
                
                # Strategy: record calls where at least one end is in repo
                # This preserves "repo -> non-repo -> repo" chains via snapshots
                if caller_file is not None and (effective_is_repo or caller_is_repo):
                    # Optimization: only snapshot when caller is non-repo and callee is repo
                    call_stack_snapshot = None
                    if not caller_is_repo and effective_is_repo:
                        call_stack_snapshot = tuple(call_stack)
                    
                    # Create call record
                    record = CallRecord(
                        caller_file=caller_file,
                        caller_name=caller_name,
                        caller_line=caller_line,
                        callee_file=norm_file,
                        callee_name=norm_name,
                        callee_line=norm_line,
                        call_stack_snapshot=call_stack_snapshot
                    )
                    
                    # Deduplicate using string key
                    record_key = f"{caller_file}:{caller_name}:{caller_line}->{norm_file}:{norm_name}:{norm_line}"
                    
                    if record_key not in self.call_records_set:
                        with self.trace_lock:
                            self.call_records_set.add(record_key)
                            self.call_records.append(record)
                        
                        if self.debug:
                            caller_repo = "✓" if self._is_repo_file(caller_file) else "✗"
                            self._debug(
                                f"Record[{caller_repo}->✓]: {caller_name}@{os.path.basename(caller_file)} "
                                f"-> {current_name}@{os.path.basename(current_file)}"
                            )
                        
                        # Print progress every 1000 unique calls
                        if len(self.call_records) % 1000 == 0:
                            self._print(f"Recorded {len(self.call_records)} unique call relations...")
            
            elif event == 'return':
                # Pop current function from call stack
                call_stack = self._get_call_stack()
                if call_stack:
                    top_file, top_name, top_line = call_stack[-1]
                    # Ensure stack top matches current function
                    if top_file == current_file and top_name == current_name:
                        call_stack.pop()
                        
        except Exception as e:
            if self.debug:
                self._debug(f"Tracing error: {e}")
                traceback.print_exc()
            
        return self._trace_calls
    
    def _build_function_definition_mapping(self):
        """
        Build function definition mapping: (file, function_name) -> definition_line

        Returns:
            dict: mapping dictionary
        """
        function_definition_mapping = {}
        
        # Collect all files to analyze
        files_to_analyze = set()
        for record in self.call_records:
            files_to_analyze.add(record.caller_file)
            files_to_analyze.add(record.callee_file)
        
        # Analyze each file with AST to get definition line numbers
        for file_path in files_to_analyze:
            try:
                if not self._is_repo_file(file_path):
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                tree = ast.parse(source_code, filename=file_path)
                
                # Walk AST to collect function/class definition line numbers
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        key = (file_path, node.name)
                        function_definition_mapping[key] = node.lineno
            except Exception as e:
                # Skip if AST parsing fails
                continue
        
        return function_definition_mapping
        
    def _normalize_call_records(self):
        """
        Normalize CallRecord line numbers:
        - caller_line: from call site to caller definition line
        - callee_line: from execution line to callee definition line

        Both lines represent the function definition location.
        """
        # Build function definition mapping
        function_definition_mapping = self._build_function_definition_mapping()
        
        # Second pass: update line numbers to definition lines
        updated_count = 0
        for record in self.call_records:
            caller_key = (record.caller_file, record.caller_name)
            callee_key = (record.callee_file, record.callee_name)
            
            # Update caller_line to caller definition line
            if caller_key in function_definition_mapping:
                record.caller_line = function_definition_mapping[caller_key]
                updated_count += 1
            
            # Update callee_line to callee definition line
            if callee_key in function_definition_mapping:
                record.callee_line = function_definition_mapping[callee_key]
                updated_count += 1
        
    
    def _is_repo_file(self, filepath: str) -> bool:
        try:
            repo_root = str(self.repo_root)

            # Filter out dynamically generated or system code (starting with <)
            if not filepath or filepath.startswith('<'):
                return False

            # Handle files not ending in .py or .pyx
            if not (filepath.endswith('.py') or filepath.endswith('.pyx')):
                # Filter obvious non-source files
                if any(filepath.endswith(ext) for ext in ['.html', '.js', '.css', '.json', '.xml', '.txt', '.md', '.jinja', '.rst', '.rst_t']):
                    return False
                # Keep others (e.g. <string>, module name)

            # Check if path is under repo root
            if filepath.startswith('/'):
                if filepath.startswith(repo_root):
                    return True
            else:
                try:
                    resolved = Path(filepath).resolve()
                    if str(resolved).startswith(repo_root):
                        return True
                except Exception:
                    pass
                
            return False
        
        except Exception:
            return False

    def _build_dependency_graph(self):
        """
        Build object dependency graph from call records.

        Strategy:
        - caller and callee in repo: add edge directly
        - caller not in repo, callee in repo: use call_stack_snapshot to find nearest repo caller
        """
        self._print("Building object dependency graph...")
        
        # Clear previous object info
        self.objects.clear()
        
        self._print(f"Processing {len(self.call_records)} call records...")
        
        # Build function definition mapping (for normalizing call_stack_snapshot lines)
        function_definition_mapping = self._build_function_definition_mapping()
        
        processed_edges = set()  # Avoid duplicate edges
        
        for record in self.call_records:
            caller_is_repo = self._is_repo_file(record.caller_file)
            callee_is_repo = self._is_repo_file(record.callee_file)
            
            if caller_is_repo and callee_is_repo:
                # Case 1: both in repo -> add edge directly
                self._add_edge_to_graph(record, processed_edges)
            elif not caller_is_repo and callee_is_repo:
                # Case 2: caller not in repo, callee in repo
                # Use call_stack_snapshot to find a repo caller above
                repo_callers_found = []
                
                if record.call_stack_snapshot:
                    # Traverse call stack upward to find nearest repo caller
                    # Note: the last element is the callee itself; skip it
                    for stack_entry in reversed(record.call_stack_snapshot[:-1]):
                        stack_file, stack_name, stack_line = stack_entry
                        if self._is_repo_file(stack_file):
                            # Normalize stack_line to definition line
                            caller_key = (stack_file, stack_name)
                            normalized_line = function_definition_mapping.get(caller_key, stack_line)
                            
                            # Found repo caller; create transitive edge
                            repo_caller_record = CallRecord(
                                caller_file=stack_file,
                                caller_name=stack_name,
                                caller_line=normalized_line,  # use normalized line number
                                callee_file=record.callee_file,
                                callee_name=record.callee_name,
                                callee_line=record.callee_line
                            )
                            repo_callers_found.append(repo_caller_record)
                            break  # Only take the nearest one
                
                # Add edges for found repo callers
                for repo_caller_record in repo_callers_found:
                    self._add_edge_to_graph(repo_caller_record, processed_edges)
            
            # Other cases (repo -> non-repo, non-repo -> non-repo) -> skip
        
        self._print(f"Build complete: {len(self.objects)} objects, {len(processed_edges)} edges")
    
    def _add_edge_to_graph(self, record: CallRecord, processed_edges: set):
        """
        Add one call record to the dependency graph.

        Args:
            record: Call record (caller and callee should be in repo)
            processed_edges: Set of processed edges
        """
        # Create object identifiers
        caller_id = f"{record.caller_file}::{record.caller_name}::{record.caller_line}"
        callee_id = f"{record.callee_file}::{record.callee_name}::{record.callee_line}"
        
        # Avoid duplicate processing
        edge_key = (caller_id, callee_id)
        if edge_key in processed_edges:
            return
        processed_edges.add(edge_key)
        
        # Create/update caller object
        if caller_id not in self.objects:
            self.objects[caller_id] = ObjInfo(
                id=caller_id,
                name=record.caller_name,
                file=record.caller_file,
                line=record.caller_line,
                edges=set(),
                call_count=0,
                is_pass2pass=False,
            )
        
        # Create/update callee object
        if callee_id not in self.objects:
            self.objects[callee_id] = ObjInfo(
                id=callee_id,
                name=record.callee_name,
                file=record.callee_file,
                line=record.callee_line,
                edges=set(),
                call_count=0,
                is_pass2pass=False,
            )
        
        # Add dependency edge: caller -> callee
        self.objects[caller_id].edges.add(callee_id)
        
        # Update callee call count
        self.objects[callee_id].call_count += 1

    def _serialize_objects(self) -> Dict[str, Dict[str, Any]]:
        """Serialize ObjInfo objects into JSON-friendly format."""
        serializable_objects = {}
        for obj_id, obj_info in self.objects.items():
            serializable_objects[obj_id] = {
                'id': obj_info.id,
                'name': obj_info.name,
                'file': obj_info.file,
                'line': obj_info.line,
                'edges': list(obj_info.edges),  # convert set to list for JSON
                'call_count': obj_info.call_count,
                'is_pass2pass': obj_info.is_pass2pass,
            }
        return serializable_objects
    
    def _serialize_imports(self) -> List[Dict[str, Any]]:
        """Serialize ImportRecord into JSON-friendly format."""
        serializable_imports = []
        for import_record in self.import_records:
            serializable_imports.append({
                'defined_file': import_record.defined_file,
                'defined_line': import_record.defined_line,
                'defined_name': import_record.defined_name,
                'importer_file': import_record.importer_file,
                'importer_line': import_record.importer_line,
                'imported_name': import_record.imported_name,
            })
        return serializable_imports

def main():
    """Main entry."""
    parser = argparse.ArgumentParser(description='Trace function/class dependency calls during pytest execution')
    parser.add_argument('test_file', help='Path to the test file to execute')
    parser.add_argument('repo_root', help='Path to the repository root')
    parser.add_argument('-o', '--output', default='pytest_trace.json', help='Output JSON file (default: pytest_trace.json)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--pytest-args', nargs=argparse.REMAINDER, help='Extra pytest arguments')
    
    args = parser.parse_args()
    
    # Check test file exists
    if not os.path.exists(args.test_file):
        print(f"❌ Error: test file {args.test_file} does not exist")
        return 1
    
    print("Starting dynamic dependency analysis")
    print(f"Test file: {args.test_file}")
    print(f"Output file: {args.output}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    # Create tracer and run
    tracer = DependencyTracer(args.repo_root, debug=args.debug, quiet=True)
    exit_code = tracer.trace_pytest_execution(
        test_file=args.test_file,
        output_file=args.output,
        pytest_args=args.pytest_args
    )
    
    return exit_code

if __name__ == "__main__":
    exit(main())