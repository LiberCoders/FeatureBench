"""
Import Hook Patch - patch to trace all import operations
"""
import sys
import os
import builtins
import inspect
import traceback
from pathlib import Path
from typing import Optional


_MISSING = object()


class ImportHookPatcher:
    """Import Hook patch - used to trace `from xxx import yyy` operations."""
    
    def __init__(self, tracer, ImportRecord):
        """
        Initialize Import Hook patch.

        Args:
            tracer: DependencyTracer instance
            ImportRecord: ImportRecord dataclass
        """
        self.tracer = tracer
        self.ImportRecord = ImportRecord
        self._original_import = builtins.__import__
        self._hooked = False
    
    def hook(self):
        """Install Import Hook."""
        if not self._hooked:
            builtins.__import__ = self._custom_import
            self._hooked = True
            if self.tracer.debug:
                print("✓ Import Hook installed")
    
    def unhook(self):
        """Uninstall Import Hook."""
        if self._hooked:
            builtins.__import__ = self._original_import
            self._hooked = False
            if self.tracer.debug:
                print("✓ Import Hook uninstalled")
    
    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Custom __import__ hook - trace all import operations.

        Args:
            name: module name
            globals: global namespace
            locals: local namespace
            fromlist: from xxx import list
            level: relative import level (0=absolute, >0=relative)
        """
        # Call original import
        module = self._original_import(name, globals, locals, fromlist, level)
        
        try:
            # Get the location of the import caller
            importer_file = None
            importer_line = None
            frame = sys._getframe(1)  # Get caller frame
            if frame:
                importer_file = frame.f_code.co_filename
                importer_line = frame.f_lineno

            # Only record imports from within the repo (importer in repo)
            if not self.tracer._is_repo_file(importer_file):
                return module

            importer_line = self._normalize_importer_line(importer_file, importer_line)
            
            if fromlist:  # from xxx import yyy
                for item in fromlist:
                    if item == '*':
                        # from xxx import * case, skip
                        continue
                        
                    obj = None

                    # Prefer submodule: from pkg import submodule
                    parent_name = getattr(module, '__name__', None)
                    if parent_name:
                        submodule_name = f"{parent_name}.{item}"
                        obj = sys.modules.get(submodule_name)

                    if obj is None:
                        try:
                            # Use getattr_static to avoid side effects from module __getattr__
                            obj = inspect.getattr_static(module, item, _MISSING)
                        except Exception:
                            continue

                        if obj is _MISSING:
                            continue

                    # Get object definition location and real name
                    defined_file = None
                    defined_line = None
                    defined_name = None
                    try:
                        if inspect.isclass(obj) or inspect.isfunction(obj):
                            # If the object is wrapped by a decorator, try to unwrap it
                            # e.g. @torch.inference_mode() wraps functions
                            actual_obj = obj
                            if hasattr(obj, '__wrapped__'):
                                actual_obj = obj.__wrapped__

                            defined_name = getattr(actual_obj, '__qualname__', None)
                            if defined_name:
                                defined_name = defined_name.replace('<locals>.', '')

                            source_lines = None
                            first_line = None

                            try:
                                defined_file = inspect.getfile(actual_obj)
                                source_lines, first_line = inspect.getsourcelines(actual_obj)
                            except (OSError, TypeError):
                                if inspect.isclass(actual_obj):
                                    fallback = self._fallback_class_source_info(actual_obj)
                                    if fallback:
                                        defined_file, defined_line = fallback
                                # Keep functions consistent with previous behavior

                            if defined_file and defined_line is None:
                                if source_lines is not None and first_line is not None:
                                    defined_line = self._extract_definition_line(source_lines, first_line)
                                elif first_line is not None:
                                    defined_line = first_line

                        elif inspect.ismodule(obj):
                            # Be conservative for namespace packages: skip if no __file__ or origin is namespace
                            module_file = getattr(obj, '__file__', None)
                            spec = getattr(obj, '__spec__', None)
                            if module_file is None:
                                continue
                            if spec is not None and getattr(spec, 'origin', None) == 'namespace':
                                continue

                            defined_file = module_file
                            defined_line = -1  # Module-level, mark with -1
                            defined_name = getattr(obj, '__name__', None)
                    except Exception:
                        pass

                    # Only record imports for repo objects (filter stdlib/third-party)
                    if defined_file and not self.tracer._is_repo_file(defined_file):
                        continue
 
                    # Create import record
                    record = self.ImportRecord(
                        defined_file=defined_file,
                        defined_line=defined_line,
                        defined_name=defined_name,
                        importer_file=importer_file,
                        importer_line=importer_line,
                        imported_name=item  # name used in import
                    )

                    # Deduplicate
                    record_key = f"{importer_file}:{importer_line}:{name}:{item}"
                    if record_key not in self.tracer.import_records_set:
                        with self.tracer.trace_lock:
                            self.tracer.import_records_set.add(record_key)
                            self.tracer.import_records.append(record)

                        if self.tracer.debug:
                            print(f"Record import: from {name} import {item} @ {os.path.basename(importer_file or '?')}:{importer_line}")
                            if defined_file and defined_name:
                                print(f"  → Real definition: {defined_name} @ {defined_file}:{defined_line}")
        
        except Exception as e:
            if self.tracer.debug:
                print(f"Import Hook error: {e}")
                traceback.print_exc()
        
        return module

    def _extract_definition_line(self, source_lines, first_line: int) -> int:
        """Extract the line number of a def/class from source lines."""
        for offset, line in enumerate(source_lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'async def ')):
                return first_line + offset
        return first_line

    def _fallback_class_source_info(self, cls):
        """Fallback: locate class definition via methods when inspect fails."""
        class_name = cls.__qualname__.split('.')[-1]

        for attr in cls.__dict__.values():
            candidate = attr
            if isinstance(candidate, (staticmethod, classmethod)):
                candidate = candidate.__func__
            candidate = getattr(candidate, '__wrapped__', candidate)

            if not inspect.isfunction(candidate):
                continue

            try:
                file_path = inspect.getfile(candidate)
                _, first_line = inspect.getsourcelines(candidate)
            except (OSError, TypeError):
                continue

            class_line = self._find_class_def_lineno(file_path, class_name, first_line)
            if class_line is not None:
                return file_path, class_line

        module = inspect.getmodule(cls)
        if module:
            file_path = getattr(module, '__file__', None)
            if file_path:
                class_line = self._find_class_def_lineno(file_path, class_name)
                if class_line is not None:
                    return file_path, class_line
                return file_path, -1

        return None

    def _find_class_def_lineno(self, file_path: str, class_name: str, upper_bound: int | None = None):
        """Find the line number of a class definition in a file."""
        try:
            lines = Path(file_path).read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return None

        search_limit = upper_bound if upper_bound is not None else len(lines)
        search_limit = min(search_limit, len(lines))

        for idx in range(search_limit - 1, -1, -1):
            stripped = lines[idx].lstrip()
            if stripped.startswith('class ') and stripped[6:].startswith(class_name):
                return idx + 1

        if upper_bound is not None and upper_bound < len(lines):
            for idx in range(upper_bound, len(lines)):
                stripped = lines[idx].lstrip()
                if stripped.startswith('class ') and stripped[6:].startswith(class_name):
                    return idx + 1

        return None

    def _normalize_importer_line(self, importer_file: str, importer_line: int) -> int:
        """Backtrack multi-line imports to ensure the line points to the statement start."""
        if not importer_file or importer_line is None:
            return importer_line

        try:
            # Convert source to a list
            lines = Path(importer_file).read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            print('Unable to read importer file to normalize line number:', importer_file)
            return importer_line

        # Convert to list index
        index = importer_line - 1
        if index < 0 or index >= len(lines):
            return importer_line

        # If it's a single-line import, return directly
        current = lines[index].lstrip()
        if current.startswith(("from ", "import ")):
            return importer_line

        # Traverse upward to find the line starting with from/import
        for candidate in range(index - 1, -1, -1):
            stripped = lines[candidate].lstrip()
            if stripped.startswith(("from ", "import ")):
                return candidate + 1
            # If line is non-empty and doesn't end with \, , , ( , [, stop (preceding is a normal statement)
            if stripped and not stripped.endswith(("\\", ",", "(", "[")):
                break

        return importer_line

