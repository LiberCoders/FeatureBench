"""
Mask generator - build masked code from classification results.
"""
import ast
import json
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, Set, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import re

from featurebench.mask.signature_extractor import (
    ModuleSignature,
    extract_signature
)
from featurebench.mask.mask_validator import MaskValidator, ValidationResult
from featurebench.utils.logger import create_mask_generator_logger
from featurebench.utils.command_utils import apply_uv_run_prefix
from featurebench.utils.parser.pytest_parser import PytestParser

logger = logging.getLogger(__name__)


@dataclass
class MaskResult:
    """Result of a mask operation."""
    file_path: str                              # File path (container path)
    masked_code: str                            # Masked code (top + specific objects removed)
    top_objects: List[str]                      # Removed top object full IDs
    specific_objects: List[str]                 # Removed specific object full IDs
    signature_info: ModuleSignature             # File signature info
    validation_result: Optional[ValidationResult] = None  # Validation result
    success: bool = True
    error_message: Optional[str] = None


class MaskGenerator:
    """Mask generator."""
    
    def __init__(
        self,
        config,
        repo_manager,
        image_manager,
        storage_manager,
        data_item,
        classification_summary: dict,
        logger
    ):
        self.config = config
        self.repo_manager = repo_manager
        self.image_manager = image_manager
        self.storage_manager = storage_manager
        self.data_item = data_item
        self.classification_summary = classification_summary
        self.logger = logger

        # Load import records (merge f2p + p2p traces)
        self.import_records = self._load_import_records()
        
        # Scan and cache static imports from .pyx/.pxd/.pyi files
        self.extension_imports = self._scan_extension_imports()
        # Cache module exports to avoid re-parsing.
        self.module_exports_cache: Dict[str, Set[str]] = {}
        
    def _iter_trace_files(self) -> List[str]:
        """Collect dynamic trace files for this item (f2p + p2p)."""
        trace_files: List[str] = []
        seen: Set[str] = set()

        primary_trace = getattr(self.data_item, 'dynamic_trace_file', None)
        if primary_trace:
            trace_files.append(primary_trace)
            seen.add(primary_trace)

        extra_traces = getattr(self.data_item, 'dynamic_trace_files', None) or []
        for extra_trace in extra_traces:
            if not extra_trace or extra_trace in seen:
                continue
            trace_files.append(extra_trace)
            seen.add(extra_trace)

        return trace_files

    def _load_import_records(self) -> List[Dict[str, Any]]:
        """Load import records from all relevant dynamic traces."""
        merged_records: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, int, str]] = set()

        for trace_file in self._iter_trace_files():
            if not trace_file:
                continue

            if not Path(trace_file).exists():
                self.logger.debug("Dynamic trace file missing: %s", trace_file)
                continue

            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)
            except Exception as exc:
                self.logger.warning("Failed to load import records: %s (%s)", exc, trace_file)
                continue

            for record in trace_data.get('imports', []) or []:
                importer_file = record.get('importer_file')
                importer_line = record.get('importer_line')
                imported_name = record.get('imported_name')

                key = (importer_file, importer_line, imported_name)
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                merged_records.append(record)

        return merged_records
    
    def _scan_extension_imports(self) -> Dict[str, Set[str]]:
        """
        Scan .pyx/.pxd/.pyi files and extract static import dependencies.
        
        Returns:
            Dict[module::symbol, Set[ref_file_path]]: reverse index of extension-file references
            Example: {"sklearn.utils.extmath::row_norms": {"/testbed/sklearn/cluster/_k_means_common.pyx"}}
        """
        
        extension_imports = {}
        repo_root = self.data_item.repo_root
        
        # Only scan .pyx/.pxd/.pyi files
        extension_patterns = [
            '*.pyx', '*.pxd', '*.pyi',
            '*.pyx.in', '*.pxd.in', '*.pyi.in',
            '*.pyx.tp', '*.pxd.tp', '*.pyi.tp'
        ]
        extension_files = set()
        for pattern in extension_patterns:
            extension_files.update(repo_root.rglob(pattern))
        
        if not extension_files:
            return extension_imports
        
        # Regex for import statements (multi-line and parentheses supported)
        # Match: from a.b.c import x, y, z
        # Match: from a.b.c cimport x, y
        # Match: from a.b.c import (x, y, z)
        # Match: from a.b.c import (\n    x,\n    y\n)
        # DOTALL lets '.' match newlines; non-capturing groups handle bracketed imports
        from_import_pattern = re.compile(
            r'^\s*from\s+([\w.]+)\s+(?:c?import)\s+(?:\(([^)]+)\)|(.+?)$)',
            re.MULTILINE | re.DOTALL
        )
        
        for file_path in sorted(extension_files):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Preprocess: merge backslash line continuations
                # Replace "...\\\n..." with "... ..."
                content = re.sub(r'\\\s*\n\s*', ' ', content)
                
                # Compute module path for current file (for relative imports)
                # Example: /path/to/sklearn/cluster/_k_means.pyx -> sklearn.cluster
                rel_path = file_path.relative_to(repo_root)
                module_rel_path = rel_path
                # Strip template/generator suffixes to get real module path
                while module_rel_path.suffix in {'.tp', '.in'}:
                    module_rel_path = module_rel_path.with_suffix('')
                # Strip extension suffix
                while module_rel_path.suffix in {'.pyx', '.pxd', '.pyi'}:
                    module_rel_path = module_rel_path.with_suffix('')

                # Directory part used for package name resolution
                rel_dir = module_rel_path.parent
                # Filter empty parts or '.' so top-level file yields empty package
                package_parts = [part for part in rel_dir.parts if part and part != '.']
                current_package = '.'.join(package_parts)
                package_depth = len(package_parts)
                
                # Extract all from xxx import yyy statements
                for match in from_import_pattern.finditer(content):
                    module_name = match.group(1)  # Module name, e.g. sklearn.utils._cython_blas or .utils
                    # group(2) is bracketed imports, group(3) is non-bracketed
                    imports_str = match.group(2) or match.group(3)
                    
                    if not imports_str:
                        continue
                    
                    # Handle relative imports: convert .module or ..module to absolute path
                    if module_name.startswith('.'):
                        # Compute relative level
                        level = len(module_name) - len(module_name.lstrip('.'))
                        relative_module = module_name.lstrip('.')
                        
                        # Compute how many levels to go up
                        levels_up = level - 1
                        
                        # Check for package boundary overflow
                        # If levels_up >= package depth, it escapes the top-level package
                        if levels_up >= package_depth:
                            # Escapes top-level package; skip
                            pkg_display = current_package or '<root>'
                            tqdm.write(
                                f"⚠️ Relative import escapes package boundary: {file_path} - {module_name} "
                                f"(current package: {pkg_display}, needs {levels_up} levels up, only {package_depth} levels)"
                            )
                            continue
                        
                        # Compute base package after moving up
                        if levels_up == 0:
                            # from .xxx - current package
                            base_package = current_package
                        else:
                            # from ..xxx or ...xxx - walk up
                            base_package = '.'.join(package_parts[:-levels_up])
                        
                        # Build absolute module path
                        if relative_module:
                            module_name = f"{base_package}.{relative_module}" if base_package else relative_module
                        else:
                            module_name = base_package
                    
                    # Remove inline comments (content after # on each line)
                    lines = imports_str.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        if '#' in line:
                            line = line.split('#')[0]
                        cleaned_lines.append(line)
                    imports_str = '\n'.join(cleaned_lines)
                    
                    # Parse imported symbols (handle aliases and multiple imports)
                    # e.g.: x, y as yy, z -> [x, y, z]
                    # e.g.: (\n    x,\n    y\n) -> [x, y]
                    symbols = []
                    for item in imports_str.split(','):
                        item = item.strip()
                        if not item:
                            continue
                        
                        # Remove alias (e.g., "x as xx" -> "x")
                        if ' as ' in item:
                            symbol = item.split(' as ')[0].strip()
                        else:
                            symbol = item
                        
                        if not symbol or symbol == '*':
                            continue
                        
                        symbols.append(symbol)
                    
                    # Record import relations per symbol
                    for symbol in symbols:
                        # Build module::symbol
                        obj_qualified_name = f"{module_name}::{symbol}"
                        
                        if obj_qualified_name not in extension_imports:
                            extension_imports[obj_qualified_name] = set()
                        
                        # Convert to container path (repo_root maps to /testbed)
                        container_path = str(file_path).replace(str(repo_root), '/testbed')
                        extension_imports[obj_qualified_name].add(container_path)
                        
            except Exception as e:
                tqdm.write(f"❌ Failed to scan extension file {file_path}: {e}")
                continue
        
        if extension_imports:
            pass
            # tqdm.write(f"Found {len(extension_imports)} symbols referenced by extension modules")
        
        return extension_imports
        
    def run(self):
        """
        Run the mask generation workflow.

        Returns:
            Tuple[Dict[file_path, MaskResult], bool, int]: (mask results, passed, total deleted LOC)
        """
        try:
            test_file_name = Path(self.data_item.file_path).name
            # tqdm.write(f"🏃 Start mask generation: {self.data_item.repo} - {test_file_name}")
            
            # Generate masks
            mask_results = self._generate_masks()
            
            # Save masked files
            self.storage_manager.save_masked_files(
                repo=self.data_item.repo,
                test_file=self.data_item.file_path,
                mask_results=mask_results,
                logger=self.logger
            )

            # Save mask diff and get total deleted LOC
            deleted_lines = self.storage_manager.save_mask_diffs(
                repo=self.data_item.repo,
                test_file=self.data_item.file_path,
                mask_results=mask_results,
                repo_manager=self.repo_manager,
                logger=self.logger
            )
            
            # Summarize results
            success_count = sum(1 for r in mask_results.values() if r.success)
            perfect_count = sum(
                1 for r in mask_results.values() 
                if r.success and r.validation_result and r.validation_result.is_perfect
            )
            total_count = len(mask_results)
            
            if success_count == total_count and perfect_count == total_count:
                is_passed = True
                pass
                # tqdm.write(f"✅ {self.data_item.repo}: Mask generation complete - {test_file_name} ({perfect_count}/{total_count} perfect matches)")
            else:
                is_passed = False
                tqdm.write(
                    f"⚠️ {self.data_item.repo}: Mask generation partially complete - {test_file_name} "
                    f"(success={success_count}/{total_count}, perfect={perfect_count}/{total_count})"
                )
            
            return mask_results, is_passed, deleted_lines
            
        except Exception as e:
            error_msg = f"Mask generation failed - {test_file_name}: {e}"
            # tqdm.write(f"❌ {self.data_item.repo}: {error_msg}")
            raise RuntimeError(error_msg) from e
        
    def _generate_masks(self) -> Dict[str, MaskResult]:
        """
        Generate masks from classification results (iterative deletion).

        Strategy:
        1. while loop: continue until no more objects can be deleted
        2. for loop: prefer deleting leaf objects each round
        3. After each deletion, run pytest --collect in Docker to validate
        4. Commit on success, rollback on failure
        5. Clean up related import statements

        Returns:
            Dict[file_path, MaskResult]: mask result dict
        """
        results = {}
        
        # tqdm.write('\n' * 3)
        # tqdm.write(f'New sample: {self.data_item.repo} - {Path(self.data_item.file_path).name}')

        # Get all patch files
        patch_files = self.classification_summary.get('patch_files', [])
        
        if not patch_files:
            tqdm.write(f"No files found to mask: {self.data_item.repo} - {Path(self.data_item.file_path).name}")
            return results
        
        # Collect all objects to remove
        all_top_objects = self.classification_summary.get('top_objects', [])
        all_specific_objects = self.classification_summary.get('specific_objects', [])
        all_objects_to_remove = list(set(all_top_objects + all_specific_objects))

        # Filter public API objects (module __all__)
        all_objects_to_remove = self._filter_public_api_objects(all_objects_to_remove)

        # Filter objects referenced by extension modules (.pyx/.pxd/.pyi)
        # .pyx Cython source, .pxd Cython header, .pyi Python type stubs
        all_objects_to_remove = self._filter_extension_protected_objects(all_objects_to_remove)

        # Protect critical magic methods (e.g., __len__) to avoid breaking ABC minimum implementations
        all_objects_to_remove = self._filter_critical_special_methods(all_objects_to_remove)

        # Build import map
        # file::line -> [(importer_file, importer_line, imported_name, defined_name), ...]
        import_map = self._build_import_map()

        # Build dependency graph
        # file::name::line -> {file1::name1::line1, file2::name2::line2}, dependents of name are name1, name2
        dependency_graph = self._build_dependency_graph(all_objects_to_remove)
        
        # print("Objects to remove: ")
        # for obj in all_objects_to_remove:
        #     print(f" - {obj}")
        # tqdm.write(f'Objects to remove: {len(all_objects_to_remove)}')

        # Create temp directory for modified files
        temp_dir = tempfile.mkdtemp(prefix="fb_mask_")
        
        try:
            # Collect files to modify: patch_files + all files importing removed objects
            files_to_track = set(patch_files)
            
            # Extract all importer files from import_map
            for obj_id in all_objects_to_remove:
                parts = obj_id.split('::')
                file_path_container = parts[0]
                line_number = int(parts[2])
                obj_key = f"{file_path_container}::{line_number}"
                
                import_locations = import_map.get(obj_key, [])
                for importer_file, _, _, _ in import_locations:
                    files_to_track.add(importer_file)
            
            # Copy all tracked files to temp directory
            temp_files_map = {}  # Container path -> temp file path
            original_signatures = {}  # Preserve original signatures
            
            # Iterate files to modify and copy to temp files
            for file_path_container in files_to_track:
                host_path = self.repo_manager.convert_container_path_to_local(
                    self.data_item.repo, file_path_container
                )
                # Create temp file (unique name to avoid conflicts)
                safe_name = file_path_container.replace('/', '_').replace('.', '_') + '.py'
                temp_file_path = Path(temp_dir) / safe_name
                shutil.copy2(host_path, temp_file_path)
                temp_files_map[file_path_container] = str(temp_file_path)
                
                # Extract original signatures after copy
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    original_code = f.read()
                original_signatures[file_path_container] = extract_signature(original_code, file_path_container)
            
            # Iterative deletion
            successfully_deleted = []  # Successfully deleted object IDs
            remaining_objects = all_objects_to_remove.copy()  # Remaining objects to delete
            
            iteration = 0
            
            # Reuse container - start only once
            container_id = None
            try:
                container_id = self.image_manager.run_container(
                    specs_name=self.data_item.repo,
                    working_dir="/testbed",
                    prepare_env=True  # Prepare env on first start
                )
            
                while remaining_objects:
                    iteration += 1
                    # tqdm.write(f"Iteration {iteration}: {len(remaining_objects)} objects remaining")
                    
                    # Get leaf objects to try this round
                    leaf_objects = self._get_leaf_objects(remaining_objects, dependency_graph)
                    # tqdm.write(f"Found {len(leaf_objects)} leaf objects, attempting batch delete")

                    deleted_this_round = []
                    
                    # Try batch deletion first
                    # Reset container env
                    reset_success = self.image_manager.reset_container_env(
                        container_id=container_id,
                        specs_name=self.data_item.repo,
                        timeout=None
                    )
                    
                    if not reset_success:
                        # tqdm.write("❌ Failed to reset container env, skipping this round")
                        break
                    
                    # Backup all temp files
                    backup = {}
                    for temp_fpath in temp_files_map.values():
                        with open(temp_fpath, 'r', encoding='utf-8') as f:
                            backup[temp_fpath] = f.read()
                    
                    try:
                        # Batch delete all leaf objects
                        self._batch_delete_objects_and_imports(
                            leaf_objects,
                            temp_files_map,
                            import_map
                        )
                        
                        # Copy files to container
                        self.image_manager.copy_to_container(
                            container_id=container_id,
                            src_path=None,
                            dest_path=None,
                            use_tar=True,
                            files_mapping=temp_files_map
                        )
                        
                        # Run pytest --collect validation
                        collect_success, log_file_path = self._run_pytest_collect(
                            container_id,
                            self.data_item.file_path
                        )

                        if collect_success:
                            # Batch delete succeeded
                            deleted_this_round = leaf_objects
                            successfully_deleted.extend(leaf_objects)
                            # tqdm.write(f"✅ Batch delete succeeded: {len(leaf_objects)} objects")
                        else:
                            # Batch delete failed
                            # 1) Extract error symbols from log
                            error_symbols = self._extract_error_symbols_from_log(log_file_path)
                            
                            # 2) Find object IDs by symbol and remove from remaining list
                            protected_objects = []
                            if error_symbols:
                                for symbol in error_symbols:
                                    matched_objs = self._find_objects_by_symbol_name(leaf_objects, symbol)
                                    protected_objects.extend(matched_objs)
                                
                                # Remove protected objects from remaining list
                                if protected_objects:
                                    # tqdm.write(f"🛡️ Identified {len(error_symbols)} error symbols, protecting {len(protected_objects)} objects")
                                    for obj_id in protected_objects:
                                        if obj_id in remaining_objects:
                                            remaining_objects.remove(obj_id)
                                    
                                    # Remove protected objects from this round's leaves
                                    leaf_objects = [obj for obj in leaf_objects if obj not in protected_objects]
                            
                            # 3. If objects remain, use binary search deletion
                            if leaf_objects:
                                # tqdm.write(f"❌ Batch deletion failed; start binary search deletion (remaining {len(leaf_objects)} objects)...")
                                deleted_this_round = self._binary_search_failed_objects(
                                    failed_batch=leaf_objects,
                                    container_id=container_id,
                                    temp_files_map=temp_files_map,
                                    import_map=import_map,
                                    backup=backup,
                                    remaining_objects=remaining_objects  # Pass remaining_objects for dynamic updates
                                )
                                successfully_deleted.extend(deleted_this_round)
                                
                                if deleted_this_round:
                                    pass
                                    # tqdm.write(f"✅ Binary search complete: deleted {len(deleted_this_round)}/{len(leaf_objects)} objects")
                                else:
                                    # No objects deleted; must restore backup
                                    # tqdm.write(f"⚠️ Binary search complete: no objects deletable, restoring backup")
                                    for temp_fpath, content in backup.items():
                                        with open(temp_fpath, 'w', encoding='utf-8') as f:
                                            f.write(content)
                            else:
                                # All objects are protected; restore backup
                                for temp_fpath, content in backup.items():
                                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                                        f.write(content)
                    
                    except Exception as e:
                    # Batch deletion exception; rollback
                        tqdm.write(f"❌ Batch deletion error: {e}")
                        for temp_fpath, content in backup.items():
                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(content)
                    
                    # Update remaining objects list
                    for obj_id in deleted_this_round:
                        if obj_id in remaining_objects:
                            remaining_objects.remove(obj_id)
                    
                    # If no objects deleted this round, exit loop
                    if not deleted_this_round:
                        # tqdm.write(f"No objects deleted this round; stop iterating. Remaining: {remaining_objects}")
                        break
            
                # ============ P2P validation stage ============
                # tqdm.write(f"\n{'='*60}")
                # tqdm.write(f"Iterative deletion complete; start P2P validation (deleted {len(successfully_deleted)} objects)")
                # tqdm.write(f"{'='*60}\n")
                
                # Reset container environment
                reset_success = self.image_manager.reset_container_env(
                    container_id=container_id,
                    specs_name=self.data_item.repo,
                    timeout=None
                )
                
                if not reset_success:
                    pass
                    # tqdm.write(f"❌ Failed to reset container env before P2P validation")
                else:
                    # Reapply all deletions (reuse batch deletion logic)
                    self._batch_delete_objects_and_imports(
                        successfully_deleted,
                        temp_files_map,
                        import_map
                    )
                    
                    # Copy files to container
                    self.image_manager.copy_to_container(
                        container_id=container_id,
                        src_path=None,
                        dest_path=None,
                        use_tar=True,
                        files_mapping=temp_files_map
                    )
                    
                    # Run P2P validation
                    p2p_success, failed_p2p, failed_p2p_log = self._run_p2p_validation(
                        container_id,
                        temp_files_map
                    )
                    
                    if not p2p_success:
                        # P2P validation failed; extract error symbols to protect objects and reduce binary search work
                        # tqdm.write(f"\n⚠️ P2P validation failed! Parsing log and starting binary search...\n")

                        # If failure log exists, parse and protect known error-causing objects
                        if failed_p2p_log:
                            try:
                                error_symbols = self._extract_error_symbols_from_log(failed_p2p_log)
                            except Exception:
                                error_symbols = set()

                            if error_symbols:
                                protected_objects = []
                                for symbol in error_symbols:
                                    matched_objs = self._find_objects_by_symbol_name(successfully_deleted, symbol)
                                    protected_objects.extend(matched_objs)

                                if protected_objects:
                                    # Remove non-deletable objects from successfully_deleted
                                    for obj_id in protected_objects:
                                        if obj_id in successfully_deleted:
                                            successfully_deleted.remove(obj_id)
                                    # tqdm.write(f"🛡️ From P2P log found {len(error_symbols)} error symbols, protected {len(protected_objects)} objects, skipping their binary search")

                        # Reset temp files to original content and build backup
                        backup: Dict[str, str] = {}
                        for file_path_container, temp_fpath in temp_files_map.items():
                            host_path = self.repo_manager.convert_container_path_to_local(
                                self.data_item.repo,
                                file_path_container
                            )
                            with open(host_path, 'r', encoding='utf-8') as f:
                                original_code = f.read()

                            backup[temp_fpath] = original_code

                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(original_code)

                        # Use binary search to find truly safe objects
                        actually_safe_to_delete = self._binary_search_with_p2p(
                            failed_batch=successfully_deleted.copy(),
                            container_id=container_id,
                            temp_files_map=temp_files_map,
                            import_map=import_map,
                            backup=backup,
                            remaining_objects=[]
                        )

                        # Reset files and reapply safe deletions to keep content in sync
                        for file_path_container, temp_fpath in temp_files_map.items():
                            host_path = self.repo_manager.convert_container_path_to_local(
                                self.data_item.repo,
                                file_path_container
                            )
                            with open(host_path, 'r', encoding='utf-8') as f:
                                original_code = f.read()

                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(original_code)

                        if actually_safe_to_delete:
                            self._batch_delete_objects_and_imports(
                                actually_safe_to_delete,
                                temp_files_map,
                                import_map
                            )

                        # Update successfully_deleted list
                        objects_to_restore = [obj for obj in successfully_deleted if obj not in actually_safe_to_delete]
                        if objects_to_restore:
                            pass
                            # tqdm.write(f"🔄 P2P post-validation restored {len(objects_to_restore)} objects; final deleted {len(actually_safe_to_delete)}\n")
                        successfully_deleted = actually_safe_to_delete
                    else:
                        pass
                        # tqdm.write(f"✅ P2P validation passed!\n")
            
            finally:
                total = len(all_objects_to_remove)
                deleted_set = set(successfully_deleted)
                deleted_count = len(deleted_set)
                protected_count = total - deleted_count
                tqdm.write(
                    f"{self.data_item.repo} - {Path(self.data_item.file_path).name}: total {total} objects, "
                    f"deleted {deleted_count}, protected {protected_count}"
                )
                # Ensure container is stopped at the end
                if container_id:
                    self.image_manager.stop_container(container_id, force=False)
            

            # Generate final results (read from temp files)
            # Note: need to process all tracked files, not only patch_files
            # because import files were also modified (imports removed)
            for file_path in files_to_track:
                try:
                    # Read final content from temp file
                    temp_file_path = temp_files_map.get(file_path)
                    if not temp_file_path:
                        raise ValueError(f"Temp file not found: {file_path}")
                    
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        masked_code = f.read()
                    
                    # Use previously saved original signature
                    original_signature = original_signatures.get(file_path)

                    # Collect deleted objects for this file
                    top_deleted = [obj for obj in successfully_deleted if obj in all_top_objects and obj.startswith(file_path + '::')]
                    specific_deleted = [obj for obj in successfully_deleted if obj in all_specific_objects and obj.startswith(file_path + '::')]
                    
                    # Validate
                    validation_result = MaskValidator.validate(
                        original_signature=original_signature,
                        masked_code=masked_code,
                        should_remove_set=set(top_deleted + specific_deleted)
                    )

                    results[file_path] = MaskResult(
                        file_path=file_path,
                        masked_code=masked_code,
                        top_objects=sorted(top_deleted),
                        specific_objects=sorted(specific_deleted),
                        signature_info=original_signature,
                        validation_result=validation_result,
                        success=True
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate final result {file_path}: {e}")
                    results[file_path] = MaskResult(
                        file_path=file_path,
                        masked_code="",
                        top_objects=[],
                        specific_objects=[],
                        signature_info=ModuleSignature(file_path=file_path),
                        success=False,
                        error_message=str(e)
                    )
        
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                tqdm.write(f"⚠️ Failed to clean temp directory {temp_dir}: {e}")
        
        return results
    
    def _apply_mask(
        self,
        source_code: str,
        file_path: str,
        objects_to_remove_ids: Set[str]
    ) -> str:
        """
        Apply mask to source code.

        Strategy: delete specified objects and replace with blank lines (keep line numbers stable).

        Args:
            source_code: Source code
            file_path: File path (container path), used to build object IDs
            objects_to_remove_ids: Full object IDs to remove (including line numbers),
                             e.g. {'/testbed/file.py::Class.method::123', '/testbed/file.py::Class.method::456'}

        Returns:
            str: masked code
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.error(f"Failed to parse code: {e}")
            return source_code
            
        source_lines = source_code.splitlines()
        
        # Collect all nodes to delete (ignore parent/child relationship)
        replacements = []
        
        def find_nested_definitions(node):
            """Find function/class definitions in all child nodes."""
            nested = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nested.append(child)
                else:
                    nested.extend(find_nested_definitions(child))
            return nested
        
        def collect_nodes(nodes, parent_qualified_name=None, inherit_delete=False):
            """Recursively collect nodes to delete.

            Args:
                nodes: Node list at current level
                parent_qualified_name: Qualified name of parent node
                inherit_delete: Whether to inherit deletion flag from parent (delete children if parent is deleted)
            """
            for node in nodes:
                # Handle function/class definitions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Build qualified name
                    if parent_qualified_name:
                        qualified_name = f"{parent_qualified_name}.{node.name}"
                    else:
                        qualified_name = node.name
                    
                    # Build full object ID
                    node_lineno = node.lineno
                    obj_id = f"{file_path}::{qualified_name}::{node_lineno}"
                    
                    # Determine deletion: explicitly listed or inherited from parent
                    should_delete = (obj_id in objects_to_remove_ids) or inherit_delete
                    
                    if should_delete:
                        # Compute deletion range
                        start_line = self._get_node_start_line(node)
                        end_line = self._get_node_end_line(node, source_lines)
                        replacements.append((start_line, end_line, obj_id))
                    
                    # Recurse into nested definitions; pass deletion flag
                    nested_defs = find_nested_definitions(node)
                    collect_nodes(nested_defs, parent_qualified_name=qualified_name, inherit_delete=should_delete)
                
                # Handle control-flow blocks (if/else/try/with/for/while) and recurse into bodies
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    # Body
                    if hasattr(node, 'body'):
                        collect_nodes(node.body, parent_qualified_name=parent_qualified_name, inherit_delete=inherit_delete)
                    
                    # Else part (if/for/while/try)
                    if hasattr(node, 'orelse') and node.orelse:
                        collect_nodes(node.orelse, parent_qualified_name=parent_qualified_name, inherit_delete=inherit_delete)
                    
                    # Except handlers (try)
                    if hasattr(node, 'handlers'):
                        for handler in node.handlers:
                            collect_nodes(handler.body, parent_qualified_name=parent_qualified_name, inherit_delete=inherit_delete)
                    
                    # Finally block (try)
                    if hasattr(node, 'finalbody'):
                        collect_nodes(node.finalbody, parent_qualified_name=parent_qualified_name, inherit_delete=inherit_delete)
        
        # Start from top level
        collect_nodes(tree.body)
        
        # Sort by line number (front to back)
        replacements.sort(key=lambda x: x[0])
        
        # Apply deletion: replace with blank lines
        for start_line, end_line, obj_id in replacements:
            # Ensure indices are valid
            if start_line < 0 or start_line >= len(source_lines):
                continue
            if end_line < 0:
                end_line = start_line
            if end_line >= len(source_lines):
                end_line = len(source_lines) - 1
            
            # Replace with blank lines (keep line numbers stable)
            for i in range(start_line, end_line + 1):
                source_lines[i] = ''
        
        result_code = '\n'.join(source_lines)
        
        # Fix empty classes/control blocks
        result_code = self._fix_empty_structures(result_code)
        
        return result_code
        
    def _get_node_start_line(self, node: ast.AST) -> int:
        """Get node start line (including decorators)."""
        if hasattr(node, 'decorator_list') and node.decorator_list:
            return node.decorator_list[0].lineno - 1
        return node.lineno - 1
        
    def _get_node_end_line(self, node: ast.AST, source_lines: List[str]) -> int:
        """Get node end line."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            # Use precise end line from AST
            return node.end_lineno - 1
            
        # Fallback: if no end_lineno, use indentation detection
        # Note: this may be less accurate; Python 3.8+ recommended
        start_line = node.lineno - 1
        if start_line >= len(source_lines):
            return start_line
            
        def_line = source_lines[start_line]
        base_indent = len(def_line) - len(def_line.lstrip())
        
        current_line = start_line + 1
        while current_line < len(source_lines):
            line = source_lines[current_line]
            # Only check indentation for non-empty lines
            if line.strip():
                line_indent = len(line) - len(line.lstrip())
                    # If indentation <= base, we've reached next sibling or outer definition
                if line_indent <= base_indent:
                        # Return previous non-empty line position
                    return current_line - 1
            current_line += 1
            
        # If end of file, return last line
        return len(source_lines) - 1
    
    def _fix_empty_classes(self, code: str, *, skip_validation: bool = False) -> str:
        """
        Fix empty classes: add pass statements.

        Note: do not proactively remove all empty classes because original code may
        contain docstring-only classes (valid code). Only add pass when deletions
        cause syntax errors.

        Args:
            code: Code that may contain empty classes
			
        Returns:
            str: Fixed code
        """
        try:
            # Try parsing first; if no syntax errors, return directly
            tree = ast.parse(code)
            return code
        except SyntaxError:
            # Syntax error might be caused by empty classes; try adding pass
            lines = code.splitlines()
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # Check if this is a class definition
                if line.strip().startswith('class ') and line.strip().endswith(':'):
                    # Get class definition indentation
                    class_indent = len(line) - len(line.lstrip())
                    
                    # Traverse class body, check if it only contains blank lines
                    j = i + 1
                    is_empty_class = True  # Assume empty class until content found
                    candidate_indices: List[int] = []  # Record blank/comment line indices
                    
                    while j < len(lines):
                        body_line = lines[j]
                        stripped_body = body_line.strip()
                        
                        # Blank line
                        if not stripped_body:
                            candidate_indices.append(j)
                            j += 1
                            continue
                        
                        # Current line indentation
                        line_indent = len(body_line) - len(body_line.lstrip())
                        
                        # If indentation returns to class level or less, class body ends
                        if line_indent <= class_indent:
                            break
                        
                        # Comment-only line; ignore content but record candidate position
                        if stripped_body.startswith('#'):
                            candidate_indices.append(j)
                            j += 1
                            continue
                        
                        # Any other content means not empty
                        is_empty_class = False
                        break
                    
                    if is_empty_class:
                        class_def = line.strip()
                        self.logger.debug(f"Fix empty class (add pass): {class_def}")
                        
                        # Class body indentation (class indent + 4 spaces)
                        class_indent = len(lines[i]) - len(lines[i].lstrip())
                        body_indent = class_indent + 4
                        body_indent_str = ' ' * body_indent
                        
                        # Prefer blank line; then comment line; otherwise insert a new line
                        blank_index = next((idx for idx in candidate_indices if not lines[idx].strip()), None)
                        comment_index = next(
                            (idx for idx in candidate_indices if lines[idx].strip().startswith('#')),
                            None
                        )

                        if blank_index is not None:
                            lines[blank_index] = f'{body_indent_str}pass'
                        elif comment_index is not None:
                            comment_text = lines[comment_index].strip()[1:].strip()
                            if comment_text:
                                lines[comment_index] = f'{body_indent_str}pass  # {comment_text}'
                            else:
                                lines[comment_index] = f'{body_indent_str}pass'
                        else:
                            # No blank/comment line, insert pass placeholder
                            lines.insert(i + 1, f'{body_indent_str}pass')
                
                i += 1
            
            result = '\n'.join(lines)

            if skip_validation:
                return result
            
            # Validate fixed code
            try:
                ast.parse(result)
                return result
            except SyntaxError:
                # If still invalid, return original code and warn
                self.logger.warning("Syntax error remains after fixing empty class; returning original code")
                return code

    def _fix_empty_control_blocks(self, code: str, *, skip_validation: bool = False) -> str:
        """
        Fix empty control blocks (if/for/while/try/with, etc.) caused by deletions.
        """
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass

        lines = code.splitlines()
        block_keywords = (
            'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:',
            'with ', 'async with ', 'async for ', 'match ', 'case '
        )

        def is_block_header(text: str) -> bool:
            stripped = text.strip()
            if not stripped.endswith(':'):
                return False
            if stripped.startswith('class ') or stripped.startswith('def '):
                return False
            for keyword in block_keywords:
                if stripped.startswith(keyword) or stripped == keyword.rstrip(' '):
                    return True
            return False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                i += 1
                continue

            if is_block_header(line):
                block_indent = len(line) - len(line.lstrip())
                candidate_indices: List[int] = []
                block_is_empty = True
                j = i + 1

                while j < len(lines):
                    body_line = lines[j]
                    stripped_body = body_line.strip()

                    if not stripped_body:
                        candidate_indices.append(j)
                        j += 1
                        continue

                    body_indent = len(body_line) - len(body_line.lstrip())
                    if body_indent <= block_indent:
                        break

                    if stripped_body.startswith('#'):
                        candidate_indices.append(j)
                        j += 1
                        continue

                    block_is_empty = False
                    break

                if block_is_empty:
                    body_indent_str = ' ' * (block_indent + 4)
                    blank_index = next((idx for idx in candidate_indices if not lines[idx].strip()), None)
                    if blank_index is not None:
                        lines[blank_index] = f"{body_indent_str}pass"
                    else:
                        comment_index = next(
                            (idx for idx in candidate_indices if lines[idx].strip().startswith('#')),
                            None
                        )
                        if comment_index is not None:
                            comment_text = lines[comment_index].strip()[1:].strip()
                            if comment_text:
                                lines[comment_index] = f"{body_indent_str}pass  # {comment_text}"
                            else:
                                lines[comment_index] = f"{body_indent_str}pass"
                        else:
                            lines.insert(i + 1, f"{body_indent_str}pass")
                            i += 1

            i += 1

        result = '\n'.join(lines)

        if skip_validation:
            return result

        try:
            ast.parse(result)
            return result
        except SyntaxError:
            self.logger.warning("Syntax error remains after fixing empty control blocks; returning original code")
            return code

    def _fix_empty_structures(self, code: str) -> str:
        """Repair empty classes/blocks sequentially, then re-validate syntax."""
        original_code = code
        result = self._fix_empty_classes(code, skip_validation=True)
        result = self._fix_empty_control_blocks(result, skip_validation=True)

        try:
            ast.parse(result)
            return result
        except SyntaxError:
            self.logger.warning("Syntax error remains after fixing empty structures; returning original code")
            return original_code
 
    def _build_import_map(self) -> Dict[str, List[Tuple[str, int, str, str]]]:
        """
        Build a mapping from simplified object identifier to import locations.

        Returns:
            {
                "/testbed/file.py::123": [
                    (importer_file, importer_line, imported_name, defined_name)
                    ...
                ]
            }
        """
        import_map = {}
        
        for record in self.import_records:
            defined_file = record.get('defined_file')
            defined_line = record.get('defined_line')
            defined_name = record.get('defined_name')
            imported_name = record.get('imported_name')
            importer_file = record.get('importer_file')
            importer_line = record.get('importer_line')
            
            if not all([defined_file, defined_line, defined_name, imported_name, importer_file, importer_line]):
                continue
            
            # Build object ID (simplified: name + line number)
            # Note: full qualified name (e.g., OuterClass.InnerClass.method) is unavailable here
            # so use file path + line number as unique identifier
            obj_key = f"{defined_file}::{defined_line}"
            
            if obj_key not in import_map:
                import_map[obj_key] = []

            # Save importer_file, importer_line, imported_name, defined_name
            import_map[obj_key].append((importer_file, importer_line, imported_name, defined_name))
        
        return import_map
    
    def _delete_import_from_line(
        self,
        file_path: str,
        line_number: int,
        object_name: str
    ) -> bool:
        """
        Delete import of a specific object at a specific line in a file.

        Args:
            file_path: File path (host path)
            line_number: Line number (1-indexed)
            object_name: Object name to delete

        Returns:
            bool: Whether deletion succeeded
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number < 1 or line_number > len(lines):
                self.logger.debug(f"Line number out of range: {file_path}:{line_number} (total lines: {len(lines)})")
                return False
            
            line_idx = line_number - 1

            # Some import statements span multiple lines (parentheses or line continuation);
            # collect downward until a full import/from statement can be parsed.
            # Note: remove indentation because these lines may be inside try/if blocks,
            # which can cause "unexpected indent" errors during parsing.
            end_idx = line_idx
            first_line = lines[line_idx]
            if not first_line.strip():
                # Object already deleted; first line empty means no import to remove
                return False
            indent = len(first_line) - len(first_line.lstrip())
            
            # Collect dedented code for AST parsing
            collected_lines = first_line.lstrip().rstrip('\n')
            parse_failed = False
            while True:
                try:
                    tree = ast.parse(collected_lines)
                    if tree.body:
                        stmt = tree.body[0]
                        # Ensure parsed statement is import/importfrom
                        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                            break
                        else:
                            # Parsed other statement; not an import
                            tqdm.write(f"❌ Parsed non-import statement: {file_path}:{line_number}")
                            return False
                    else:
                        # Try reading next line
                        raise SyntaxError('empty body')
                except SyntaxError as e:
                    # If EOF reached or threshold exceeded, give up
                    end_idx += 1
                    if end_idx >= len(lines) or end_idx - line_idx > 200:
                        tqdm.write(f"❌ Failed to parse full import statement {file_path}:{line_number} - {e}")
                        parse_failed = True
                        break
                    # Append next line (dedented) and continue parsing
                    next_line = lines[end_idx].lstrip().rstrip('\n')
                    collected_lines += '\n' + next_line

            if parse_failed:
                return False

            # Handle from xxx import yyy
            if isinstance(stmt, ast.ImportFrom):
                # Collect imports other than the object to delete
                remaining_imports = [
                    alias.name for alias in stmt.names
                    if alias.name != object_name and alias.name != '*'
                ]
                
                if '*' in [alias.name for alias in stmt.names]:
                    # from xxx import * - skip
                    tqdm.write(f"Skip 'import *' statement: {file_path}:{line_number}")
                    return False
                
                if not remaining_imports:
                    # Delete all lines of the import statement (may be multi-line), replace with blanks
                    for i in range(line_idx, end_idx + 1):
                        lines[i] = '\n'
                else:
                    # Rewrite to keep remaining imports only (write to first line, clear others)
                    indent_str = ' ' * indent
                    # Relative import: stmt.level is dot count, stmt.module is module name
                    dots = '.' * (stmt.level or 0)
                    module = stmt.module or ''
                    new_import = f"{indent_str}from {dots}{module} import {', '.join(remaining_imports)}\n"
                    lines[line_idx] = new_import
                    for i in range(line_idx + 1, end_idx + 1):
                        lines[i] = '\n'
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
            
            # Handle import xxx (uncommon, but supported)
            elif isinstance(stmt, ast.Import):
                remaining_imports = [
                    alias.name for alias in stmt.names
                    if alias.name != object_name
                ]
                
                if not remaining_imports:
                    for i in range(line_idx, end_idx + 1):
                        lines[i] = '\n'
                else:
                    indent_str = ' ' * indent
                    new_import = f"{indent_str}import {', '.join(remaining_imports)}\n"
                    lines[line_idx] = new_import
                    for i in range(line_idx + 1, end_idx + 1):
                        lines[i] = '\n'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True
            
            return False
            
        except Exception as e:
            tqdm.write(f"❌ Failed to delete import {file_path}:{line_number} - {e}")
            return False
    
    def _run_pytest_collect(self, container_id: str, test_file: str, timeout: int = None):
        """
        Run pytest --collect-only in the container to validate collection.

        Args:
            container_id: Docker container ID
            test_file: Test file path (container path)
            timeout: Timeout in seconds; if None, read from config

        Returns:
            Tuple[bool, Optional[str]]: (success, log file path)
        """
        # Get timeout from config
        if timeout is None:
            timeout = self.data_item.specs.get('timeout_collect', 300)  # Default 300s
            if isinstance(timeout, (int, float)) and timeout < 0:
                # Allow setting -1 for no limit
                timeout = None
        
        # Create log file path using logger.py helper
        log_file_path = create_mask_generator_logger(
            specs_name=self.data_item.repo,
            test_file=self.data_item.file_path,
            container_id=container_id,
            logs_dir=self.config.actual_output_dir / "logs"
        )
        
        try:
            cmd = f"pytest --collect-only {test_file}"
            cmd = apply_uv_run_prefix(cmd, self.data_item.specs)
            result = self.image_manager.exec_in_container(
                container_id,
                cmd,
                timeout=timeout,
                log_file_path=str(log_file_path)  # Use image_manager logging
            )
            
            if result.returncode != 0:
                # Parse and print error types
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                error_types = PytestParser.parse_error_types(log_content)
                error_str = ", ".join(error_types)
                # tqdm.write(f"❌ pytest --collect failed ({error_str}): {test_file}")
                # tqdm.write(f"   Log: {log_file_path}")

            return result.returncode == 0, log_file_path
            
        except subprocess.TimeoutExpired:
            # tqdm.write(f"❌ pytest --collect timed out: {test_file}\nLog: {log_file_path}")
            return False, log_file_path
            
        except Exception as e:
            # tqdm.write(f"❌ pytest --collect error: {test_file}\nLog: {log_file_path}")
            return False, log_file_path
    
    def _run_p2p_validation(
        self,
        container_id: str,
        temp_files_map: Dict[str, str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run full p2p test list in container for validation.

        This runs after collect validation succeeds to catch runtime-only dependency issues.
        Unlike P2PValidator, it reuses the existing container instead of starting a new one.

        Args:
            container_id: Docker container ID (reused)
            temp_files_map: Temp file mapping (container path -> host temp file path)

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (
                whether all p2p tests passed,
                first failing p2p file path (container path),
                log file path for the failed test (host)
            )
        """
        p2p_list = getattr(self.data_item, 'p2p_list', None)
        
        if not p2p_list or len(p2p_list) == 0:
            # No p2p tests; return success
            return True, None, None
        
        test_cmd = apply_uv_run_prefix(self.data_item.specs.get("test_cmd"), self.data_item.specs)
        timeout_test = self.data_item.specs.get("timeout_run", 300)
        
        # Ensure all p2p test files are unmasked (use original versions)
        for p2p_file in p2p_list:
            p2p_host_path = self.repo_manager.convert_container_path_to_local(
                self.data_item.repo,
                p2p_file
            )
            
            # If p2p file is in temp_files_map (modified), copy original first
            if p2p_file in temp_files_map:
                self.image_manager.copy_to_container(
                    container_id,
                    p2p_host_path,
                    p2p_file
                )
        
        # Run p2p tests one by one
        for p2p_file in p2p_list:
            # Create log file
            from featurebench.utils.logger import create_p2p_validator_logger
            log_file_path = create_p2p_validator_logger(
                specs_name=self.data_item.repo,
                test_file=p2p_file,
                logs_dir=self.config.actual_output_dir / "logs"
            )
            
            try:
                # Build full test command
                full_test_cmd = f"{test_cmd} {p2p_file}"
                
                test_result = self.image_manager.exec_in_container(
                    container_id,
                    full_test_cmd,
                    timeout=timeout_test,
                    log_file_path=str(log_file_path)
                )
                
                # Parse test result
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    test_output = f.read()
                
                pytest_result = PytestParser.parse_output(test_output)
                
                # Check success (return code 0 or 1 means tests ran)
                # and all tests must pass (pass_rate == 1.0)
                success = pytest_result.return_code in [0, 1] and pytest_result.pass_rate == 1.0
                
                if not success:
                    # A p2p test failed
                    if pytest_result.return_code not in [0, 1]:
                        error_msg = f"P2P test error (return code: {pytest_result.return_code})"
                    else:
                        error_msg = f"P2P test failed (pass rate: {pytest_result.pass_rate:.2%})"
                    
                    tqdm.write(
                        # f"❌ Mask P2P validation failed: {p2p_file} - {error_msg}\n"
                        # f"   Log: {log_file_path}"
                    )
                    # Return failed p2p file and log path for upstream parsing/protection
                    return False, p2p_file, str(log_file_path)
                
            except subprocess.TimeoutExpired:
                # tqdm.write(f"❌ Mask P2P validation timed out: {p2p_file}\n   Log: {log_file_path}")
                return False, p2p_file, str(log_file_path)
            
            except Exception as e:
                # tqdm.write(f"❌ Mask P2P validation error: {p2p_file} - {e}\n   Log: {log_file_path}")
                return False, p2p_file, str(log_file_path)
        
        # All p2p tests passed
        return True, None, None
    
    def _extract_error_symbols_from_log(self, log_file_path: str) -> Set[str]:
        """
        Extract error-causing symbol names from a log file.

        Args:
            log_file_path: Log file path

        Returns:
            Set[str]: Symbols that caused errors, e.g. {'_binop', '_f', 'some_func'}
        """
        error_symbols = set()
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            # 1. Match NameError: name 'xxx' is not defined
            name_error_pattern = re.compile(r"NameError:\s+name\s+'([^']+)'\s+is\s+not\s+defined", re.IGNORECASE)
            for match in name_error_pattern.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            # 2. Match AttributeError: module 'xxx' has no attribute 'yyy'
            #    or AttributeError: type object 'XXX' has no attribute 'yyy'
            #    or AttributeError: 'XXX' object has no attribute 'yyy'
            attr_error_pattern = re.compile(
                r"AttributeError:\s+(?:module|type\s+object|'[^']+'(?:\s+object)?)\s+(?:'[^']+'\s+)?has\s+no\s+attribute\s+'([^']+)'",
                re.IGNORECASE
            )
            for match in attr_error_pattern.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            # 3. Match ImportError: cannot import name 'xxx' from 'yyy'
            import_error_pattern = re.compile(
                r"ImportError:\s+cannot\s+import\s+name\s+'([^']+)'",
                re.IGNORECASE
            )
            for match in import_error_pattern.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            # 4. Match ModuleNotFoundError: No module named 'xxx'
            #    Extract the last segment as a possible symbol (for submodules)
            module_error_pattern = re.compile(
                r"ModuleNotFoundError:\s+No\s+module\s+named\s+'([^']+)'",
                re.IGNORECASE
            )
            for match in module_error_pattern.finditer(log_content):
                module_name = match.group(1)
                # Extract last segment, e.g. 'pkg.subpkg.module' -> 'module'
                symbol = module_name.split('.')[-1]
                error_symbols.add(symbol)
            
            # 5. Match TypeError: 'xxx' object is not callable (function/class may be deleted)
            #    or TypeError: xxx() missing required argument 'yyy'
            type_error_callable = re.compile(r"TypeError:\s+'([^']+)'\s+object\s+is\s+not\s+callable", re.IGNORECASE)
            for match in type_error_callable.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            type_error_missing = re.compile(r"TypeError:\s+([^\s(]+)\(\)", re.IGNORECASE)
            for match in type_error_missing.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            # 6. Match symbols mentioned in ValueError (rare but sometimes useful)
            #    e.g. ValueError: unknown function 'xxx'
            value_error_pattern = re.compile(
                r"ValueError:.*?(?:function|method|class|attribute|name)\s+'([^']+)'",
                re.IGNORECASE
            )
            for match in value_error_pattern.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
            # 7. Match UnboundLocalError: local variable 'xxx' referenced before assignment
            unbound_error_pattern = re.compile(
                r"UnboundLocalError:.*?variable\s+'([^']+)'",
                re.IGNORECASE
            )
            for match in unbound_error_pattern.finditer(log_content):
                symbol = match.group(1)
                error_symbols.add(symbol)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse log file {log_file_path}: {e}")
        
        return error_symbols
    
    def _find_objects_by_symbol_name(self, objects_list: List[str], symbol_name: str) -> List[str]:
        """
        Find all matching object IDs in a list by symbol name.

        Args:
            objects_list: Object ID list, e.g. ["/testbed/file.py::Class.method::123", ...]
            symbol_name: Symbol name, e.g. '_binop', '_f', 'fillna'

        Returns:
            List[str]: Matching object IDs
        """
        matched_objects = []
        
        for obj_id in objects_list:
            # Parse object ID: /testbed/file.py::qualified_name::line
            parts = obj_id.split('::')
            if len(parts) < 3:
                continue
            
            qualified_name = parts[1]  # e.g. 'Class.method' or '_binop' or 'ExtensionArray.fillna'
            
            # Check symbol name match (exact or last segment match)
            # e.g. symbol_name='_binop' should match 'ExtensionScalarOpsMixin._create_method._binop'
            # e.g. symbol_name='fillna' should match 'ExtensionArray.fillna'
            if qualified_name == symbol_name or qualified_name.endswith('.' + symbol_name):
                matched_objects.append(obj_id)
        
        return matched_objects
    
    def _binary_search_with_p2p(
        self,
        failed_batch: List[str],
        container_id: str,
        temp_files_map: Dict[str, str],
        import_map: Dict[str, List[Tuple[str, int, str, str]]],
        backup: Dict[str, str],
        remaining_objects: List[str]
    ) -> List[str]:
        """
        Use p2p-validated binary search to find which objects in a failed batch can be deleted.

        Similar to `_binary_search_failed_objects`, but validates with full p2p tests
        instead of only pytest --collect-only.

        Args:
            failed_batch: Objects that failed batch deletion
            container_id: Container ID
            temp_files_map: Temp file mapping
            import_map: Import mapping
            backup: Backup of temp file contents
            remaining_objects: Remaining objects to delete (for dynamic protection updates)

        Returns:
            List of objects successfully deleted
        """
        if len(failed_batch) == 0:
            return []
        
        if len(failed_batch) == 1:
            # Only one object left; cannot split further, likely not deletable
            # Restore backup to avoid unvalidated deletion
            for temp_fpath, content in backup.items():
                with open(temp_fpath, 'w', encoding='utf-8') as f:
                    f.write(content)
            # tqdm.write(f"  ⚠️ P2P binary search: cannot delete object: {failed_batch[0]}")
            return []
        
        # Split in half
        mid = len(failed_batch) // 2
        left_batch = failed_batch[:mid]
        right_batch = failed_batch[mid:]
        
        # tqdm.write(f"  🔍 P2P binary search: left half {len(left_batch)} objects")
        
        # Test left half
        # 1. Restore backup
        for temp_fpath, content in backup.items():
            with open(temp_fpath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 2. Delete left half
        try:
            # Reset container environment
            self.image_manager.reset_container_env(
                container_id=container_id,
                specs_name=self.data_item.repo,
                timeout=60
            )

            # Delete left half
            self._batch_delete_objects_and_imports(
                left_batch,
                temp_files_map,
                import_map
            )
            
            # Copy files to container
            self.image_manager.copy_to_container(
                container_id=container_id,
                src_path=None,
                dest_path=None,
                use_tar=True,
                files_mapping=temp_files_map
            )
            
            # Run p2p validation
            left_success, failed_p2p, left_log = self._run_p2p_validation(
                container_id,
                temp_files_map
            )
            # If left half fails, extract error symbols and protect objects
            if not left_success and left_log:
                try:
                    error_symbols = self._extract_error_symbols_from_log(left_log)
                except Exception:
                    error_symbols = set()

                if error_symbols:
                    protected_objects = []
                    for symbol in error_symbols:
                        matched_objs = self._find_objects_by_symbol_name(left_batch, symbol)
                        protected_objects.extend(matched_objs)

                    if protected_objects:
                        # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from left-half log")
                        # Remove protected objects from remaining_objects and left_batch
                        for obj_id in protected_objects:
                            if obj_id in remaining_objects:
                                remaining_objects.remove(obj_id)
                        left_batch = [obj for obj in left_batch if obj not in protected_objects]
            
        except Exception as e:
            tqdm.write(f"  ❌ P2P test error (left half): {e}")
            left_success = False
        
            # Decide strategy based on left-half result
        if left_success:
                # Left half succeeded; test right half
                # tqdm.write(f"  ✅ P2P left half succeeded; test right half {len(right_batch)} objects")
            
                # Continue deletions on top of left half
            try:
                self.image_manager.reset_container_env(
                    container_id=container_id,
                    specs_name=self.data_item.repo,
                    timeout=60
                )

                self._batch_delete_objects_and_imports(
                    right_batch,
                    temp_files_map,
                    import_map
                )
                
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=None,
                    dest_path=None,
                    use_tar=True,
                    files_mapping=temp_files_map
                )
                
                right_success, failed_p2p, right_log = self._run_p2p_validation(
                    container_id,
                    temp_files_map
                )
                # If right half fails, extract error symbols and protect objects
                if not right_success and right_log:
                    try:
                        error_symbols = self._extract_error_symbols_from_log(right_log)
                    except Exception:
                        error_symbols = set()

                    if error_symbols:
                        protected_objects = []
                        for symbol in error_symbols:
                            matched_objs = self._find_objects_by_symbol_name(right_batch, symbol)
                            protected_objects.extend(matched_objs)

                        if protected_objects:
                            # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from right-half log")
                            for obj_id in protected_objects:
                                if obj_id in remaining_objects:
                                    remaining_objects.remove(obj_id)
                            right_batch = [obj for obj in right_batch if obj not in protected_objects]
                
                if right_success:
                        # Both halves succeeded
                    return left_batch + right_batch
                else:
                        # Left succeeded, right failed -> recurse on right half
                    right_result = self._binary_search_with_p2p(
                        right_batch,
                        container_id,
                        temp_files_map,
                        import_map,
                        backup,
                        remaining_objects
                    )

                    if left_batch or right_result:
                        for temp_fpath, content in backup.items():
                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(content)

                        self._batch_delete_objects_and_imports(
                            left_batch + right_result,
                            temp_files_map,
                            import_map
                        )

                    return left_batch + right_result
            
            except Exception as e:
                tqdm.write(f"  ❌ P2P test error (right half): {e}")
                # Roll back and reapply left half to keep state consistent
                for temp_fpath, content in backup.items():
                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                        f.write(content)

                if left_batch:
                    self._batch_delete_objects_and_imports(
                        left_batch,
                        temp_files_map,
                        import_map
                    )

                return left_batch
        
        else:
            # Left half failed; recurse on left, then test right
            # tqdm.write(f"  ❌ P2P left half failed; recurse")
            left_result = self._binary_search_with_p2p(
                left_batch,
                container_id,
                temp_files_map,
                import_map,
                backup,
                remaining_objects
            )
            
            # After recursion, reapply deletions from left_result: restore first, then delete
            if left_result:
                for temp_fpath, content in backup.items():
                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                self._batch_delete_objects_and_imports(
                    left_result,
                    temp_files_map,
                    import_map
                )
            
            # Test right half
            # tqdm.write(f"  🔍 P2P test right half {len(right_batch)} objects")
            
            try:
                # Continue deleting right half
                self.image_manager.reset_container_env(
                    container_id=container_id,
                    specs_name=self.data_item.repo,
                    timeout=60
                )

                self._batch_delete_objects_and_imports(
                    right_batch,
                    temp_files_map,
                    import_map
                )
                
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=None,
                    dest_path=None,
                    use_tar=True,
                    files_mapping=temp_files_map
                )
                
                right_success, failed_p2p, right_log = self._run_p2p_validation(
                    container_id,
                    temp_files_map
                )
                # If right half fails, extract error symbols and protect objects
                if not right_success and right_log:
                    try:
                        error_symbols = self._extract_error_symbols_from_log(right_log)
                    except Exception:
                        error_symbols = set()

                    if error_symbols:
                        protected_objects = []
                        for symbol in error_symbols:
                            matched_objs = self._find_objects_by_symbol_name(right_batch, symbol)
                            protected_objects.extend(matched_objs)

                        if protected_objects:
                            # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from right-half log")
                            for obj_id in protected_objects:
                                if obj_id in remaining_objects:
                                    remaining_objects.remove(obj_id)
                            right_batch = [obj for obj in right_batch if obj not in protected_objects]
                
                if right_success:
                    # Right half succeeded on top of left_result
                    return left_result + right_batch
                else:
                    # Right half failed; recurse
                    right_result = self._binary_search_with_p2p(
                        right_batch,
                        container_id,
                        temp_files_map,
                        import_map,
                        backup,
                        remaining_objects
                    )

                    if left_result or right_result:
                        for temp_fpath, content in backup.items():
                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(content)
                        
                        self._batch_delete_objects_and_imports(
                            left_result + right_result,
                            temp_files_map,
                            import_map
                        )

                    return left_result + right_result
            
            except Exception as e:
                tqdm.write(f"  ❌ P2P test error (right half): {e}")
                # Only keep left_result
                if left_result:
                    for temp_fpath, content in backup.items():
                        with open(temp_fpath, 'w', encoding='utf-8') as f:
                            f.write(content)
                    
                    self._batch_delete_objects_and_imports(
                        left_result,
                        temp_files_map,
                        import_map
                    )
                
                return left_result
    
    def _filter_public_api_objects(
        self,
        objects_to_remove: List[str]
    ) -> List[str]:
        """Filter out public APIs declared in __all__."""

        protected: List[str] = []
        remaining: List[str] = []

        for obj_id in objects_to_remove:
            parts = obj_id.split('::')
            if len(parts) < 3:
                remaining.append(obj_id)
                continue

            file_path = parts[0]
            obj_name = parts[1]

            if self._is_public_export(file_path, obj_name):
                protected.append(obj_id)
                self.logger.debug(
                    f"Object {obj_id} is declared in __all__ as public API; protected."
                )
            else:
                remaining.append(obj_id)

        if protected:
            pass
            # tqdm.write(f"{len(protected)} objects declared in module __all__; added to protection list")

        return remaining

    def _filter_extension_protected_objects(
        self, 
        objects_to_remove: List[str]
    ) -> List[str]:
        """
        Filter out objects referenced by extension modules (.pyx/.pxd/.pyi).

        Args:
            objects_to_remove: Object IDs to delete, e.g. ["/testbed/file.py::name::line", ...]

        Returns:
            Filtered deletable object list
        """
        protected = []
        can_remove = []
        
        for obj_id in objects_to_remove:
            # Parse object ID: /testbed/sklearn/utils/extmath.py::row_norms::60
            parts = obj_id.split('::')
            if len(parts) < 3:
                can_remove.append(obj_id)
                continue
            
            file_path = parts[0]  # /testbed/sklearn/utils/extmath.py
            obj_name = parts[1]   # row_norms
            
            # Convert file path to module path
            # /testbed/sklearn/utils/extmath.py -> sklearn.utils.extmath
            if file_path.startswith('/testbed/'):
                rel_path = file_path[len('/testbed/'):]  # sklearn/utils/extmath.py
                if rel_path.endswith('.py'):
                    rel_path = rel_path[:-3]  # sklearn/utils/extmath
                module_path = rel_path.replace('/', '.')  # sklearn.utils.extmath
            else:
                # Conversion failed; conservatively keep deletable
                tqdm.write(f"Failed to parse module path; skip protection check: {obj_id}")
                can_remove.append(obj_id)
                continue
            
            # Build module::name and consider re-exports from parent modules
            module_candidates = []
            if module_path:
                module_candidates.append(module_path)
                parts = module_path.split('.')
                # Include all parent modules to cover __init__.py re-exports
                for i in range(len(parts) - 1, 0, -1):
                    parent_module = '.'.join(parts[:i])
                    module_candidates.append(parent_module)
            else:
                module_candidates.append(module_path)

            found_protected = False
            for candidate_module in module_candidates:
                qualified_name = f"{candidate_module}::{obj_name}"
                # Check whether referenced by extension modules
                if qualified_name in self.extension_imports:
                    protected.append(obj_id)
                    referencing_files = self.extension_imports[qualified_name]
                    self.logger.debug(
                        f"Object {obj_id} is referenced by extension modules; protected (match {qualified_name})."
                        f"Referencing files: {', '.join(referencing_files)}"
                    )
                    found_protected = True
                    break

            if not found_protected:
                can_remove.append(obj_id)
        
        if protected:
            pass
            # tqdm.write(f'{len(protected)} objects referenced by extension modules added to protection list')
        
        return can_remove

    def _filter_critical_special_methods(
        self,
        objects_to_remove: List[str]
    ) -> List[str]:
        """Protect magic methods critical to abstract base class semantics."""

        critical_methods = {
            "__len__",
            "__iter__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
        }

        protected: List[str] = []
        remaining: List[str] = []

        for obj_id in objects_to_remove:
            parts = obj_id.split("::")
            if len(parts) < 2:
                remaining.append(obj_id)
                continue

            symbol = parts[1]
            method_name = symbol.split(".")[-1]

            if method_name in critical_methods:
                protected.append(obj_id)
            else:
                remaining.append(obj_id)

        if protected:
            pass
            # tqdm.write(f"{len(protected)} critical magic methods added to protection list")

        return remaining

    def _is_public_export(self, container_path: str, symbol: str) -> bool:
        """Check whether a symbol is declared in module __all__."""

        if container_path in self.module_exports_cache:
            return symbol in self.module_exports_cache[container_path]

        exports: Set[str] = set()
        try:
            host_path = self.repo_manager.convert_container_path_to_local(
                self.data_item.repo, container_path
            )
            with open(host_path, 'r', encoding='utf-8') as f:
                source = f.read()

            exports.update(self._extract_all_names(source))

        except Exception as exc:
            self.logger.debug(
                "Failed to parse __all__: %s (file: %s)", exc, container_path
            )

        self.module_exports_cache[container_path] = exports
        return symbol in exports

    @staticmethod
    def _extract_all_names(source: str) -> Set[str]:
        """Extract symbol names from __all__ in source code."""

        exports: Set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return exports

        for node in ast.walk(tree):
            # Handle "__all__ = [...]"
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                ):
                    names = MaskGenerator._literal_string_list(node.value)
                    exports.update(names)

            # Handle "__all__ += [...]"
            elif isinstance(node, ast.AugAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "__all__"
                    and isinstance(node.op, ast.Add)
                ):
                    names = MaskGenerator._literal_string_list(node.value)
                    exports.update(names)

            # Handle "__all__.extend([...])"
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "__all__"
                    and call.func.attr in {"extend", "append"}
                    and call.args
                ):
                    if call.func.attr == "append":
                        names = MaskGenerator._literal_string_list(call.args[0])
                    else:
                        names = MaskGenerator._literal_string_list(call.args[0])
                    exports.update(names)

        return exports

    @staticmethod
    def _literal_string_list(node: ast.AST) -> Set[str]:
        """Parse string constants from a node; return empty on complex cases."""

        values: Set[str] = set()
        try:
            evaluated = ast.literal_eval(node)
        except Exception:
            return values

        if isinstance(evaluated, str):
            values.add(evaluated)
        elif isinstance(evaluated, (list, tuple, set, frozenset)):
            for item in evaluated:
                if isinstance(item, str):
                    values.add(item)
        return values
    
    def _build_dependency_graph(self, objects_to_remove: List[str]) -> Dict[str, Set[str]]:
        """
        Build dependency graph between objects (based on dynamic trace edges).

        Args:
            objects_to_remove: Object IDs to delete

        Returns:
            Dependency graph: {obj_id: {objects that depend on it}}
            Example: {
                "/testbed/a.py::func_a::10": {"/testbed/b.py::func_b::20"},
                "/testbed/b.py::func_b::20": set()
            }
        """
        # Convert to set for fast lookup (O(1) vs O(n))
        objects_to_remove = set(objects_to_remove)
        
        # Read dependency relations from dynamic trace file
        with open(self.data_item.dynamic_trace_file, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        all_objects = trace_data.get('objects', {})
        
        # Build reverse dependency graph: obj_id -> {obj_ids that depend on it}
        # If A.edges contains B, then A depends on B
        dependency_graph = {obj_id: set() for obj_id in objects_to_remove}
        
        for obj_id, obj_info in all_objects.items():
            # Only consider objects to remove
            if obj_id not in objects_to_remove:
                continue
            
            # Traverse all dependencies (edges) of the object
            edges = obj_info.get('edges', [])
            for edge_id in edges:
                # If the dependent object is also in the removal list
                if edge_id in objects_to_remove:
                    # edge_id is depended on by obj_id
                    if edge_id not in dependency_graph:
                        dependency_graph[edge_id] = set()
                    dependency_graph[edge_id].add(obj_id)
        
        return dependency_graph
    
    def _get_leaf_objects(
        self,
        remaining_objects: List[str],
        dependency_graph: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Get leaf objects (not depended on by other objects to delete).

        Args:
            remaining_objects: Remaining objects to delete
            dependency_graph: Dependency graph

        Returns:
            List of leaf objects
        """
        remaining = set(remaining_objects)
        leaves = []
        
        for obj_id in remaining_objects:
            # Check whether any remaining object depends on it
            dependents = dependency_graph.get(obj_id, set())
            if not (dependents & remaining):  # no remaining objects depend on it
                leaves.append(obj_id)
        
        return leaves
    
    def _mask_single_object_in_temp_file(
        self,
        temp_file_path: str,
        file_path_container: str,
        obj_id: str
    ) -> None:
        """
        Delete a single object in a temp file.

        Args:
            temp_file_path: Temp file path (host path)
            file_path_container: Container path (for obj_id generation)
            obj_id: Object ID (format: "/testbed/file.py::Class.method::123")
        """
        # Read temp file
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Delete this object
        masked_code = self._apply_mask(
            source_code=source_code,
            file_path=file_path_container,
            objects_to_remove_ids={obj_id}
        )
        
        # Write back to temp file
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(masked_code)
    
    def _batch_delete_objects_and_imports(
        self,
        obj_ids: List[str],
        temp_files_map: Dict[str, str],
        import_map: Dict[str, List[Tuple[str, int, str, str]]]
    ) -> None:
        """
        Batch delete multiple objects and their import statements.

        Args:
            obj_ids: Object IDs to delete
            temp_files_map: Container path -> temp file path map
            import_map: Object -> import locations map
        """
        # Collect files to modify and objects to delete
        files_to_objects = {}  # {file_path_container: [obj_ids]}
        files_to_imports = {}  # {importer_file_container: [(line, name), ...]}
        
        for obj_id in obj_ids:
            parts = obj_id.split('::')
            file_path_container = parts[0]
            line_number = int(parts[2])
            
            # Collect object file
            if file_path_container not in files_to_objects:
                files_to_objects[file_path_container] = []
            files_to_objects[file_path_container].append(obj_id)
            
            # Collect import info
            obj_key = f"{file_path_container}::{line_number}"
            import_locations = import_map.get(obj_key, [])
            
            for importer_file, importer_line, imported_name, _ in import_locations:
                if importer_file not in files_to_imports:
                    files_to_imports[importer_file] = []
                files_to_imports[importer_file].append((importer_line, imported_name))
        
        # 1. Delete all objects
        for file_path_container, obj_ids_in_file in files_to_objects.items():
            temp_file_path = temp_files_map.get(file_path_container)
            if not temp_file_path:
                continue
            
            # Read temp file
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Delete all objects in this file
            masked_code = self._apply_mask(
                source_code=source_code,
                file_path=file_path_container,
                objects_to_remove_ids=set(obj_ids_in_file)
            )
            
            # Write back to temp file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(masked_code)
        
        # 2. Delete all import statements
        for importer_file, imports in files_to_imports.items():
            temp_importer_path = temp_files_map.get(importer_file)
            if not temp_importer_path:
                continue
            
            # Sort by line number descending (delete from bottom to top)
            imports_sorted = sorted(set(imports), key=lambda x: x[0], reverse=True)
            
            for importer_line, imported_name in imports_sorted:
                self._delete_import_from_line(
                    temp_importer_path,
                    importer_line,
                    imported_name
                )
    
    def _binary_search_failed_objects(
        self,
        failed_batch: List[str],
        container_id: str,
        temp_files_map: Dict[str, str],
        import_map: Dict[str, List[Tuple[str, int, str, str]]],
        backup: Dict[str, str],
        remaining_objects: List[str]
    ) -> List[str]:
        """
        Use binary search to find which objects can be deleted from a failed batch.

        Optimization: if deletion fails, extract error symbols from logs and protect
        those objects to avoid re-testing known failures in subsequent splits.

        Args:
            failed_batch: Objects that failed batch deletion
            container_id: Container ID
            temp_files_map: Temp file map
            import_map: Import map
            backup: Backup temp file contents
            remaining_objects: Remaining objects to delete (for dynamic protection updates)

        Returns:
            List of successfully deleted objects
        """
        if len(failed_batch) == 0:
            return []
        
        if len(failed_batch) == 1:
            # Only one object left, cannot split further; it cannot be deleted
            # Restore backup to avoid unverified deletions
            for temp_fpath, content in backup.items():
                with open(temp_fpath, 'w', encoding='utf-8') as f:
                    f.write(content)
            # tqdm.write(f"  ⚠️ Cannot delete object: {failed_batch[0]}")
            return []
        
        # Split into halves
        mid = len(failed_batch) // 2
        left_batch = failed_batch[:mid]
        right_batch = failed_batch[mid:]
        
        # tqdm.write(f"  🔍 Binary test: left half {len(left_batch)} objects")
        
        # Test left half
        # 1. Restore backup
        for temp_fpath, content in backup.items():
            with open(temp_fpath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 2. Delete left half
        try:
            # Reset container env
            self.image_manager.reset_container_env(
                container_id=container_id,
                specs_name=self.data_item.repo,
                timeout=60
            )

            # Delete left half
            self._batch_delete_objects_and_imports(
                left_batch,
                temp_files_map,
                import_map
            )
            
            # Copy files to container
            self.image_manager.copy_to_container(
                container_id=container_id,
                src_path=None,
                dest_path=None,
                use_tar=True,
                files_mapping=temp_files_map
            )
            
            # Run pytest --collect validation
            left_success, left_log_path = self._run_pytest_collect(
                container_id,
                self.data_item.file_path
            )
            
            # If left half failed, try extracting error symbols and protect them
            if not left_success:
                error_symbols = self._extract_error_symbols_from_log(left_log_path)
                if error_symbols:
                    protected_objects = []
                    for symbol in error_symbols:
                        matched_objs = self._find_objects_by_symbol_name(left_batch, symbol)
                        protected_objects.extend(matched_objs)
                    
                    if protected_objects:
                        # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from left-half log")
                        # Remove protected objects from remaining_objects and left_batch
                        for obj_id in protected_objects:
                            if obj_id in remaining_objects:
                                remaining_objects.remove(obj_id)
                        left_batch = [obj for obj in left_batch if obj not in protected_objects]
            
        except Exception as e:
            tqdm.write(f"  ❌ Left-half test exception: {e}")
            left_success = False
        
        # Decide strategy based on left-half result
        if left_success:
            # Left half succeeded; continue with right half
            # tqdm.write(f"  ✅ Left half succeeded; continue testing right half {len(right_batch)} objects")
            
            # Delete right half on top of left-half deletions
            try:
                self.image_manager.reset_container_env(
                    container_id=container_id,
                    specs_name=self.data_item.repo,
                    timeout=60
                )

                self._batch_delete_objects_and_imports(
                    right_batch,
                    temp_files_map,
                    import_map
                )
                
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=None,
                    dest_path=None,
                    use_tar=True,
                    files_mapping=temp_files_map
                )
                
                right_success, right_log_path = self._run_pytest_collect(
                    container_id,
                    self.data_item.file_path
                )
                
                if right_success:
                    # Both halves succeeded
                    return left_batch + right_batch
                else:
                    # Right half failed; try extracting error symbols
                    error_symbols = self._extract_error_symbols_from_log(right_log_path)
                    if error_symbols:
                        protected_objects = []
                        for symbol in error_symbols:
                            matched_objs = self._find_objects_by_symbol_name(right_batch, symbol)
                            protected_objects.extend(matched_objs)
                        
                        if protected_objects:
                            # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from right-half logs")
                            # Remove protected objects from remaining_objects and right_batch
                            for obj_id in protected_objects:
                                if obj_id in remaining_objects:
                                    remaining_objects.remove(obj_id)
                            right_batch = [obj for obj in right_batch if obj not in protected_objects]
                    
                    # Left succeeded, right failed -> recurse on right half
                    right_result = self._binary_search_failed_objects(
                        right_batch,
                        container_id,
                        temp_files_map,
                        import_map,
                        backup,
                        remaining_objects
                    )

                    if left_batch or right_result:
                        for temp_fpath, content in backup.items():
                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(content)

                        self._batch_delete_objects_and_imports(
                            left_batch + right_result,
                            temp_files_map,
                            import_map
                        )

                    return left_batch + right_result
            
            except Exception as e:
                tqdm.write(f"  ❌ Right-half test exception: {e}")
                # Roll back and re-apply left half to keep state consistent
                for temp_fpath, content in backup.items():
                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                        f.write(content)

                if left_batch:
                    self._batch_delete_objects_and_imports(
                        left_batch,
                        temp_files_map,
                        import_map
                    )

                return left_batch
        
        else:
            # Left half failed; recurse on left, then test right
            # tqdm.write("  ❌ Left half failed, recurse")
            left_result = self._binary_search_failed_objects(
                left_batch,
                container_id,
                temp_files_map,
                import_map,
                backup,
                remaining_objects
            )
            
            # Re-apply left_result deletions after recursion
            if left_result:
                for temp_fpath, content in backup.items():
                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                self._batch_delete_objects_and_imports(
                    left_result,
                    temp_files_map,
                    import_map
                )
            
            # Test right half
            # tqdm.write(f"  🔍 Test right half {len(right_batch)} objects")
            
            try:
                # Continue deleting right half
                self._batch_delete_objects_and_imports(
                    right_batch,
                    temp_files_map,
                    import_map
                )
                
                self.image_manager.reset_container_env(
                    container_id=container_id,
                    specs_name=self.data_item.repo,
                    timeout=60
                )
                
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=None,
                    dest_path=None,
                    use_tar=True,
                    files_mapping=temp_files_map
                )
                
                combined_success, combined_log_path = self._run_pytest_collect(
                    container_id,
                    self.data_item.file_path
                )
                
                if combined_success:
                    # Both halves succeeded
                    return left_result + right_batch
                else:
                    # Right half failed; try extracting error symbols
                    error_symbols = self._extract_error_symbols_from_log(combined_log_path)
                    if error_symbols:
                        protected_objects = []
                        for symbol in error_symbols:
                            matched_objs = self._find_objects_by_symbol_name(right_batch, symbol)
                            protected_objects.extend(matched_objs)
                        
                        if protected_objects:
                            # tqdm.write(f"  🛡️ Protected {len(protected_objects)} objects from right-half logs")
                            # Remove protected objects from remaining_objects and right_batch
                            for obj_id in protected_objects:
                                if obj_id in remaining_objects:
                                    remaining_objects.remove(obj_id)
                            right_batch = [obj for obj in right_batch if obj not in protected_objects]
                    
                    # Left succeeded, right failed -> recurse on right half
                    # Restore to state with only left_result deleted
                    for temp_fpath, content in backup.items():
                        with open(temp_fpath, 'w', encoding='utf-8') as f:
                            f.write(content)
                    
                    if left_result:
                        self._batch_delete_objects_and_imports(
                            left_result,
                            temp_files_map,
                            import_map
                        )
                    
                    right_result = self._binary_search_failed_objects(
                        right_batch,
                        container_id,
                        temp_files_map,
                        import_map,
                        backup,
                        remaining_objects
                    )
                    
                    # Re-apply full results
                    if left_result or right_result:
                        for temp_fpath, content in backup.items():
                            with open(temp_fpath, 'w', encoding='utf-8') as f:
                                f.write(content)
                        
                        self._batch_delete_objects_and_imports(
                            left_result + right_result,
                            temp_files_map,
                            import_map
                        )
                    
                    return left_result + right_result
            
            except Exception as e:
                tqdm.write(f"  ❌ Combined test exception: {e}")
                # Restore backup
                for temp_fpath, content in backup.items():
                    with open(temp_fpath, 'w', encoding='utf-8') as f:
                        f.write(content)
                # Re-apply left_result
                if left_result:
                    self._batch_delete_objects_and_imports(
                        left_result,
                        temp_files_map,
                        import_map
                    )
                return left_result
