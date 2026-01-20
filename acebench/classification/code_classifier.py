from typing import Dict, Set, Optional, Any, Tuple, List
import logging
import ast
from collections import defaultdict, Counter
from pathlib import Path
import os
import random
import json
from collections import deque
from copy import deepcopy
from acebench.utils.config import Config
from acebench.utils.repo_manager import RepoManager
from acebench.utils.storage import StorageManager
from acebench.classification.file_analyzer import FunctionClassVisitor, analyze_dependencies, build_class_info
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CodeClassifier:
    """Main code classifier."""
    
    def __init__(
        self, 
        config: Config,
        repo_manager: RepoManager,
        storage_manager: StorageManager,
        data_item,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.data_item = data_item
        self.logger = logger

        # Collect all dynamic trace files for current item (f2p + p2p)
        self._trace_files: List[str] = self._gather_trace_files()

        self.test_file = data_item.file_path.split('/')[-1]     # e.g. test_model.py
        self.test_name = self.test_file.split('.')[0].lstrip('test_')   # e.g. model
        
        # Load dynamic trace data
        self.objects = self._get_all_objects(data_item.dynamic_trace_file)
        # Drop invalid objects
        self.objects = self._discard_invalid_objects(self.objects)
        # Set is_pass2pass for all objects
        self.objects = self._get_p2p_properties(self.objects)

        # Get top objects
        self.top_objects = deepcopy(data_item.updated_top_objects)
        self.all_top_objects_candidates = deepcopy(data_item.all_top_objects_candidates)

        # Load objects that should not be deleted (protected)
        self.collect_objects = self._get_collect_objects()

        # Get all_obj_info; start/end lines are per-object ranges
        # e.g. {'/testbed/model.py::MyModel::10': ((10, 45), 'Class')}
        self.all_obj_info: Dict[str, Tuple[Tuple[int, int], str]] = self._get_all_obj_info(self.objects)
        # Read code line count limit range
        self.max_line_code = random.randint(int(data_item.specs.get('max_code_line_lower_bound', '3000')), int(data_item.specs.get('max_code_line_upper_bound', '5000')))

        # Code line count stats
        self.lines_code_level1: int = 0
        self.lines_code_level2: int = 0
        # Classification results
        # e.g. {'/testbed/model.py': 'left'/'patch'}
        self.file_classifier: Dict[str, str] = defaultdict(lambda: "left")
        # e.g. {'/testbed/model.py::MyModel::10': ('Class', 'top'/'specific'/'others')}
        self.obj_classifier: Dict[str, Tuple[str, str]] = defaultdict(lambda: ("Func", "others"))

    def run(self):
        """Run the code classification flow."""
        try:
            # tqdm.write(f"🏃 Start code classification: {self.data_item.repo} - {self.test_file}")
            
            # Run code classification
            self.obj_classifier,self.file_classifier, self.lines_code_level1, self.lines_code_level2 = self._classify_with_bfs(self.objects, self.top_objects, self.all_obj_info)
            
            # Promote specific objects to top
            self._update_top_objects_by_candidates()

            # Write classification results
            self.summary = self.get_classification_summary()
            code_classification_result_path = self.storage_manager.create_code_classification_result_path(self.data_item.repo, self.data_item.file_path)
            with open(code_classification_result_path, 'w', encoding='utf-8') as f:
                json.dump(self.summary, f, ensure_ascii=False, indent=4)
            
            # Output classification summary
            # tqdm.write(f"✅ {self.data_item.repo}: Code classification complete - {self.test_file}")

            patch_files_count = self.summary['file_stats']['patch_count']
            top_objects_count = self.summary['obj_stats']['top_count']
            specific_objects_count = self.summary['obj_stats']['specific_count']
            
            # tqdm.write(f"   Patch files: {patch_files_count}, Top objects: {top_objects_count}, Specific objects: {specific_objects_count}")
            # tqdm.write(f"   Code lines Level1: {self.lines_code_level1}, Level2: {self.lines_code_level2}")
            
            return self.summary
            
        except Exception as e:
            error_msg = f"Code classification failed - {self.test_file}: {e}"
            tqdm.write(f"❌ {self.data_item.repo}: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _get_host_path(self, file_path: str) -> str:
        """Get local file path without reusing self.data_item.repo."""
        return self.repo_manager.convert_container_path_to_local(self.data_item.repo, file_path)

    def _is_repo_file(self, file_path: str) -> bool:
        """Check whether a container path belongs to current repo."""
        if not file_path:
            return False
        try:
            host_path = self._get_host_path(file_path)
        except Exception:
            return False
        return os.path.exists(host_path)

    def _get_p2p_properties(self, objects: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Set is_pass2pass for objects in self.objects."""
        p2p_objects_set = set()
        p2p_dynamic_trace_file_list = self.data_item.dynamic_trace_files
        for p2p_dynamic_trace_file in p2p_dynamic_trace_file_list:
            with open(p2p_dynamic_trace_file, 'r', encoding='utf-8') as f:
                p2p_dynamic_trace = json.load(f)
            for obj_id, obj_info in p2p_dynamic_trace.get('objects', {}).items():
                p2p_objects_set.add(obj_id)
        new_objects = deepcopy(objects)
        for obj_id, obj_info in new_objects.items():
            if obj_id in p2p_objects_set:
                new_objects[obj_id]['is_pass2pass'] = True
        return new_objects

    def _update_top_objects_by_candidates(self):
        """Update top objects based on candidate set."""
        old_obj_classifier = deepcopy(self.obj_classifier)
        new_obj_classifier = deepcopy(self.obj_classifier)
        for obj_id, obj_class in old_obj_classifier.items():
            if obj_class[1] == "specific" and obj_id in self.all_top_objects_candidates:
                new_obj_classifier[obj_id] = (obj_class[0], "top")
                logger.debug(f"Promote specific object {obj_id} to top")
        self.obj_classifier = new_obj_classifier

    def _classify_with_bfs(
        self,
        objects: Dict[str, Dict[str, Any]], # All dynamically traced objects
        top_objects: Set[str],  # Set of top IDs from dynamic trace
        all_obj_info: Dict[str, Tuple[Tuple[int, int], str]]    # Line range, type
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str], int, int]:

        # FIXME: lv2 logic is similar to lv1; consider merging to simplify?
        # lv2 processing logic
        obj_classifier_lv2, file_classifier_lv2, lines_code_level2 = self._get_lines_code_level2(objects, top_objects, all_obj_info, self.test_name, float('inf'), self.test_file)

        obj_classifier = defaultdict(lambda: ("Func", "others"))
        file_classifier = defaultdict(lambda: "left")
        
        # Initialize all classification results
        for obj_id, obj_info in objects.items():
            obj_classifier[obj_id] = ("Func", "others")
            file_classifier[obj_info['file']] = "left"

        # Classify all top objects and accumulate total LOC
        # Note: if a top object is also pass2pass, it should not be masked; skip
        # Note: if a top object is traced in collect stage, it should not be masked; skip
        lines_code_level1 = 0
        for obj_id in top_objects:
            # Skip pass2pass objects
            if objects[obj_id].get('is_pass2pass', False):
                continue
            # Skip protected objects
            if obj_id in self.collect_objects:
                continue
            obj_classifier[obj_id] = (all_obj_info[obj_id][1], "top")
            lines_code_level1 += all_obj_info[obj_id][0][1] - all_obj_info[obj_id][0][0]
        
        # Use BFS to classify all specific objects
        bfs_queue = deque(top_objects)
        have_visted = set(top_objects)
        while bfs_queue:
            obj_id = bfs_queue.popleft()
            obj_info = objects[obj_id]

            for child_id in obj_info['edges']:
                # Skip if already visited
                if child_id in have_visted:
                    continue
                # If pass2pass, enqueue for traversal and mark visited
                elif objects[child_id]['is_pass2pass']:
                    bfs_queue.append(child_id)
                    have_visted.add(child_id)
                # If protected, enqueue for traversal and mark visited
                elif child_id in self.collect_objects:
                    bfs_queue.append(child_id)
                    have_visted.add(child_id)
                # If not pass2pass, not in collect, and classified as specific, enqueue and mark visited
                elif is_specific_obj(child_id, all_obj_info, self.test_name, lines_code_level1, self.max_line_code, self.test_file):
                    lines_code_level1 += all_obj_info[child_id][0][1] - all_obj_info[child_id][0][0]
                    bfs_queue.append(child_id)
                    have_visted.add(child_id)
                    obj_classifier[child_id] = (all_obj_info[child_id][1], "specific")
                # If not pass2pass and classified as others, do not enqueue
                else:
                    have_visted.add(child_id)

        # Mark files containing specific/top objects as patch
        for obj_id, obj_info in objects.items():
            if obj_classifier[obj_id][1] == "specific" or obj_classifier[obj_id][1] == "top":
                file_classifier[obj_info['file']] = "patch"

        # Downgrade: if a specific is a prefix of a top (same file, ignore line), cancel the specific
        obj_classifier, file_classifier, lines_code_level1 = self._downgrade_specific_prefix_of_top(
            obj_classifier,
            file_classifier,
            all_obj_info,
            lines_code_level1,
        )

        # NOTE: removed; new dynamic trace can patch to handle untracked third-party calls
        # Post-processing safety: if same-file dependencies of specific/top are others, promote to specific

        # # File path set
        # all_files = set()
        # # Mapping from file path to object list
        # # e.g. {'/testbed/model.py': ['MyModel::10', 'forward::15', '_helper::20']}
        # file_obj_dict = defaultdict(list)
        # for obj_id in all_obj_info:
        #     all_files.add(obj_id.split('::')[0])
        #     file_obj_dict[obj_id.split('::')[0]].append('::'.join(obj_id.split('::')[1:]))

        # # Keep scanning until no new objects are promoted to specific
        # changed = True
        # while changed:
        #     changed = False
        #     for file in all_files:
        #         if file_classifier[file] == "left":
        #             continue
        #         # Traverse all classes and functions
        #         all_objs = file_obj_dict[file]
        #         for obj in all_objs:
        #             # Only handle specific or top objects
        #             if obj_classifier[f'{file}::{obj}'][1] in ("specific", "top"):
        #                 # Get dependency list for this object
        #                 host_file = self._get_host_path(file)
        #                 dependencies = analyze_dependencies(obj, host_file)

        #                 # Process dependencies and decide whether to promote to specific
        #                 for dep_obj in dependencies:
        #                     # Compose id (missing line number)
        #                     dep_key = f'{file}::{dep_obj}'
        #                     # Check if dependency is pass2pass
        #                     is_dep_pass2pass = False
        #                     for obj_id, obj_info in objects.items():
        #                         if dep_key == obj_id:
        #                             if obj_info['is_pass2pass']:
        #                                 is_dep_pass2pass = True
        #                                 break
        #                     # If dependency is pass2pass, do not classify as specific
        #                     if is_dep_pass2pass:
        #                         continue
        #                     # If dependency exists and is others, promote to specific
        #                     if dep_obj in all_objs and obj_classifier.get(dep_key, ("Func", "others"))[1] == "others":
        #                         obj_classifier[dep_key] = (all_obj_info[dep_key][1], "specific")
        #                         changed = True

        return obj_classifier, file_classifier, lines_code_level1, lines_code_level2

    def _downgrade_specific_prefix_of_top(
        self,
        obj_classifier: Dict[str, Tuple[str, str]],
        file_classifier: Dict[str, str],
        all_obj_info: Dict[str, Tuple[Tuple[int, int], str]],
        lines_code_level1: int,
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str], int]:
        """If a specific is a prefix of a top in the same file, downgrade to others.

        Example: specific=/file.py::A::10, top=/file.py::A.B::20 -> specific is downgraded.
        Line numbers only distinguish objects and are not used for prefix checks.
        """

        # Collect top qualified names by file (remove line numbers)
        top_prefixes: Dict[str, Set[str]] = defaultdict(set)  # Per-file top qualified names (no line)
        for obj_id, (_obj_type, cls) in obj_classifier.items():  # Traverse all classifications
            if cls != "top":  # Only top objects
                continue
            parts = obj_id.split("::")  # Split into path, qualified name, line
            if len(parts) < 3:  # Skip invalid format
                continue
            file_path = parts[0]  # File path
            qual_name = "::".join(parts[1:-1])  # Qualified name without line
            if qual_name:  # Only add non-empty
                top_prefixes[file_path].add(qual_name)  # Record per-file top qualified name

        downgraded: Set[str] = set()  # Track downgraded specifics
        updated_classifier = dict(obj_classifier)  # Copy for modification

        # Find specifics to downgrade
        for obj_id, (obj_type, cls) in obj_classifier.items():  # Traverse all objects
            if cls != "specific":  # Only specifics
                continue
            parts = obj_id.split("::")  # Split path/qualified name/line
            if len(parts) < 3:  # Skip invalid format
                continue
            file_path = parts[0]  # File containing object
            qual_name = "::".join(parts[1:-1])  # Qualified name without line
            if not qual_name:  # Skip empty qualified name
                continue
            for top_qual in top_prefixes.get(file_path, ()):  # All top qualified names in file
                if top_qual.startswith(qual_name):  # If specific is a prefix of top
                    downgraded.add(obj_id)  # Record downgrade
                    updated_classifier[obj_id] = (obj_type, "others")  # Set to others
                    break  # One hit is enough

        if not downgraded:  # No downgrade; return original result
            return obj_classifier, file_classifier, lines_code_level1

        # Recompute file_classifier (remove patch marks after downgrades)
        new_file_classifier: Dict[str, str] = defaultdict(lambda: "left")  # New file classification
        for obj_id, (_obj_type, cls) in updated_classifier.items():  # Traverse updated classification
            file_path = obj_id.split("::")[0]  # File path
            if cls in ("top", "specific"):  # Mark patch if top/specific exists
                new_file_classifier[file_path] = "patch"
            else:
                new_file_classifier.setdefault(file_path, "left")  # Keep default left

        # Recalculate level1 LOC (top/specific only)
        new_lines_code_level1 = 0  # Reset counter
        for obj_id, (_obj_type, cls) in updated_classifier.items():  # Traverse all objects
            if cls not in ("top", "specific"):  # Count top/specific only
                continue
            if obj_id not in all_obj_info:
                continue
            span = all_obj_info[obj_id][0]  # Get line range tuple
            new_lines_code_level1 += span[1] - span[0]  # Accumulate lines

        return updated_classifier, new_file_classifier, new_lines_code_level1  # Return updated results


    def _get_lines_code_level2(self, objects: Dict[str, Dict[str, Any]], top_objects: Set[str], all_obj_info: Dict[str, Tuple[Tuple[int, int], str]], test_name: str, max_line_code: int, test_file: str) -> int:

        obj_classifier = defaultdict(lambda: ("Func", "others"))
        file_classifier = defaultdict(lambda: "left")
        
        """Classify all specific objects via BFS."""
        # Initialize all classification results
        for obj_id, obj_info in objects.items():
            obj_classifier[obj_id] = ("Func", "others")
            file_classifier[obj_info['file']] = "left"

        # Classify all top objects and record total LOC
        lines_code_level2 = 0
        for obj_id in top_objects:
            obj_classifier[obj_id] = (all_obj_info[obj_id][1], "top")
            lines_code_level2 += all_obj_info[obj_id][0][1] - all_obj_info[obj_id][0][0]
        
        # Use BFS to classify all specific objects
        bfs_queue = deque(top_objects)
        have_visted = set(top_objects)
        while bfs_queue:
            obj_id = bfs_queue.popleft()
            obj_info = objects[obj_id]

            for child_id in obj_info['edges']:
                # Skip if already visited
                if child_id in have_visted:
                    continue
                # If classified as specific, enqueue and mark visited (ignore pass2pass here)
                elif is_specific_obj(child_id, all_obj_info, test_name, lines_code_level2, max_line_code, test_file):
                    lines_code_level2 += all_obj_info[child_id][0][1] - all_obj_info[child_id][0][0]
                    bfs_queue.append(child_id)
                    have_visted.add(child_id)
                    obj_classifier[child_id] = (all_obj_info[child_id][1], "specific")
                # If classified as others, do not enqueue
                else:
                    have_visted.add(child_id)

        # Mark files containing specific/top as patch
        for obj_id, obj_info in objects.items():
            if obj_classifier[obj_id][1] == "specific" or obj_classifier[obj_id][1] == "top":
                file_classifier[obj_info['file']] = "patch"

        # NOTE: removed; new dynamic trace can patch to handle untracked third-party calls
        # If same-file dependencies of specific/top are others, promote to specific
        # all_files = set()
        # file_obj_dict = defaultdict(list)
        # for obj_id in all_obj_info:
        #     all_files.add(obj_id.split('::')[0])
        #     file_obj_dict[obj_id.split('::')[0]].append('::'.join(obj_id.split('::')[1:]))

        # changed = True
        # while changed:
        #     changed = False
        #     for file in all_files:
        #         if file_classifier[file] == "left":
        #             continue
        #         # Traverse all classes and functions
        #         all_objs = file_obj_dict[file]
        #         for obj in all_objs:
        #             # Only handle specific or top objects
        #             if obj_classifier[f'{file}::{obj}'][1] in ("specific", "top"):
        #                 # Get dependency list for the object
        #                 host_file = self._get_host_path(file)
        #                 dependencies = analyze_dependencies(obj, host_file)

        #                 for dep_obj in dependencies:
        #                     dep_key = f'{file}::{dep_obj}'
        #                     # If dependency exists and is others, promote to specific
        #                     if dep_obj in all_objs and obj_classifier.get(dep_key, ("Func", "others"))[1] == "others":
        #                         obj_classifier[dep_key] = (all_obj_info[dep_key][1], "specific")
        #                         changed = True

        return obj_classifier, file_classifier, lines_code_level2

    def _get_all_objects(self, dynamic_trace_file: str) -> Dict[str, Dict[str, Any]]:
        """Get all objects from dynamic trace files."""
        with open(dynamic_trace_file, 'r', encoding='utf-8') as f:
            dynamic_trace = json.load(f)
        return dynamic_trace.get('objects', {})

    def _gather_trace_files(self) -> List[str]:
        """Collect dynamic trace paths for this item (f2p + p2p)."""
        trace_files: List[str] = []
        seen: Set[str] = set()

        primary_trace = getattr(self.data_item, 'dynamic_trace_file', None)
        if primary_trace:
            seen.add(primary_trace)
            trace_files.append(primary_trace)
            if not os.path.exists(primary_trace) and self.logger:
                self.logger.warning("Dynamic trace file not found: %s", primary_trace)

        extra_traces = getattr(self.data_item, 'dynamic_trace_files', None) or []
        for extra_trace in extra_traces:
            if not extra_trace or extra_trace in seen:
                continue
            seen.add(extra_trace)
            trace_files.append(extra_trace)
            if not os.path.exists(extra_trace) and self.logger:
                self.logger.warning("Dynamic trace file not found: %s", extra_trace)

        return trace_files

    def _get_collect_objects(self) -> Set[str]:
        collect_objects: Set[str] = set()
        for trace_file in self._trace_files:
            collect_objects.update(self._collect_objects_from_trace(trace_file))
        return collect_objects
    
    def _collect_objects_from_trace(self, dynamic_trace_file: str) -> Set[str]:
        """
        Get all object IDs from the collect-phase dynamic trace file.

        These objects are used during pytest --collect-only; removing them
        would cause NameError, so they cannot be classified as top/specific.

        Special handling for module-level imports:
        - If collect imports a module object (e.g., from ..common import _aliases)
        - Use AST to analyze the importer file and find top-level attribute access
          of that module (e.g., _aliases.clip)
        - Protect those top-level accessed attributes

        Args:
            dynamic_trace_file: Path to the normal dynamic trace file.

        Returns:
            Set[str]: Set of object IDs traced during collect phase.
        """
        try:
            # Derive collect trace file path from normal trace path
            trace_path = Path(dynamic_trace_file)
            collect_trace_file = trace_path.parent / f"{trace_path.stem}_collect{trace_path.suffix}"
            
            if not collect_trace_file.exists():
                # Return empty set if collect trace file is missing
                return set()
            
            with open(collect_trace_file, 'r', encoding='utf-8') as f:
                collect_trace = json.load(f)
            
            # Get directly traced objects
            collect_objects = set(collect_trace.get('objects', {}).keys())
            
            # F2P collect results likely contain true top, so ignore f2p pytest --collect
            primary_trace = getattr(self.data_item, 'dynamic_trace_file', None)
            if primary_trace:
                current_trace = os.path.normpath(str(dynamic_trace_file))
                primary_trace_norm = os.path.normpath(str(primary_trace))
                if current_trace == primary_trace_norm:
                    collect_objects = set()

            # Handle module-level imports (e.g., from ..common import _aliases)
            # Use AST to find which attributes are accessed
            imports = collect_trace.get('imports', [])

            module_imports = {}  # {importer_file: {module_alias: module_file}}
            symbol_imports: Dict[str, Dict[str, List[Tuple[str, str, int]]]] = defaultdict(lambda: defaultdict(list))
            # {importer_file: {symbol_alias: [(defined_file, defined_name, defined_line), ...]}}

            reexport_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
            
            imports_by_importer = defaultdict(list)

            for imp in imports:
                defined_file = imp.get('defined_file', '')
                defined_line = imp.get('defined_line')
                imported_name = imp.get('imported_name', '')
                importer_file = imp.get('importer_file', '')
                defined_name = imp.get('defined_name') or imported_name

                if not imported_name or not importer_file or not defined_file:
                    continue

                if not self._is_repo_file(defined_file):
                    continue

                imports_by_importer[importer_file].append(imp)

                # If defined_line == -1, this is an import of a module object
                if defined_line == -1:
                    if importer_file not in module_imports:
                        module_imports[importer_file] = {}
                    module_imports[importer_file][imported_name] = defined_file
                else:
                    # Record re-export mapping for recursive resolution of real definitions
                    reexport_map[importer_file][imported_name].append((defined_file, defined_name))
                    symbol_imports[importer_file][imported_name].append((defined_file, defined_name, defined_line))

            # Run AST analysis for files that import modules or external objects
            if module_imports or symbol_imports:
                accessed_objects = self._analyze_module_attribute_access(module_imports, reexport_map, symbol_imports)
                collect_objects.update(accessed_objects)

            # Collect all files to analyze (importers + imported)
            all_files = set()
            # Importer files
            all_files.update(imports_by_importer.keys())
            # Imported files
            for imp in imports:
                defined_file = imp.get('defined_file', '')
                if defined_file and self._is_repo_file(defined_file):
                    all_files.add(defined_file)
            
            # Protect objects used in top-level statements during collect phase
            if all_files:
                accessed_names = self._analyze_top_level_name_references(imports_by_importer, all_files)
                collect_objects.update(accessed_names)
            
            # If an abstract base class has @abstract methods, protect those methods in subclasses
            collect_objects = self._expand_collect_class_methods(collect_objects)

            return collect_objects
            
        except Exception as e:
            # Return empty set on read failure; do not affect main flow
            if self.logger:
                self.logger.warning(f"Failed to read collect trace file: {e}")
            return set()

    def _expand_collect_class_methods(self, collect_objects: Set[str]) -> Set[str]:
        """Expand collect classes and add methods to avoid missing abstract methods."""

        expanded: Set[str] = set(collect_objects)
        definitions_cache: Dict[str, Dict[str, Tuple[int, int, str]]] = {}
        class_info_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for obj_id in list(collect_objects):
            parts = obj_id.split("::")
            if len(parts) < 3:
                continue

            file_path, class_name = parts[0], parts[1]

            # Skip if already a method or not a repo file
            if "." in class_name or not self._is_repo_file(file_path):
                continue

            try:
                host_path = self._get_host_path(file_path)
            except Exception:
                continue

            # Build file AST info cache
            if host_path not in definitions_cache:
                try:
                    definitions_cache[host_path] = self._analyze_single_file(host_path) or {}
                except Exception:
                    definitions_cache[host_path] = {}

            # Build file class info cache
            if host_path not in class_info_cache:
                class_info_cache[host_path] = build_class_info(host_path)

            definitions = definitions_cache[host_path]
            if not definitions:
                continue
            
            # Select methods to keep in this class
            required_methods = self._select_required_methods(class_name, class_info_cache[host_path])

            # Traverse all objects in this file and match this class
            prefix = f"{class_name}."
            for def_key, meta in definitions.items():
                if not def_key.startswith(prefix):
                    continue
                # Only add function/method definitions
                if len(meta) >= 3 and meta[2] != "Func":
                    continue
                simple_method = def_key.split("::")[0].split(".")[-1]
                if required_methods is not None and simple_method not in required_methods:
                    continue

                expanded.add(f"{file_path}::{def_key}")

        return expanded
    
    def _select_required_methods(
        self,
        class_name: str,
        class_info: Dict[str, Dict[str, Any]],
    ) -> Optional[Set[str]]:
        """Compute methods to keep for a class; None means keep all."""

        # Cache abstract methods across class inheritance chains
        memo: Dict[str, Optional[Set[str]]] = {}
        visiting: Set[str] = set()

        def dfs(name: str) -> Optional[Set[str]]:
            if name in memo:
                return memo[name]

            if name in visiting:
                memo[name] = None
                return None

            if name not in class_info:
                memo[name] = None
                return None

            visiting.add(name)
            try:
                info = class_info[name]
                # Collect abstract methods for this class
                required: Set[str] = set(info.get("abstract_methods", set()))

                # Traverse all base classes and include their abstract methods
                for base in info.get("bases", []):
                    base_required = dfs(base)
                    if base_required is None:
                        memo[name] = None
                        return None
                    required.update(base_required)

                memo[name] = required
                return required
            finally:
                visiting.discard(name)

        # Get all @abstract methods across the class inheritance chain
        required_methods = dfs(class_name)
        if required_methods is None:
            return None

        # Get all methods defined within class_name
        defined = class_info.get(class_name, {}).get("defined_methods", set())
        
        # Intersect the two sets and keep them
        return required_methods.intersection(defined)
    
    def _analyze_module_attribute_access(
        self,
        module_imports: Dict[str, Dict[str, str]],
        reexport_map: Dict[str, Dict[str, List[Tuple[str, str]]]],
        symbol_imports: Dict[str, Dict[str, List[Tuple[str, str, int]]]],
    ) -> Set[str]:
        """
        Use AST to analyze module attribute access and find concrete attributes
        accessed via module objects or symbol aliases.

        Example:
        - from ..common import _aliases
        - clip = get_xp(np)(_aliases.clip)  # module top-level, executed on import

        Detect _aliases.clip and mark common/_aliases.py::clip as used.

        Note: Only track top-level attribute access because those execute on import.
        Access inside functions or conditionals is not protected (may not execute).

        Args:
            module_imports: {importer_file: {module_alias: module_file}}
            reexport_map: {module_file: {exported_name: [(target_file, target_name), ...]}}
            symbol_imports: {importer_file: {symbol_alias: [(defined_file, defined_name, defined_line), ...]}}

        Returns:
            Set[str]: Set of accessed object IDs.
        """
        accessed_objects = set()
        module_definition_cache: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        symbol_attr_cache: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
        file_definition_cache: Dict[str, Dict[str, Tuple[int, int, str]]] = {}

        # For each file that imports module objects or symbol aliases
        all_importers = set(module_imports.keys()) | set(symbol_imports.keys())
        for importer_file in all_importers:
            module_aliases = module_imports.get(importer_file, {})
            symbol_aliases = symbol_imports.get(importer_file, {})

            alias_info: Dict[str, Tuple[str, Any]] = {}

            for alias, module_file in module_aliases.items():
                alias_info[alias] = ("module", module_file)

            for alias, definitions in symbol_aliases.items():
                if alias in alias_info:
                    existing_kind, existing_payload = alias_info[alias]
                    if existing_kind == "module":
                        alias_info[alias] = ("mixed", {
                            "module": existing_payload,
                            "symbols": list(definitions),
                        })
                    elif existing_kind == "symbol":
                        merged = list(existing_payload) + list(definitions)
                        alias_info[alias] = ("symbol", merged)
                    elif existing_kind == "mixed":
                        payload = existing_payload
                        payload.setdefault("symbols", [])
                        payload["symbols"].extend(definitions)
                        alias_info[alias] = ("mixed", payload)
                else:
                    alias_info[alias] = ("symbol", list(definitions))

            if not alias_info:
                continue

            # Convert to host path
            host_path = self._get_host_path(importer_file)
            if not os.path.exists(host_path):
                continue

            try:
                with open(host_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=host_path)

                # Only traverse top-level module statements
                for node in tree.body:
                    # Collect all attribute accesses in this top-level statement
                    module_attrs = self._extract_module_attrs_from_node(node, alias_info)

                    for alias, attr_name in module_attrs:
                        alias_kind, payload = alias_info.get(alias, (None, None))
                        if alias_kind is None:
                            continue

                        resolved_defs: List[Tuple[str, str]] = []

                        if alias_kind == "module":
                            module_file = payload
                            resolved_defs = self._resolve_exported_attr(
                                module_file,
                                attr_name,
                                reexport_map,
                                module_definition_cache,
                                set(),
                            )
                        elif alias_kind == "symbol":
                            seen: Set[Tuple[str, str]] = set()
                            for defined_file, defined_name, _ in payload:
                                symbol_defs = self._resolve_symbol_attr(
                                    defined_file,
                                    defined_name,
                                    attr_name,
                                    symbol_attr_cache,
                                    file_definition_cache,
                                )
                                for item in symbol_defs:
                                    if item not in seen:
                                        resolved_defs.append(item)
                                        seen.add(item)
                        elif alias_kind == "mixed":
                            module_file = payload.get("module")
                            seen: Set[Tuple[str, str]] = set()
                            if module_file:
                                module_defs = self._resolve_exported_attr(
                                    module_file,
                                    attr_name,
                                    reexport_map,
                                    module_definition_cache,
                                    set(),
                                )
                                for item in module_defs:
                                    if item not in seen:
                                        resolved_defs.append(item)
                                        seen.add(item)
                            for defined_file, defined_name, _ in payload.get("symbols", []):
                                symbol_defs = self._resolve_symbol_attr(
                                    defined_file,
                                    defined_name,
                                    attr_name,
                                    symbol_attr_cache,
                                    file_definition_cache,
                                )
                                for item in symbol_defs:
                                    if item not in seen:
                                        resolved_defs.append(item)
                                        seen.add(item)
                        else:
                            continue

                        for resolved_file, def_key in resolved_defs:
                            obj_id = f"{resolved_file}::{def_key}"
                            accessed_objects.add(obj_id)

            except Exception as e:
                tqdm.write(f"❌ AST analysis failed for file {host_path}: {e}")

        return accessed_objects

    def _resolve_exported_attr(
        self,
        module_file: str,
        attr_name: str,
        reexport_map: Dict[str, Dict[str, List[Tuple[str, str]]]],
        cache: Dict[Tuple[str, str], List[Tuple[str, str]]],
        resolving: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Resolve module attributes with multi-level re-exports."""
        """Resolve attr_name to real definitions in module_file or re-exports."""

        key = (module_file, attr_name)
        if key in cache:
            return cache[key]
        # If already visited, a cycle exists; return empty list
        if key in resolving:
            return []

        resolving.add(key)
        results: List[Tuple[str, str]] = []

        # Try matching within the current module file first
        if self._is_repo_file(module_file):
            module_host_path = self._get_host_path(module_file)
            if os.path.exists(module_host_path):
                try:
                    module_definitions = self._analyze_single_file(module_host_path) or {}
                except Exception as e:
                    module_definitions = {}
                    tqdm.write(f"❌ AST analysis failed for file {module_host_path}: {e}")

                for def_key in module_definitions.keys():
                    qualified_name = def_key.split('::')[0]
                    simple_name = qualified_name.split('.')[-1]
                    if qualified_name == attr_name or simple_name == attr_name:
                        results.append((module_file, def_key))

        # If not matched in current module, try re-exported files
        if not results:
            for target_file, target_name in reexport_map.get(module_file, {}).get(attr_name, []):
                if not self._is_repo_file(target_file):
                    continue
                resolved = self._resolve_exported_attr(
                    target_file,
                    target_name,
                    reexport_map,
                    cache,
                    resolving,
                )
                if resolved:
                    results.extend(resolved)

        resolving.discard(key)
        cache[key] = results
        return results

    def _resolve_symbol_attr(
        self,
        defined_file: str,
        defined_name: str,
        attr_name: str,
        cache: Dict[Tuple[str, str, str], List[Tuple[str, str]]],
        definition_cache: Dict[str, Dict[str, Tuple[int, int, str]]],
    ) -> List[Tuple[str, str]]:
        """Resolve attribute references for symbol aliases."""

        key = (defined_file, defined_name, attr_name)
        if key in cache:
            return cache[key]

        results: List[Tuple[str, str]] = []

        if not self._is_repo_file(defined_file):
            cache[key] = results
            return results

        try:
            host_path = self._get_host_path(defined_file)
        except Exception:
            cache[key] = results
            return results

        if not os.path.exists(host_path):
            cache[key] = results
            return results

        definitions = definition_cache.get(host_path)
        if definitions is None:
            try:
                definitions = self._analyze_single_file(host_path) or {}
            except Exception as e:
                definitions = {}
                tqdm.write(f"❌ AST analysis failed for file {host_path}: {e}")
            definition_cache[host_path] = definitions

        # def_key format is usually 'QualifiedName::line-start-end'
        base_qualified = defined_name.split("::")[0]
        target_qualified = f"{base_qualified}.{attr_name}"
        target_simple = attr_name

        for def_key in definitions.keys():
            qualified = def_key.split("::")[0]
            simple = qualified.split(".")[-1]
            if qualified == target_qualified or simple == target_simple:
                results.append((defined_file, def_key))

        # If no exact match, at least keep the original symbol to avoid removal
        if not results:
            results.append((defined_file, defined_name))

        cache[key] = results
        return results

    def _resolve_imported_attribute(
        self,
        defined_file: str,
        defined_name: str,
        defined_line: int,
        attr_suffix: str,
    ) -> Set[str]:
        """Resolve attribute chain for imported symbols to object IDs."""

        resolved: Set[str] = set()

        if not attr_suffix:
            return resolved

        if not self._is_repo_file(defined_file):
            return resolved

        try:
            host_path = self._get_host_path(defined_file)
        except Exception:
            return resolved

        if not os.path.exists(host_path):
            return resolved

        definition_cache = getattr(self, "_import_definition_cache", None)
        if definition_cache is None:
            definition_cache = {}
            self._import_definition_cache = definition_cache

        definitions = definition_cache.get(host_path)
        if definitions is None:
            try:
                definitions = self._analyze_single_file(host_path) or {}
            except Exception as e:
                definitions = {}
                tqdm.write(f"❌ AST analysis failed for file {host_path}: {e}")
            definition_cache[host_path] = definitions

        if not definitions:
            return resolved

        base_qualified = defined_name.split("::")[0]
        target_qualified = f"{base_qualified}.{attr_suffix}"
        target_simple = attr_suffix.split(".")[-1]

        for def_key in definitions.keys():
            qualified = def_key.split("::")[0]
            simple = qualified.split(".")[-1]
            if qualified == target_qualified:
                resolved.add(f"{defined_file}::{def_key}")
            elif qualified.startswith(f"{base_qualified}.") and simple == target_simple:
                resolved.add(f"{defined_file}::{def_key}")

        if not resolved:
            resolved.add(f"{defined_file}::{defined_name}::{defined_line}")

        return resolved

    def _analyze_top_level_name_references(
        self,
        imports_by_importer: Dict[str, List[Dict[str, Any]]],
        all_files: Set[str],
    ) -> Set[str]:
        """Analyze top-level name references across all files (importers + imported).

        Handle two file types uniformly:
        1. Importer files: protect top-level referenced external imports + local definitions
        2. Imported files: protect top-level referenced local definitions

        Args:
            imports_by_importer: {importer_file: [import_records]}, only importer files
            all_files: Set of files to analyze (importers + imported)

        Returns:
            Set[str]: Object IDs to protect
        """
        protected = set()

        # Traverse all files
        for file_path in all_files:
            if not self._is_repo_file(file_path):
                continue

            host_path = self._get_host_path(file_path)
            if not os.path.exists(host_path):
                continue

            # Build external import mapping for importer files
            name_to_defs = defaultdict(list)
            if file_path in imports_by_importer:
                for imp in imports_by_importer[file_path]:
                    imported_name = imp.get('imported_name')
                    defined_file = imp.get('defined_file')
                    defined_name = imp.get('defined_name') or imported_name
                    defined_line = imp.get('defined_line')

                    if not imported_name or defined_line is None:
                        continue
                    if not self._is_repo_file(defined_file):
                        continue
                    name_to_defs[imported_name].append((defined_file, defined_name, defined_line))

            try:
                with open(host_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source, filename=host_path)
            except Exception:
                continue

            # Map local definitions to object IDs using simple name (tail) as key
            local_definitions: Dict[str, List[str]] = defaultdict(list)
            try:
                local_defs = self._analyze_single_file(host_path) or {}
            except Exception:
                local_defs = {}
            
            for def_key in local_defs:
                qualified_name = def_key.split("::")[0]
                obj_id = f"{file_path}::{def_key}"
                simple_name = qualified_name.split(".")[-1]
                if obj_id not in local_definitions[simple_name]:
                    local_definitions[simple_name].append(obj_id)

            # Traverse top-level statements, extract referenced names, and protect objects
            for node in tree.body:
                for name in self._extract_names_from_top_level_node(node):
                    # Protect external imported objects (importer files only)
                    if name in name_to_defs:
                        for defined_file, defined_name, defined_line in name_to_defs[name]:
                            obj_id = f"{defined_file}::{defined_name}::{defined_line}"
                            protected.add(obj_id)
                        continue

                    prefix_matches = [alias for alias in name_to_defs if name.startswith(f"{alias}.")]
                    if prefix_matches:
                        for alias in prefix_matches:
                            suffix = name[len(alias) + 1 :]
                            for defined_file, defined_name, defined_line in name_to_defs[alias]:
                                resolved_ids = self._resolve_imported_attribute(
                                    defined_file,
                                    defined_name,
                                    defined_line,
                                    suffix,
                                )
                                if resolved_ids:
                                    protected.update(resolved_ids)
                                else:
                                    obj_id = f"{defined_file}::{defined_name}::{defined_line}"
                                    protected.add(obj_id)
                    # Protect locally defined objects (all files)
                    if name in local_definitions:
                        for obj_id in local_definitions[name]:
                            protected.add(obj_id)

        return protected

    def _extract_names_from_top_level_node(self, node: ast.AST) -> Set[str]:
        """Extract names referenced on the RHS of a top-level statement.

        Includes:
        - RHS of assignments
        - Expression statements
        - Iterables in for-loops
        - Decorators for functions/classes
        """
        names: Set[str] = set()

        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        elif isinstance(node, ast.Expr):
            value = node.value
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            # Top-level for-loop iterables are evaluated on import
            # Example: for name, metric in [(label, func), ...]:
            value = node.iter

        if value is not None:
            for child in ast.walk(value):
                if isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.Attribute):
                    dotted = self._attribute_to_name(child)
                    if dotted:
                        names.add(dotted)
        
        # Handle decorators: @pytest.mark.parametrize("metric", [f1_score, ...])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                for child in ast.walk(decorator):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
                    elif isinstance(child, ast.Attribute):
                        dotted = self._attribute_to_name(child)
                        if dotted:
                            names.add(dotted)

        # Decorators and class-level assignments execute during import
        if isinstance(node, ast.ClassDef):
            for class_node in node.body:
                names.update(self._extract_names_from_class_level_node(class_node))

        return names

    def _extract_names_from_class_level_node(self, node: ast.AST) -> Set[str]:
        """Extract name references executed during class definition."""
        names: Set[str] = set()

        value = None
        # Class-level assignment statements
        if isinstance(node, ast.Assign):    # x = some_func()
            value = node.value
        elif isinstance(node, ast.AnnAssign):   # x: int = some_func()
            value = node.value
        elif isinstance(node, ast.Expr):    # some_func()
            value = node.value

        if value is not None:
            for child in ast.walk(value):
                if isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.Attribute):
                    dotted = self._attribute_to_name(child)
                    if dotted:
                        names.add(dotted)

        # Decorators of methods or inner classes execute at definition time
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                for child in ast.walk(decorator):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
                    elif isinstance(child, ast.Attribute):
                        dotted = self._attribute_to_name(child)
                        if dotted:
                            names.add(dotted)

            # Default arguments are evaluated at definition time; protect referenced names
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + [
                    d for d in node.args.kw_defaults if d is not None
                ]
                for default in defaults:
                    for child in ast.walk(default):
                        if isinstance(child, ast.Name):
                            names.add(child.id)
                        elif isinstance(child, ast.Attribute):
                            dotted = self._attribute_to_name(child)
                            if dotted:
                                names.add(dotted)

        # Recursively handle nested class definitions
        if isinstance(node, ast.ClassDef):
            for inner in node.body:
                names.update(self._extract_names_from_class_level_node(inner))

        return names

    def _attribute_to_name(self, node: ast.AST) -> Optional[str]:
        """Convert an Attribute node into dotted-name string."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._attribute_to_name(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return None
    
    def _extract_module_attrs_from_node(
        self,
        node: ast.AST,
        aliases: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """Extract module attribute refs evaluated at import time."""

        # Expressions evaluated on import are broad: decorators, defaults, class body, etc.
        # They may trigger module attribute access; full scan is needed for collect deps.

        def collect_from_expr(expr: Optional[ast.AST]) -> List[Tuple[str, str]]:
            attrs: List[Tuple[str, str]] = []
            if expr is None:
                return attrs
            for child in ast.walk(expr):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                    alias = child.value.id
                    if alias in aliases:
                        attrs.append((alias, child.attr))
            return attrs

        module_attrs: List[Tuple[str, str]] = []

        if isinstance(node, ast.Assign):
            module_attrs.extend(collect_from_expr(node.value))

        elif isinstance(node, ast.AnnAssign):
            module_attrs.extend(collect_from_expr(node.value))

        elif isinstance(node, ast.Expr):
            module_attrs.extend(collect_from_expr(node.value))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators, return annotations, and defaults are evaluated at definition time
            for decorator in node.decorator_list:
                module_attrs.extend(collect_from_expr(decorator))
            module_attrs.extend(collect_from_expr(node.returns))

            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            if hasattr(node.args, "posonlyargs"):
                defaults += [arg.annotation for arg in node.args.posonlyargs if arg.annotation]
            defaults += [arg.annotation for arg in node.args.args if arg.annotation]
            defaults += [arg.annotation for arg in node.args.kwonlyargs if arg.annotation]
            if node.args.vararg and node.args.vararg.annotation:
                defaults.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                defaults.append(node.args.kwarg.annotation)
            defaults = [expr for expr in defaults if expr is not None]
            for expr in defaults:
                module_attrs.extend(collect_from_expr(expr))

        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                module_attrs.extend(collect_from_expr(decorator))
            for base in node.bases:
                module_attrs.extend(collect_from_expr(base))
            for keyword in node.keywords:
                module_attrs.extend(collect_from_expr(keyword.value))

            # Class body executes at definition time; recurse into top-level statements
            # Otherwise, class attribute assignments and other import-time logic may be missed.
            for class_node in node.body:
                module_attrs.extend(self._extract_module_attrs_from_node(class_node, aliases))

        return module_attrs

    def _get_all_obj_info(self, objects: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[Tuple[int, int], str]]:
        """Use AST analysis to get object spans and types."""
        all_obj_info = {}
        # Collect all involved files
        all_files = set()
        for obj_id, obj_info in objects.items():
            host_path = self._get_host_path(obj_info['file'])
            all_files.add(host_path)
        
        if self.logger:
            self.logger.debug(f"Starting AST analysis: {len(all_files)} files")
        else:
            logger.debug(f"Starting AST analysis: {len(all_files)} files")
        
        file_content_dict = {}
        for file_path in all_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                file_definitions = self._analyze_single_file(file_path)
                file_content_dict[file_path] = file_definitions
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
                continue
        if self.logger:
            self.logger.debug("AST analysis completed")
        else:
            logger.debug("AST analysis completed")
        # Record all_obj_info
        for obj_key, obj_info in objects.items():
            host_path = self._get_host_path(obj_info['file'])
            obj_file_definitions = file_content_dict[host_path]
            success_find = False
            for key, val in obj_file_definitions.items():
                if "::".join(obj_key.split("::")[1:]) == key:
                    all_obj_info[obj_key] = ((val[0], val[1]), val[2])
                    success_find = True
                    break
            if not success_find:
                logger.warning(f"Definition for object {obj_key} not found in file {host_path}")
        return all_obj_info

    def _analyze_single_file(self, file_path: str):
        """Analyze a single file to extract class/function defs."""
        # Only analyze Python files; skip HTML/templates and other non-code files
        # if not file_path.endswith('.py'):
        #     return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Failed to parse AST for file {file_path}: {e}")
            return
            
        visitor = FunctionClassVisitor()
        visitor.visit(tree)
        
        return visitor.definitions


    def _discard_invalid_objects(self, objects: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Discard invalid objects."""
        new_objects = deepcopy(objects)
        # Drop objects starting with '<' or paths not starting with '/'
        bck_objects = deepcopy(new_objects)
        for obj_id, obj_info in bck_objects.items():
            for edge_id in obj_info['edges']:
                if not edge_id.startswith('/') or edge_id.split('::')[1].startswith('<'):
                    new_objects[obj_id]['edges'].remove(edge_id)
        for obj_id, obj_info in bck_objects.items():
            if obj_info['name'].startswith("<") or not obj_info['file'].startswith("/"):
                del new_objects[obj_id]
            
        # Drop objects from non-.py files
        bck_objects = deepcopy(new_objects)
        for obj_id, obj_info in bck_objects.items():
            for edge_id in obj_info['edges']:
                if not edge_id.split('::')[0].endswith(".py"):
                    new_objects[obj_id]['edges'].remove(edge_id)
        
        for obj_id, obj_info in bck_objects.items():
            if not obj_id.split('::')[0].endswith(".py"):
                del new_objects[obj_id]

        return new_objects


    def get_classification_summary(self) -> Dict[str, Any]:
        """
        Generate a statistical summary of classification results.

        Returns:
            Dict[str, Any]: Dictionary with the following statistics:
                - file_stats: file classification stats
                - obj_stats: object classification stats
                - patch_files: list of patch-type files
                - top_objects: list of top objects
                - specific_objects: list of specific objects
        """
        # File classification stats
        file_counts = Counter(self.file_classifier.values())
        file_stats = {
            "total_files": len(self.file_classifier),
            "patch_count": file_counts.get("patch", 0),
            "left_count": file_counts.get("left", 0),
            "distribution": dict(file_counts)
        }
        
        # Object classification stats (only the second element)
        obj_classification_values = [classification[1] for classification in self.obj_classifier.values()]
        obj_counts = Counter(obj_classification_values)
        obj_stats = {
            "total_objects": len(self.obj_classifier),
            "top_count": obj_counts.get("top", 0),
            "specific_count": obj_counts.get("specific", 0),
            "others_count": obj_counts.get("others", 0),
            "distribution": dict(obj_counts)
        }
        
        # Collect specific file/object types
        patch_files = [file for file, classification in self.file_classifier.items() 
                      if classification == "patch"]
        
        top_objects = [obj for obj, classification in self.obj_classifier.items() 
                      if classification[1] == "top"]
        
        specific_objects = [obj for obj, classification in self.obj_classifier.items() 
                           if classification[1] == "specific"]
        
        # Assemble summary dict
        summary = {
            "specs_name": self.data_item.repo,
            "test_file": self.data_item.file_path,
            "file_stats": file_stats,
            "obj_stats": obj_stats,
            "patch_files": patch_files,
            "top_objects": top_objects,
            "specific_objects": specific_objects,
            "lines_code_level1": self.lines_code_level1,
            "lines_code_level2": self.lines_code_level2,  
        }

        return summary


def is_specific_obj(obj_id: str, all_obj_info: Dict[str, Tuple[Tuple[int, int], str]], test_name: str, lines_code: int, max_line_code: int, test_file: str) -> bool:
    # If <module>, it is not specific
    if obj_id.split('::')[-1] == "<module>":
        return False

    # FIXME: This should ideally be handled in a generic way
    # Special case: transformers utils are not specific
    if '/transformers/utils/' in obj_id.split('::')[0] or '/tests/' in obj_id.split('::')[0] or 'modeling_utils.py' in obj_id.split('::')[0] or '/transformers/integrations/' in obj_id.split('::')[0]:
        return False

    # If test-related object, it is not specific
    if Path(obj_id.split('::')[0]).name == Path(test_file).name or Path(obj_id.split('::')[0]).name == 'conftest.py' or Path(obj_id.split('::')[0]).name == '__init__.py':
        return False
    
    # If obj_id is a subset of test_name, treat as specific
    if obj_id.split('::')[-1] in test_name or test_name in obj_id.split('::')[-1]:
        return True

    # If lines exceed max_line_code, it is not specific
    if (all_obj_info[obj_id][0][1] - all_obj_info[obj_id][0][0])+lines_code > max_line_code:
        return False
    
    # If lines do not exceed max_line_code, treat as specific
    return True


