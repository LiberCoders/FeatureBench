from copy import deepcopy
import os
import yaml
import shutil
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import hashlib
import subprocess
import tempfile
from tqdm import tqdm

from featurebench.conversion.render_task import get_task
from featurebench.utils.config import Config
from featurebench.utils.repo_manager import RepoManager
from featurebench.utils.storage import StorageManager
from featurebench.mask.mask_generator import MaskResult
from featurebench.docker.image_manager import ImageManager
from featurebench.conversion.llm_prompt_generator import LLMPromptGenerator
from featurebench.conversion.llm_testfile_adjuster import LLMTestfileAdjuster
from featurebench.post_verify.level2_validator import Level2ValidationResult, Level2Validator
from featurebench.llm.llm_exceptions import LLMException

class CaseConverter:
    def __init__(self, 
        config: Config,
        repo_manager: RepoManager,
        storage_manager: StorageManager,
        image_manager: ImageManager,
        data_item,
        classification_summary: dict,
        mask_results: Dict[str, MaskResult],
        deleted_lines: Optional[int] = None,
        mask_file_count: Optional[int] = None,
        mask_object_count: Optional[int] = None,
        generate_level1: bool = False,
        generate_level2: bool = False,
        logger = None
    ):
        self.config = config
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.image_manager = image_manager
        self.data_item = data_item
        self.classification_summary = classification_summary
        self.mask_results = mask_results
        self.deleted_lines = deleted_lines
        self.mask_file_count = mask_file_count
        self.mask_object_count = mask_object_count
        self.generate_level1 = generate_level1
        self.generate_level2 = generate_level2
        self.logger = logger
        self.adjusted_test_file_path_local = None
        self.case_dirs: Dict[int, str] = {}

        self.llm_prompt_for_case = data_item.specs.get('llm_prompt_for_case', True)
        if self.llm_prompt_for_case:
            self.llm_prompt_generator = LLMPromptGenerator(config, repo_manager, storage_manager, data_item, classification_summary, mask_results, logger)

        if self.generate_level2:
            self.llm_testfile_adjuster = LLMTestfileAdjuster(config, repo_manager, storage_manager, data_item, logger)

        # Initialize render_config for task/prompt.md rendering
        self.render_config = self._init_render_config()


    def _init_render_config(self) -> dict:
        """Initialize render_config."""
        render_config = {
            "black_links": self.data_item.specs.get('black_links', []),
            "commit": self.data_item.specs.get('commit', None),
            "install": self.data_item.specs.get('install', ''),
            "docker_specs": self.data_item.specs.get('docker_specs', {}),
            "install": self.data_item.specs.get('install', ''),
            "library_name": self.data_item.specs.get('library_name', ''),
            "pip_packages": self.data_item.specs.get('pip_packages', []),
            "pre_install": self.data_item.specs.get('pre_install', []),
            "repository": self.data_item.specs.get('repository', ''),
            "repo_name": str(self.data_item.repo_root.name),
            "technical_docs": self.data_item.specs.get('technical_docs', []),
            "test_cmd": self.data_item.specs.get('test_cmd', ''),
            "timeout": self.data_item.specs.get('timeout_run', 600),
            "base_image": self.image_manager.image_info[self.data_item.repo]['base_image'],
            "instance_image": self.image_manager.image_info[self.data_item.repo]['instance_image'],
            "task_name": str(self.data_item.repo_root.name).replace('-', '_') + '_' + str(Path(self.data_item.file_path).name).replace('.py', '').replace('test_', '')
        }
        return render_config

    def run(self):
        """
        Convert to test cases and generate level1/level2 folders.
        
        Returns:
            Tuple: (lv1_error_message, success_lv1, lv2_result, success_lv2, is_llm_error)
        """
        levels = []
        if self.generate_level1:
            levels.append(1)
        if self.generate_level2:
            levels.append(2)
        
        # Initialize return values
        lv1_error_message = None
        success_lv1 = False
        lv2_result = None
        success_lv2 = False
        is_llm_error = False  # Mark whether error is LLM-related
        
        # Handle each level
        for level in levels:
            try:
                if level == 1:
                    # Generate level 1
                    self._generate_level_case(level)
                    success_lv1 = True
                elif level == 2:
                    # Generate level 2 (with post-verify)
                    lv2_result, success_lv2 = self._generate_level_case(level)
                    
            except LLMException as e:
                # Catch LLM-related exceptions and mark as LLM error
                is_llm_error = True
                error_message = f"LLM error: {str(e)}"
                
                if level == 1:
                    lv1_error_message = f"Level 1 generation failed: {error_message}"
                    success_lv1 = False
                    # Remove level 1 directory
                    level_dir = Path(self.storage_manager.output_dir) / 'cases' / self._get_repo_commit_dirname() / self._get_test_file_hash_name(1)
                    if level_dir.exists():
                        shutil.rmtree(level_dir)
                        if self.logger:
                            tqdm.write(f"\nRemoved failed Level 1 directory: {level_dir}, reason: {error_message}")
                    
                elif level == 2:
                    lv2_error_message = f"Level 2 generation failed: {error_message}"
                    success_lv2 = False
                    # Create error result object
                    from featurebench.post_verify.level2_validator import Level2ValidationResult
                    lv2_result = Level2ValidationResult(
                        success=False,
                        error_message=lv2_error_message,
                        pass_rate=0.0
                    )
                    # Remove level 2 directory
                    level_dir = Path(self.storage_manager.output_dir) / 'cases' / self._get_repo_commit_dirname() / self._get_test_file_hash_name(2)
                    if level_dir.exists():
                        shutil.rmtree(level_dir)
                        if self.logger:
                            tqdm.write(f"\nRemoved failed Level 2 directory: {level_dir}, reason: {error_message}")
                
            except Exception as e:
                # Catch non-LLM exceptions and remove directories
                error_message = str(e)
                
                if level == 1:
                    lv1_error_message = f"Level 1 generation failed: {error_message}"
                    success_lv1 = False
                    # Remove level 1 directory
                    level_dir = Path(self.storage_manager.output_dir) / 'cases' / self._get_repo_commit_dirname() / self._get_test_file_hash_name(1)
                    if level_dir.exists():
                        shutil.rmtree(level_dir)
                        if self.logger:
                            tqdm.write(f"\nRemoved failed Level 1 directory: {level_dir}, reason: {error_message}")
                    
                elif level == 2:
                    lv2_error_message = f"Level 2 generation failed: {error_message}"
                    success_lv2 = False
                    # Create error result object
                    from featurebench.post_verify.level2_validator import Level2ValidationResult
                    lv2_result = Level2ValidationResult(
                        success=False,
                        error_message=lv2_error_message,
                        pass_rate=0.0
                    )
                    # Remove level 2 directory
                    level_dir = Path(self.storage_manager.output_dir) / 'cases' / self._get_repo_commit_dirname() / self._get_test_file_hash_name(2)
                    if level_dir.exists():
                        shutil.rmtree(level_dir)
                        if self.logger:
                            tqdm.write(f"\nRemoved failed Level 2 directory: {level_dir}, reason: {error_message}")
        
        return lv1_error_message, success_lv1, lv2_result, success_lv2, is_llm_error

    def _post_validate_level2(self) -> Tuple[Level2ValidationResult, bool]:
        """Post-validate level2."""
        
        # Build level2 directory path
        level2_dir = Path(self.storage_manager.output_dir) / 'cases' / self._get_repo_commit_dirname() / self._get_test_file_hash_name(2)

        level2_validator = Level2Validator(
            config=self.config,
            repo_manager=self.repo_manager,
            image_manager=self.image_manager,
            storage_manager=self.storage_manager,
            data_item=self.data_item,
            level2_dir=level2_dir,
            logger=self.logger
        )
        lv2_result, success_lv2 = level2_validator.run()
        preserve_dir = self._should_preserve_level2_dir(lv2_result)
        if (not success_lv2) and (not preserve_dir):
            # Remove level2 directory
            if level2_dir.exists():
                shutil.rmtree(level2_dir)
                if self.logger:
                    tqdm.write(f"\nRemoved Level 2 directory after post-validation failure: {level2_dir}, reason: {lv2_result.error_message}")
        return lv2_result, success_lv2

    def _should_preserve_level2_dir(self, lv2_result: Optional[Level2ValidationResult]) -> bool:
        """Determine if level2 artifacts should stay on failure (e.g., high pass rate)."""
        if not lv2_result:
            return False
        error_message = (lv2_result.error_message or '').strip()
        return 'Level2 post-validation pass rate too high' in error_message

    
    def _generate_level_case(self, level: int):
        """
        Generate test cases for a specific level.
        
        Args:
            level: Test case level (1 or 2)
            
        Returns:
            If level == 2, returns (lv2_result, success_lv2)
            If level == 1, returns None
        """
        # New directory structure: cases/usr__repo.commit[:8]/test_file_name.hash[:8].lv{level}/
        repo_commit_dir = self._get_repo_commit_dirname()
        test_file_hash_name = self._get_test_file_hash_name(level)
        
        level_dir = Path(self.storage_manager.output_dir) / 'cases' / repo_commit_dir / test_file_hash_name
        level_dir.mkdir(parents=True, exist_ok=True)
        self.case_dirs[level] = str(level_dir)
        mask_results = self.mask_results

        # 1. Generate config.yaml
        config_path = self._generate_config_yaml(level, level_dir, mask_results)
        
        # 2. Generate problem_statement.md (formerly prompt.md)
        self._generate_problem_statement(config_path, level_dir)
        
        # 3. Generate patch files (in container)
        if level == 1:
            self._generate_patch_diff_lv1(level_dir)
            self._generate_test_patch_diff_lv1(level_dir)
        elif level == 2:
            self._generate_test_patch_diff_lv2(level_dir)
        
        # 4. Generate instance.json
        self._generate_instance_json(level, level_dir)
        
        # 5. If level 2, run post-validation
        if level == 2:
            lv2_result, success_lv2 = self._post_validate_level2()
            return lv2_result, success_lv2
        
        return None

    def _generate_config_yaml(self, level: int, level_dir: Path, mask_results: Dict[str, MaskResult]) -> str:
        """Generate config.yaml file."""
        # Read base config
        render_config = self.render_config
        
        # Update task_level
        render_config['task_level'] = level
        
        # Fill interface descriptions from mask_results
        render_config = self._fill_interface_descriptions(level, render_config, mask_results)
        
        # Update config with examples (e.g., test_code_example)
        render_config = self._update_config_with_examples(render_config)

        # Update task_statement
        if self.llm_prompt_for_case:
            render_config = self._update_task_statement(render_config, level)

        # Write config file
        config_path = level_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(render_config, f, default_flow_style=False, allow_unicode=True)
        
        return str(config_path)
    
    def _fill_interface_descriptions(self, level: int, render_config: dict, mask_results: Dict[str, MaskResult]) -> dict:
        """Fill interface descriptions from mask_results."""
        # interface_codes: Dict, e.g. {'/testbed/func.py': 'def func(): ...'}
        interface_codes = self._generate_interface_code(mask_results)
        
        # If interface_codes has no content, interface description generation failed
        interface_codes = {key: value for key, value in interface_codes.items() if value}
        if not interface_codes:
            raise RuntimeError("No interface descriptions were generated for this case")
        for idx, (interface_code_path, interface_code) in enumerate(interface_codes.items(), 1):
            render_config[f'interface_description{idx}'] = f"Below is **Interface Description {idx}**"
            render_config[f'interface_code{idx}'] = interface_code
            render_config[f'interface_code_path{idx}'] = interface_code_path
        return render_config
    
    def _update_task_statement(self, render_config: dict, level: int) -> dict:
        """Update task_statement."""
        task_statement = self.storage_manager.load_llm_task_statement(self.data_item.repo, self.data_item.file_path)
        if not task_statement:
            top_objects_ids = self.classification_summary['top_objects']
            task_statement = self.llm_prompt_generator.generate_task_statement_with_llm(top_objects_ids)
            self.storage_manager.save_llm_task_statement(self.data_item.repo, self.data_item.file_path, task_statement)
        render_config['task_statement'] = task_statement
        return render_config

    
    def _generate_interface_code(self, mask_results: Dict[str, MaskResult]) -> Dict:
        """Generate interface code and return signatures per file."""
        interface_codes = {}
        
        for file_path, result in mask_results.items():
            if not result.success or not result.top_objects:
                continue

            # Parse top-object hierarchy
            top_structure = self._parse_top_objects_structure(result)

            if self.llm_prompt_for_case:
                try:
                    result, top_structure = self.llm_prompt_generator.generate_docstring_with_llm(file_path, result, top_structure)
                    # Update self.mask_results for later masking, and top_structure for task_statement generation
                    self.mask_results[file_path] = deepcopy(result)
                    self.top_structure = deepcopy(top_structure)

                except LLMException:
                    # Re-raise LLM-related exceptions
                    raise
                except Exception as e:
                    raise LLMException(f"LLM prompt generation failed: {e}") 
            
            # Generate interface code
            code_str = self._build_interface_code(top_structure)
            
            interface_codes[file_path] = code_str
            
        return interface_codes
    
    def _parse_top_objects_structure(self, result: MaskResult) -> Dict:
        """
        Parse top-object structure into a tree.
        
        This is a pruning step: keep only paths from the root to top objects,
        with top objects as leaves (do not expand children).
        
        Build based on ModuleSignature trees (nested_functions and nested_classes),
        not by qualified_name parsing.
        
        Args:
            result: MaskResult object, includes top_objects list and signature_info
            
        Returns:
            Dict: Tree structure based on ModuleSignature (nested_functions and nested_classes),
            Example
            {
                'Class1': {
                    'type': 'class',
                    'signature': ClassSignature object,
                    'children': {
                        'method1': {'type': 'function', 'signature': ..., 'children': {}},
                    }
                }
                'func1':{
                    'type': 'function',
                    'signature': FunctionSignature object,
                    'children': {}
                }
            }
        """
        from featurebench.mask.signature_extractor import FunctionSignature, ClassSignature
        
        top_structure = {}
        signature_info = result.signature_info
        top_objects = set(result.top_objects)
        
        if not top_objects:
            return top_structure
        
        # Helper: find object by full_id and return the path from root
        def find_path_to_object(target_full_id: str):
            """
            Find target_full_id in signature_info tree and return the path.
            The path is a list of (name, type, signature).
            """
            # Search top-level functions
            for func in signature_info.functions:
                if func.full_id == target_full_id:
                    return [(func.name, 'function', func)]
                # Recursively search nested objects
                path = find_in_function(func, target_full_id, [(func.name, 'function', func)])
                if path:
                    return path
            
            # Search top-level classes
            for cls in signature_info.classes:
                if cls.full_id == target_full_id:
                    return [(cls.name, 'class', cls)]
                # Recursively search nested objects
                path = find_in_class(cls, target_full_id, [(cls.name, 'class', cls)])
                if path:
                    return path
            
            return None
        
        def find_in_class(cls: ClassSignature, target_full_id: str, current_path: list):
            """Recursive search in class; current_path includes current class."""
            # Search nested functions (methods)
            for func in cls.nested_functions:
                if func.full_id == target_full_id:
                    return current_path + [(func.name, 'function', func)]
                # Recursive search
                path = find_in_function(func, target_full_id, current_path + [(func.name, 'function', func)])
                if path:
                    return path
            
            # Search nested classes
            for nested_cls in cls.nested_classes:
                if nested_cls.full_id == target_full_id:
                    return current_path + [(nested_cls.name, 'class', nested_cls)]
                # Recursive search
                path = find_in_class(nested_cls, target_full_id, current_path + [(nested_cls.name, 'class', nested_cls)])
                if path:
                    return path
            
            return None
        
        def find_in_function(func: FunctionSignature, target_full_id: str, current_path: list):
            """Recursive search in function; current_path includes current function."""
            # Search nested functions
            for nested_func in func.nested_functions:
                if nested_func.full_id == target_full_id:
                    return current_path + [(nested_func.name, 'function', nested_func)]
                # Recursive search
                path = find_in_function(nested_func, target_full_id, current_path + [(nested_func.name, 'function', nested_func)])
                if path:
                    return path
            
            # Search nested classes
            for nested_cls in func.nested_classes:
                if nested_cls.full_id == target_full_id:
                    return current_path + [(nested_cls.name, 'class', nested_cls)]
                # Recursive search
                path = find_in_class(nested_cls, target_full_id, current_path + [(nested_cls.name, 'class', nested_cls)])
                if path:
                    return path
            
            return None
        
        # For each top object, find its path in signature_info and build top_structure
        for full_id in sorted(top_objects):
            path = find_path_to_object(full_id)
            if not path:
                # Not found; skip
                continue
            
            # Build tree along the path
            current_level = top_structure
            
            for i, (name, obj_type, sig) in enumerate(path):
                # Check if node exists
                if name not in current_level:
                    # Create node
                    current_level[name] = {
                        'type': obj_type,
                        'signature': sig,
                        'children': {}
                    }
                
                # If this is the last level (top object itself)
                if i == len(path) - 1:
                    # Ensure leaf node (clear children)
                    current_level[name]['children'] = {}
                else:
                    # Not last level; keep going
                    current_level = current_level[name]['children']
        
        return top_structure

    def _build_interface_code(self, top_structure: Dict) -> str:
        """
        Generate interface code string from the top-object structure.
        
        Args:
            top_structure: Structure dict for top objects
            
        Returns:
            str: Generated interface code string with signatures and # <your code> stubs
        """
        from featurebench.mask.signature_extractor import FunctionSignature, ClassSignature
        
        lines = []
        
        def render_function(func_sig: FunctionSignature, indent: int = 0, children: Dict = None) -> List[str]:
            """Render function signature; recurse into children when needed."""
            result_lines = []
            indent_str = '    ' * indent
            
            # Add decorators
            for decorator in func_sig.decorators:
                # ast.unparse does not include '@'; add manually
                if not decorator.startswith('@'):
                    decorator = '@' + decorator
                result_lines.append(f"{indent_str}{decorator}")
            
            # Build function definition line
            func_prefix = "async def" if func_sig.is_async else "def"
            args_str = ', '.join(func_sig.args) if func_sig.args else ''
            
            func_def = f"{indent_str}{func_prefix} {func_sig.name}({args_str})"
            if func_sig.return_annotation:
                func_def += f" -> {func_sig.return_annotation}"
            func_def += ":"
            
            result_lines.append(func_def)
            
            # Add docstring
            if func_sig.docstring:
                docstring_lines = func_sig.docstring.split('\n')
                if len(docstring_lines) == 1:
                    result_lines.append(f'{indent_str}    """{func_sig.docstring}"""')
                else:
                    result_lines.append(f'{indent_str}    """')
                    for line in docstring_lines:
                        result_lines.append(f'{indent_str}    {line}')
                    result_lines.append(f'{indent_str}    """')
            
            # Render children (only top objects in children)
            if children:
                for child_name, child_node in children.items():
                    result_lines.append("")  # Blank line separator
                    if child_node['type'] == 'function':
                        result_lines.extend(
                            render_function(
                                child_node['signature'],
                                indent + 1,
                                child_node.get('children', {})
                            )
                        )
                    elif child_node['type'] == 'class':
                        result_lines.extend(
                            render_class(
                                child_node['signature'],
                                indent + 1,
                                child_node.get('children', {})
                            )
                        )
            else:
                # Add placeholder when no children
                result_lines.append(f"{indent_str}    # <your code>")
            
            return result_lines
        
        def render_class(cls_sig: ClassSignature, indent: int = 0, children: Dict = None) -> List[str]:
            """Render class signature."""
            result_lines = []
            indent_str = '    ' * indent
            
            # Add decorators
            for decorator in cls_sig.decorators:
                # ast.unparse does not include '@'; add manually
                if not decorator.startswith('@'):
                    decorator = '@' + decorator
                result_lines.append(f"{indent_str}{decorator}")
            
            # Build class definition line
            class_def = f"{indent_str}class {cls_sig.name}"
            if cls_sig.bases:
                bases_str = ', '.join(cls_sig.bases)
                class_def += f"({bases_str})"
            class_def += ":"
            
            result_lines.append(class_def)
            
            # Add docstring
            if cls_sig.docstring:
                docstring_lines = cls_sig.docstring.split('\n')
                if len(docstring_lines) == 1:
                    result_lines.append(f'{indent_str}    """{cls_sig.docstring}"""')
                else:
                    result_lines.append(f'{indent_str}    """')
                    for line in docstring_lines:
                        result_lines.append(f'{indent_str}    {line}')
                    result_lines.append(f'{indent_str}    """')
            
            # Add class attributes
            if cls_sig.attributes:
                for attr_name, attr_value in cls_sig.attributes.items():
                    if isinstance(attr_value, str) and attr_value.startswith('# Type:'):
                        # Type annotation
                        type_str = attr_value[8:].strip()
                        result_lines.append(f"{indent_str}    {attr_name}: {type_str}")
                    else:
                        # Attribute with default value
                        result_lines.append(f"{indent_str}    {attr_name} = {repr(attr_value)}")
            
            # Render children (only top objects in children)
            if children:
                for child_name, child_node in children.items():
                    result_lines.append("")  # Blank line separator
                    if child_node['type'] == 'function':
                        result_lines.extend(render_function(child_node['signature'], indent + 1))
                    elif child_node['type'] == 'class':
                        result_lines.extend(render_class(
                            child_node['signature'], 
                            indent + 1,
                            child_node.get('children', {})
                        ))
            else:
                # Add placeholder when no children
                result_lines.append(f"{indent_str}    # <your code>")
            
            return result_lines
        
        # Generate code by traversing top_structure
        for obj_name, node in sorted(top_structure.items()):
            if node['type'] == 'function':
                lines.extend(render_function(
                    node['signature'],
                    children=node.get('children', {})
                ))
            elif node['type'] == 'class':
                lines.extend(render_class(
                    node['signature'],
                    children=node.get('children', {})
                ))
            lines.append("")  # Blank line separator between objects
        
        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()
        
        return '\n'.join(lines)
    

    def _get_test_file_object_top_objects_dict(self, data_item) -> Dict[str, List[str]]:
        """Build {test_file_object: [top_objects]} mapping."""
        # Read dynamic trace JSON
        with open(data_item.dynamic_trace_file, 'r', encoding='utf-8') as f:
            dynamic_trace_dict = json.load(f)
            objects = dynamic_trace_dict.get('objects', {})

        test_file = data_item.file_path

        # Build {test-file object: [top objects]} mapping
        test_file_object_top_objects_dict = {}
        for object_id, object_info in objects.items():
            if object_info['file'] == test_file and not object_info['name'].startswith('<'):
                test_file_object_top_objects_dict[object_id] = [edge_id for edge_id in object_info['edges'] if edge_id in self.classification_summary['top_objects']]

        return test_file_object_top_objects_dict

    def _insert_agent_code_import(self, file_content: str) -> str:
        """
        Insert import agent_code into file content.
        If the file has from __future__ import statements, insert after the last one.
        Otherwise insert at the beginning.
        
        Args:
            file_content: File content string
            
        Returns:
            File content after insertion
        """
        lines = file_content.split('\n')
        
        # Find the last from __future__ import line
        last_future_import_idx = -1
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('from __future__ import'):
                last_future_import_idx = i
        
        # Insert import agent_code
        import_statement = 'import agent_code'
        
        if last_future_import_idx != -1:
            # Insert after from __future__ import
            lines.insert(last_future_import_idx + 1, import_statement)
        else:
            # Insert at the beginning of the file
            lines.insert(0, import_statement)
        
        return '\n'.join(lines)

    def _update_config_with_examples(self, config: dict) -> dict:
        """
        Update config with example fields like test_code_example.
        
        Args:
            config: Config dict
            level: Current level
            
        Returns:
            Updated config dict
        """
        code_key = 'interface_code1'
        code_path_key = 'interface_code_path1'
        
        # Ensure keys exist (defensive)
        if code_key not in config or code_path_key not in config:
            # If missing, use empty defaults
            config['interface_code_example'] = ''
            config['interface_code_example_path'] = ''
            return config
        
        interface_code = config[code_key]
        interface_code_path = config[code_path_key]
        
        # Ensure interface_code is not empty (defensive)
        if not interface_code:
            config['interface_code_example'] = ''
            config['interface_code_example_path'] = interface_code_path
            return config
        
        your_code_idx = interface_code.find('<your code>')
                    
        if your_code_idx != -1:
            # Contains <your code>; truncate after it
            end_idx = your_code_idx + len('<your code>')
            interface_code_example = interface_code[:end_idx] + '\n...'
        else:
            # No <your code>; take first 200 chars
            interface_code_example = interface_code[:200] + '\n...' if len(interface_code) > 200 else interface_code + '\n...'
        
        config['interface_code_example'] = interface_code_example
        config['interface_code_example_path'] = interface_code_path
        return config

    def _get_repo_commit_dirname(self) -> str:
        """
        Generate repo_commit dir name: usr__repo.commit[:8]
        Example: lightning-ai__pytorch-lightning.1234abcd
        Requires full commit hash from the container
        """
        repository = self.data_item.specs.get('repository', '')
        
        # Get full commit hash from container (first 8 chars)
        commit_short = self._get_full_commit_hash()
        
        # Replace / with __
        repo_name = repository.replace('/', '__')
        
        return f"{repo_name}.{commit_short}"
    
    def _get_full_commit_hash(self) -> str:
        """
        Get full commit hash in container and return first 8 chars.
        Uses git rev-parse to get the full commit.
        """
        full_hash = self._get_full_commit_hash_complete()
        return full_hash[:8] if full_hash != 'unknown' else 'unknown'
    
    def _get_full_commit_hash_complete(self) -> str:
        """
        Get full commit hash in container (no truncation).
        Uses git rev-parse to get the full commit.
        """
        commit = self.data_item.specs.get('commit', '')
        if not commit:
            return 'unknown'
        
        # Check cache
        if hasattr(self, '_cached_full_commit_hash'):
            return self._cached_full_commit_hash
        
        container_id = None
        try:
            # Start a temporary container
            container_id = self.image_manager.run_container(
                specs_name=self.data_item.repo,
                working_dir="/testbed",
                prepare_env=True
            )
            
            # Run git rev-parse in container to get full commit hash
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command=f"git rev-parse {commit}",
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                full_commit = result.stdout.strip()
                # Cache result
                self._cached_full_commit_hash = full_commit
                return full_commit
            else:
                # If failed, use original commit
                if self.logger:
                    self.logger.warning(f"Failed to get full commit hash: {result.stderr}; using original commit")
                self._cached_full_commit_hash = commit
                return commit
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error getting full commit hash: {e}; using original commit")
            self._cached_full_commit_hash = commit
            return commit
        finally:
            # Cleanup container
            if container_id:
                self.image_manager.stop_container(container_id)

    def _get_test_file_hash_name(self, level: int) -> str:
        """
        Generate test file hash name: test_file_name.hash[:8].lv{level}
        Example: test_model.a1b2c3d4.lv1
        """
        test_file_path = Path(self.data_item.file_path)
        test_file_name = test_file_path.stem  # Strip .py
        
        # Hash the test file relative path
        test_file_relative_path = self._get_test_file_relative_path()
        path_hash = hashlib.md5(test_file_relative_path.encode()).hexdigest()[:8]
        
        return f"{test_file_name}.{path_hash}.lv{level}"

    def _get_test_file_relative_path(self) -> str:
        """
        Get test file path relative to repo root.
        Example: tests/test_model.py
        """
        # data_item.file_path is a container path like /testbed/tests/test_model.py
        # Strip the /testbed/ prefix
        container_path = self.data_item.file_path
        if container_path.startswith('/testbed/'):
            return container_path[9:]  # Strip '/testbed/'
        return container_path

    def _generate_instance_id(self, level: int) -> str:
        """
        Generate instance_id: usr__repo.commit[:8].test_file_name.hash[:8].lv{level}
        Example: lightning-ai__pytorch-lightning.1234abcd.test_model.a1b2c3d4.lv1
        """
        repo_commit_dir = self._get_repo_commit_dirname()
        test_file_name = Path(self.data_item.file_path).stem
        test_file_relative_path = self._get_test_file_relative_path()
        path_hash = hashlib.md5(test_file_relative_path.encode()).hexdigest()[:8]
        
        return f"{repo_commit_dir}.{test_file_name}.{path_hash}.lv{level}"

    def _generate_problem_statement(self, config_path: str, level_dir: Path):
        """Generate problem_statement.md (formerly prompt.md)."""
        try:
            rendered_task = get_task(config_path)
            problem_statement_path = level_dir / "problem_statement.md"
            with open(problem_statement_path, 'w', encoding='utf-8') as f:
                f.write(rendered_task.get('prompt', ''))
        except Exception as e:
            raise RuntimeError(f"Failed to generate problem_statement.md: {e}")

    def _generate_patch_diff_lv1(self, level_dir: Path):
        """
        Generate patch.diff for level1 (records masking changes).
        Overwrite files in container with masked_files and run git diff.
        """
        container_id = None
        try:
            # Start a new container
            container_id = self.image_manager.run_container(
                specs_name=self.data_item.repo,
                working_dir="/testbed",
                prepare_env=True
            )
            
            # Clean workspace: configure git and commit all changes
            # 1. Configure git user info
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git config user.email "fb@bench.com" && git config user.name "FeatureBench"',
                timeout=30
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to configure git user info: {result.stderr}")
            
            # 2. Commit all current changes to clean workspace
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git add -A && git commit -m "Initial commit for FeatureBench evaluation" --allow-empty',
                timeout=60
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to clean workspace: {result.stderr}")
            
            # Copy masked_files into container to overwrite originals
            files_mapping = {}
            for file_path, mask_result in self.mask_results.items():
                if mask_result.success and mask_result.masked_code:
                    # file_path is a container path like /testbed/src/model.py
                    # Create temp file for masked_code
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                        tmp_file.write(mask_result.masked_code)
                        tmp_file_path = tmp_file.name
                    files_mapping[file_path] = tmp_file_path
            
            # Batch copy into container
            if files_mapping:
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=None,
                    dest_path=None,
                    use_tar=True,
                    files_mapping=files_mapping
                )
                
                # Clean up temp files
                for tmp_file_path in files_mapping.values():
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            
            # Run git diff in container
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command="git diff",
                timeout=60
            )
            
            if result.returncode == 0:
                # Save patch.diff
                patch_path = level_dir / "patch.diff"
                with open(patch_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                # if self.logger:
                #     self.logger.info(f"Generated patch.diff successfully: {patch_path}")
            else:
                raise RuntimeError(f"Failed to generate patch.diff: {result.stderr}")
                
        except Exception as e:
            raise RuntimeError(f"Error generating patch.diff: {e}")
        finally:
            # Stop and remove container
            if container_id:
                self.image_manager.stop_container(container_id)

    def _generate_test_patch_diff_lv1(self, level_dir: Path):
        """
        Generate test_patch.diff for level1 (records test file removal).
        Delete test file in container and run git diff.
        """
        container_id = None
        try:
            # Start a new container
            container_id = self.image_manager.run_container(
                specs_name=self.data_item.repo,
                working_dir="/testbed",
                prepare_env=True
            )
            
            # Clean workspace: configure git and commit all changes
            # 1. Configure git user info
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git config user.email "fb@bench.com" && git config user.name "FeatureBench"',
                timeout=30
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to configure git user info: {result.stderr}")
            
            # 2. Commit all current changes to clean workspace
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git add -A && git commit -m "Initial commit for FeatureBench evaluation" --allow-empty',
                timeout=60
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to clean workspace: {result.stderr}")
            
            # Delete test file
            test_file_path = self.data_item.file_path  # container path
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command=f"rm -f {test_file_path}",
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to delete test file: {result.stderr}")
            
            # Run git diff in container
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command="git diff",
                timeout=60
            )
            
            if result.returncode == 0:
                # Save test_patch.diff
                patch_path = level_dir / "test_patch.diff"
                with open(patch_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                # if self.logger:
                #     self.logger.info(f"Generated test_patch.diff successfully: {patch_path}")
            else:
                raise RuntimeError(f"Failed to generate test_patch.diff: {result.stderr}")
                
        except Exception as e:
            raise RuntimeError(f"Error generating test_patch.diff (lv1): {e}")
        finally:
            # Stop and remove container
            if container_id:
                self.image_manager.stop_container(container_id)

    def _generate_test_patch_diff_lv2(self, level_dir: Path):
        """
        Generate test_patch.diff for level2 (records test file adjustments).
        Replace the test file in container and run git diff.
        """
        container_id = None
        try:
            # Start a new container
            container_id = self.image_manager.run_container(
                specs_name=self.data_item.repo,
                working_dir="/testbed",
                prepare_env=True
            )
            
            # Clean workspace: configure git and commit all changes
            # 1. Configure git user info
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git config user.email "fb@bench.com" && git config user.name "FeatureBench"',
                timeout=30
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to configure git user info: {result.stderr}")
            
            # 2. Commit all current changes to clean workspace
            result = self.image_manager.exec_in_container(
                container_id=container_id,
                command='cd /testbed && git add -A && git commit -m "Initial commit for FeatureBench evaluation" --allow-empty',
                timeout=60
            )
            if result.returncode != 0:
                self.logger.warning(f"Failed to clean workspace: {result.stderr}")
            
            # Get adjusted test file content
            test_file_object_top_objects_dict = self._get_test_file_object_top_objects_dict(self.data_item)
            adjusted_test_file_content = self.llm_testfile_adjuster.adjust_test_file(test_file_object_top_objects_dict)
            adjusted_test_file_content = self._insert_agent_code_import(adjusted_test_file_content)
            
            # Create temp file with adjusted content (for container copy)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(adjusted_test_file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Copy into container to overwrite original test file
                test_file_path = self.data_item.file_path  # container path
                self.image_manager.copy_to_container(
                    container_id=container_id,
                    src_path=tmp_file_path,
                    dest_path=test_file_path,
                    is_directory=False
                )
                
                # Run git diff in container
                result = self.image_manager.exec_in_container(
                    container_id=container_id,
                    command="git diff",
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Save test_patch.diff
                    patch_path = level_dir / "test_patch.diff"
                    with open(patch_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    
                    # if self.logger:
                    #     self.logger.info(f"Generated test_patch.diff successfully: {patch_path}")
                else:
                    raise RuntimeError(f"Failed to generate test_patch.diff: {result.stderr}")
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
        except Exception as e:
            raise RuntimeError(f"Error generating test_patch.diff (lv2): {e}")
        finally:
            # Stop and remove container
            if container_id:
                self.image_manager.stop_container(container_id)

    def _generate_instance_json(self, level: int, level_dir: Path):
        """
        Generate instance.json.

        Includes:
        - instance_id: str
        - FAIL_TO_PASS: list (test file relative path)
        - PASS_TO_PASS: list (lv1: p2p list, lv2: empty list)
        - image_name: str
        - repo: str (usr/repo)
        - base_commit: str (full commit hash, not 8 chars)
        - repo_settings: dict (loaded from python.py)
        """
        # Generate instance_id
        instance_id = self._generate_instance_id(level)
        
        # Get test file relative path
        test_file_relative_path = self._get_test_file_relative_path()
        
        # FAIL_TO_PASS: test file path
        fail_to_pass = [test_file_relative_path]
        
        # PASS_TO_PASS
        if level == 1:
            # level1: p2p file list
            p2p_list = self.data_item.p2p_list if hasattr(self.data_item, 'p2p_list') else []
            pass_to_pass = []
            for p2p_path in p2p_list:
                # p2p_path is a container path; convert to relative path
                if p2p_path.startswith('/testbed/'):
                    pass_to_pass.append(p2p_path[9:])
                else:
                    pass_to_pass.append(p2p_path)
        else:
            # level2: empty list
            pass_to_pass = []
        
        # Get image name
        image_name = self.image_manager.image_info[self.data_item.repo]['instance_image']
        
        # Get repo and full commit hash
        repository = self.data_item.specs.get('repository', '')
        base_commit = self._get_full_commit_hash_complete()  # Full commit hash
        
        # Get repo_settings (loaded from python.py)
        repo_settings = self.data_item.specs
        
        # Get top_objects (collected from mask_results; used in problem_statement interface)
        top_objects = []
        for file_path, result in self.mask_results.items():
            if result.success and result.top_objects:
                top_objects.extend(result.top_objects)
        
        # Build instance.json content
        instance_dockerfile = None
        base_dockerfile = None
        setup_env_script = None
        try:
            instance_dockerfile = self.image_manager.render_instance_dockerfile(self.data_item.repo)
        except Exception:
            instance_dockerfile = None
        try:
            base_dockerfile = self.image_manager.render_base_dockerfile(self.data_item.repo)
        except Exception:
            base_dockerfile = None
        try:
            setup_env_script = self.image_manager.render_setup_env_script(self.data_item.repo)
        except Exception:
            setup_env_script = None

        instance_data = {
            'instance_id': instance_id,
            'FAIL_TO_PASS': fail_to_pass,
            'PASS_TO_PASS': pass_to_pass,
            'image_name': image_name,
            'repo': repository,
            'base_commit': base_commit,
            'repo_settings': repo_settings,
            'lines': self.deleted_lines,
            'files': self.mask_file_count,
            'functions': self.mask_object_count,
            'f2p_test_points': getattr(self.data_item, 'test_count_run', None),
            'p2p_test_points': getattr(self.data_item, 'p2p_test_points', None),
            'upd_time': getattr(self.data_item, 'last_modified', None),
            'create_time': getattr(self.data_item, 'first_commit', None),
            'top_objects': top_objects,
            'instance_dockerfile': instance_dockerfile,
            'base_dockerfile': base_dockerfile,
            'build_setup_env_sh': setup_env_script,
        }
        
        # Write file
        instance_json_path = level_dir / "instance.json"
        with open(instance_json_path, 'w', encoding='utf-8') as f:
            json.dump(instance_data, f, indent=2, ensure_ascii=False)
        
        # if self.logger:
        #     self.logger.info(f"Generated instance.json successfully: {instance_json_path}")