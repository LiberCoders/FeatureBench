import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

from featurebench.mask.mask_generator import MaskResult
from featurebench.mask.signature_extractor import FunctionSignature, ClassSignature
from featurebench.llm.llm_caller import LLMCaller
from featurebench.llm.llm_exceptions import LLMException


class LLMPromptGenerator(LLMCaller):
    """Generate prompts and docstrings with LLM."""
    
    def __init__(
        self,
        config,
        repo_manager,
        storage_manager,
        data_item,
        classification_summary,
        mask_results,
        logger
    ):
        # Initialize LLMCaller base
        super().__init__(config, logger)
        
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.data_item = data_item
        self.classification_summary = classification_summary
        self.mask_results = mask_results

        # Load templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load LLM prompt templates."""
        template_path = Path(__file__).parent.parent / "resources" / "templates" / "llm_prompt_templates.yaml"
        
        if not template_path.exists():
            self.logger.warning(f"Template file not found: {template_path}")
            return {
                'function_docstring_template': '',
                'class_docstring_template': '',
                'task_statement_template': ''
            }
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = yaml.safe_load(f)
            return templates
        except Exception as e:
            raise LLMException(f"Failed to load templates: {e}")
    


    def generate_docstring_with_llm(
        self,
        file_path: str,
        result: MaskResult,
        top_structure: Dict,
    ) -> Tuple[MaskResult, Dict]:
        """
        Generate docstring with LLM.
        
        Args:
            file_path: File path (container path)
            result: MaskResult object
            top_structure: Structure dict for top objects
            
        Returns:
            Updated (result, top_structure)
        """
        # Read file content
        local_file_path = self.repo_manager.convert_container_path_to_local(self.data_item.repo, file_path)
        with open(local_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Get fully qualified names of all top objects
        top_objects = set(result.top_objects)
        
        # Load docstrings from cache
        llm_docstring_result_cache = self.storage_manager.load_llm_top_docstring(
            self.data_item.repo, 
            self.data_item.file_path,
            file_path
        )
        
        # Track whether new docstrings were generated (save cache)
        cache_updated = False
        
        # Step 1: DFS through top_structure to generate missing docstrings
        def dfs_generate_docstring(node_name: str, node: Dict) -> None:
            """DFS traversal to generate docstrings."""
            nonlocal cache_updated
            
            signature = node.get('signature')
            if not signature:
                return
            
            full_id = signature.full_id
            
            # Check if this node is a top object
            if full_id in top_objects:
                # Skip if already in cache
                if full_id in llm_docstring_result_cache:
                    self.logger.debug(f"Loaded docstring from cache: {full_id}")
                else:
                    # Use LLM to generate docstring
                    # self.logger.info(f"Generate docstring via LLM: {full_id}")
                    try:
                        docstring = self._call_llm_for_docstring(
                            full_id=full_id,
                            obj_type=node.get('type'),
                            file_content=file_content
                        )
                        llm_docstring_result_cache[full_id] = docstring
                        cache_updated = True
                        # self.logger.info(f"✅ Generated docstring: {full_id}")
                    except LLMException:
                        # Re-raise LLM-related exceptions
                        raise
                    except Exception as e:
                        raise LLMException(f"Failed to generate docstring {full_id}: {e}")
            
            # Recurse into child nodes
            children = node.get('children', {})
            for child_name, child_node in children.items():
                dfs_generate_docstring(child_name, child_node)
        
        # Iterate all top-level nodes in top_structure
        for node_name, node in top_structure.items():
            dfs_generate_docstring(node_name, node)
        
        # Step 2: Save cache if updated
        if cache_updated:
            try:
                self.storage_manager.save_llm_top_docstring(
                    self.data_item.repo,
                    self.data_item.file_path,
                    file_path,
                    llm_docstring_result_cache
                )
                # self.logger.info(f"✅ Saved LLM docstring cache: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save LLM docstring cache: {e}")
        
        # Step 3: DFS to fill cached docstrings into top_structure and result
        def dfs_update_docstring(node: Dict) -> None:
            """DFS update for docstrings."""
            signature = node.get('signature')
            if not signature:
                return
            
            full_id = signature.full_id
            
            # If cached, update docstring
            if full_id in llm_docstring_result_cache:
                new_docstring = llm_docstring_result_cache[full_id]
                if new_docstring:
                    signature.docstring = new_docstring
                    # self.logger.debug(f"Update docstring: {full_id}")
            
            # Recurse into child nodes
            children = node.get('children', {})
            for child_node in children.values():
                dfs_update_docstring(child_node)
        
        # Update docstrings in top_structure
        for node in top_structure.values():
            dfs_update_docstring(node)
        
        # Update docstrings in result.signature_info
        # Recursively update ModuleSignature functions/classes
        def update_signature_info_docstring(
            sig_obj: Any,
        ) -> None:
            """Recursively update docstrings in signature_info."""
            # Handle FunctionSignature
            if isinstance(sig_obj, FunctionSignature):
                if sig_obj.full_id in llm_docstring_result_cache:
                    new_docstring = llm_docstring_result_cache[sig_obj.full_id]
                    if new_docstring:
                        sig_obj.docstring = new_docstring
                
                # Recurse into nested functions/classes
                for nested_func in sig_obj.nested_functions:
                    update_signature_info_docstring(nested_func)
                for nested_cls in sig_obj.nested_classes:
                    update_signature_info_docstring(nested_cls)
            
            # Handle ClassSignature
            elif isinstance(sig_obj, ClassSignature):
                if sig_obj.full_id in llm_docstring_result_cache:
                    new_docstring = llm_docstring_result_cache[sig_obj.full_id]
                    if new_docstring:
                        sig_obj.docstring = new_docstring
                
                # Recurse into nested methods/classes
                for nested_func in sig_obj.nested_functions:
                    update_signature_info_docstring(nested_func)
                for nested_cls in sig_obj.nested_classes:
                    update_signature_info_docstring(nested_cls)
        
        # Update signature_info in result
        for func in result.signature_info.functions:
            update_signature_info_docstring(func)
        for cls in result.signature_info.classes:
            update_signature_info_docstring(cls)
        
        return result, top_structure
    
    def generate_task_statement_with_llm(
        self,
        top_objects_ids: List[str]
    ) -> str:
        """
        Generate a task statement using the LLM.
        
        Args:
            top_objects_ids: list of top object IDs (e.g. '/testbed/file.py::Class.method::123')
            
        Returns:
            str: generated task statement
        """
        if not self.llm_client or not self.llm_model:
            raise LLMException("LLM client is not initialized")
        
        def _find_sig(module_sig, target_full_id):
            def match_function(func_sig):
                if func_sig.full_id == target_full_id:
                    return func_sig
                for nf in func_sig.nested_functions:
                    found = match_function(nf)
                    if found:
                        return found
                for nc in func_sig.nested_classes:
                    found = match_class(nc)
                    if found:
                        return found
                return None

            def match_class(cls_sig):
                if cls_sig.full_id == target_full_id:
                    return cls_sig
                for mf in cls_sig.nested_functions:
                    found = match_function(mf)
                    if found:
                        return found
                for nc in cls_sig.nested_classes:
                    found = match_class(nc)
                    if found:
                        return found
                return None

            for func in module_sig.functions:
                found = match_function(func)
                if found:
                    return found
            for cls in module_sig.classes:
                found = match_class(cls)
                if found:
                    return found
            return None

        def _render_object(full_id: str) -> str:
            file_path = full_id.split('::')[0]
            mask_result = self.mask_results.get(file_path)
            sig_info = mask_result.signature_info if mask_result else None
            if sig_info:
                sig = _find_sig(sig_info, full_id)
                if sig and getattr(sig, "content", None):
                    return sig.content
            return f"# Source not found: {full_id}"
        
        # Format top_objects_ids into string list
        # Extract qualified name from obj_id (second part)
        top_objects_names = []
        for obj_id in top_objects_ids:
            parts = obj_id.split('::')
            if len(parts) >= 2:
                # Format: file_path::qualified_name::line_number
                # Extract qualified_name with file path
                file_path = parts[0]
                qualified_name = parts[1]
                top_objects_names.append(f"{qualified_name} (from {file_path})")
            else:
                top_objects_names.append(obj_id)
        
        # Format list to string
        top_objects_list_str = '\n'.join([f"  - {name}" for name in top_objects_names])
        
        # Use only top object bodies to reduce context length
        top_file_content_str = ""
        for obj_id in top_objects_ids:
            snippet = _render_object(obj_id)
            top_file_content_str += f"\n\n{'='*80}\n"
            top_file_content_str += f"Object: {obj_id}\n"
            top_file_content_str += f"{'='*80}\n"
            top_file_content_str += f"```python\n{snippet}\n```\n"
        
        # Generate prompt from template
        template = self.templates.get('task_statement_template', '')
        if not template:
            raise LLMException("task_statement_template template is missing")
        
        prompt = template.format(
            top_objects_ids=top_objects_list_str,  # Insert object list
            top_file_content_dict=top_file_content_str  # Insert source context
        )

        # Persist task-statement prompt for debugging
        try:
            specs_name = getattr(self.data_item, "repo", "") or "unknown_repo"
            test_file_path = getattr(self.data_item, "file_path", "") or "unknown_file"
            self.storage_manager.save_llm_task_statement_prompt(specs_name, test_file_path, prompt)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to save task statement prompt: {e}")
        
        # Call LLM API
        try:
            # self.logger.info(f"Generate task statement with {len(top_objects_ids)} top objects")
            
            # Use base call_llm with default llm_config params
            response = self.call_llm(
                messages=[{"role": "user", "content": prompt}]
            )
            
            task_statement = self.get_llm_response_text(response)
            # self.logger.info("✅ Generated task statement")
            return task_statement
            
        except LLMException:
            # Re-raise LLM-related exceptions
            raise
        except Exception as e:
            raise LLMException(f"LLM API call failed: {e}")


    def _call_llm_for_docstring(
        self,
        full_id: str,
        obj_type: str,
        file_content: str
    ) -> str:
        """
        Call the LLM API to generate a docstring.
        
        Args:
            full_id: Unique object identifier
            obj_type: Object type ('function' or 'class')
            file_content: File content
            
        Returns:
            Generated docstring
        """
        if not self.llm_client or not self.llm_model:
            raise LLMException("LLM client is not initialized")
        
        # Choose template by object type
        if obj_type == 'function':
            template = self.templates.get('function_docstring_template', '')
            prompt = template.format(
                function_qualified_name=full_id.split('::')[1],
                file_content=file_content
            )
        elif obj_type == 'class':
            template = self.templates.get('class_docstring_template', '')
            prompt = template.format(
                class_qualified_name=full_id.split('::')[1],
                file_content=file_content
            )
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
        
        # Call LLM API
        try:
            # Use base call_llm with default llm_config params
            response = self.call_llm(
                messages=[{"role": "user", "content": prompt}]
            )
            
            docstring = self.get_llm_response_text(response)
            return docstring
            
        except LLMException:
            # Re-raise LLM-related exceptions
            raise
        except Exception as e:
            raise LLMException(f"LLM API call failed: {e}")