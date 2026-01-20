from typing import Dict,  Any, List, Tuple, Optional
import logging
from pathlib import Path
import ast
import yaml
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from acebench.classification.file_analyzer import FunctionClassVisitor
from acebench.llm.llm_caller import LLMCaller
from acebench.llm.llm_exceptions import LLMException


class LLMTestfileAdjuster(LLMCaller):
    """Adjust LV2 test files with LLM."""
    
    def __init__(
        self,
        config,
        repo_manager,
        storage_manager,
        data_item,
        logger
    ):
        # Initialize LLMCaller base
        super().__init__(config, logger)
        
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.data_item = data_item

        # Load templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load LLM prompt templates."""
        template_path = Path(__file__).parent.parent / "resources" / "templates" / "llm_prompt_templates.yaml"
        
        if not template_path.exists():
            self.logger.warning(f"Template file not found: {template_path}")
            return {
                'test_file_adjustment_template': ''
            }
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = yaml.safe_load(f)
            return templates
        except Exception as e:
            raise LLMException(f"Failed to load templates: {e}")
    
    
    def _get_local_file_path(self, file_path: str) -> str:
        """Get local file path."""
        local_file_path = self.repo_manager.convert_container_path_to_local(self.data_item.repo, file_path)
        return local_file_path
    
    def _detect_indentation(self, code: str) -> Optional[str]:
        """
        Detect indentation of the first line.
        
        Args:
            code: Code string
            
        Returns:
            Indentation string (spaces/tabs) or None if no indent
        """
        lines = code.splitlines()
        if not lines:
            return None
        
        first_line = lines[0]
        # Match leading whitespace with regex
        match = re.match(r'^(\s+)', first_line)
        if match:
            return match.group(1)
        return None
    
    def _remove_indentation(self, code: str, indent: str) -> str:
        """
        Remove the specified indentation from each line of code.
        
        Args:
            code: Code string
            indent: Indentation string to remove
            
        Returns:
            Code with indentation removed
        """
        lines = code.splitlines()
        dedented_lines = []
        for line in lines:
            if line.startswith(indent):
                dedented_lines.append(line[len(indent):])
            else:
                # If a line doesn't start with the indent (e.g., blank), keep as-is
                dedented_lines.append(line)
        return '\n'.join(dedented_lines)
    
    def _add_indentation(self, code: str, indent: str) -> str:
        """
        Add the specified indentation to each line of code.
        
        Args:
            code: Code string
            indent: Indentation string to add
            
        Returns:
            Code with indentation added
        """
        lines = code.splitlines()
        indented_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                indented_lines.append(indent + line)
            else:  # Keep empty lines as-is
                indented_lines.append(line)
        return '\n'.join(indented_lines)

    def adjust_test_file(self, test_file_object_top_objects_dict: Dict[str, List[str]]) -> str:
        """
        Adjust test file by replacing repo object refs with agent_code imports.
        
        Args:
            test_file_object_top_objects_dict: {test_file_object_id: [top_object_ids]}
            
        Returns:
            Adjusted test file content
        """
        # Get repo name
        repo_name = self.data_item.specs.get('library_name', '').replace('-', '_')

        # Build object_need_agent_code_prefix by mapping full_id to top-level qualified name
        object_need_agent_code_prefix = {}
        for test_file_object_id, top_objects in test_file_object_top_objects_dict.items():
            object_need_agent_code_prefix[test_file_object_id] = set(top_object.split('::')[1].split('.')[0] for top_object in top_objects)
        # Drop empty sets
        object_need_agent_code_prefix = {k: v for k, v in object_need_agent_code_prefix.items() if v}

        test_file_local_path = self._get_local_file_path(self.data_item.file_path)
        with open(test_file_local_path, 'r', encoding='utf-8') as f:
            test_file_content = f.read()
            test_file_content_lines = test_file_content.splitlines()
        try:
            tree = ast.parse(test_file_content)
        except SyntaxError as e:
            self.logger.error(f"Failed to parse AST for {test_file_local_path}: {e}")
            return test_file_content

        # Get object start/end lines in test file: {id: [start, end]}
        visitor = FunctionClassVisitor()
        visitor.visit(tree)
        test_file_object_start_end_line_dict = {}
        for test_file_object_id, top_objects in test_file_object_top_objects_dict.items():
            # visitor.definitions: qualified_name::start_line -> (start, end, type)
            for qualified_name, (start_line, end_line, _) in visitor.definitions.items():
                if qualified_name == "::".join(test_file_object_id.split('::')[1:]):
                    test_file_object_start_end_line_dict[test_file_object_id] = [int(start_line), int(end_line)]
                    break
        
        # If no objects need adjustment, return original
        if not object_need_agent_code_prefix:
            tqdm.write("No objects to adjust; returning original file content")
            return test_file_content
        
        # Parallel run: call LLM for each non-empty target object
        adjusted_code_segments = {}
        max_workers = min(10, len(object_need_agent_code_prefix))  # Max 10 workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_object_id = {}
            for test_file_object_id, objects_to_prefix in object_need_agent_code_prefix.items():
                if test_file_object_id not in test_file_object_start_end_line_dict:
                    self.logger.warning(f"Line info not found for object {test_file_object_id}; skipping")
                    continue
                
                start_line, end_line = test_file_object_start_end_line_dict[test_file_object_id]
                # Extract original code (line numbers start at 1; list index at 0)
                original_code_lines = test_file_content_lines[start_line - 1:end_line]
                original_code = '\n'.join(original_code_lines)
                
                future = executor.submit(
                    self._adjust_code_segment_with_retry,
                    test_file_object_id,
                    original_code,
                    list(objects_to_prefix),
                    len(original_code_lines),
                    repo_name
                )
                future_to_object_id[future] = (test_file_object_id, start_line, end_line)
            
            # Collect results
            for future in as_completed(future_to_object_id):
                test_file_object_id, start_line, end_line = future_to_object_id[future]
                try:
                    adjusted_code = future.result()
                    adjusted_code_segments[test_file_object_id] = {
                        'code': adjusted_code,
                        'start_line': start_line,
                        'end_line': end_line
                    }
                except LLMException:
                    # Re-raise LLM-related exceptions
                    raise
                except Exception as e:
                    # Convert other exceptions to LLMException
                    raise LLMException(f"Unexpected error while adjusting object {test_file_object_id}: {e}")
        
        # Replace adjusted code segments back into file content
        for test_file_object_id, segment_info in adjusted_code_segments.items():
            adjusted_code = segment_info['code']
            start_line = segment_info['start_line']
            end_line = segment_info['end_line']
            
            adjusted_code_lines = adjusted_code.splitlines()
            # Replace lines
            test_file_content_lines[start_line - 1:end_line] = adjusted_code_lines
        
        # Join final content and return
        final_content = '\n'.join(test_file_content_lines)
        return final_content
    
    def _adjust_code_segment_with_retry(
        self,
        test_file_object_id: str,
        original_code: str,
        objects_to_prefix: List[str],
        expected_line_count: int,
        repo_name: str,
        max_retries: int = 3
    ) -> str:
        """
        Adjust a code segment with the LLM, retrying on mismatch.
        
        Args:
            test_file_object_id: Test object ID
            original_code: Original code (may be indented)
            objects_to_prefix: List of objects to replace
            expected_line_count: Expected line count
            repo_name: Repo name
            max_retries: Max retry attempts
            
        Returns:
            Adjusted code (preserves original indentation)
            
        Raises:
            LLMException: If all retries fail
        """
        # Detect and remove indent from first line
        indent = self._detect_indentation(original_code)
        if indent:
            # Strip indent before LLM processing
            code_without_indent = self._remove_indentation(original_code, indent)
            self.logger.debug(f"Detected indentation (length {len(indent)}); removed before LLM")
        else:
            code_without_indent = original_code
        
        last_error_msg = ""
        
        for attempt in range(max_retries):
            try:
                # Call LLM with de-indented code
                # Pass current attempt and max retries
                adjusted_code = self._call_llm_for_adjustment(
                    code_without_indent,
                    objects_to_prefix,
                    repo_name,
                    attempt_number=attempt + 1,
                    max_attempts=max_retries
                )
                
                # Validate line count
                adjusted_lines = adjusted_code.splitlines()
                if len(adjusted_lines) != expected_line_count:
                    last_error_msg = f"Line count mismatch (expected {expected_line_count}, got {len(adjusted_lines)})"
                    self.logger.warning(
                        f"Object {test_file_object_id} adjustment {last_error_msg}; "
                        f"retry {attempt + 1}/{max_retries}"
                    )
                    if attempt == max_retries - 1:
                        raise LLMException(
                            f"Object {test_file_object_id} failed after max retries: {last_error_msg}"
                        )
                    continue
                
                # Validate content: check object references were replaced
                # Note: compare de-indented versions
                is_valid, error_msg = self._validate_adjustment(
                    code_without_indent,
                    adjusted_code
                )
                
                if is_valid:
                    # If originally indented, add indent back
                    if indent:
                        adjusted_code_with_indent = self._add_indentation(adjusted_code, indent)
                        self.logger.debug("Re-applied indentation to adjusted code")
                        return adjusted_code_with_indent
                    else:
                        return adjusted_code
                else:
                    last_error_msg = error_msg
                    self.logger.warning(
                        f"Object {test_file_object_id} adjustment invalid: {error_msg}, "
                        f"retry {attempt + 1}/{max_retries}"
                    )
                    if attempt == max_retries - 1:
                        raise LLMException(
                            f"Object {test_file_object_id} failed after max retries: {last_error_msg}"
                        )
                    
            except LLMException as e:
                # LLM-related exceptions: propagate (no retry)
                raise
            except Exception as e:
                last_error_msg = str(e)
                self.logger.warning(f"Adjustment failed; retry {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise LLMException(
                        f"Object {test_file_object_id} LLM call failed (after {max_retries} retries): {e}"
                    )
        
        # Should not reach here; raise just in case
        raise LLMException(f"Object {test_file_object_id} LLM adjustment failed: {last_error_msg}")
    
    def _validate_adjustment(
        self,
        original_code: str,
        adjusted_code: str
    ) -> Tuple[bool, str]:
        """
        Validate adjusted code (line count only).
        
        Args:
            original_code: Original code
            adjusted_code: Adjusted code
            
        Returns:
            (is_valid, error_message)
        """
        # Validate line count
        original_lines = original_code.splitlines()
        adjusted_lines = adjusted_code.splitlines()
        
        if len(original_lines) != len(adjusted_lines):
            return False, f"Line count mismatch: original {len(original_lines)} lines, adjusted {len(adjusted_lines)} lines"
        
        # Pass if line counts match
        return True, ""
    
    def _call_llm_for_adjustment(
        self,
        original_code: str,
        objects_to_prefix: List[str],
        repo_name: str,
        attempt_number: int = 1,
        max_attempts: int = 3
    ) -> str:
        """
        Call the LLM API to adjust code.
        
        Args:
            original_code: Original code
            objects_to_prefix: Objects to replace
            repo_name: Repo name
            attempt_number: Current attempt (1-based)
            max_attempts: Max attempts
            
        Returns:
            Adjusted code
        """
        if not self.llm_client or not self.llm_model:
            raise LLMException("LLM client is not initialized")
        
        # Generate prompt from template
        template = self.templates.get('test_file_adjustment_template', '')
        if not template:
            raise LLMException("test_file_adjustment_template template is missing")
        
        # Format object list
        objects_list_str = '\n'.join([f"  - {obj}" for obj in objects_to_prefix])
        
        prompt = template.format(
            repo_name=repo_name,
            objects_to_prefix=objects_list_str,
            original_code=original_code
        )
        
        # If not first attempt, add retry hint
        if attempt_number > 1:
            retry_warning = f"""
⚠️ **IMPORTANT - This is attempt {attempt_number} of {max_attempts}**

Your previous attempt failed because you modified content other than just the specified object references. 

**CRITICAL REMINDERS:**
1. You MUST ONLY replace references to the specified objects from "{repo_name}" with "agent_code" imports
2. DO NOT modify references from external libraries (e.g., torch, numpy, etc.)
3. DO NOT modify comments, whitespace, indentation, variable names, or any other code
4. The line count MUST remain exactly the same
5. The ONLY changes should be replacing "{repo_name}.ObjectName" or standalone "ObjectName" (from {repo_name}) with "agent_code.ObjectName"

Please be extremely careful this time and ONLY replace the specified object references!

---

"""
            prompt = retry_warning + prompt
        
        # Call LLM API
        try:
            # Use base call_llm with default llm_config params
            response = self.call_llm(
                messages=[{"role": "user", "content": prompt}]
            )
            
            adjusted_code = self.get_llm_response_text(response)
            
            # Remove possible markdown code fences
            if adjusted_code.startswith('```python'):
                adjusted_code = adjusted_code[len('```python'):].strip()
            if adjusted_code.startswith('```'):
                adjusted_code = adjusted_code[3:].strip()
            if adjusted_code.endswith('```'):
                adjusted_code = adjusted_code[:-3].strip()
            
            return adjusted_code
            
        except LLMException:
            # Re-raise LLM-related exceptions
            raise
        except Exception as e:
            raise LLMException(f"LLM API call failed: {e}")

