"""Top-object classifier using LLM."""

import json
import logging
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast

from acebench.classification.file_analyzer import FunctionClassVisitor
from acebench.utils.logger import print_build_report, save_llm_api_call
from acebench.llm.llm_caller import LLMCaller


class LLMTopClassifier(LLMCaller):
    """Classify top objects with an LLM."""
	
    def __init__(
		self,
        config,
		repo_manager,
		storage_manager,
		logger: Optional[logging.Logger] = None
	):
        """
        Initialize the top-object classifier.

        Args:
            config: Config object (logs_dir, llm_config, etc.)
            repo_manager: Repo manager
            storage_manager: Storage manager
            logger: Logger instance
        """
        # Initialize parent LLMCaller
        super().__init__(config, logger)
        
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.logs_dir = config.logs_dir

        # Load test results from storage
        repo_names = list(self.repo_manager.loaded_repos.keys())
        self.test_results: Dict[str, Dict[str, List]] = self.storage_manager.load_all_test_results(repo_names)
        
        # Initialize top_results dict
        # Shape: {specs_name: {test_file_path: {'top_objects': Set[...], 'all_top_objects_candidates': Set[...]}}}
        self.top_results: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}

    def run(self, max_workers: Optional[int] = None) -> None:
        """
        Run the top classification workflow.

        Args:
            max_workers: Max parallel workers; auto-select if None
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Starting Top classification phase...")
        self.logger.info("=" * 60)
        
        # Run LLM classification
        self._run_llm_classification(max_workers)


    def _run_llm_classification(self, max_workers: Optional[int] = None) -> None:
        """
        Run the LLM classification workflow.

        Args:
            max_workers: Max parallel workers; auto-select if None
        """
        # Set default parallelism
        if max_workers is None:
            max_workers = min(len(self.repo_manager.loaded_repos), os.cpu_count() or 1, 5)
        
        self.logger.info(f"Running LLM classification for {len(self.repo_manager.loaded_repos)} repos with {max_workers} parallel workers")
        
        # Count total test files (passed and with dynamic_trace_file in f2p_files)
        total_test_files = 0
        for specs_name in self.repo_manager.loaded_repos.keys():
            if specs_name in self.test_results:
                f2p_files = self.test_results[specs_name].get('f2p_files', [])
                # Only count files that passed and have dynamic_trace_file
                traced_files = [f for f in f2p_files if f.get('passed', False) and f.get('dynamic_trace_file')]
                if self.config.debug_sample:
                    traced_files = [
                        f for f in traced_files
                        if self.config.is_sample_selected(f.get('file_path', ''))
                    ]
                total_test_files += len(traced_files)
        
        self.logger.info(f"Total test files: {total_test_files}")
        
        # Failed repo info
        failed_repos: List[Tuple[str, str]] = []
        
        # Shared progress bar
        pbar = tqdm(total=total_test_files, desc="LLM classification", unit="file")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all LLM classification tasks
            future_to_repo = {}
            for specs_name, repo_info in self.repo_manager.loaded_repos.items():
                specs = repo_info['specs']
                # Override cache config with config.get_cache_config (debug supported)
                use_cache = self.config.get_cache_config('top', specs.get('llm_cache', True))
                max_depth = specs.get('max_depth_top', 5)
                batchsize = specs.get('batchsize_top', 5)
                
                future = executor.submit(
                    self._classify_repo_tests,
                    specs_name,
                    use_cache,
                    max_depth,
                    batchsize,
                    pbar
                )
                future_to_repo[future] = specs_name
            
            # Track progress
            completed_repos = 0
            total_repos = len(self.repo_manager.loaded_repos)
            
            # Handle completed tasks
            for future in as_completed(future_to_repo):
                specs_name = future_to_repo[future]
                completed_repos += 1
                
                try:
                    classified_count = future.result()
                    
                    pbar.set_postfix_str(f"Latest completed: {specs_name}")
                    # tqdm.write(f"✅ {specs_name}: LLM classification done (classified files={classified_count}) (repo progress: {completed_repos}/{total_repos})")
                    
                except Exception as e:
                    # Errors are logged inside; only collect failure info here
                    failed_repos.append((specs_name, str(e)))
                    pbar.set_postfix_str(f"Latest failed: {specs_name}")
        
        pbar.close()
        
        # Wait for all save tasks to complete
        self.storage_manager.wait_for_save_completion(shutdown=False)
        
        # Reload latest data from storage for reporting
        repo_names = list(self.test_results.keys())
        self.test_results = self.storage_manager.load_all_test_results(repo_names)
        
        # Aggregate classification results
        success_info = {}
        for specs_name in self.test_results.keys():
            f2p_files = self.test_results[specs_name].get('f2p_files', [])
            classified_files = [f for f in f2p_files if f.get('top_objects')]
            success_info[specs_name] = {
                'Top classified files': len(classified_files)
            }
        
        # Output final report
        print_build_report(
			logger=self.logger,
			total_repos=list(self.repo_manager.loaded_repos.items()),
			failed_repos=failed_repos,
			success_info=success_info,
            operation_name="Top classification"
        )

        # If any repo failed, raise an error
        # if failed_repos:
        #     failed_names = [name for name, _ in failed_repos]
        #     raise RuntimeError(f"Repos with failed LLM classification: {', '.join(failed_names)}")

        self.logger.info("LLM classification phase completed")

    def _classify_repo_tests(
        self,
        specs_name: str,
        use_cache: bool = True,
        max_depth: int = 5,
        batchsize: int = 5,
        pbar: Optional[tqdm] = None
    ) -> int:
        """
        Run LLM classification for test files in a single repo (f2p_files only).

        Args:
            specs_name: Repo spec name
            use_cache: Whether to use cache
            pbar: Progress bar instance

        Returns:
            int: Number of successfully classified test files
        """
        # tqdm.write(f"🏃 Starting LLM classification for {specs_name}...")
        
        if specs_name not in self.test_results:
            tqdm.write(f"❌ {specs_name}: test discovery results not found; run test discovery first")
            raise RuntimeError(f"{specs_name}: test discovery results not found; run test discovery first")
        
        # Only handle f2p_files that passed and have dynamic_trace_file
        f2p_files = self.test_results[specs_name].get('f2p_files', [])
        traced_files = [f for f in f2p_files if f.get('passed', False) and f.get('dynamic_trace_file')]
        if self.config.debug_sample:
            traced_files = [
                f for f in traced_files
                if self.config.is_sample_selected(f.get('file_path', ''))
            ]
		
        if not traced_files:
            if self.config.debug_sample:
                tqdm.write(f"🔍 {specs_name}: no matching tests after debug_sample filter, skipping")
                return 0
            error_msg = f"{specs_name}: no tests passed and contain dynamic_trace_file"
            tqdm.write(f"⚠️ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Initialize result dict
        self.top_results[specs_name] = {}
        
        # Set parallelism
        max_file_workers = min(len(traced_files), os.cpu_count() or 1, 10)  # file-level parallelism
        
        classified_count = 0
        
        with ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            future_to_file = {}
            
            for file_info in traced_files:
                file_path = file_info['file_path']
                file_test_points = file_info.get('test_points', [])
                dynamic_trace_file = file_info.get('dynamic_trace_file', '')

                # Check cache: if using cache and retry_top_classification exists and is False, skip
                if use_cache and ('retry_top_classification' in file_info) and (not file_info['retry_top_classification']):
                    # Already has classification result; skip
                    self.top_results[specs_name][file_path] = {'top_objects': set(file_info['top_objects']), 'all_top_objects_candidates': set(file_info['all_top_objects_candidates']), 'retry_top_classification': False}
                    classified_count += 1
                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                    continue
                
                # Check required information
                if not file_test_points:
                    self.logger.debug(f"Skipping test file with no test_points: {file_path}")
                    file_info['top_objects'] = []  # Save as empty list
                    if pbar:
                        pbar.update(1)
                    continue
                
                future = executor.submit(
                    self._classify_single_test,
                    specs_name,
                    file_path,
                    dynamic_trace_file,
                    file_test_points if isinstance(file_test_points, list) else list(file_test_points),
                    max_depth,
                    batchsize,
                )
                future_to_file[future] = file_info
            
            # Collect results
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                file_path = file_info['file_path']
                
                try:
                    top_objects_results = future.result()

                    retry_top_classification = top_objects_results['retry_top_classification']
                    # Convert Set[str] to List[str] for JSON serialization
                    top_objects_list = list(top_objects_results['top_objects']) if top_objects_results['top_objects'] else []
                    all_top_objects_candidates_list = list(top_objects_results['all_top_objects_candidates']) if top_objects_results['all_top_objects_candidates'] else []
                    
                    if top_objects_list:
                        classified_count += 1
                    
                except Exception:
                    # Errors are logged in _classify_single_test; record status only
                    top_objects_list = []
                    all_top_objects_candidates_list = []
                    retry_top_classification = True
                
                # Save to storage using incremental update
                self.storage_manager.save_files_status(
                    specs_name=specs_name,
                    update_single_file={
                        'file_path': file_path,
                        'updates': {'top_objects': top_objects_list, 'all_top_objects_candidates': all_top_objects_candidates_list, 'retry_top_classification': retry_top_classification}
                    }
                )
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
        
        return classified_count

    def _classify_single_test(
        self,
        specs_name: str,
        test_file_path: str,
        dynamic_trace_file: str,
        test_points: List[str],
        max_depth: int = 5,
        batchsize: int = 5
    ) -> Dict[str, Set[str]]:
        """
        Run LLM classification for a single test file.

        Args:
            specs_name: Repo spec name
            test_file_path: Local test file path
            dynamic_trace_file: Dynamic trace file path
            test_points: List of test points for the test file

        Returns:
            Dict[str, Set[str]]: top objects set and all candidates set
            Format: {'top_objects': Set[str], 'all_top_objects_candidates': Set[str], 'retry_top_classification': bool}
        """

        # tqdm.write(f"🏃 {specs_name}: classify test file {test_file_path}")

        # Call LLM to get top objects
        top_objects_results = self._get_top_objects_llm(
            dynamic_trace_file=dynamic_trace_file,
            test_points=test_points,
            test_file=test_file_path,
            specs_name=specs_name,
            max_depth=max_depth,
            batchsize=batchsize
        )
        
        # tqdm.write(f"✅ {specs_name}: classification done; top imports in {test_file_path}: {len(top_imports)}")
        self.top_results[specs_name][test_file_path] = top_objects_results
        return top_objects_results

    def _get_top_objects_llm(self, dynamic_trace_file: str, test_points: List[str], test_file: str, specs_name: str, max_depth: int = 5, batchsize: int = 5) -> Dict[str, Set[str]]:
        """
        Use LLM to get top objects.

        Args:
            dynamic_trace_file: Dynamic trace file path
            test_points: List of test points, ['test_correctness', ...]
            test_file: Test file path, e.g. /testbed/test/transformers/test_jsd.py
            specs_name: Repo spec name, e.g. SPECS_LIGER_KERNEL
            max_depth: Maximum traversal depth
            batchsize: Objects per batch
        Returns:
            Dict[str, Set[str]]: top objects set and all candidates set
            Format: {'top_objects': Set[str], 'all_top_objects_candidates': Set[str], 'retry_top_classification': bool}
        """
        # Load dynamic trace JSON
        with open(dynamic_trace_file, 'r', encoding='utf-8') as f:
            dynamic_trace_dict = json.load(f)
            all_objects = dynamic_trace_dict.get('objects', {})
        
        
        top_objects = set()
        all_top_objects_candidates = set()
        all_top_objects_candidates_not_filtered = set()

        # Get full test point IDs and initialize BFS queue
        test_points_ids = set()
        for object_id, object_info in all_objects.items():
            if object_info['name'] in test_points:
                test_points_ids.add(object_id)
        
        # BFS by layer; classify each layer in batches
        current_layer = list(test_points_ids)
        visited_objects = set(test_points_ids)
        # Track traversal depth
        depth_dict = {}
        for obj_id in test_points_ids:
            depth_dict[obj_id] = 0
        
        current_depth = 0
        
        # Traverse layer by layer
        while current_layer and current_depth < max_depth:
            self.logger.debug(f"Processing layer {current_depth} with {len(current_layer)} objects")
            
            # Collect all child nodes in current layer
            next_layer_candidates = []
            for object_id in current_layer:
                if object_id not in all_objects:
                    continue
                for child_id in all_objects[object_id]['edges']:
                    # Skip if visited
                    if child_id in visited_objects:
                        continue
                    # Ensure child_id exists in all_objects
                    if child_id not in all_objects:
                        continue
                    visited_objects.add(child_id)
                    next_layer_candidates.append(child_id)
                    depth_dict[child_id] = current_depth + 1
            
            if not next_layer_candidates:
                self.logger.debug(f"No child nodes at layer {current_depth}; BFS ends")
                break
            
            self.logger.debug(f"Layer {current_depth} collected {len(next_layer_candidates)} candidate children")
            all_top_objects_candidates_not_filtered.update(next_layer_candidates)

            # Rule-based pre-filtering
            rule_filtered_candidates = [
                obj_id for obj_id in self._select_top_objects_rule(
                    next_layer_candidates, 
                    all_objects, 
                    test_points
                )
            ]
            
            self.logger.debug(f"{len(rule_filtered_candidates)} candidates remain after rule filtering")
            
            if not rule_filtered_candidates:
                # If no candidates after rule filtering, continue to next layer
                current_layer = next_layer_candidates
                current_depth += 1
                continue
            
            # Add candidates to all-top-candidates set
            all_top_objects_candidates.update(rule_filtered_candidates)
            
            # Split candidates into batches
            batches = []
            for i in range(0, len(rule_filtered_candidates), batchsize):
                batch = rule_filtered_candidates[i:i+batchsize]
                batches.append(batch)
            
            self.logger.debug(f"Split {len(rule_filtered_candidates)} objects into {len(batches)} batches for LLM classification")
            
            # Parallel batch classification
            layer_top_objects = set()
            max_batch_workers = min(len(batches), os.cpu_count() or 1, 3)  # batch-level parallelism
            
            with ThreadPoolExecutor(max_workers=max_batch_workers) as executor:
                future_to_batch = {}
                for batch in batches:
                    future = executor.submit(
                        self._select_top_objects_llm,
                        batch,
                        all_objects,
                        test_file,
                        specs_name
                    )
                    future_to_batch[future] = batch
                
                # Collect batch classification results
                for future in as_completed(future_to_batch):
                    try:
                        batch_top_objects = future.result()
                        if batch_top_objects:
                            layer_top_objects.update(batch_top_objects)
                            self.logger.debug(f"Batch returned {len(batch_top_objects)} top objects")
                    except Exception as e:
                        tqdm.write(f"❌ {test_file} Top classification failed; saving empty list to cache (rerun will retry). Error: {e}")
                        return {'top_objects': set(), 'all_top_objects_candidates': all_top_objects_candidates, 'retry_top_classification': True}
            
            # Merge top objects for this layer
            top_objects.update(layer_top_objects)
            self.logger.debug(f"Layer {current_depth} found {len(layer_top_objects)} top objects; total {len(top_objects)}")
            
            # Build next layer: exclude identified top objects (do not traverse their children)
            next_layer = []
            for obj_id in next_layer_candidates:
                if obj_id not in layer_top_objects:
                    next_layer.append(obj_id)
            
            self.logger.debug(f"Next layer will traverse {len(next_layer)} objects (excluded {len(layer_top_objects)} top objects)")
            
            current_layer = next_layer
            current_depth += 1

        # If no top objects, log warnings
        if not all_top_objects_candidates_not_filtered:
            tqdm.write(f"⚠️ {specs_name}: BFS found no top-object candidates (before rule filtering); rerun will skip this case. Test file: {test_file}")
            return {'top_objects': set(), 'all_top_objects_candidates': set(), 'retry_top_classification': False}
        elif not all_top_objects_candidates:
            tqdm.write(f"⚠️ {specs_name}: No top-object candidates after rule filtering (before LLM); rerun will skip this case. Test file: {test_file}")
            return {'top_objects': set(), 'all_top_objects_candidates': set(), 'retry_top_classification': False}
        elif not top_objects:
            tqdm.write(f"⚠️ {specs_name}: No top objects found after LLM; rerun will skip this case. Test file: {test_file}")
            return {'top_objects': set(), 'all_top_objects_candidates': all_top_objects_candidates, 'retry_top_classification': False}
        else:
            self.logger.debug(f"BFS completed; found {len(top_objects)} top objects")
            return {'top_objects': top_objects, 'all_top_objects_candidates': all_top_objects_candidates, 'retry_top_classification': False}


    def _select_top_objects_rule(self, object_ids: List[str], all_objects: Dict[str, Dict[str, Any]], test_points: List[str]) -> List[str]:
        """
        Select top objects using rules.

        Args:
            object_ids: List of object IDs
            all_objects: Object dictionary
            test_points: List of test points
        """
        top_objects = []
        for object_id in object_ids:
            if self._is_top_object_rule(object_id, all_objects, test_points):
                top_objects.append(object_id)
        return top_objects

    def _select_top_objects_llm(self, object_ids: List[str], all_objects: Dict[str, Dict[str, Any]], test_file: str, specs_name: str) -> List[str]:
        """
        Select top objects using LLM.

        Args:
            object_ids: List of object IDs
            all_objects: Object dictionary
            test_file: Test file path
            specs_name: Repo spec name
        """
        top_objects = []
        
        # Convert test_file to local path
        test_file_local_path = self.repo_manager.convert_container_path_to_local(specs_name, test_file)
        # Read test file content
        with open(test_file_local_path, 'r', encoding='utf-8') as f:
            test_file_content = f.read()

        # Get object implementations: object_id -> object_content
        object_implementations = {}
        for object_id in object_ids:
            object_info = all_objects.get(object_id, {})
            object_file = object_info.get('file', 'Unknown')
            object_file_local_path = self.repo_manager.convert_container_path_to_local(specs_name, object_file)
            object_implementations[object_id] = self._get_object_implementation(object_id, object_file_local_path)

        # Build prompt
        prompt = self._build_prompt_for_select_top_objects(object_ids, object_implementations, all_objects, test_file, test_file_content)

        # Build message list
        msg_list = [
            {"role": "system", "content": "You are an expert Python code analyzer specialized in identifying which objects are the actual targets being tested in test files."},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM API
        start_time = time.time()
        
        # Use parent call_llm with default llm_config params
        completion = self.call_llm(messages=msg_list)
        
        end_time = time.time()
        
        # Save API call record
        save_llm_api_call(
            msg_list=msg_list,
            completion=completion,
            start_time=start_time,
            end_time=end_time,
            test_file=test_file,
            llm_model=self.llm_model,
            llm_client=self.llm_client,
            logs_dir=self.logs_dir,
            logger=self.logger
        )
        
        # Parse response
        response = completion.choices[0].message.content
        
        # Save LLM analysis report
        try:
            analysis_path = self.storage_manager.save_llm_analysis_top_batch(
                specs_name=specs_name,
                object_ids=object_ids,
                prompt=prompt,
                test_file_path=test_file,
                llm_response=response
            )
            self.logger.debug(f"LLM batch analysis report saved to: {analysis_path}")
        except Exception as e:
            tqdm.write(f"❌ {specs_name}: Failed to save LLM batch analysis report: {e}")
        
        # Parse response and get top objects list
        top_objects = self._parse_batch_response(response, object_ids)
        
        return top_objects

    def _get_object_implementation(self, object_id: str, object_file_local_path: str) -> str:
        """
        Get object implementation.

        Args:
            object_id: Object ID
            object_file_local_path: Object file path
        """
        with open(object_file_local_path, 'r', encoding='utf-8') as f:
            object_file_content = f.read()
            object_file_content_lines = object_file_content.splitlines()
        try:
            tree = ast.parse(object_file_content)
        except SyntaxError as e:
            self.logger.error(f"Failed to parse AST for file {object_file_local_path}: {e}")
            return
            
        visitor = FunctionClassVisitor()
        visitor.visit(tree)
        
        visitor.definitions
        # visitor.definitions format: qualified_name::start_line -> (start_line, end_line, type)
        for qualified_name, (start_line, end_line, _) in visitor.definitions.items():
            if qualified_name == "::".join(object_id.split('::')[1:]):
                object_start_line = int(start_line)
                object_end_line = int(end_line)
                break
        object_content = "\n".join(object_file_content_lines[object_start_line-1:object_end_line])
        return object_content

    def _build_prompt_for_select_top_objects(self, object_ids: List[str], object_implementations: Dict[str, str], all_objects: Dict[str, Dict[str, Any]], test_file: str, test_file_content: str) -> str:
        """
        Build prompt for selecting top objects.

        Args:
            object_ids: List of object IDs
            object_implementations: Object implementations dict
            all_objects: All objects dict
            test_file: Test file path
            test_file_content: Test file content
        """
        # Get test file name
        test_file_name = Path(test_file).name
        
        # Build candidate object list
        candidate_objects_info = []
        for idx, object_id in enumerate(object_ids, 1):
            object_info = all_objects.get(object_id, {})
            object_name = object_info.get('name', 'Unknown')
            object_file = object_info.get('file', 'Unknown')
            object_implementation = object_implementations.get(object_id, 'Implementation not available')
            
            candidate_info = f"""
**Candidate {idx}:**
- Object ID: `{object_id}`
- Object Name: `{object_name}`
- Object File: `{object_file}`
- Object Implementation:
```python
{object_implementation}
```
"""
            candidate_objects_info.append(candidate_info)
        
        candidates_section = "\n".join(candidate_objects_info)
        
        prompt = f"""
Task: From a list of candidate objects, identify which ones are "tested objects" in the context of a Python test file.

**Definition of "Tested Object":**
A "tested object" is an object that the test file is specifically designed to test. It represents the core functionality or feature being validated, NOT utility functions, test helpers, or infrastructure code.

**Test File Information:**
- Test file path: `{test_file}`
- Test file name: `{test_file_name}`

**Test File Content:**
```python
{test_file_content}
```

---

**Candidate Objects to Classify:**

{candidates_section}

---

**Classification Guidelines:**

**Tested Objects (should be selected):**
- Core algorithms, classes, or functions that the test file is designed to validate
- Main interfaces or APIs being tested
- Key components whose behavior is the primary focus of the test

**Non-Tested Objects (should NOT be selected):**
- Utility functions from test utilities (e.g., `test.utils.*`, `pytest.*`)
- Common tools defined in the codebase (e.g., `infer_device()`, `assert_verbose_allclose()`)

---

**Examples for Reference:**

**Example Scenario:**
- Test file: `/testbed/test/transformers/test_jsd.py` (testing JSD algorithm)

**Should Select (Tested Objects):**
- `/testbed/src/liger_kernel/transformers/jsd.py::LigerJSD.forward::64` - Core JSD implementation
- `/testbed/src/liger_kernel/transformers/jsd.py::LigerJSD.__init__::59` - JSD class initialization

**Should NOT Select (Non-Tested Objects):**
- `/testbed/test/transformers/test_jsd.py::_test_correctness_once::91` - Helper function in test file
- `/testbed/src/liger_kernel/ops/utils.py::ensure_contiguous.wrapper::34` - General utility

---

**Your Task:**

Please analyze each candidate object and determine which ones are tested objects for the given test file.

Provide your response in the following structured format:

## Analysis

For each candidate object, briefly explain whether it should be selected as a tested object and why.

## Final Answer

Provide your final selection in the following JSON format:

```json
{{
    "tested_object_ids": [
        "object_id_1",
        "object_id_2"
    ],
    "reasoning": "Brief summary of the selection criteria applied"
}}
```

**Important Notes:**
- The `tested_object_ids` list should contain ONLY the object IDs that are tested objects
- If none of the candidates are tested objects, return an empty list: `"tested_object_ids": []`
- Include the full object ID exactly as provided in the candidate list
- Unless it's obvious or you're pretty sure that a candidate object is a general purpose tool, you need to categorize it as tested object

Now, please begin your analysis for the candidate objects listed above.
"""
        
        return prompt

    def _is_top_object_rule(self, object_id: str, all_objects: Dict[str, Dict[str, Any]], test_points: List[str]) -> bool:
        """
        Determine whether an object is top using rules.

        Args:
            all_objects: All objects dict
            object_id: Object ID
            test_points: List of test points

        Returns:
            bool: Whether it is a top object
        """
        is_possible_top = True
        object_info = all_objects.get(object_id, {})
        if object_info:
            # If it is a test point, it is not top
            if object_info['name'] in test_points:
                is_possible_top = False
            # If under test folder, it is not top
            elif '/test/' in object_info['file'] or '/tests/' in object_info['file']:
                is_possible_top = False
            # If in conftest.py, it is not top
            elif 'conftest.py' in object_info['file']:
                is_possible_top = False
            # If file does not end with .py, it is not top
            elif not object_info['file'].endswith('.py'):
                is_possible_top = False
            # If file path does not start with '/', it is not top
            elif not object_info['file'].startswith('/'):
                is_possible_top = False
            # If name starts with '<', it is not top
            elif object_info['name'].startswith('<'):
                is_possible_top = False
            elif '__init__.py' in object_info['file']:
                is_possible_top = False
            # Otherwise it may be top
            else:
                is_possible_top = True
        return is_possible_top


    def _parse_batch_response(self, response: str, object_ids: List[str]) -> List[str]:
        """
        Parse LLM batch selection response and extract tested_object_ids.

        Args:
            response: Full LLM response text
            object_ids: Candidate object IDs (for validation)

        Returns:
            List[str]: Top object IDs
        """
        # Find ```json ... ``` code block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            
            # Extract tested_object_ids field
            top_object_ids = result.get('tested_object_ids', [])
            
            # Validate returned object IDs against candidate list
            valid_top_ids = []
            for obj_id in top_object_ids:
                if obj_id in object_ids:
                    valid_top_ids.append(obj_id)
                else:
                    raise ValueError(f"⚠️ LLM returned object ID not in candidate list: {obj_id}")
            
            # Record reasoning (for debugging)
            reasoning = result.get('reasoning', 'No reasoning provided')
            self.logger.debug(f"LLM batch classification: selected {len(valid_top_ids)}/{len(object_ids)} objects, reasoning={reasoning}")
            
            return valid_top_ids
        else:
            # If no JSON code block, try to extract tested_object_ids directly
            # This is a fallback
            top_ids_match = re.search(r'"tested_object_ids"\s*:\s*\[(.*?)\]', response, re.DOTALL)
            if top_ids_match:
                ids_str = top_ids_match.group(1)
                # Extract all strings in quotes
                found_ids = re.findall(r'"([^"]+)"', ids_str)
                valid_ids = [obj_id for obj_id in found_ids if obj_id in object_ids]
                self.logger.debug(f"Using fallback parsing: found {len(valid_ids)} objects")
                return valid_ids
            else:
                raise ValueError("⚠️ Failed to parse LLM batch response; no valid JSON found.")