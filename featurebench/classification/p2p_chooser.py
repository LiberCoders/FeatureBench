"""P2P chooser for selecting pass-to-pass tests."""

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import random

from featurebench.utils.logger import print_build_report, create_dynamic_tracer_logger, create_p2p_chooser_logger


class P2PChooser:
    """Choose p2p tests based on trace data."""
    
    def __init__(
        self,
        config,
        repo_manager,
        storage_manager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize P2P chooser.
        
        Args:
            repo_manager: Repo manager
            image_manager: Image manager
            storage_manager: Storage manager
            logs_dir: Log directory
            logger: Logger
        """
        self.config = config
        self.repo_manager = repo_manager
        self.storage_manager = storage_manager
        self.logs_dir = config.logs_dir
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Load test results from file
        repo_names = list(self.repo_manager.loaded_repos.keys())
        self.test_results: Dict[str, Dict[str, List]] = self.storage_manager.load_all_test_results(repo_names)

        # Store updated top_objects: {specs_name: {test_file: updated_top_objects}}
        self.updated_top_objects: Dict[str, Dict[str, List]] = {}

        # Store p2p list results: {specs_name: {test_file: p2p_list}}
        self.p2p_list_results: Dict[str, Dict[str, str]] = {}
    
    def run(self, max_workers: Optional[int] = None) -> None:
        """
        Parallel p2p selection workflow.

        Args:
            max_workers: Max parallel workers; auto-select if None
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Starting p2p selection stage...")
        self.logger.info("=" * 60)
        
        # Set default parallelism
        if max_workers is None:
            max_workers = min(len(self.repo_manager.loaded_repos), os.cpu_count() or 1, 5)
        
        self.logger.info(
            f"Running p2p selection for {len(self.repo_manager.loaded_repos)} repos with {max_workers} workers"
        )
        
        # Count total test files (passed, with dynamic_trace_file and top_objects)
        total_test_files = 0
        for specs_name in self.repo_manager.loaded_repos.keys():
            if specs_name in self.test_results:
                f2p_files = self.test_results[specs_name].get('f2p_files', [])
                # Only count files meeting all conditions
                valid_files = [f for f in f2p_files 
                              if f.get('passed', False) 
                              and f.get('dynamic_trace_file') 
                              and f.get('top_objects')]
                if self.config.debug_sample:
                    valid_files = [
                        f for f in valid_files
                        if self.config.is_sample_selected(f.get('file_path', ''))
                    ]
                total_test_files += len(valid_files)
        
        self.logger.info(f"Total f2p files: {total_test_files}")
        
        # Failed repo info
        failed_repos: List[Tuple[str, str]] = []
        
        # Shared progress bar
        pbar = tqdm(total=total_test_files, desc="p2p selection", unit="file")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all p2p selection tasks
            future_to_repo = {}
            for specs_name, repo_info in self.repo_manager.loaded_repos.items():
                specs = repo_info['specs']
                # Override cache config (debug supported)
                use_cache = self.config.get_cache_config('p2p', specs.get('p2p_cache', True))
                
                future = executor.submit(
                    self._choose_p2p_list,
                    specs_name,
                    specs,
                    use_cache,
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
                    p2p_list_count = future.result()
                    
                    pbar.set_postfix_str(f"Latest completed: {specs_name}")
                    # tqdm.write(f"✅ {specs_name}: p2p selection completed (selected files={p2p_list_count}) (repo progress: {completed_repos}/{total_repos})")
                    
                except Exception as e:
                # Errors are logged inside; only collect failure info here
                    failed_repos.append((specs_name, str(e)))
                pbar.set_postfix_str(f"Latest failed: {specs_name}")
        
        pbar.close()
        
        # Wait for all save tasks to complete
        self.storage_manager.wait_for_save_completion(shutdown=False)
        
        # Reload latest data for report
        repo_names = list(self.test_results.keys())
        self.test_results = self.storage_manager.load_all_test_results(repo_names)
        
        # Summarize p2p selection results
        success_info = {}
        for specs_name in self.test_results.keys():
            f2p_files = self.test_results[specs_name].get('f2p_files', [])
            selected_files = [f for f in f2p_files if f.get('p2p_list')]
            success_info[specs_name] = {
                'p2p selected files': len(selected_files)
            }
        
        print_build_report(
            logger=self.logger,
            total_repos=list(self.repo_manager.loaded_repos.items()),
            failed_repos=failed_repos,
            success_info=success_info,
            operation_name="p2p selection"
        )
        
        # If there are failed repos, raise an exception
        # if failed_repos:
        #     failed_names = [name for name, _ in failed_repos]
        #     raise RuntimeError(f"Repos failed in p2p selection: {', '.join(failed_names)}")
        
        self.logger.info("p2p selection stage completed")
    
    def _choose_p2p_list(
        self,
        specs_name: str,
        specs: Dict,
        use_cache: bool = True,
        pbar: Optional[tqdm] = None
    ) -> int:
        """
        Run p2p selection for all test files in a single repo.

        Args:
            specs_name: Repo spec name
            specs: Repo specs config
            use_cache: Whether to use cache
            pbar: Progress bar instance

        Returns:
            int: Number of files selected for p2p
        """
        # tqdm.write(f"🏃 Starting p2p selection for {specs_name}...")
        
        # Get test file list
        if specs_name not in self.test_results:
            error_msg = f"{specs_name}: test discovery results not found; run test discovery first"
            tqdm.write(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        p2p_files = self.test_results[specs_name].get('p2p_files', [])
        f2p_files = self.test_results[specs_name].get('f2p_files', [])
        
        # Only handle f2p files that passed tests and have dynamic_trace_file + top_objects
        valid_f2p_files = [f for f in f2p_files 
                          if f.get('passed', False) 
                          and f.get('dynamic_trace_file') 
                          and f.get('top_objects')]
        if self.config.debug_sample:
            valid_f2p_files = [
                f for f in valid_f2p_files
                if self.config.is_sample_selected(f.get('file_path', ''))
            ]
		
        if not valid_f2p_files:
            if self.config.debug_sample:
                tqdm.write(f"🔍 {specs_name}: no usable test files after debug_sample filter, skipping")
                return 0
            error_msg = (
                f"{specs_name}: no tests meet criteria (must pass tests, have dynamic_trace_file and top_objects)"
            )
            tqdm.write(f"⚠️ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Get min/max p2p file list sizes
        min_p2p_files = specs['min_p2p_files']
        max_p2p_files = specs['max_p2p_files']
        
        # Initialize result dicts
        self.p2p_list_results[specs_name] = {}
        self.updated_top_objects[specs_name] = {}
        
        # Set parallelism
        max_workers = min(len(valid_f2p_files), os.cpu_count() or 1, 10)
        
        p2p_list_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            
            for file_info in valid_f2p_files:
                file_path = file_info['file_path']
                
                # Cache check: if p2p_list exists and has value, skip
                if use_cache and 'p2p_list' in file_info and file_info['p2p_list']:
                    # Already have p2p list
                    self.p2p_list_results[specs_name][file_path] = file_info['p2p_list']
                    p2p_list_count += 1
                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                    continue
                
                future = executor.submit(
                    self._choose_p2p_single_test,
                    specs_name,
                    file_path,
                    min_p2p_files,
                    max_p2p_files
                )
                future_to_file[future] = file_info
            
            # Collect results
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                file_path = file_info['file_path']
                
                try:
                    choose_result = future.result()
                    
                    p2p_list = choose_result['p2p_list']
                    updated_top_objects = choose_result['updated_top_objects']
                    
                    if p2p_list:
                        self.p2p_list_results[specs_name][file_path] = p2p_list
                        self.updated_top_objects[specs_name][file_path] = updated_top_objects
                        p2p_list_count += 1
                    
                except Exception:
                    # Errors are logged in _choose_p2p_single_test; only record status here
                    p2p_list = None
                    updated_top_objects = None
                
                # Save using incremental update mode
                self.storage_manager.save_files_status(
                    specs_name=specs_name,
                    update_single_file={
                        'file_path': file_path,
                        'updates': {
                            'p2p_list': p2p_list,
                            'updated_top_objects': updated_top_objects
                        }
                    }
                )
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
        
        return p2p_list_count
    
    def _choose_p2p_single_test(
        self,
        specs_name: str,
        test_file: str,
        min_p2p_files: int,
        max_p2p_files: int,
    ) -> Optional[Dict]:
        """
        Run p2p selection for a single test file.

        Args:
            specs_name: Repo spec name
            test_file: Test file path (container path)
            min_p2p_files: Minimum p2p file list size
            max_p2p_files: Maximum p2p file list size

        Returns:
            Optional[dict]: p2p list result and updated_top_objects, None on failure
                Format: {'p2p_list': List[str], 'updated_top_objects': List[str]}
        """
        # Create dedicated logger
        file_logger = create_p2p_chooser_logger(
            specs_name=specs_name,
            test_file=test_file,
            logs_dir=self.logs_dir
        )
        
        # tqdm.write(f"🏃 {specs_name}: selecting p2p files for {test_file}, log: {file_logger.log_file_path}")
        
        try:
            # 1. Get top_objects for current test file
            current_top_objects = self._get_current_test_top_objects(specs_name, test_file)
            if current_top_objects is None:
                error_msg = f"Failed to get top_objects for test file {test_file}"
                tqdm.write(f"❌ {specs_name}: {error_msg}")
                raise RuntimeError(error_msg)
            
            file_logger.info(f"Top_objects for current test file: {len(current_top_objects)}")
            file_logger.info(f"Top objects: {current_top_objects}")

            if not current_top_objects:
                file_logger.info("Top objects empty for current test file; returning empty result")
                return {
                    'p2p_list': None,
                    'updated_top_objects': []
                }
            
            # 2. Get all candidate p2p files
            p2p_files = self.test_results[specs_name].get('p2p_files', [])
            if not p2p_files:
                error_msg = "No p2p test files found"
                tqdm.write(f"❌ {specs_name}: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Only select p2p files that passed tests and have dynamic_trace_file
            valid_p2p_files = [f for f in p2p_files 
                              if f.get('passed', False) 
                              and f.get('dynamic_trace_file')]
            
            if not valid_p2p_files:
                error_msg = "No p2p test files meet criteria (passed tests and have dynamic_trace_file)"
                tqdm.write(f"❌ {specs_name}: {error_msg}")
                raise RuntimeError(error_msg)
            
            file_logger.info(
                f"Found {len(valid_p2p_files)} eligible p2p test files (passed tests and have dynamic_trace_file)"
            )
            
            # 3. Filter other test files in same repo (exclude current file)
            same_repo_files = []
            for file_info in valid_p2p_files:
                p2p_file_path = file_info['file_path']
                if p2p_file_path != test_file:
                    same_repo_files.append(p2p_file_path)
            
            if not same_repo_files:
                error_msg = "No other usable p2p test files found"
                tqdm.write(f"❌ {specs_name}: {error_msg}")
                raise RuntimeError(error_msg)
            
            file_logger.info(f"Selected {len(same_repo_files)} candidate p2p files (excluding current file)")
            
            # 4. Count top_objects usage across p2p trace files
            file_logger.info("Counting top_objects usage in p2p files")
            top_objects_call_count = self._count_top_objects_usage(
                current_top_objects, same_repo_files, specs_name, file_logger
            )
            file_logger.info(f"Top objects usage count complete: {top_objects_call_count}")
            
            # 5. Create mutable copy of current_top_objects to prune frequent objects
            working_top_objects = current_top_objects.copy()
            file_logger.info(f"Start p2p selection, initial top_objects: {len(working_top_objects)}")
            
            # 6. Loop to find enough valid_p2p_files
            max_overlap_tops = 0  # No overlap by default
            iteration = 0
            while working_top_objects:
                iteration += 1
                file_logger.info(f"Round {iteration}, current top_objects: {len(working_top_objects)}")
                
                # Filter p2p files by overlap constraint
                valid_p2p_files = []
                for p2p_file in same_repo_files:
                    # Load dynamic trace results for p2p file
                    traced_objects = self._load_dynamic_trace_objects(specs_name, p2p_file, file_logger)
                    if traced_objects is None:
                        file_logger.info(f"Failed to load trace results for {p2p_file}, skipping")
                        continue

                    # Compute overlapping top objects
                    overlap_count = self._count_top_objects_overlap(working_top_objects, traced_objects)
                    
                    if overlap_count <= max_overlap_tops:
                        valid_p2p_files.append(p2p_file)
                        file_logger.info(
                            f"✅ p2p file {Path(p2p_file).name} has {overlap_count} overlapping top objects; ok"
                        )
                    else:
                        file_logger.info(
                            f"❌ p2p file {Path(p2p_file).name} has {overlap_count} overlapping top objects; limit {max_overlap_tops}"
                        )

                file_logger.info(f"Round selected {len(valid_p2p_files)} eligible p2p files")

                # Check minimum p2p file requirement
                if len(valid_p2p_files) >= min_p2p_files:
                    # Requirement met; exit loop
                    file_logger.info(f"Minimum p2p file requirement met ({min_p2p_files}), exiting loop")
                    break
                    
                # Not enough; remove most frequently used top object
                if not working_top_objects:
                    # No top objects left; exit loop
                    file_logger.info("No more top objects to remove; exiting loop")
                    break

                # Find most frequently used object in working_top_objects
                max_count = -1
                most_frequent_obj = None
                for obj in working_top_objects:
                    count = top_objects_call_count.get(obj, 0)
                    if count > max_count:
                        max_count = count
                        most_frequent_obj = obj
                
                if most_frequent_obj:
                    working_top_objects.remove(most_frequent_obj)
                    file_logger.info(
                        f"Removed most frequent top object '{most_frequent_obj}' (count: {max_count}), "
                        f"remaining {len(working_top_objects)}"
                    )
                else:
                    # Should not happen; safety fallback
                    file_logger.info("No removable top object found; exiting loop")
                    break

            # 7. If working_top_objects empty, return empty result
            if not working_top_objects:
                file_logger.info(
                    f"After removing all top objects, still insufficient p2p files (need {min_p2p_files}); return empty"
                )
                return {
                    'p2p_list': None,
                    'updated_top_objects': []
                }
            
            # 8. Randomly select from eligible files
            if not valid_p2p_files:
                file_logger.info(
                    f"No p2p files meet overlap criteria (max {max_overlap_tops}); return empty"
                )
                return {
                    'p2p_list': None,
                    'updated_top_objects': working_top_objects
                }
            
            num_to_select = min(len(valid_p2p_files), max_p2p_files)
            if len(valid_p2p_files) < max_p2p_files:
                file_logger.info(
                    f"Only {len(valid_p2p_files)} eligible p2p files (< {max_p2p_files}); selecting all"
                )
            
            selected_p2p = random.sample(valid_p2p_files, num_to_select)
            file_logger.info(f"Randomly selected {len(selected_p2p)} p2p files: {[Path(p).name for p in selected_p2p]}")

            # 9. Build p2p_list string (comma-separated paths)
            file_logger.info(f"Final p2p_list: {selected_p2p}")
            file_logger.info(f"Final updated_top_objects: {working_top_objects}")
            
            # tqdm.write(f"✅ {specs_name}: selected {len(selected_p2p)} pass2pass files for {Path(test_file).name}, log: {file_logger.log_file_path}")
            
            return {
                'p2p_list': selected_p2p,
                'updated_top_objects': working_top_objects
            }
            
        except Exception as e:
            error_msg = f"p2p selection error - {test_file}: {e}"
            file_logger.error(f"ERROR: {error_msg}")
            tqdm.write(
                f"❌ {specs_name}: p2p selection failed - {Path(test_file).name}, "
                f"error: {str(e)}..., log: {file_logger.log_file_path}"
            )
            raise RuntimeError(f"{error_msg}, log: {file_logger.log_file_path}") from e
    
    def _get_current_test_top_objects(self, specs_name: str, test_file: str) -> Optional[List[str]]:
        """
        Get top_objects for the current test file.

        Args:
            specs_name: Repo spec name
            test_file: Test file path

        Returns:
            Optional[List[str]]: top_objects list, None on failure
        """
        try:
            # Get top_objects from test_results
            if specs_name not in self.test_results:
                return None
            
            f2p_files = self.test_results[specs_name].get('f2p_files', [])
            for file_info in f2p_files:
                if file_info.get('file_path') == test_file:
                    return file_info.get('top_objects')
            
            return None
            
        except Exception as e:
            tqdm.write(f"❌ {specs_name}: failed to get top_objects for {test_file}: {str(e)}")
            return None
    
    def _count_top_objects_usage(self, top_objects: List[str], p2p_files: List[str], specs_name: str, file_logger: logging.Logger = None) -> Dict[str, int]:
        """
        Count how often each top_object appears across p2p trace files.

        Args:
            top_objects: Top objects for current test file
            p2p_files: Candidate p2p file paths
            specs_name: Repo spec name
            file_logger: File logger

        Returns:
            Dict[str, int]: usage count for each top object
        """
        usage_count = {obj: 0 for obj in top_objects}
        
        # Iterate p2p files and count occurrences for each top object
        for p2p_file in p2p_files:
            # Load traced object names from p2p file
            traced_objects = self._load_dynamic_trace_objects(specs_name, p2p_file, file_logger)
            if traced_objects is None:
                continue

            # Check whether each top object appears in this p2p trace
            for top_obj in top_objects:
                # Match with same logic as _count_top_objects_overlap
                if top_obj in traced_objects:
                    usage_count[top_obj] += 1
        
        file_logger.debug(f"Top objects usage count: {usage_count}")
        return usage_count
    
    def _load_dynamic_trace_objects(self, specs_name: str, test_file: str, file_logger: logging.Logger = None) -> Optional[Set[str]]:
        """
        Load traced object set from a dynamic trace file.

        Args:
            specs_name: Repo spec name
            test_file: Test file path
            file_logger: File logger

        Returns:
            Set of traced object names, None if load fails
        """
        try:
            # Get dynamic trace file path from test_results
            p2p_files = self.test_results[specs_name].get('p2p_files', [])
            trace_file_path = None
            
            for file_info in p2p_files:
                if file_info['file_path'] == test_file:
                    trace_file_path = file_info.get('dynamic_trace_file')
                    break
            
            if not trace_file_path:
                file_logger.warning(f"No dynamic trace path found for test file {test_file}")
                return None
            
            trace_file = Path(trace_file_path)
            if not trace_file.exists():
                file_logger.warning(f"Dynamic trace file not found: {trace_file}")
                return None
                
            with open(trace_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                objects = data.get('objects', {})
                
                # Extract all object ids to build identifier set
                traced_objects = set()
                for obj_id, obj_info in objects.items():
                    # Use object id as identifier
                    traced_objects.add(obj_id)
                
                return traced_objects
                
        except Exception as e:
            file_logger.error(f"Failed to load dynamic trace file {test_file}: {e}")
            tqdm.write(
                f"❌ {specs_name}: failed to load dynamic trace file - {Path(test_file).name}, "
                f"error: {str(e)[:50]}..."
            )
            raise RuntimeError(f"Dynamic trace file load failed - file: {test_file}, error: {e}") from e
    
    def _count_top_objects_overlap(self, current_top_objects: List[str], traced_objects: Set[str]) -> int:
        """
        Count overlap between current test top objects and traced objects.

        Args:
            current_top_objects: Top object list for current test file
            traced_objects: Traced object set

        Returns:
            Number of overlapping objects
        """
        if not current_top_objects or not traced_objects:
            return 0
            
        # Compute overlap
        current_set = set(obj.split('.')[-1] for obj in current_top_objects)
        overlap_count = len(current_set.intersection(traced_objects))
        
        return overlap_count
