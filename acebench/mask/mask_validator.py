"""
Mask validator for verifying masking correctness.
"""
from typing import Set
from dataclasses import dataclass

from acebench.mask.signature_extractor import (
    ModuleSignature,
    FunctionSignature,
    ClassSignature,
    extract_signature
)


@dataclass
class ValidationResult:
    """Validation result."""
    all_original_objects: Set[str]      # All qualified objects in original file
    all_masked_objects: Set[str]        # All qualified objects in masked file
    actually_removed: Set[str]          # Objects removed in masked (present in original only)
    expected_to_remove: Set[str]        # Objects expected to be removed (incl. nested/empty containers)
    should_keep: Set[str]               # Objects that should be kept
    wrongly_removed: Set[str]           # Objects removed but should be kept
    missed_removal: Set[str]            # Objects expected to remove but still present
    is_perfect: bool                    # Perfect match (no wrong removals or misses)


class MaskValidator:
    """Mask validator to verify mask correctness."""
    
    @staticmethod
    def validate(
        original_signature: ModuleSignature,
        masked_code: str,
        should_remove_set: Set[str]
    ) -> ValidationResult:
        """
        Validate masking correctness.
        
        Args:
            original_signature: Original file signature info
            masked_code: Masked code
            should_remove_set: Object IDs to remove ("/path/to/file.py::qualified_name::line_number")
            
        Returns:
            ValidationResult: Validation result
        """
        # Collect all objects in original file (full ID format)
        all_original_objects = MaskValidator._collect_all_full_ids(original_signature)
        
        # Collect all objects in masked file
        # Masking preserves line numbers by replacing with blank lines
        # Note: must pass file_path or full IDs won't match
        masked_signature = extract_signature(masked_code, original_signature.file_path)
        all_masked_objects = MaskValidator._collect_all_full_ids(masked_signature)
        
        # Compute actually removed objects
        actually_removed = all_original_objects - all_masked_objects
        
        # Collect objects expected to be removed (incl. nested)
        expected_to_remove = set()
        for full_id in should_remove_set:
            nested_objects = MaskValidator._find_object_and_nested(
                original_signature,
                full_id
            )
            expected_to_remove.update(nested_objects)
        
        # Compute objects that should be kept
        should_keep = all_original_objects - expected_to_remove
        
        # Compute wrong removals and missed removals
        wrongly_removed = actually_removed & should_keep
        missed_removal = expected_to_remove - actually_removed
        
        # Post-process: allow "reasonable removals"
        # If a container (class/function):
        # 1) is not in expected_to_remove
        # 2) all nested contents are in expected_to_remove
        # 3) and it was removed (in actually_removed)
        # -> this can be reasonable (removing all methods may break syntax)
        # -> move it from wrongly_removed into expected_to_remove
        
        reasonable_removals = MaskValidator._find_reasonable_container_removals(
            original_signature,
            expected_to_remove,
            wrongly_removed
        )
        
        # Add reasonably removed containers into expected_to_remove
        expected_to_remove.update(reasonable_removals)
        
        # Recompute should_keep, wrongly_removed, missed_removal
        should_keep = all_original_objects - expected_to_remove
        wrongly_removed = actually_removed & should_keep
        missed_removal = expected_to_remove - actually_removed
        
        # Determine perfect match
        is_perfect = (actually_removed == expected_to_remove) and len(wrongly_removed) == 0
        
        return ValidationResult(
            all_original_objects=all_original_objects,
            all_masked_objects=all_masked_objects,
            actually_removed=actually_removed,
            expected_to_remove=expected_to_remove,
            should_keep=should_keep,
            wrongly_removed=wrongly_removed,
            missed_removal=missed_removal,
            is_perfect=is_perfect
        )
    
    @staticmethod
    def _collect_all_full_ids(sig: ModuleSignature) -> Set[str]:
        """
        Recursively collect full IDs from a module signature.
        
        Format: "/path/to/file.py::qualified_name::line_number"
        Example: "/testbed/litgpt/model.py::GPT.max_seq_length::41"
        
        Returns:
            Set[str]: Full object ID set
        """
        names = set()
        file_path = sig.file_path
        
        # Collect top-level functions
        for func in sig.functions:
            names.add(f"{file_path}::{func.qualified_name}::{func.line_number}")
            names.update(MaskValidator._collect_from_function(func, file_path))
        
        # Collect top-level classes
        for cls in sig.classes:
            names.add(f"{file_path}::{cls.qualified_name}::{cls.line_number}")
            names.update(MaskValidator._collect_from_class(cls, file_path))
        
        return names
    
    @staticmethod
    def _collect_from_function(func: FunctionSignature, file_path: str) -> Set[str]:
        """Recursively collect nested defs in a function (full IDs)."""
        names = set()
        for nested_func in func.nested_functions:
            names.add(f"{file_path}::{nested_func.qualified_name}::{nested_func.line_number}")
            names.update(MaskValidator._collect_from_function(nested_func, file_path))
        for nested_cls in func.nested_classes:
            names.add(f"{file_path}::{nested_cls.qualified_name}::{nested_cls.line_number}")
            names.update(MaskValidator._collect_from_class(nested_cls, file_path))
        return names
    
    @staticmethod
    def _collect_from_class(cls: ClassSignature, file_path: str) -> Set[str]:
        """Recursively collect nested defs in a class (full IDs)."""
        names = set()
        for nested_func in cls.nested_functions:
            names.add(f"{file_path}::{nested_func.qualified_name}::{nested_func.line_number}")
            names.update(MaskValidator._collect_from_function(nested_func, file_path))
        for nested_cls in cls.nested_classes:
            names.add(f"{file_path}::{nested_cls.qualified_name}::{nested_cls.line_number}")
            names.update(MaskValidator._collect_from_class(nested_cls, file_path))
        return names
    
    @staticmethod
    def _find_object_and_nested(
        sig: ModuleSignature,
        full_id: str
    ) -> Set[str]:
        """
        Find an object in the signature tree and return it with nested full IDs.
        
        Args:
            sig: Module signature
            full_id: Full object ID ("/path/file.py::qualified_name::line")
            
        Returns:
            Set[str]: Object plus nested full IDs (empty if not found)
        """
        # Parse full ID: "/testbed/file.py::GPT.max_seq_length::45"
        parts = full_id.split("::")
        if len(parts) != 3:
            print("Malformed full ID:", full_id)
            return set()  # Invalid format
        
        file_path, qualified_name, line_str = parts
        try:
            line_number = int(line_str)
        except ValueError:
            print("Failed to parse line number in full ID:", full_id)
            return set()  # Failed to parse line number
        
        # Search in top-level functions
        for func in sig.functions:
            if func.qualified_name == qualified_name and func.line_number == line_number:
                result = {full_id}
                result.update(MaskValidator._collect_from_function(func, file_path))
                return result
            # Recursively search nested definitions
            nested_result = MaskValidator._find_in_function(func, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        # Search in top-level classes
        for cls in sig.classes:
            if cls.qualified_name == qualified_name and cls.line_number == line_number:
                result = {full_id}
                result.update(MaskValidator._collect_from_class(cls, file_path))
                return result
            # Recursively search nested definitions
            nested_result = MaskValidator._find_in_class(cls, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        # Not found: return empty set (dynamic trace may reference missing objects)
        print("Full ID not found:", full_id)
        return set()
    
    @staticmethod
    def _find_in_function(func: FunctionSignature, file_path: str, qualified_name: str, line_number: int) -> Set[str]:
        """Recursively search nested defs in a function (full IDs)."""
        for nested_func in func.nested_functions:
            if nested_func.qualified_name == qualified_name and nested_func.line_number == line_number:
                result = {f"{file_path}::{qualified_name}::{line_number}"}
                result.update(MaskValidator._collect_from_function(nested_func, file_path))
                return result
            nested_result = MaskValidator._find_in_function(nested_func, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        for nested_cls in func.nested_classes:
            if nested_cls.qualified_name == qualified_name and nested_cls.line_number == line_number:
                result = {f"{file_path}::{qualified_name}::{line_number}"}
                result.update(MaskValidator._collect_from_class(nested_cls, file_path))
                return result
            nested_result = MaskValidator._find_in_class(nested_cls, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        return None
    
    @staticmethod
    def _find_in_class(cls: ClassSignature, file_path: str, qualified_name: str, line_number: int) -> Set[str]:
        """Recursively search nested defs in a class (full IDs)."""
        for nested_func in cls.nested_functions:
            if nested_func.qualified_name == qualified_name and nested_func.line_number == line_number:
                result = {f"{file_path}::{qualified_name}::{line_number}"}
                result.update(MaskValidator._collect_from_function(nested_func, file_path))
                return result
            nested_result = MaskValidator._find_in_function(nested_func, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        for nested_cls in cls.nested_classes:
            if nested_cls.qualified_name == qualified_name and nested_cls.line_number == line_number:
                result = {f"{file_path}::{qualified_name}::{line_number}"}
                result.update(MaskValidator._collect_from_class(nested_cls, file_path))
                return result
            nested_result = MaskValidator._find_in_class(nested_cls, file_path, qualified_name, line_number)
            if nested_result:
                return nested_result
        
        return None
    
    @staticmethod
    def _find_reasonable_container_removals(
        sig: ModuleSignature,
        expected_to_remove: Set[str],
        wrongly_removed: Set[str]
    ) -> Set[str]:
        """
        Find containers that are "reasonably removed."
        
        Scenario: removing all methods can make a class empty (no docstring),
        so _fix_empty_classes may remove the class to fix syntax errors.
        Even if the class isn't in should_remove_set, removal can be reasonable.
        
        Criteria:
        1) Container is in wrongly_removed
        2) All nested contents are in expected_to_remove
        3) Container has nested contents (not originally empty)
        
        Args:
            sig: Module signature
            expected_to_remove: Objects to remove (full IDs)
            wrongly_removed: Wrongly removed objects (full IDs)
            
        Returns:
            Set[str]: Full IDs of reasonably removed containers
        """
        reasonable = set()
        file_path = sig.file_path
        
        def check_class(cls: ClassSignature):
            """Check whether class removal is reasonable."""
            # Collect nested content for class (full IDs)
            all_nested = set()
            for nested_func in cls.nested_functions:
                all_nested.add(f"{file_path}::{nested_func.qualified_name}::{nested_func.line_number}")
            for nested_cls in cls.nested_classes:
                all_nested.add(f"{file_path}::{nested_cls.qualified_name}::{nested_cls.line_number}")
            
            # Class full ID
            cls_full_id = f"{file_path}::{cls.qualified_name}::{cls.line_number}"
            
            # Conditions:
            # 1) class in wrongly_removed
            # 2) class has nested content
            # 3) all nested content in expected_to_remove
            if (cls_full_id in wrongly_removed and 
                all_nested and 
                all_nested.issubset(expected_to_remove)):
                reasonable.add(cls_full_id)
            
            # Recursively check nested classes
            for nested_cls in cls.nested_classes:
                check_class(nested_cls)
            for nested_func in cls.nested_functions:
                check_function(nested_func)
        
        def check_function(func: FunctionSignature):
            """Check whether function removal is reasonable."""
            # Collect nested content for function (full IDs)
            all_nested = set()
            for nested_func in func.nested_functions:
                all_nested.add(f"{file_path}::{nested_func.qualified_name}::{nested_func.line_number}")
            for nested_cls in func.nested_classes:
                all_nested.add(f"{file_path}::{nested_cls.qualified_name}::{nested_cls.line_number}")
            
            # Function full ID
            func_full_id = f"{file_path}::{func.qualified_name}::{func.line_number}"
            
            # Same conditions apply
            if (func_full_id in wrongly_removed and 
                all_nested and 
                all_nested.issubset(expected_to_remove)):
                reasonable.add(func_full_id)
            
            # Recursively check nested content
            for nested_func in func.nested_functions:
                check_function(nested_func)
            for nested_cls in func.nested_classes:
                check_class(nested_cls)
        
        # Start checking from top-level
        for cls in sig.classes:
            check_class(cls)
        for func in sig.functions:
            check_function(func)
        
        return reasonable
