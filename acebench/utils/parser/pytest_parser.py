"""
Pytest output parser - parses pytest test results.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PytestResult:
    """Pytest test result."""
    passed: int = 0          # Number of passed tests
    failed: int = 0          # Number of failed tests
    skipped: int = 0         # Number of skipped tests
    deselected: int = 0      # Number of tests deselected during collection
    error: int = 0           # Number of errored tests
    xfailed: int = 0         # Number of expected failures
    xpassed: int = 0         # Number of unexpected passes
    total: int = 0           # Total tests
    pass_rate: float = 0.0   # Pass rate = passed / (passed + failed + error)
    return_code: int = -1    # pytest exit code (0=pass, 1=fail, 2=interrupt, 3=internal, 4=usage, 5=no tests)
    errors_detail: List[Tuple[str, str]] = field(default_factory=list)  # Error details [(type, reason), ...]
    
    def __post_init__(self):
        """Compute totals and pass rate."""
        self.total = self.passed + self.failed + self.skipped + self.error + self.xfailed + self.xpassed
        
        # Compute pass rate: passed / (passed + failed + error)
        denominator = self.passed + self.failed + self.error
        if denominator > 0:
            self.pass_rate = self.passed / denominator
        else:
            self.pass_rate = 0.0


class PytestParser:
    """Pytest output parser."""
    
    @staticmethod
    def parse_output(output: str) -> PytestResult:
        """
        Parse pytest output to get test results.
        
        Args:
            output: pytest stdout or log content
            
        Returns:
            PytestResult: Parsed test results
            
        Example output formats:
            ====== 2 passed, 1 failed, 1 skipped in 0.05s ======
            ====== 5 passed in 0.03s ======
            ====== 1 failed, 1 error in 0.10s ======
            ====== 3 passed, 1 xfailed in 0.08s ======
        """
        result = PytestResult()
        
        # Pytest summary usually appears in the last '='-delimited line
        # Format: ====== X passed, Y failed, Z skipped in N.NNs ======
        
        # Search from the end to avoid header separators
        # Only scan the last 2000 chars (enough for summary)
        search_text = output[-2000:] if len(output) > 2000 else output
        
        # Match patterns (supports multiple formats)
        result_patterns = [
            r'={3,}\s*(.*?)\s+in\s+[\d.]+s',           # ====== ... in N.NNs (may include extra text)
            r'={3,}\s*(.*?)\s*={3,}',                  # ====== ... ======
            r'([\d]+\s+passed.*?)(?:\n|$)',            # Simplified format
        ]
        
        result_text = None
        for pattern in result_patterns:
            matches = list(re.finditer(pattern, search_text, re.MULTILINE | re.IGNORECASE))
            if not matches:
                continue

            for match in reversed(matches):
                candidate = match.group(1)
                if PytestParser._looks_like_result_summary(candidate):
                    result_text = candidate
                    break

            if result_text:
                break
        
        if not result_text:
            # If no result line found, fall back to whole output
            result_text = output
        
        # Parse counts
        # Format: "X passed", "Y failed", "Z skipped", etc.
        counters = {
            'passed': r'(\d+)\s+passed',
            'failed': r'(\d+)\s+failed',
            'skipped': r'(\d+)\s+skipped',
            'deselected': r'(\d+)\s+deselected',
            'error': r'(\d+)\s+error',
            'xfailed': r'(\d+)\s+xfailed',
            'xpassed': r'(\d+)\s+xpassed',
        }
        
        for key, pattern in counters.items():
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                setattr(result, key, int(match.group(1)))

        # Support summary blocks like "Results (9.67s):\n 1 failed\n 83 passed"
        results_block = re.search(
            r'Results\s*\([^)]*\):\s*(?:\r?\n)(.+?)(?:\n{2,}|$)',
            search_text,
            re.IGNORECASE | re.DOTALL,
        )
        if results_block:
            block_text = results_block.group(1)
            for line in block_text.splitlines():
                line = line.strip()
                match = re.match(
                    r'(\d+)\s+(passed|failed|error|skipped|deselected|xfailed|xpassed)',
                    line,
                    re.IGNORECASE,
                )
                if match:
                    count = int(match.group(1))
                    key = match.group(2).lower()
                    setattr(result, key, count)
        
        # Parse exit code (from log tail)
        # Format: "Return code: 0" or "Exit code: 2"
        return_code_match = re.search(r'(?:Return code|Exit code):\s*(\d+)', output, re.IGNORECASE)
        if return_code_match:
            result.return_code = int(return_code_match.group(1))
        
        # Parse error details
        result.errors_detail = PytestParser._parse_errors(output)
        
        # Compute totals and pass rate
        result.__post_init__()
        
        return result
    
    @staticmethod
    def _looks_like_result_summary(text: str) -> bool:
        """Check whether text looks like a pytest summary line."""
        normalized = text.lower()
        keywords = (
            "passed",
            "failed",
            "skipped",
            "deselected",
            "error",
            "errors",
            "xfailed",
            "xpassed",
            "warnings",
            "no tests ran",
            "no tests collected",
        )
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _parse_errors(output: str) -> List[Tuple[str, str]]:
        """
        Parse error details from pytest output.
        
        Args:
            output: pytest stdout or log content
            
        Returns:
            List[Tuple[str, str]]: Error list, each item is (error_type, error_reason)
            
        Example error formats:
            E   AttributeError: module 'agent_code' has no attribute 'WindowedFile'
            E   SyntaxError: from __future__ imports must occur at the beginning of the file
        """
        errors = []
        
        # Find ERRORS and FAILURES sections
        # These sections usually start with "==== ERRORS ====" or "==== FAILURES ===="
        error_sections = []
        
        # Match ERRORS section
        errors_match = re.search(r'={3,}\s*ERRORS\s*={3,}(.*?)(?:={3,}|$)', output, re.DOTALL | re.IGNORECASE)
        if errors_match:
            error_sections.append(errors_match.group(1))
        
        # Match FAILURES section
        failures_match = re.search(r'={3,}\s*FAILURES\s*={3,}(.*?)(?:={3,}|$)', output, re.DOTALL | re.IGNORECASE)
        if failures_match:
            error_sections.append(failures_match.group(1))
        
        # Extract concrete error messages from each section
        for section in error_sections:
            # Find lines starting with "E   ", format: E   <ErrorType>: <message>
            # Extract error type and message
            error_pattern = r'^E\s+(\w+(?:Error|Exception|Warning)):\s*(.+?)$'
            matches = re.finditer(error_pattern, section, re.MULTILINE)
            
            for match in matches:
                error_type = match.group(1).strip()
                error_message = match.group(2).strip()
                errors.append((error_type, error_message))
        
        return errors
    
    @staticmethod
    def parse_error_types(output: str) -> List[str]:
        """
        Parse error types from pytest output (ImportError, SyntaxError, etc.).
        
        Args:
            output: pytest stdout or log content
            
        Returns:
            List[str]: List of error types, e.g., ["ImportError", "SyntaxError"]
        """
        
        error_types = []
        seen = set()  # Dedupe
        
        # Strip ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_output = ansi_escape.sub('', output)
        
        # Common Python error types
        error_patterns = [
            'ImportError',
            'ModuleNotFoundError',
            'NameError',
            'SyntaxError',
            'IndentationError',
            'AttributeError',
            'TypeError',
            'ValueError',
            'KeyError',
            'IndexError',
            'FileNotFoundError',
        ]
        
        # Locate error type (usually in lines starting with "E   ")
        for line in clean_output.split('\n'):
            # Look for "E   ErrorType:" format
            if line.startswith('E   '):
                for error_type in error_patterns:
                    if error_type in line and error_type not in seen:
                        error_types.append(error_type)
                        seen.add(error_type)
                        break  # Stop after first match to avoid duplicates
        
        return error_types

