"""Constants for FeatureBench evaluation harness."""

from enum import Enum
from pathlib import Path

# Directory constants
FB_BENCH_DIR = Path(__file__).parent.parent

# Docker constants
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"

# Log file names
LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
LOG_REPORT = "report.json"
LOG_PATCH = "patch.diff"

# Keys for prediction/instance data
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"
KEY_N_ATTEMPT = "n_attempt"

# Test result constants
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"

# Encoding
UTF8 = "utf-8"

# Default pytest command
DEFAULT_PYTEST_CMD = "pytest -rA -p no:cacheprovider --color=no"


# Test Status Enum - PASSED, FAILED, ERROR, XFAIL, SKIPPED
class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


# Evaluation Type Enum
class EvalType(Enum):
    PASS_AND_FAIL = "pass_and_fail"  # Default mode
    FAIL_ONLY = "fail_only"


# Repos that use FAIL_ONLY evaluation mode
# For these repos, only FAILED and ERROR count as failure; everything else silently passes
FAIL_ONLY_REPOS: set[str] = {
    # Add repo names here that should use FAIL_ONLY mode
}