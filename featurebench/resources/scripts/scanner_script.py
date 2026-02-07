"""Test discovery script - runs in Docker container to collect test info."""

import subprocess
import json
import sys
from pathlib import Path


def main(test_discovery_cmd: list, output_file: str):
	"""
	Run test discovery and save results.

	Args:
		test_discovery_cmd: test discovery command list
		output_file: output JSON file path
	"""
	try:
		# Run pytest --collect-only
		cmd_str = " ".join(test_discovery_cmd)
		full_command = f"source ~/.bashrc && conda activate testbed && cd /testbed && {cmd_str}"
		
		result = subprocess.run(
			["bash", "-c", full_command],
			capture_output=True,
			text=True
		)
		
		# Debug: print raw pytest output
		# print("=" * 60)
		# print("DEBUG: pytest stdout:")
		# print("=" * 60)
		# print(result.stdout)
		# print("=" * 60)
		# print("DEBUG: pytest stderr:")
		# print("=" * 60)
		# print(result.stderr)
		# print("=" * 60)
		# print(f"DEBUG: pytest return code: {result.returncode}")
		# print("=" * 60)
		
		# Use set to store test points and deduplicate (handles parameterized tests)
		file_tests = {}
		
		# Parse output
		if result.stdout:
			for line in result.stdout.splitlines():
				line = line.strip()
				if "::" in line:
					parts = line.split("::")
					if len(parts) >= 2:
						file_path = parts[0]
						
						# Only process Python test files
						if not file_path.endswith('.py'):
							continue
						
						full_path = str(Path("/testbed") / file_path)
						
						if Path(full_path).exists():
							if full_path not in file_tests:
								file_tests[full_path] = set()
							
							if len(parts) == 2:
								# Standalone test function: keep function name (strip parameters [...])
								test_name = parts[1].split("[")[0]
								file_tests[full_path].add(test_name)
							elif len(parts) >= 3:
								# Test method in class (including nested classes):
								# - Normal: TestClass::test_method → TestClass.test_method
								# - Nested: TestOuter::TestInner::test_method → TestOuter.TestInner.test_method
								# Join class names and method by dots (strip parameters [...])
								class_parts = parts[1:-1]  # all class names
								method_name = parts[-1].split("[")[0]  # Last part is the method name
								full_method = ".".join(class_parts) + "." + method_name
								file_tests[full_path].add(full_method)
		
		# Save results
		result_data = {}
		for file_path, test_points_set in file_tests.items():
			# Convert to sorted list
			test_points = sorted(test_points_set)
			
			result_data[file_path] = {
				"test_count": len(test_points),
				"test_points": test_points
			}
		
		with open(output_file, "w") as f:
			json.dump(result_data, f, indent=2)
		
		print(f"Test discovery completed, found {len(result_data)} test files")
		
		# Print detailed info to logs
		if result_data:
			print("\nDetailed test file list:")
			for file_path, info in sorted(result_data.items()):
				print(f"  {file_path}")
				print(f"    - Test count: {info['test_count']}")
				print(
					f"    - Test points: {', '.join(info['test_points'][:5])}" +
					(f" ... (total {len(info['test_points'])})" if len(info['test_points']) > 5 else "")
				)
		
	except Exception as e:
		print(f"Test discovery failed: {e}", file=sys.stderr)
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	# Read config from CLI args
	if len(sys.argv) < 3:
		print("Usage: python scanner_script.py <output_file> <test_discovery_cmd...>", file=sys.stderr)
		print("Example: python scanner_script.py /tmp/results.json pytest --collect-only -q", file=sys.stderr)
		sys.exit(1)
	
	output_file = sys.argv[1]
	test_discovery_cmd = sys.argv[2:]
	
	main(test_discovery_cmd, output_file)
