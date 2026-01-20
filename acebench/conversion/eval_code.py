import json
import re
from pathlib import Path
import argparse

def transform_pytest_report(input_data):
    # Handle summary
    passed = input_data.get("summary", {}).get("passed", 0)
    failed = input_data.get("summary", {}).get("failed", 0)
    skipped = input_data.get("summary", {}).get("skipped", 0)
    deselected = input_data.get("summary", {}).get("deselected", 0)
    xfailed = input_data.get("summary", {}).get("xfailed", 0)
    xpassed = input_data.get("summary", {}).get("xpassed", 0)
    exitcode = input_data.get("exitcode", 0)
    collected = input_data.get("summary", {}).get("collected", 0)

    # Compute actual error count, excluding known statuses
    known_states = passed + failed + skipped + deselected + xfailed + xpassed
    error = max(0, collected - known_states)

    summary = {
        "total": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "error": error,
        "return_code": exitcode,
        "success": exitcode == 0
    }

    # Handle test cases and failures
    test_cases = {}
    failures = {}

    # Handle normal test cases
    for test in input_data.get("tests", []):
        nodeid = test.get("nodeid")
        outcome = test.get("outcome")

        if nodeid and outcome:
            test_cases[nodeid] = outcome

            # Only real failures go into failures
            if outcome == "failed":
                # Extract failure info
                failure_msg = "Unknown error"
                call_data = test.get("call", {})

                if "crash" in call_data:
                    failure_msg = call_data["crash"].get("message", failure_msg)
                elif "longrepr" in call_data:
                    # Extract last line starting with 'E '
                    lines = call_data["longrepr"].split('\n')
                    for line in reversed(lines):
                        if line.startswith('E '):
                            failure_msg = line[2:].strip()
                            break

                failures[nodeid] = failure_msg

    # Handle collector import errors (tests empty but collectors failed)
    if not input_data.get("tests") and input_data.get("collectors"):
        for collector in input_data.get("collectors", []):
            if collector.get("outcome") == "failed":
                nodeid = collector.get("nodeid", "")
                longrepr = collector.get("longrepr", "")
                
                # If no file path nodeid, try extracting from longrepr
                if not nodeid and longrepr:
                    # Generic file path extraction: look for quoted path
                    match = re.search(r"'([^']*\.py)'", longrepr)
                    if match:
                        file_path = match.group(1)
                        # Remove common workspace prefix for brevity
                        for prefix in ['/workspace/', '/tmp/', '/home/', '/usr/', '/opt/']:
                            if file_path.startswith(prefix):
                                # Find the first segment that looks like repo root
                                parts = file_path[len(prefix):].split('/')
                                if len(parts) > 1:
                                    # Keep path from first "test" dir or from second dir
                                    for i, part in enumerate(parts):
                                        if 'test' in part.lower() or i == 1:
                                            nodeid = '/'.join(parts[i:])
                                            break
                                    else:
                                        nodeid = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
                                break
                        else:
                            # If no prefix match, use relative path
                            nodeid = file_path
                
                # If still no nodeid, use default
                if not nodeid:
                    nodeid = "collection_error"
                
                # Extract error info, prefer last line starting with 'E '
                failure_msg = "Collection failed"
                if longrepr:
                    lines = longrepr.split('\n')
                    for line in reversed(lines):
                        if line.startswith('E '):
                            failure_msg = line[2:].strip()
                            break
                    
                    # If no 'E ' line, use first non-empty line
                    if failure_msg == "Collection failed":
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('=') and not line.startswith('-'):
                                failure_msg = line
                                break
                
                test_cases[nodeid] = "failed"
                failures[nodeid] = failure_msg

    # If no tests but nonzero exit code
    if not test_cases and exitcode != 0:
        test_cases["pytest_execution_error"] = "failed"
        failures["pytest_execution_error"] = f"Pytest execution failed with exit code {exitcode}"
        
        # Update summary to reflect actual status
        summary["total"] = 0
        summary["failed"] = 0
        summary["skipped"] = 0
        summary["error"] = 0

    # Return converted report
    return {
        "summary": summary,
        "test_cases": test_cases,
        "failures": failures
    }

def main():
    parser = argparse.ArgumentParser(description="Convert pytest JSON report to the standard format")
    parser.add_argument('--input', type=str, default="repo_tmp.json", help="Input file name, default: repo_tmp.json")
    parser.add_argument('--output', type=str, default="repo.json", help="Output file name, default: repo.json")
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    input_file = current_dir / args.input
    output_file = current_dir / args.output

    # Check if input file exists
    if not input_file.exists():
        print("Error: input file not found!")
        return

    try:
        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # Convert data
        transformed_data = transform_pytest_report(original_data)

        # Write output JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully generated {args.output}")

    except json.JSONDecodeError:
        print("Error: input file is not valid JSON!")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
