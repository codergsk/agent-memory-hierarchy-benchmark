"""Regenerate the HTML report from a saved benchmark_results_*.json file.

Usage:
    python3 generate_report_from_data.py [results.json] [output.html]

Defaults to the newest benchmark_results_*.json in the current directory.
Lets you iterate on report styling without re-running the (paid) benchmark.
"""
import glob
import sys

from benchmarks.metrics import BenchmarkResults
from reports.report_generator import generate_report


def main() -> None:
    """Load saved results and render the report."""
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        candidates = sorted(glob.glob("benchmark_results_*.json"))
        if not candidates:
            sys.exit("No benchmark_results_*.json found. Run run_benchmark.py first.")
        src = candidates[-1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.replace(
        "benchmark_results_", "benchmark_report_").replace(".json", ".html")
    results = BenchmarkResults.load(src)
    path = generate_report(results, out)
    print(f"Report written: {path} (from {src})")


if __name__ == "__main__":
    main()
