"""Runs the quarantined bot's own test suite in a separate process.

execution_capital and gold_monitor both define top-level packages named
config/data/engine/strategies, so their tests cannot share one interpreter;
the sub-suite runs with execution_capital as its import root.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXECUTION_CAPITAL = os.path.join(ROOT, "execution_capital")


def test_execution_capital_suite_passes():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=EXECUTION_CAPITAL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"execution_capital tests failed:\n{result.stdout}\n{result.stderr}"
    )
