"""Path setup for the quarantined bot's own test suite.

These tests run in their own pytest process (spawned by
tests/test_execution_capital.py) because execution_capital and gold_monitor
both define top-level packages named config/data/engine/strategies and cannot
be imported into the same interpreter.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
