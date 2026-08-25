"""Technical indicators - thin re-export of the single shared implementation.

The real code lives in common/indicators.py at the repo root, used by both
gold_monitor and execution_capital (no more duplicated indicator math).
execution_capital treats its own directory as the import root, so the repo
root is added to sys.path here before importing the shared package.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.indicators import Indicators

__all__ = ["Indicators"]
