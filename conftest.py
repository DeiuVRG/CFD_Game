"""Pytest path setup.

gold_monitor is an application with its own import root (it inserts its own
directory on sys.path at runtime), so tests need both the repo root and the
gold_monitor directory importable. gold_monitor comes first so that its
config/data/engine packages win name resolution.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
GOLD_MONITOR = os.path.join(ROOT, "gold_monitor")

for path in (ROOT, GOLD_MONITOR):
    if path not in sys.path:
        sys.path.insert(0, path)
