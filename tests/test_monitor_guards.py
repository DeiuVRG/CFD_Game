"""Guards that make the monitor safe to run as a headless service (systemd,
cron, nohup): the keyboard listener must never spin on a non-interactive
stdin."""
import os
import sys
import threading
import time

import pytest

from engine.monitor_engine import MonitorEngine


def bare_engine() -> MonitorEngine:
    """MonitorEngine without __init__ (no Discord, no SQLite, no network)."""
    eng = MonitorEngine.__new__(MonitorEngine)
    eng.is_running = True
    eng._pending_command = None
    return eng


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX select() path")
def test_keyboard_listener_returns_on_eof(monkeypatch):
    """stdin = /dev/null (what systemd gives a service): select() reports it
    readable forever and read() returns '' -> the listener must return, not
    busy-loop."""
    eng = bare_engine()
    with open(os.devnull) as devnull:
        monkeypatch.setattr(sys, "stdin", devnull)
        t = threading.Thread(target=eng._keyboard_listener, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "listener kept spinning on EOF stdin"
    eng.is_running = False


def test_stdin_interactive_false_for_devnull(monkeypatch):
    with open(os.devnull) as devnull:
        monkeypatch.setattr(sys, "stdin", devnull)
        assert MonitorEngine._stdin_is_interactive() is False


def test_stdin_interactive_false_when_missing(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    assert MonitorEngine._stdin_is_interactive() is False
