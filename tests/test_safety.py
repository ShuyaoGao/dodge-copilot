import os
import signal
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT
from dodge_copilot.io.sim import SimKeyboardSink
from dodge_copilot.safety import ReleaseGuard

KEYMAP = {"up": "W", "down": "S", "left": "A", "right": "D"}


def test_context_manager_releases_on_exception():
    sink = SimKeyboardSink(KEYMAP)
    with pytest.raises(RuntimeError):
        with ReleaseGuard(sink):
            sink.set_move(1, 1)
            raise RuntimeError("崩了")
    assert sink.down == set()


def test_sigint_releases_then_raises_keyboardinterrupt():
    sink = SimKeyboardSink(KEYMAP)
    g = ReleaseGuard(sink).install()
    try:
        sink.set_move(0, -1)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
        assert sink.down == set()
    finally:
        g.uninstall()


def _run_child(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from dodge_copilot.inputs import KeyboardSink
        from dodge_copilot.safety import ReleaseGuard

        class PrintSink(KeyboardSink):
            def _key_down(self, k): print("down", k, flush=True)
            def _key_up(self, k): print("up", k, flush=True)

        sink = PrintSink({{"up": "W", "down": "S", "left": "A", "right": "D"}})
        ReleaseGuard(sink).install()
        sink.set_move(1, 0)
        """
    ) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)


def test_uncaught_exception_releases_via_atexit():
    r = _run_child("raise RuntimeError('崩了')")
    assert r.returncode != 0
    assert r.stdout.split() == ["down", "D", "up", "D"]


def test_sigterm_releases():
    r = _run_child("import os, signal, time\nos.kill(os.getpid(), signal.SIGTERM)\ntime.sleep(5)")
    assert "up D" in r.stdout
