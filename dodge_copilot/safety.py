"""不管怎么退出，都先把按着的键松开。

三道保险：
1. 主循环外面包 try/finally（见 app.py）；
2. atexit：正常退出、未捕获异常时都会调；
3. 信号：Ctrl+C、被结束进程（Windows 上还有 Ctrl+Break、关控制台窗口）。
"""
from __future__ import annotations

import atexit
import signal
import sys
from typing import Callable

from .inputs import InputSink


class ReleaseGuard:
    def __init__(self, *sinks: InputSink) -> None:
        self.sinks = list(sinks)
        self._installed = False
        self._old_handlers: dict[int, object] = {}

    def release(self) -> None:
        for s in self.sinks:
            try:
                s.release_all()
            except Exception:
                pass

    def install(self) -> "ReleaseGuard":
        if self._installed:
            return self
        atexit.register(self.release)
        for sig in self._signals():
            try:
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._make_handler(sig))
            except (ValueError, OSError):  # 不在主线程等情况，装不上就算了，还有 atexit
                pass
        if sys.platform == "win32":
            _install_console_ctrl_handler(self.release)
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        atexit.unregister(self.release)
        for sig, h in self._old_handlers.items():
            try:
                signal.signal(sig, h)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass
        self._old_handlers.clear()
        self._installed = False

    def __enter__(self) -> "ReleaseGuard":
        return self.install()

    def __exit__(self, *exc: object) -> None:
        self.release()
        self.uninstall()

    @staticmethod
    def _signals() -> list[int]:
        names = ["SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"]
        return [getattr(signal, n) for n in names if hasattr(signal, n)]

    def _make_handler(self, sig: int) -> Callable:
        def handler(signum, frame):  # noqa: ANN001
            self.release()
            old = self._old_handlers.get(sig)
            if callable(old):
                old(signum, frame)
            elif sig == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(128 + signum)

        return handler


_console_handlers: list = []  # 防止回调被垃圾回收


def _install_console_ctrl_handler(release: Callable[[], None]) -> None:
    """Windows：关掉控制台窗口时 Python 的信号收不到，要用这个。"""
    try:
        import ctypes
        from ctypes import wintypes

        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(_ctrl_type: int) -> bool:
            release()
            return False  # 交给下一个处理器（默认是退出）

        cb = HandlerRoutine(_handler)
        _console_handlers.append(cb)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, True)
    except Exception:
        pass
