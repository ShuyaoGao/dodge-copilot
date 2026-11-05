"""接管和交还（需求 3.1）。

状态只有两个：玩家控制 / 助手接管中。
热键监听线程和主循环线程会同时调用这里，所以全部操作都在一把锁里做：
一旦切回"玩家控制"，主循环之后的任何 apply() 都不会再按键。
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Callable, Optional

from .inputs import InputSink


class Mode(enum.Enum):
    PLAYER = "玩家控制"
    ASSIST = "助手接管中"


class TakeoverController:
    def __init__(
        self,
        sink: InputSink,
        auto_resume_after_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        on_change: Optional[Callable[[Mode, str], None]] = None,
    ) -> None:
        self.sink = sink
        self.auto_resume_after_s = auto_resume_after_s
        self.clock = clock
        self.on_change = on_change
        self._lock = threading.RLock()
        self._mode = Mode.PLAYER
        self._stopped = threading.Event()
        self._user_keys: set[str] = set()
        self._overridden_at: Optional[float] = None  # 被玩家抢回控制的时间
        self._user_released_at: Optional[float] = None
        self.switch_log: list[tuple[float, Mode, str]] = []

    # ------------------------------------------------------------ 状态
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def assisting(self) -> bool:
        return self._mode is Mode.ASSIST

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def wait_stopped(self, timeout: Optional[float] = None) -> bool:
        return self._stopped.wait(timeout)

    # ------------------------------------------------------------ 事件（热键线程调用）
    def toggle(self) -> Mode:
        """接管键：玩家控制 <-> 助手接管。"""
        with self._lock:
            if self.stopped:
                return self._mode
            if self._mode is Mode.ASSIST:
                self._set(Mode.PLAYER, "按了接管键，交还")
            else:
                self._overridden_at = None
                self._set(Mode.ASSIST, "按了接管键，接管")
            return self._mode

    def engage(self, reason: str = "接管") -> None:
        with self._lock:
            if not self.stopped and self._mode is not Mode.ASSIST:
                self._overridden_at = None
                self._set(Mode.ASSIST, reason)

    def release(self, reason: str = "交还") -> None:
        with self._lock:
            if self._mode is not Mode.PLAYER:
                self._set(Mode.PLAYER, reason)

    def user_direction(self, key: str, down: bool) -> None:
        """玩家自己按下/松开了方向键（或推了摇杆）。"""
        with self._lock:
            if down:
                self._user_keys.add(key)
                self._user_released_at = None
                if self._mode is Mode.ASSIST:
                    self._overridden_at = self.clock()
                    self._set(Mode.PLAYER, f"玩家按了 {key}，自动交还")
            else:
                self._user_keys.discard(key)
                if not self._user_keys:
                    self._user_released_at = self.clock()

    def estop(self) -> None:
        """急停：立刻松开所有键，通知主循环退出。"""
        with self._lock:
            self._set(Mode.PLAYER, "急停")
            self._stopped.set()
            self.sink.release_all()

    # ------------------------------------------------------------ 主循环调用
    def tick(self) -> None:
        """处理"松手后自动恢复接管"（默认关闭）。"""
        if self.auto_resume_after_s is None:
            return
        with self._lock:
            if (
                self._mode is Mode.PLAYER
                and self._overridden_at is not None
                and not self._user_keys
                and self._user_released_at is not None
                and self.clock() - self._user_released_at >= self.auto_resume_after_s
            ):
                self._overridden_at = None
                self._set(Mode.ASSIST, "玩家松手，自动恢复接管")

    def apply(self, dx: float, dy: float) -> bool:
        """助手想往 (dx, dy) 走。只有接管中才真的按键。返回是否按了。"""
        with self._lock:
            if self._mode is not Mode.ASSIST or self.stopped:
                return False
            self.sink.set_move(dx, dy)
            return True

    # ------------------------------------------------------------ 内部
    def _set(self, mode: Mode, reason: str) -> None:
        if mode is Mode.PLAYER:
            # 先松键再改状态：哪怕下面回调出错，键也已经松开了
            self.sink.release_all()
        if mode is self._mode:
            return
        self._mode = mode
        self.switch_log.append((self.clock(), mode, reason))
        if self.on_change is not None:
            try:
                self.on_change(mode, reason)
            except Exception:
                pass
