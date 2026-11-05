"""输出端：把"往哪个方向走"变成按键或摇杆。

所有后端都要做到两件事：
1. 记住自己按下了哪些键；
2. release_all() 能把它们全部松开，并且可以重复调用（崩溃、退出时都会调）。
"""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod

DIRECTIONS = ("up", "down", "left", "right")

# 8 个扇区（从正右开始，顺时针，因为屏幕 y 轴向下）对应要按的方向键
_SECTOR_KEYS = (
    {"right"},
    {"right", "down"},
    {"down"},
    {"down", "left"},
    {"left"},
    {"left", "up"},
    {"up"},
    {"up", "right"},
)


def quantize_8(dx: float, dy: float, deadzone: float = 0.3) -> set[str]:
    """把连续方向变成 8 个方向之一（用方向键表示）。太小就当不动。"""
    if math.hypot(dx, dy) < deadzone:
        return set()
    sector = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
    return set(_SECTOR_KEYS[sector])


def keys_to_vector(keys: set[str]) -> tuple[float, float]:
    """方向键组合 -> 单位方向（斜着走也是长度 1）。"""
    x = ("right" in keys) - ("left" in keys)
    y = ("down" in keys) - ("up" in keys)
    n = math.hypot(x, y)
    return (x / n, y / n) if n else (0.0, 0.0)


class InputSink(ABC):
    """助手的"手"。"""

    @abstractmethod
    def set_move(self, dx: float, dy: float) -> None:
        """往 (dx, dy) 方向走，长度 0-1。(0, 0) 表示不动。"""

    @abstractmethod
    def release_all(self) -> None:
        """松开所有按着的键 / 摇杆归中。必须能重复调用。"""

    def close(self) -> None:
        self.release_all()


class KeyboardSink(InputSink):
    """用方向键走位。子类只需实现 _key_down / _key_up。"""

    def __init__(self, keymap: dict[str, str], deadzone: float = 0.3) -> None:
        self.keymap = dict(keymap)
        self.deadzone = deadzone
        self.pressed: set[str] = set()  # 逻辑方向
        self._lock = threading.RLock()

    def set_move(self, dx: float, dy: float) -> None:
        want = quantize_8(dx, dy, self.deadzone)
        with self._lock:
            for d in sorted(self.pressed - want):
                self._key_up(self.keymap[d])
                self.pressed.discard(d)
            for d in sorted(want - self.pressed):
                self._key_down(self.keymap[d])
                self.pressed.add(d)

    def release_all(self) -> None:
        with self._lock:
            for d in sorted(self.pressed):
                try:
                    self._key_up(self.keymap[d])
                except Exception:  # 松键失败也要继续松别的
                    pass
            self.pressed.clear()

    @abstractmethod
    def _key_down(self, key: str) -> None: ...

    @abstractmethod
    def _key_up(self, key: str) -> None: ...


class StickSink(InputSink):
    """用虚拟手柄左摇杆走位。子类只需实现 _set_stick。"""

    def __init__(self) -> None:
        self.stick = (0.0, 0.0)
        self._lock = threading.RLock()

    def set_move(self, dx: float, dy: float) -> None:
        n = math.hypot(dx, dy)
        if n > 1:
            dx, dy = dx / n, dy / n
        with self._lock:
            self.stick = (dx, dy)
            self._set_stick(dx, dy)

    def release_all(self) -> None:
        with self._lock:
            self.stick = (0.0, 0.0)
            try:
                self._set_stick(0.0, 0.0)
            except Exception:
                pass

    @abstractmethod
    def _set_stick(self, x: float, y: float) -> None: ...
