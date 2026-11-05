"""每帧耗时统计：截屏、识别、决策、输入各花多少毫秒。"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

STAGES = ("capture", "perceive", "decide", "act")
STAGE_NAMES = {"capture": "截屏", "perceive": "识别", "decide": "决策", "act": "输入"}


class FrameTimer:
    def __init__(self, window: int = 120) -> None:
        self.window = window
        self.samples: dict[str, deque[float]] = {s: deque(maxlen=window) for s in STAGES}
        self.totals: deque[float] = deque(maxlen=window)
        self.frame_starts: deque[float] = deque(maxlen=window)
        self._current: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._current[name] = (time.perf_counter() - t0) * 1000

    def begin_frame(self) -> None:
        self._current = {}
        self.frame_starts.append(time.perf_counter())

    def end_frame(self) -> None:
        for s in STAGES:
            self.samples[s].append(self._current.get(s, 0.0))
        self.totals.append(sum(self._current.values()))

    def mean_ms(self, stage: str) -> float:
        d = self.samples[stage] if stage != "total" else self.totals
        return sum(d) / len(d) if d else 0.0

    def fps(self) -> float:
        if len(self.frame_starts) < 2:
            return 0.0
        span = self.frame_starts[-1] - self.frame_starts[0]
        return (len(self.frame_starts) - 1) / span if span > 0 else 0.0

    def summary(self, ascii: bool = False) -> str:
        """ascii=True 时用英文（OpenCV 在画面上写不了中文）。"""
        if ascii:
            parts = [f"{s} {self.mean_ms(s):.1f}" for s in STAGES]
            return " | ".join(parts) + f" | total {self.mean_ms('total'):.1f} ms | {self.fps():.0f} fps"
        parts = [f"{STAGE_NAMES[s]} {self.mean_ms(s):.1f}" for s in STAGES]
        return " | ".join(parts) + f" | 合计 {self.mean_ms('total'):.1f} ms | {self.fps():.0f} 帧/秒"
