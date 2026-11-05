"""找玩家在画面上的位置。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from ..config import ColorRange


def ignore_mask(shape: tuple[int, ...], regions: list[list[int]]) -> np.ndarray:
    """返回一张 uint8 掩码：要看的地方是 255，忽略区（血条、分数等）是 0。"""
    h, w = shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x, y, rw, rh in regions:
        mask[max(0, y) : max(0, y + rh), max(0, x) : max(0, x + rw)] = 0
    return mask


class PlayerLocator(ABC):
    @abstractmethod
    def locate(self, frame: np.ndarray, hsv: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """返回玩家中心 (x, y)，找不到返回 None。hsv 是同一帧的 HSV 图（有就不用再转）。"""


class FixedCenterLocator(PlayerLocator):
    """镜头跟随的游戏：玩家永远在画面正中。"""

    def locate(self, frame: np.ndarray, hsv: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        return np.array([w / 2, h / 2])


class ColorPlayerLocator(PlayerLocator):
    """按颜色找：颜色在范围内、大小接近角色大小的连通块里，挑离上一帧位置最近的。"""

    def __init__(self, color: ColorRange, size: list[int], ignore_regions: list[list[int]]) -> None:
        self.lower = np.array(color.lower, dtype=np.uint8)
        self.upper = np.array(color.upper, dtype=np.uint8)
        self.expected_area = float(size[0] * size[1])
        self.ignore_regions = ignore_regions
        self._mask: Optional[np.ndarray] = None
        self.last: Optional[np.ndarray] = None
        self.search_radius = 4 * max(size)

    def locate(self, frame: np.ndarray, hsv: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self._mask is None or self._mask.shape != frame.shape[:2]:
            self._mask = ignore_mask(frame.shape, self.ignore_regions)
        if hsv is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 先在上一帧位置附近找（快），找不到再全图找
        if self.last is not None:
            r = int(self.search_radius)
            x, y = int(self.last[0]), int(self.last[1])
            h, w = frame.shape[:2]
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + r), min(h, y + r)
            if x1 > x0 and y1 > y0:
                found = self._search(hsv[y0:y1, x0:x1], self._mask[y0:y1, x0:x1], np.array([x0, y0]))
                if found is not None:
                    return found
        return self._search(hsv, self._mask, np.zeros(2))

    def _search(self, hsv: np.ndarray, mask: np.ndarray, origin: np.ndarray) -> Optional[np.ndarray]:
        m = cv2.inRange(hsv, self.lower, self.upper)
        m &= mask
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        best, best_score = None, float("inf")
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (0.3 * self.expected_area <= area <= 3.0 * self.expected_area):
                continue
            c = cents[i] + origin
            score = abs(np.log(area / self.expected_area))
            if self.last is not None:
                score += float(np.hypot(*(c - self.last))) / 50.0
            if score < best_score:
                best, best_score = c, score
        if best is None:
            return None
        self.last = np.array(best, dtype=float)
        return self.last.copy()
