"""通用危险物识别（不需要训练）。

- 子弹、敌人：又亮又饱和、大小适中的色块 -> 当成"点"，交给追踪器估计速度。
- 激光、闪电（包括它们较暗的预警线）、大片危险区：细长或很大的色块 -> 当成"区域"。
- 镜头跟随时：用相邻两帧的相位相关估计画面整体平移，把位置换算到"世界坐标"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import HazardConfig
from .player import ignore_mask


@dataclass
class Detection:
    pos: np.ndarray  # 画面坐标 (x, y)
    radius: float
    area: int


@dataclass
class DetectionResult:
    points: list[Detection] = field(default_factory=list)
    region_mask: Optional[np.ndarray] = None  # uint8，已经生效的危险区域是 255
    warn_mask: Optional[np.ndarray] = None  # uint8，预警中（较暗的线）的区域是 255


class CameraMotion:
    """估计镜头累计移动了多少。世界坐标 = 画面坐标 + offset。"""

    def __init__(self, scale: float = 0.5, brightness_cap: int = 90) -> None:
        self.scale = scale
        # 玩家、子弹又亮又不跟背景一起动，会把估计拉偏；把亮的地方压平，只看背景纹理
        self.brightness_cap = brightness_cap
        self.offset = np.zeros(2)
        self._prev: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self.last_shift = np.zeros(2)

    def reset(self) -> None:
        self.offset = np.zeros(2)
        self._prev = None

    def update(self, gray: np.ndarray, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        """gray：亮度图。exclude：uint8 掩码，非 0 的地方（玩家、界面）不参与估计。"""
        gray = np.minimum(gray, self.brightness_cap)
        if exclude is not None:
            gray = gray.copy()
            gray[exclude > 0] = int(np.median(gray))
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        if self._window is None or self._window.shape != small.shape:
            self._window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            self._prev = None
        if self._prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(self._prev, small, self._window)
            # 画面内容往左挪 = 镜头往右走
            self.last_shift = -np.array([dx, dy]) / self.scale
            self.offset = self.offset + self.last_shift
        self._prev = small
        return self.offset.copy()


class HazardDetector:
    def __init__(self, cfg: HazardConfig, ignore_regions: list[list[int]], player_size: list[int]) -> None:
        self.cfg = cfg
        self.ignore_regions = ignore_regions
        self.player_size = player_size
        self._ignore: Optional[np.ndarray] = None
        self._lo = np.array([0, cfg.min_saturation, cfg.min_brightness], np.uint8)
        self._hi = np.array([179, 255, 255], np.uint8)
        self._line_lo = np.array([0, cfg.line_min_saturation, cfg.line_min_brightness], np.uint8)
        self._colors = [(np.array(c.lower, np.uint8), np.array(c.upper, np.uint8)) for c in cfg.colors]

    def _base_mask(self, hsv: np.ndarray, player: Optional[np.ndarray]) -> np.ndarray:
        if self._ignore is None or self._ignore.shape != hsv.shape[:2]:
            self._ignore = ignore_mask(hsv.shape, self.ignore_regions)
        m = self._ignore.copy()
        if player is not None:  # 玩家自己不算危险物
            hw = self.player_size[0] // 2 + 2
            hh = self.player_size[1] // 2 + 2
            x, y = int(round(player[0])), int(round(player[1]))
            m[max(0, y - hh) : max(0, y + hh), max(0, x - hw) : max(0, x + hw)] = 0
        return m

    def _is_region(self, area: int, w: int, h: int) -> bool:
        cfg = self.cfg
        if area > cfg.max_area:
            return True
        long, short = max(w, h), max(1, min(w, h))
        fill = area / float(w * h)
        return long >= cfg.line_min_length and (long / short >= 3 or fill < 0.3)

    def detect(self, hsv: np.ndarray, player: Optional[np.ndarray]) -> DetectionResult:
        cfg = self.cfg
        base = self._base_mask(hsv, player)

        main = cv2.inRange(hsv, self._lo, self._hi)
        for lo, hi in self._colors:
            main |= cv2.inRange(hsv, lo, hi)
        main &= base
        n, labels, stats, cents = cv2.connectedComponentsWithStats(main, connectivity=8)
        res = DetectionResult()
        region_ids = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cfg.min_area:
                continue
            w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            if self._is_region(area, w, h):
                region_ids.append(i)
            else:
                res.points.append(Detection(cents[i].astype(float), max(w, h) / 2.0, area))
        region = np.zeros(hsv.shape[:2], np.uint8)
        if region_ids:
            region[np.isin(labels, region_ids)] = 255

        # 较暗的预警线：只认细长的。线都很长，用半分辨率找就够了，快 4 倍
        line = cv2.inRange(hsv, self._line_lo, self._hi) & base
        small = cv2.resize(line, (line.shape[1] // 2, line.shape[0] // 2), interpolation=cv2.INTER_AREA)
        small = (small > 0).astype(np.uint8)
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
        line_ids = []
        for i in range(1, n2):
            w, h = int(stats2[i, cv2.CC_STAT_WIDTH]), int(stats2[i, cv2.CC_STAT_HEIGHT])
            area = int(stats2[i, cv2.CC_STAT_AREA])
            if 2 * max(w, h) >= cfg.line_min_length and self._is_region(4 * area, 2 * w, 2 * h):
                line_ids.append(i)
        if line_ids:
            lm = np.isin(labels2, line_ids).astype(np.uint8) * 255
            lm = cv2.resize(lm, (line.shape[1], line.shape[0]), interpolation=cv2.INTER_NEAREST)
            # 已经亮起来的部分算"生效"，其余算"预警"
            warn = lm & line & ~cv2.dilate(main, np.ones((5, 5), np.uint8))
            region |= lm & line & ~warn
            if warn.any():
                res.warn_mask = warn
        if region.any():
            res.region_mask = region
        return res
