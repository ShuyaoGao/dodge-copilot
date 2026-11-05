"""看懂画面：玩家在哪、危险物在哪、往哪飞。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import Profile
from .hazards import CameraMotion, Detection, DetectionResult, HazardDetector
from .player import ColorPlayerLocator, FixedCenterLocator, PlayerLocator, ignore_mask
from .tracker import Track, Tracker


@dataclass
class Observation:
    t: float  # 秒
    frame: Optional[np.ndarray]  # BGR 画面；不需要画面的策略可以是 None
    player: Optional[np.ndarray] = None  # 玩家在画面上的位置 (x, y)
    hazards: list[Track] = field(default_factory=list)  # 点状危险物（世界坐标）
    region_mask: Optional[np.ndarray] = None  # 区域类危险物（画面坐标），255 = 危险
    region_dist: Optional[np.ndarray] = None  # 到最近危险区域的距离（像素），半分辨率
    region_dist_scale: int = 2  # region_dist[y // 2, x // 2] 对应画面 (x, y)
    warn_mask: Optional[np.ndarray] = None  # 预警中的区域（还没生效）
    warn_dist: Optional[np.ndarray] = None
    cam_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 世界 = 画面 + 它
    detections: list[Detection] = field(default_factory=list)

    @property
    def frame_size(self) -> Optional[tuple[int, int]]:
        if self.frame is None:
            return None
        return self.frame.shape[1], self.frame.shape[0]


class Perception:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.player_locator = build_player_locator(profile)
        hz = profile.hazards
        self.detector = HazardDetector(hz, profile.ignore_regions, profile.player.size)
        self.tracker = Tracker(hz.track_gate_px, hz.track_max_misses)
        self.camera = CameraMotion() if hz.motion_compensation else None
        self._last_player: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.tracker.reset()
        if self.camera:
            self.camera.reset()
        self._last_player = None

    def update(self, frame: Optional[np.ndarray], t: float) -> Observation:
        obs = Observation(t=t, frame=frame)
        if frame is None:
            return obs
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        player = self.player_locator.locate(frame, hsv)
        if player is None:
            player = self._last_player
        self._last_player = player
        obs.player = player

        if self.camera is not None:
            obs.cam_offset = self.camera.update(hsv[:, :, 2], self._camera_exclude(frame.shape, player))
        det = self.detector.detect(hsv, player)
        obs.detections = det.points
        pw = None if player is None else player + obs.cam_offset
        obs.hazards = list(self.tracker.update(det.points, obs.cam_offset, t, pw))
        if det.region_mask is not None:
            obs.region_mask = det.region_mask
            obs.region_dist = _dist_map(det.region_mask, obs.region_dist_scale)
        if det.warn_mask is not None:
            obs.warn_mask = det.warn_mask
            obs.warn_dist = _dist_map(det.warn_mask, obs.region_dist_scale)
        return obs

    def _camera_exclude(self, shape: tuple[int, ...], player: Optional[np.ndarray]) -> np.ndarray:
        m = 255 - ignore_mask(shape, self.profile.ignore_regions)
        if player is not None:
            hw, hh = self.profile.player.size[0], self.profile.player.size[1]
            x, y = int(player[0]), int(player[1])
            m[max(0, y - hh) : y + hh, max(0, x - hw) : x + hw] = 255
        return m


def _dist_map(mask: np.ndarray, k: int) -> np.ndarray:
    """每个格子到最近危险像素的距离（像素）。半分辨率算，快 4 倍。"""
    # 缩小时只要格子里有一点危险就算危险，距离再乘回去
    small = cv2.resize(cv2.dilate(mask, np.ones((k, k), np.uint8)),
                       (mask.shape[1] // k, mask.shape[0] // k), interpolation=cv2.INTER_NEAREST)
    return cv2.distanceTransform(255 - small, cv2.DIST_L2, 3) * k


def build_player_locator(profile: Profile) -> PlayerLocator:
    p = profile.player
    if p.mode == "fixed_center":
        return FixedCenterLocator()
    if p.mode == "color":
        assert p.color is not None
        return ColorPlayerLocator(p.color, p.size, profile.ignore_regions)
    raise NotImplementedError("模板匹配找玩家（player.mode: template）在 M2 跟标定工具一起做")


__all__ = [
    "CameraMotion",
    "ColorPlayerLocator",
    "Detection",
    "DetectionResult",
    "FixedCenterLocator",
    "HazardDetector",
    "Observation",
    "Perception",
    "PlayerLocator",
    "Track",
    "Tracker",
    "build_player_locator",
    "ignore_mask",
]
