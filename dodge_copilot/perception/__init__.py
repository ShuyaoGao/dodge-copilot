"""看懂画面。M0 只做"玩家在哪"，危险物识别在 M1 加。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import Profile
from .player import ColorPlayerLocator, FixedCenterLocator, PlayerLocator, ignore_mask


@dataclass
class Observation:
    t: float  # 秒
    frame: Optional[np.ndarray]  # BGR 画面；不需要画面的策略可以是 None
    player: Optional[np.ndarray] = None  # 玩家在画面上的位置 (x, y)
    hazards: list = field(default_factory=list)  # M1 填充


class Perception:
    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.player_locator = build_player_locator(profile)

    def update(self, frame: Optional[np.ndarray], t: float) -> Observation:
        obs = Observation(t=t, frame=frame)
        if frame is not None:
            obs.player = self.player_locator.locate(frame)
        return obs


def build_player_locator(profile: Profile) -> PlayerLocator:
    p = profile.player
    if p.mode == "fixed_center":
        return FixedCenterLocator()
    if p.mode == "color":
        assert p.color is not None
        return ColorPlayerLocator(p.color, p.size, profile.ignore_regions)
    raise NotImplementedError("模板匹配找玩家（player.mode: template）在 M2 跟标定工具一起做")


__all__ = [
    "ColorPlayerLocator",
    "FixedCenterLocator",
    "Observation",
    "Perception",
    "PlayerLocator",
    "build_player_locator",
    "ignore_mask",
]
