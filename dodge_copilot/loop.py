"""一帧的流程：截屏 -> 识别 -> 决策 -> 输入。真实模式和测试模式共用。"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .control import TakeoverController
from .perception import Observation, Perception
from .policy import Policy
from .stats import FrameTimer


class Copilot:
    def __init__(
        self,
        grab: Callable[[], Optional[np.ndarray]],
        perception: Perception,
        policy: Policy,
        controller: TakeoverController,
        timer: Optional[FrameTimer] = None,
        frame_time: Optional[Callable[[], float]] = None,
    ) -> None:
        """frame_time：截到的画面是哪个时刻的（测试模式模拟延迟时用）。不给就用 step() 的 t。"""
        self.frame_time = frame_time
        self.grab = grab
        self.perception = perception
        self.policy = policy
        self.controller = controller
        self.timer = timer or FrameTimer()
        self.last_obs: Optional[Observation] = None
        self.last_action: tuple[float, float] = (0.0, 0.0)

    def step(self, t: float) -> Observation:
        """跑一帧。识别和决策一直在跑（保证接管那一刻追踪是热的），只有接管中才真的按键。"""
        tm = self.timer
        tm.begin_frame()
        self.controller.tick()
        frame = None
        if self.policy.needs_frame:
            with tm.stage("capture"):
                frame = self.grab()
            if self.frame_time is not None:
                t = self.frame_time()
        with tm.stage("perceive"):
            obs = self.perception.update(frame, t)
        with tm.stage("decide"):
            action = self.policy.decide(obs)
        with tm.stage("act"):
            self.controller.apply(*action)
        tm.end_frame()
        self.last_obs = obs
        self.last_action = action
        return obs
