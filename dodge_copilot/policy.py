"""决策：根据观察结果决定往哪走。

- idle：不操作（基准）
- random：随机乱走（基准：随机选一个方向，按一小会儿，再换）
- assist：助手，看画面躲闪（见 dodge.py）
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .perception import Observation


class Policy(ABC):
    name = "base"
    needs_frame = True  # 不需要看画面的策略可以跳过截屏和识别，跑基准更快

    @abstractmethod
    def decide(self, obs: Observation) -> tuple[float, float]:
        """返回移动方向 (dx, dy)，长度 0-1。"""

    def reset(self) -> None:
        pass


class IdlePolicy(Policy):
    name = "idle"
    needs_frame = False

    def decide(self, obs: Observation) -> tuple[float, float]:
        return (0.0, 0.0)


class RandomPolicy(Policy):
    name = "random"
    needs_frame = False

    def __init__(self, seed: int = 0, hold_s: tuple[float, float] = (0.2, 0.8)) -> None:
        self.seed = seed
        self.hold_s = hold_s
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed + 7919)
        self._until = -1.0
        self._dir = (0.0, 0.0)

    def decide(self, obs: Observation) -> tuple[float, float]:
        if obs.t >= self._until:
            k = int(self.rng.integers(0, 9))  # 0-7 八个方向，8 不动
            if k == 8:
                self._dir = (0.0, 0.0)
            else:
                a = k * math.pi / 4
                self._dir = (math.cos(a), math.sin(a))
            self._until = obs.t + float(self.rng.uniform(*self.hold_s))
        return self._dir


POLICIES = {"idle": IdlePolicy, "random": RandomPolicy, "assist": None}


def build_policy(name: str, seed: int = 0, profile=None) -> Policy:
    if name == "assist":
        if profile is None:
            raise ValueError("assist 策略需要配置文件")
        from .dodge import DodgePolicy

        return DodgePolicy(profile)
    if name == "idle":
        return IdlePolicy()
    if name == "random":
        return RandomPolicy(seed)
    raise ValueError(f"不认识的策略 {name!r}，可选: {sorted(POLICIES)}")
