"""测试模式：助手直接对接测试游戏，不需要真屏幕和真键盘。

模拟真实情况：游戏看到的按键 = 玩家按的键 ∪ 助手按的键，跟操作系统里一样。
"""
from __future__ import annotations

import math

import numpy as np

from testgame import Game, GameConfig, PATTERNS

from ..config import Profile
from ..inputs import KeyboardSink, StickSink, keys_to_vector


class SimKeyboardSink(KeyboardSink):
    def __init__(self, keymap: dict[str, str]) -> None:
        super().__init__(keymap)
        self.down: set[str] = set()  # 当前按下的"物理键"
        self.events: list[tuple[str, str]] = []  # (down/up, 键)，测试用

    def _key_down(self, key: str) -> None:
        self.down.add(key)
        self.events.append(("down", key))

    def _key_up(self, key: str) -> None:
        self.down.discard(key)
        self.events.append(("up", key))


class SimStickSink(StickSink):
    def _set_stick(self, x: float, y: float) -> None:
        pass  # 状态已经记在 self.stick 里


def make_game(profile: Profile, seed: int) -> Game:
    sim = profile.sim
    assert sim is not None, "测试模式需要配置里有 sim 段"
    return Game(
        GameConfig(
            width=sim.width,
            height=sim.height,
            camera=sim.camera,
            seed=seed,
            difficulty=sim.difficulty,
            player_speed=profile.player.speed_px_s,
            player_size=profile.player.size[0],
            player_hit_radius=profile.player.hit_radius,
            patterns=tuple(sim.patterns) if sim.patterns else PATTERNS,
        )
    )


class SimBridge:
    def __init__(self, game: Game, profile: Profile) -> None:
        self.game = game
        self.keymap = dict(profile.input.keys)
        self._reverse = {v: k for k, v in self.keymap.items()}
        assert profile.sim is not None
        self.sink: SimKeyboardSink | SimStickSink
        if profile.sim.stick:
            self.sink = SimStickSink()
        else:
            self.sink = SimKeyboardSink(self.keymap)
        self.human_keys: set[str] = set()  # 玩家按着的逻辑方向

    def grab(self) -> np.ndarray:
        return self.game.render()

    def game_move(self) -> tuple[float, float]:
        keys = set(self.human_keys)
        if isinstance(self.sink, SimKeyboardSink):
            keys |= {self._reverse[k] for k in self.sink.down if k in self._reverse}
        x, y = keys_to_vector(keys)
        if isinstance(self.sink, SimStickSink):
            x += self.sink.stick[0]
            y += self.sink.stick[1]
        n = math.hypot(x, y)
        return (x / n, y / n) if n > 1 else (x, y)
