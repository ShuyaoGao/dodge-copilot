"""存活时间基准：同一组种子，不同策略各存活多久。这是衡量助手好坏的主要数字。

用法：
    python -m dodge_copilot bench --profile profiles/testgame.yaml --seeds 20 --policies idle,random
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .config import Profile
from .control import TakeoverController
from .io.sim import SimBridge, make_game
from .loop import Copilot
from .perception import Perception
from .policy import build_policy
from .stats import FrameTimer


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    survival_s: float
    death_cause: str | None  # None 表示撑到了时间上限
    timer: FrameTimer


def run_episode(profile: Profile, policy_name: str, seed: int, max_time: float = 60.0,
                fps: int = 60) -> EpisodeResult:
    """完整闭环跑一局：读测试游戏画面 -> 识别 -> 决策 -> 通过接管控制器按键 -> 游戏前进。"""
    game = make_game(profile, seed)
    bridge = SimBridge(game, profile)
    controller = TakeoverController(bridge.sink)
    policy = build_policy(policy_name, seed)
    copilot = Copilot(bridge.grab, Perception(profile), policy, controller)
    controller.engage("基准测试：全程接管")
    dt = 1.0 / fps
    try:
        while game.alive and game.time < max_time:
            copilot.step(game.time)
            game.step(bridge.game_move(), dt)
    finally:
        bridge.sink.release_all()
    return EpisodeResult(policy_name, seed, game.time, game.death_cause, copilot.timer)


def run_bench(profile: Profile, policies: list[str], seeds: list[int], max_time: float = 60.0,
              verbose: bool = True) -> dict[str, list[EpisodeResult]]:
    results: dict[str, list[EpisodeResult]] = {}
    for name in policies:
        results[name] = []
        for s in seeds:
            r = run_episode(profile, name, s, max_time)
            results[name].append(r)
            if verbose:
                cause = r.death_cause or "撑满时间"
                print(f"  {name:>8}  种子 {s:3d}  存活 {r.survival_s:6.2f} 秒  ({cause})", flush=True)
    return results


def format_table(results: dict[str, list[EpisodeResult]], max_time: float) -> str:
    lines = [
        f"| 策略 | 局数 | 平均存活（秒） | 中位数 | 最短 | 最长 | 撑满 {max_time:g} 秒的局数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rs in results.items():
        t = [r.survival_s for r in rs]
        full = sum(r.death_cause is None for r in rs)
        lines.append(
            f"| {name} | {len(t)} | {statistics.mean(t):.2f} | {statistics.median(t):.2f} | "
            f"{min(t):.2f} | {max(t):.2f} | {full} |"
        )
    return "\n".join(lines)
