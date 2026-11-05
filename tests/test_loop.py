import math

from conftest import ROOT
from dodge_copilot.bench import format_table, run_bench, run_episode
from dodge_copilot.config import load_profile


def profile():
    return load_profile(ROOT / "profiles/testgame.yaml")


def test_episode_runs_closed_loop_and_is_reproducible():
    a = run_episode(profile(), "random", seed=1, max_time=10)
    b = run_episode(profile(), "random", seed=1, max_time=10)
    assert a.survival_s == b.survival_s and a.death_cause == b.death_cause
    assert 0 < a.survival_s <= 10 + 1e-6


def test_random_policy_actually_moves_player():
    from dodge_copilot.control import TakeoverController
    from dodge_copilot.io.sim import SimBridge, make_game
    from dodge_copilot.loop import Copilot
    from dodge_copilot.perception import Perception
    from dodge_copilot.policy import build_policy

    p = profile()
    p.sim.patterns = []  # 没有危险物，只看能不能动起来
    game = make_game(p, 0)
    bridge = SimBridge(game, p)
    c = TakeoverController(bridge.sink)
    cp = Copilot(bridge.grab, Perception(p), build_policy("random", 0), c)
    start = game.player.copy()
    for i in range(120):
        if i == 60:
            c.toggle()  # 前一秒玩家控制，助手不应该动
            assert (game.player == start).all()
        cp.step(game.time)
        game.step(bridge.game_move(), 1 / 60)
    assert math.hypot(*(game.player - start)) > 20
    assert cp.timer.mean_ms("total") >= 0


def test_frame_pipeline_with_color_locator_and_timer():
    from dodge_copilot.policy import Policy

    import dodge_copilot.bench as bench

    class Watcher(Policy):
        name = "watch"
        needs_frame = True
        seen = []

        def decide(self, obs):
            self.seen.append(obs.player)
            return (0.0, 0.0)

    orig = bench.build_policy
    bench.build_policy = lambda name, seed: Watcher()
    try:
        r = run_episode(profile(), "watch", 0, max_time=1.0)
    finally:
        bench.build_policy = orig
    assert all(p is not None for p in Watcher.seen)
    assert len(Watcher.seen) >= 59
    assert r.timer.mean_ms("capture") > 0 and r.timer.mean_ms("perceive") > 0


def test_bench_table():
    res = run_bench(profile(), ["idle", "random"], [0, 1], max_time=5, verbose=False)
    t = format_table(res, 5)
    assert "| idle | 2 |" in t and "| random | 2 |" in t


def test_sim_app_headless_hotkeys_and_override():
    from dodge_copilot.app import run_sim
    from dodge_copilot.control import Mode

    p = profile()
    p.sim.patterns = []
    script = {
        5: [("keydown", "F8")],  # 接管
        30: [("keydown", "left")],  # 玩家一按方向键 -> 自动交还
        31: [("keyup", "left")],
        40: [("keydown", "F8")],  # 再接管
        50: [("keydown", "F12")],  # 急停
    }
    c = run_sim(p, "random", 0, max_frames=200, headless=True, script=script)
    reasons = [r for _, _, r in c.switch_log]
    modes = [m for _, m, _ in c.switch_log]
    assert modes == [Mode.ASSIST, Mode.PLAYER, Mode.ASSIST, Mode.PLAYER]
    assert "自动交还" in reasons[1] and reasons[-1] == "急停"
    assert c.stopped
    assert c.sink.down == set()
