import json

import numpy as np

from conftest import ROOT
from dodge_copilot.bench import run_episode
from dodge_copilot.config import load_profile
from dodge_copilot.debug import draw_debug
from dodge_copilot.dodge import DodgePolicy
from dodge_copilot.perception import Perception
from dodge_copilot.record import Recorder
from testgame import Game, GameConfig


def test_debug_overlay_draws():
    p = load_profile(ROOT / "profiles/testgame.yaml")
    perc, pol = Perception(p), DodgePolicy(p)
    g = Game(GameConfig(seed=1))
    obs = None
    for _ in range(200):
        g.step((0, 0.2), 1 / 60)
        obs = perc.update(g.render(), g.time)
        pol.decide(obs)
        if not g.alive:
            break
    img = draw_debug(obs.frame, obs, pol, text="test")
    assert img.shape == obs.frame.shape
    assert not np.array_equal(img, obs.frame)


def test_recorder(tmp_path):
    r = Recorder(tmp_path)
    r.write(None, {"x": 1})  # 没开始时不写
    d = r.start()
    for i in range(3):
        r.write(np.zeros((10, 10, 3), np.uint8), {"t": i * 0.1, "player": np.array([1.0, 2.0])})
    r.stop()
    lines = (d / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["player"] == [1.0, 2.0]
    assert len(list(d.glob("*.jpg"))) == 3


def test_assist_beats_idle_on_a_few_seeds():
    p = load_profile(ROOT / "profiles/testgame.yaml")
    for seed in (0, 1):
        idle = run_episode(p, "idle", seed, max_time=12)
        assist = run_episode(p, "assist", seed, max_time=12)
        assert assist.survival_s > idle.survival_s + 3


def test_assist_follow_camera():
    from dodge_copilot.__main__ import apply_camera

    p = load_profile(ROOT / "profiles/testgame.yaml")
    apply_camera(p, "follow")
    idle = run_episode(p, "idle", 0, max_time=12)
    assist = run_episode(p, "assist", 0, max_time=12)
    assert assist.survival_s > idle.survival_s + 3
