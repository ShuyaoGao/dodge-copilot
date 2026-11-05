import cv2
import numpy as np
import pytest

from conftest import ROOT
from dodge_copilot.config import HazardConfig, load_profile
from dodge_copilot.perception import CameraMotion, HazardDetector, Perception, Tracker
from dodge_copilot.perception.hazards import Detection
from testgame import Beam, Bullet, Game, GameConfig


def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def empty_game(**kw):
    return Game(GameConfig(seed=0, patterns=(), **kw))


def detector():
    return HazardDetector(HazardConfig(), [[0, 0, 240, 48]], [16, 16])


def test_bullets_are_points_hud_and_player_ignored():
    g = empty_game()
    for x, kind in [(100, "aimed"), (300, "fan"), (500, "homing"), (700, "chaser")]:
        g.bullets.append(Bullet(kind, np.array([x, 600.0]), np.zeros(2), 12 if kind == "chaser" else 6))
    img = g.render()
    res = detector().detect(hsv(img), g.player_screen_pos())
    xs = sorted(round(d.pos[0]) for d in res.points)
    assert xs == [100, 300, 500, 700]  # 红色"血条"和绿色玩家都没被当成危险物
    assert all(abs(d.pos[1] - 600) < 1 for d in res.points)
    assert res.region_mask is None


@pytest.mark.parametrize("kind", ["laser", "lightning"])
@pytest.mark.parametrize("t", [0.1, 0.8])  # 预警中 / 生效中
def test_beams_are_regions(kind, t):
    g = empty_game()
    pts = np.array([[0.0, 200.0], [960.0, 260.0]]) if kind == "laser" else \
        np.array([[0.0, 200.0], [300.0, 230.0], [600.0, 180.0], [960.0, 220.0]])
    warn = 0.7
    g.beams.append(Beam(kind, pts, width=12, warn_s=warn, active_s=0.5, t=t))
    res = detector().detect(hsv(g.render()), g.player_screen_pos())
    mask = res.region_mask if t > warn else res.warn_mask
    other = res.warn_mask if t > warn else res.region_mask
    assert mask is not None
    ys = np.nonzero(mask[:, 480])[0]
    assert len(ys) > 0 and 170 < ys.mean() < 260
    assert other is None or not other[:, 480].any()  # 生效 / 预警分得清
    assert len(res.points) <= 2  # 不应该被切成一堆点


def test_camera_motion_sign_and_accuracy():
    g = Game(GameConfig(seed=0, camera="follow", patterns=(), hud=False))
    cam = CameraMotion()
    ex = np.zeros((720, 960), np.uint8)
    ex[344:376, 464:496] = 255  # 玩家
    cam.update(hsv(g.render())[:, :, 2], ex)
    start = g.camera_offset.copy()
    for _ in range(60):
        g.step((0.8, -0.6), 1 / 60)
        est = cam.update(hsv(g.render())[:, :, 2], ex)
    true = g.camera_offset - start
    assert np.hypot(*(est - true)) < 10, (est, true)  # 累计 240 像素，允许 4% 漂移


def test_tracker_estimates_velocity():
    tr = Tracker()
    v = np.array([200.0, -50.0])
    for i in range(10):
        t = i / 60
        dets = [Detection(np.array([100.0, 300.0]) + v * t, 6, 100),
                Detection(np.array([600.0, 100.0]), 6, 100)]
        tracks = tr.update(dets, np.zeros(2), t)
    assert len(tracks) == 2
    moving = min(tracks, key=lambda k: k.pos[0])
    still = max(tracks, key=lambda k: k.pos[0])
    assert np.hypot(*(moving.vel - v)) < 10
    assert np.hypot(*still.vel) < 5


def test_tracker_drops_lost_tracks_and_keeps_ids():
    tr = Tracker(max_misses=2)
    tr.update([Detection(np.array([10.0, 10.0]), 5, 50)], np.zeros(2), 0.0)
    tid = tr.tracks[0].id
    tr.update([Detection(np.array([12.0, 10.0]), 5, 50)], np.zeros(2), 1 / 60)
    assert tr.tracks[0].id == tid and tr.tracks[0].hits == 2
    for k in range(3):
        tr.update([], np.zeros(2), (2 + k) / 60)
    assert tr.tracks == []


def test_perception_on_game_matches_truth():
    profile = load_profile(ROOT / "profiles/testgame.yaml")
    perc = Perception(profile)
    g = Game(GameConfig(seed=5, patterns=("aimed", "fan")))
    checked = 0
    for i in range(300):
        g.step((np.cos(i / 40), np.sin(i / 50)), 1 / 60)
        if not g.alive:
            break
        obs = perc.update(g.render(), g.time)
        if i > 100 and i % 20 == 0:
            for b in g.bullets:
                crowded = any(o is not b and np.hypot(*(o.pos - b.pos)) < 25 for o in g.bullets) \
                    or np.hypot(*(g.player - b.pos)) < 25
                if 40 < b.pos[0] < 920 and 80 < b.pos[1] < 680 and not crowded:  # 刚进画面时只露出半个，位置不准
                    d = [np.hypot(*(h.pos - b.pos)) for h in obs.hazards]
                    k = int(np.argmin(d))
                    assert d[k] < 3
                    if obs.hazards[k].hits > 5:
                        assert np.hypot(*(obs.hazards[k].vel - b.vel)) < 25
                    checked += 1
    assert checked > 10


def test_pursuer_detection():
    """追人的敌人会被认出来，直线子弹不会。"""
    g = empty_game()
    g.bullets.append(Bullet("chaser", np.array([100.0, 100.0]), np.zeros(2), 12, life=30, speed=100))
    g.bullets.append(Bullet("aimed", np.array([900.0, 100.0]), np.array([-150.0, 100.0]), 6))
    profile = load_profile(ROOT / "profiles/testgame.yaml")
    perc = Perception(profile)
    for i in range(120):
        g.step((0, 1) if i < 50 else (1, 0), 1 / 60)  # 玩家拐个弯，方位变化
        obs = perc.update(g.render(), g.time)
    chaser = min(obs.hazards, key=lambda h: np.hypot(*(h.pos - g.bullets[0].pos)))
    aimed = min(obs.hazards, key=lambda h: np.hypot(*(h.pos - g.bullets[1].pos)))
    assert chaser.pursuer
    assert not aimed.pursuer
