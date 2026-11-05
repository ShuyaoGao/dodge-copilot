import hashlib

import numpy as np
import pytest

from testgame import Beam, Bullet, Game, GameConfig, polyline_distance


def run(cfg, moves, n=300):
    g = Game(cfg)
    hashes = []
    for i in range(n):
        g.step(moves(i), 1 / 60)
        if i % 30 == 0:
            hashes.append(hashlib.md5(g.render().tobytes()).hexdigest())
    return g, hashes


def test_deterministic_with_seed():
    mv = lambda i: (1, 0) if (i // 40) % 2 else (-1, 0.5)  # noqa: E731
    _, a = run(GameConfig(seed=3), mv)
    _, b = run(GameConfig(seed=3), mv)
    _, c = run(GameConfig(seed=4), mv)
    assert a == b
    assert a != c


@pytest.mark.parametrize("camera", ["fixed", "follow"])
def test_idle_player_dies_eventually(camera):
    g = Game(GameConfig(seed=0, camera=camera))
    while g.step((0, 0), 1 / 60) and g.time < 60:
        pass
    assert not g.alive and g.death_cause is not None


def test_player_clamped_in_fixed_arena():
    g = Game(GameConfig(seed=0, patterns=()))
    for _ in range(600):
        g.step((-1, -1), 1 / 60)
    assert g.alive
    assert np.allclose(g.player, [8, 8])


def test_follow_camera_keeps_player_centered():
    g = Game(GameConfig(seed=0, camera="follow", patterns=()))
    for _ in range(120):
        g.step((1, 0.3), 1 / 60)
    assert np.allclose(g.player_screen_pos(), [480, 360])
    assert g.player[0] > 480 + 400  # 世界坐标确实走远了
    a = g.render()
    g.step((1, 0), 1 / 60)
    b = g.render()
    assert not np.array_equal(a, b)  # 背景跟着动


def test_diagonal_not_faster():
    g = Game(GameConfig(seed=0, patterns=()))
    start = g.player.copy()
    g.step((1, 1), 0.5)
    assert np.isclose(np.hypot(*(g.player - start)), 0.5 * g.config.player_speed)


def _game_with(hazard):
    g = Game(GameConfig(seed=0, patterns=()))
    if isinstance(hazard, Beam):
        g.beams.append(hazard)
    else:
        g.bullets.append(hazard)
    return g


def test_bullet_hit():
    p = np.array([480.0, 360.0])
    g = _game_with(Bullet("aimed", p + [30, 0], np.array([-600.0, 0]), 6))
    alive = True
    for _ in range(10):
        alive = g.step((0, 0), 1 / 60)
    assert not alive and g.death_cause == "aimed"


@pytest.mark.parametrize("kind", ["laser", "lightning"])
def test_beam_harmless_while_warning_then_kills(kind):
    pts = np.array([[0.0, 360.0], [960.0, 360.0]])
    g = _game_with(Beam(kind, pts, width=10, warn_s=0.5, active_s=0.3))
    for _ in range(25):  # 0.42 秒，还在预警
        assert g.step((0, 0), 1 / 60)
    for _ in range(10):
        g.step((0, 0), 1 / 60)
    assert not g.alive and g.death_cause == kind


def test_beam_expires():
    pts = np.array([[0.0, 100.0], [960.0, 100.0]])
    g = _game_with(Beam("laser", pts, width=10, warn_s=0.1, active_s=0.1))
    for _ in range(20):
        g.step((0, 0), 1 / 60)
    assert g.alive and g.beams == []


def test_homing_turns_toward_player():
    g = _game_with(Bullet("homing", np.array([480.0, 100.0]), np.array([150.0, 0.0]), 6,
                          life=5, turn_rate=2.0))
    for _ in range(30):
        g.step((0, 0), 1 / 60)
    assert g.bullets[0].vel[1] > 50  # 拐向下方的玩家了


def test_polyline_distance():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert polyline_distance(np.array([5.0, 3.0]), pts) == pytest.approx(3.0)
    assert polyline_distance(np.array([13.0, 5.0]), pts) == pytest.approx(3.0)
    assert polyline_distance(np.array([-4.0, -3.0]), pts) == pytest.approx(5.0)


def test_render_shape_and_player_color():
    g = Game(GameConfig(seed=0, patterns=()))
    img = g.render()
    assert img.shape == (720, 960, 3) and img.dtype == np.uint8
    assert tuple(img[360, 480]) == (90, 230, 90)
