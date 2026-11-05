import numpy as np

from conftest import ROOT
from dodge_copilot.config import load_profile
from dodge_copilot.perception import FixedCenterLocator, Perception, ignore_mask
from testgame import Game, GameConfig


def test_ignore_mask():
    m = ignore_mask((100, 200, 3), [[0, 0, 50, 20], [190, 90, 50, 50]])
    assert m[10, 10] == 0 and m[19, 49] == 0 and m[20, 10] == 255
    assert m[95, 195] == 0 and m[50, 100] == 255


def test_fixed_center():
    assert np.allclose(FixedCenterLocator().locate(np.zeros((720, 960, 3), np.uint8)), [480, 360])


def test_color_locator_tracks_player_among_bullets():
    profile = load_profile(ROOT / "profiles/testgame.yaml")
    perc = Perception(profile)
    g = Game(GameConfig(seed=2))
    errs = []
    for i in range(240):
        g.step((np.cos(i / 30), np.sin(i / 45)), 1 / 60)
        if not g.alive:
            break
        obs = perc.update(g.render(), g.time)
        assert obs.player is not None
        errs.append(np.hypot(*(obs.player - g.player_screen_pos())))
    assert len(errs) > 60
    assert max(errs) < 1.5
