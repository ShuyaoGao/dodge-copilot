import pytest

from conftest import ROOT
from dodge_copilot.config import ProfileError, load_profile, profile_from_dict


def test_repo_profiles_load():
    p = load_profile(ROOT / "profiles/testgame.yaml")
    assert p.name == "testgame"
    assert p.player.mode == "color" and p.player.color is not None
    assert p.sim is not None and p.sim.camera == "fixed"
    assert p.hotkeys.toggle == "F8" and p.hotkeys.estop == "F12"
    assert p.takeover.auto_resume_after_s is None
    e = load_profile(ROOT / "profiles/example.yaml")
    assert e.capture.backend == "dxcam" and e.input.keys["up"] == "W"
    assert e.sim is None


def test_defaults_minimal():
    p = profile_from_dict({"name": "x"})
    assert p.genre == "P1" and p.player.mode == "fixed_center"
    assert p.sim is not None  # 默认 sim 后端自动带上测试游戏参数


@pytest.mark.parametrize(
    "data, msg",
    [
        ({"name": "x", "playr": {}}, "不认识的字段"),
        ({"name": "x", "player": {"speed": 3}}, "不认识的字段"),
        ({"name": "x", "genre": "P9"}, "只能是"),
        ({"name": "x", "player": {"speed_px_s": "fast"}}, "数字"),
        ({"name": "x", "player": {"mode": "color"}}, "player.color"),
        ({"name": "x", "ignore_regions": [[0, 0, 10]]}, "ignore_regions"),
        ({"name": "x", "input": {"keys": {"up": "W"}}}, "缺少方向"),
        ({"name": "x", "player": {"mode": "color", "color": {"lower": [1, 2], "upper": [1, 2, 3]}}}, "3 个数"),
        ({"genre": "P1"}, "name"),
    ],
)
def test_bad_profiles(data, msg):
    with pytest.raises(ProfileError, match=msg):
        profile_from_dict(data)


def test_missing_file(tmp_path):
    with pytest.raises(ProfileError, match="找不到"):
        load_profile(tmp_path / "nope.yaml")
