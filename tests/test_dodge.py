import numpy as np

from conftest import ROOT
from dodge_copilot.config import load_profile
from dodge_copilot.dodge import DodgePolicy, danger, direction_set
from dodge_copilot.perception import Observation, Track


def profile():
    return load_profile(ROOT / "profiles/testgame.yaml")


def obs_with(player, tracks=(), region=None, size=(960, 720), warn=None):
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    o = Observation(t=0.0, frame=frame, player=np.array(player, float), hazards=list(tracks))
    if region is not None:
        import cv2

        o.region_mask = region
        k = o.region_dist_scale
        small = cv2.resize(region, (size[0] // k, size[1] // k), interpolation=cv2.INTER_NEAREST)
        o.region_dist = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3) * k
    if warn is not None:
        from dodge_copilot.perception import _dist_map

        o.warn_mask = warn
        o.warn_dist = _dist_map(warn, o.region_dist_scale)
    return o


def track(pos, vel, r=6.0):
    t = Track(1, np.array(pos, float), r)
    t.x[2:] = vel
    t.hits = 5
    return t


def test_danger_shape():
    c = np.array([-5.0, 0.0, 10.0, 100.0])
    d = danger(c)
    assert d[0] > d[1] > d[2] > d[3] >= 0
    assert d[0] >= 100


def test_direction_set():
    d = direction_set(8)
    assert d.shape == (9, 2) and np.allclose(d[0], 0)
    assert np.allclose(np.linalg.norm(d[1:], axis=1), 1)


def test_stays_put_when_safe_in_center():
    pol = DodgePolicy(profile())
    a = pol.decide(obs_with([480, 360]))
    assert np.hypot(*a) == 0


def test_sidesteps_head_on_bullet():
    pol = DodgePolicy(profile())
    # 子弹从右边正对着飞来
    a = pol.decide(obs_with([480, 360], [track([600, 360], [-300, 0])]))
    assert abs(a[1]) > 0.5  # 往上或往下闪，而不是往左跑或不动


def test_leaves_laser_line():
    region = np.zeros((720, 960), np.uint8)
    region[355:366, :] = 255  # 一条横穿玩家的激光预警线
    pol = DodgePolicy(profile())
    a = pol.decide(obs_with([480, 360], region=region))
    assert abs(a[1]) > 0.9 and abs(a[0]) < 0.1  # 垂直离开最快


def test_avoids_corner():
    pol = DodgePolicy(profile())
    a = pol.decide(obs_with([20, 20]))
    assert a[0] > 0 and a[1] > 0  # 往场地里面走


def test_escapes_pincer():
    pol = DodgePolicy(profile())
    tracks = [track([380, 360], [250, 0]), track([580, 360], [-250, 0])]
    a = pol.decide(obs_with([480, 360], tracks))
    assert abs(a[1]) > 0.5
    # 选的路线上不会撞
    plan = pol.last_plan
    for t, p in zip(pol.ts, plan.path):
        for tr in tracks:
            assert np.hypot(*(p - (tr.pos + tr.vel * t))) > tr.radius + pol.r_player


def test_inertia_reduces_flip_flop():
    p = profile()
    pol = DodgePolicy(p)
    pol.decide(obs_with([480, 360], [track([600, 360], [-300, 0])]))
    first = pol.prev_first
    # 情况几乎不变，不该换方向
    a = pol.decide(obs_with([480, 362], [track([595, 360], [-300, 0])]))
    assert pol.prev_first == first


def test_follow_mode_has_no_edge_cost():
    p = profile()
    p.player.mode = "fixed_center"
    p.hazards.motion_compensation = True
    pol = DodgePolicy(p)
    o = obs_with([480, 360])
    o.cam_offset = np.array([5000.0, -3000.0])
    assert np.hypot(*pol.decide(o)) == 0
    a = pol.decide(obs_with([480, 360], [track([5600, -2640], [-300, 0])]))
    # 镜头偏移是 0 时，那颗子弹在很远处，不影响
    assert np.hypot(*a) == 0


def test_danger_grid_shape():
    pol = DodgePolicy(profile())
    g = pol.danger_grid(obs_with([480, 360], [track([100, 100], [0, 0])]), stride=12)
    assert g.shape == (60, 80)
    assert g[100 // 12, 100 // 12] > g[50, 70]


def test_crosses_warning_line_to_escape_but_not_active_laser():
    """上面有子弹压过来，下面是预警线：可以趁预警时穿过去；换成已生效的激光就不能穿。"""
    line = np.zeros((720, 960), np.uint8)
    line[385:395, :] = 255  # 玩家下方 30 像素
    bullets = [track([x, 250], [0, 150], r=8) for x in range(0, 961, 18)]  # 一整排压下来，横着躲不开
    pol = DodgePolicy(profile())
    pol.decide(obs_with([480, 360], bullets, warn=line))
    assert pol.last_plan.path[-1][1] > 410  # 穿到线下面去了
    pol = DodgePolicy(profile())
    pol.decide(obs_with([480, 360], bullets, region=line))
    assert (pol.last_plan.path[:, 1] < 385).all()  # 不会穿过已生效的激光
