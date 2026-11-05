import threading

from dodge_copilot.control import Mode, TakeoverController
from dodge_copilot.io.sim import SimKeyboardSink

KEYMAP = {"up": "W", "down": "S", "left": "A", "right": "D"}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(**kw):
    sink = SimKeyboardSink(KEYMAP)
    return sink, TakeoverController(sink, **kw)


def test_starts_with_player_and_ignores_actions():
    sink, c = make()
    assert c.mode is Mode.PLAYER
    assert c.apply(1, 0) is False
    assert sink.down == set()


def test_toggle_engage_and_release_frees_keys():
    sink, c = make()
    assert c.toggle() is Mode.ASSIST
    assert c.apply(1, 0) is True
    assert sink.down == {"D"}
    assert c.toggle() is Mode.PLAYER
    assert sink.down == set()
    assert c.apply(0, 1) is False
    assert sink.down == set()


def test_user_direction_overrides_immediately():
    sink, c = make()
    c.toggle()
    c.apply(-1, -1)
    assert sink.down == {"A", "W"}
    c.user_direction("right", True)
    assert c.mode is Mode.PLAYER
    assert sink.down == set()
    # 之后主循环再想按键也没用
    assert c.apply(1, 0) is False
    assert sink.down == set()


def test_no_auto_resume_by_default():
    clock = FakeClock()
    sink, c = make(clock=clock)
    c.toggle()
    c.user_direction("up", True)
    c.user_direction("up", False)
    clock.t = 100
    c.tick()
    assert c.mode is Mode.PLAYER


def test_optional_auto_resume():
    clock = FakeClock()
    sink, c = make(clock=clock, auto_resume_after_s=1.0)
    c.toggle()
    c.user_direction("up", True)
    clock.t = 5
    c.tick()
    assert c.mode is Mode.PLAYER  # 还按着
    c.user_direction("up", False)
    clock.t = 5.5
    c.tick()
    assert c.mode is Mode.PLAYER  # 松手不到 1 秒
    clock.t = 6.01
    c.tick()
    assert c.mode is Mode.ASSIST


def test_auto_resume_only_after_override_not_after_manual_release():
    clock = FakeClock()
    sink, c = make(clock=clock, auto_resume_after_s=1.0)
    c.toggle()
    c.toggle()  # 玩家主动按 F8 交还
    c.user_direction("up", True)
    c.user_direction("up", False)
    clock.t = 10
    c.tick()
    assert c.mode is Mode.PLAYER


def test_estop_releases_and_stops():
    sink, c = make()
    c.toggle()
    c.apply(1, 1)
    c.estop()
    assert sink.down == set()
    assert c.stopped and c.mode is Mode.PLAYER
    assert c.toggle() is Mode.PLAYER  # 急停后不能再接管
    assert c.apply(1, 0) is False


def test_on_change_errors_do_not_leave_keys_down():
    sink = SimKeyboardSink(KEYMAP)

    def boom(mode, reason):
        raise RuntimeError("回调出错")

    c = TakeoverController(sink, on_change=boom)
    c.toggle()
    c.apply(1, 0)
    c.toggle()
    assert sink.down == set()
    assert c.switch_log[-1][1] is Mode.PLAYER


def test_concurrent_override_never_leaves_key_pressed():
    """热键线程交还的同时主循环在按键：交还之后不能有键残留。"""
    for _ in range(200):
        sink, c = make()
        c.toggle()
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                c.apply(1, 0)

        th = threading.Thread(target=loop)
        th.start()
        c.user_direction("left", True)
        stop.set()
        th.join()
        assert sink.down == set()


def test_switch_latency_under_50ms():
    import time

    sink, c = make()
    for _ in range(100):
        t0 = time.perf_counter()
        c.toggle()
        c.apply(1, 0)
        c.user_direction("up", True)
        c.user_direction("up", False)
        assert (time.perf_counter() - t0) * 1000 < 50
        assert sink.down == set()
