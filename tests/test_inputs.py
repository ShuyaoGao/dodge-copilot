import math

import pytest

from dodge_copilot.inputs import keys_to_vector, quantize_8
from dodge_copilot.io.sim import SimKeyboardSink, SimStickSink

KEYMAP = {"up": "W", "down": "S", "left": "A", "right": "D"}


@pytest.mark.parametrize(
    "vec, keys",
    [
        ((1, 0), {"right"}),
        ((0, 1), {"down"}),
        ((-1, 0), {"left"}),
        ((0, -1), {"up"}),
        ((1, 1), {"right", "down"}),
        ((-1, -1), {"left", "up"}),
        ((0.95, 0.3), {"right"}),
        ((0.1, 0.1), set()),
        ((0, 0), set()),
    ],
)
def test_quantize(vec, keys):
    assert quantize_8(*vec) == keys


def test_keys_to_vector():
    assert keys_to_vector(set()) == (0.0, 0.0)
    x, y = keys_to_vector({"up", "right"})
    assert math.isclose(math.hypot(x, y), 1.0) and x > 0 and y < 0
    assert keys_to_vector({"left", "right"}) == (0.0, 0.0)


def test_keyboard_sink_diffs_and_releases():
    s = SimKeyboardSink(KEYMAP)
    s.set_move(1, 0)
    assert s.down == {"D"}
    s.set_move(1, 1)
    assert s.down == {"D", "S"}
    s.events.clear()
    s.set_move(1, 1)  # 不变就不重复按
    assert s.events == []
    s.set_move(-1, 0)
    assert s.down == {"A"}
    s.release_all()
    assert s.down == set() and s.pressed == set()
    s.release_all()  # 可重复调用


def test_stick_sink_clamps_and_centers():
    s = SimStickSink()
    s.set_move(3, 4)
    assert math.isclose(math.hypot(*s.stick), 1.0)
    s.release_all()
    assert s.stick == (0.0, 0.0)
