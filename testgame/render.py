"""把测试游戏画成一张 BGR 图（numpy 数组），模拟"屏幕画面"。"""
from __future__ import annotations

import cv2
import numpy as np

# 颜色（BGR）
BG = (24, 18, 20)
GRID = (48, 40, 42)
DOT = (70, 60, 62)
PLAYER = (90, 230, 90)
PLAYER_EDGE = (30, 120, 30)
BULLET_COLORS = {
    "aimed": (60, 60, 255),  # 红
    "fan": (30, 160, 255),  # 橙
    "homing": (255, 80, 255),  # 品红
    "chaser": (200, 70, 150),  # 紫色敌人
}
LASER_WARN = (70, 70, 150)
LASER_CORE = (255, 255, 230)
LASER_GLOW = (255, 200, 60)
LIGHTNING_WARN = (60, 120, 130)
LIGHTNING_CORE = (200, 255, 255)
LIGHTNING_GLOW = (40, 230, 255)

TILE = 256
HUD_RECT = (0, 0, 240, 48)  # x, y, w, h：示例配置里把这块设为忽略区

_bg_cache: dict[tuple[int, int, int], np.ndarray] = {}


def _background(w: int, h: int, seed: int) -> np.ndarray:
    """比画面大一个瓦片的背景图，镜头移动时从里面切一块。"""
    key = (w, h, seed)
    if key not in _bg_cache:
        rng = np.random.default_rng(10_000 + seed)
        tile = np.full((TILE, TILE, 3), BG, dtype=np.uint8)
        for i in range(0, TILE, 64):
            tile[i, :] = GRID
            tile[:, i] = GRID
        for _ in range(40):
            x, y = rng.integers(0, TILE, size=2)
            cv2.circle(tile, (int(x), int(y)), 1, DOT, -1)
        reps_y = (h + TILE) // TILE + 1
        reps_x = (w + TILE) // TILE + 1
        _bg_cache[key] = np.tile(tile, (reps_y, reps_x, 1))
    return _bg_cache[key]


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def render(game) -> np.ndarray:
    cfg = game.config
    w, h = cfg.width, cfg.height
    off = game.camera_offset
    big = _background(w, h, cfg.seed)
    ox = int(np.floor(off[0])) % TILE
    oy = int(np.floor(off[1])) % TILE
    img = big[oy : oy + h, ox : ox + w].copy()

    # 激光、闪电（先画，子弹压在上面）
    for beam in game.beams:
        pts = (beam.points - off).round().astype(np.int32).reshape(-1, 1, 2)
        if beam.kind == "laser":
            if beam.active:
                cv2.polylines(img, [pts], False, LASER_GLOW, int(beam.width), cv2.LINE_AA)
                cv2.polylines(img, [pts], False, LASER_CORE, max(2, int(beam.width) // 3), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], False, LASER_WARN, 2, cv2.LINE_AA)
        else:
            if beam.active:
                cv2.polylines(img, [pts], False, LIGHTNING_GLOW, int(beam.width), cv2.LINE_AA)
                cv2.polylines(img, [pts], False, LIGHTNING_CORE, max(2, int(beam.width) // 3), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], False, LIGHTNING_WARN, 1, cv2.LINE_AA)

    # 玩家
    ps = game.player - off
    half = cfg.player_size // 2
    x, y = _pt(ps)
    cv2.rectangle(img, (x - half, y - half), (x + half - 1, y + half - 1), PLAYER, -1)
    cv2.rectangle(img, (x - half, y - half), (x + half - 1, y + half - 1), PLAYER_EDGE, 1)

    # 子弹和敌人
    for b in game.bullets:
        c = _pt(b.pos - off)
        r = int(round(b.radius))
        color = BULLET_COLORS[b.kind]
        if b.kind == "chaser":
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r // 2, (90, 30, 70), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    if cfg.hud:
        hx, hy, hw, hh = HUD_RECT
        cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (10, 10, 10), -1)
        cv2.putText(img, f"TIME {game.time:6.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        # 一个假的"血条"，颜色故意和子弹很像，用来检验"忽略区"
        cv2.rectangle(img, (10, 30), (10 + 200, 40), (40, 40, 220), -1)
    return img
