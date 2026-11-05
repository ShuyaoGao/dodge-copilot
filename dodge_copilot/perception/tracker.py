"""跨帧追踪点状危险物：最近邻匹配 + 匀速卡尔曼滤波，估计每个危险物的速度。

全部在世界坐标里做（镜头固定时世界坐标就是画面坐标）。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .hazards import Detection


class Track:
    MEAS_STD = 1.5  # 像素
    ACCEL_STD = 400.0  # 像素/秒²，追踪弹会拐弯，给大一点

    def __init__(self, tid: int, pos: np.ndarray, radius: float) -> None:
        self.id = tid
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 500.0**2, 500.0**2])
        self.radius = radius
        self.hits = 1
        self.misses = 0
        self.age = 0.0
        # 是不是"会追人的"：一直朝着玩家飞，而且玩家换了方位它也跟着转
        self.pursuit_cos = 0.0  # 速度方向和"指向玩家的方向"夹角余弦，平滑后的
        self.first_bearing: Optional[float] = None
        self.bearing_change = 0.0  # 玩家相对它的方位变了多少（弧度）

    @property
    def pursuer(self) -> bool:
        return self.hits >= 12 and self.pursuit_cos > 0.985 and self.bearing_change > math.radians(15)

    def observe_player(self, player: np.ndarray) -> None:
        if self.hits < 3:
            return
        to_p = player - self.pos
        dist = float(np.hypot(*to_p))
        speed = float(np.hypot(*self.x[2:]))
        if dist < 1e-6 or speed < 30:
            self.pursuit_cos *= 0.9
            return
        bearing = math.atan2(to_p[1], to_p[0])
        if self.first_bearing is None:
            self.first_bearing = bearing
        diff = abs((bearing - self.first_bearing + math.pi) % (2 * math.pi) - math.pi)
        self.bearing_change = max(self.bearing_change, diff)
        cos = float(np.dot(self.x[2:], to_p)) / (speed * dist)
        self.pursuit_cos = 0.85 * self.pursuit_cos + 0.15 * cos

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:] if self.hits >= 2 else np.zeros(2)

    @property
    def velocity_known(self) -> bool:
        return self.hits >= 2

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.ACCEL_STD**2
        g = np.array([0.5 * dt * dt, dt])
        Q1 = q * np.outer(g, g)
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += dt

    def update(self, z: np.ndarray, radius: float) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * self.MEAS_STD**2
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.radius = 0.7 * self.radius + 0.3 * radius
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, gate_px: float = 30.0, max_misses: int = 3) -> None:
        self.gate_px = gate_px
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self.tracks = []
        self._last_t = None

    def update(self, detections: list[Detection], offset: np.ndarray, t: float,
               player: Optional[np.ndarray] = None) -> list[Track]:
        """player：玩家的世界坐标（用来判断哪些危险物会追人）。"""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        for tr in self.tracks:
            tr.predict(dt)
        dets = [d.pos + offset for d in detections]

        # 贪心最近邻：所有 (轨迹, 检测) 对按距离从近到远配
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(dets):
                dist = float(np.hypot(*(tr.pos - z)))
                gate = self.gate_px + (0 if tr.velocity_known else 10)
                if dist <= gate + detections[j].radius:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            self.tracks[i].update(dets[j], detections[j].radius)
            used_t.add(i)
            used_d.add(j)

        kept = []
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.misses += 1
            if tr.misses <= self.max_misses:
                kept.append(tr)
        for j, z in enumerate(dets):
            if j not in used_d:
                kept.append(Track(self._next_id, z, detections[j].radius))
                self._next_id += 1
        self.tracks = kept
        if player is not None:
            for tr in self.tracks:
                if tr.misses == 0:
                    tr.observe_player(player)
        return self.tracks
