"""测试用弹幕小游戏：游戏逻辑。

不依赖窗口，可以无头运行。画面由 render.py 画成 numpy 数组（BGR），
助手只能拿到这张图，拿不到这里的内部状态（内部状态只给测试和统计用）。

坐标：世界坐标，x 向右，y 向下，单位像素。
- camera="fixed"：场地就是画面，四周是墙，玩家不能出界。
- camera="follow"：场地无限大，镜头跟着玩家，玩家永远在画面正中。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PATTERNS = ("aimed", "fan", "homing", "chaser", "laser", "lightning")

# 每种弹幕在难度 1、开局时的平均生成间隔（秒）
BASE_INTERVAL = {
    "aimed": 0.55,
    "fan": 2.6,
    "homing": 3.2,
    "chaser": 4.0,
    "laser": 5.0,
    "lightning": 6.0,
}


@dataclass
class GameConfig:
    width: int = 960
    height: int = 720
    camera: str = "fixed"  # "fixed" 或 "follow"
    seed: int = 0
    difficulty: float = 1.0
    ramp_s: float = 60.0  # 每过这么多秒，生成频率再翻一倍
    grace_s: float = 1.0  # 开局这段时间不生成危险物
    player_speed: float = 240.0
    player_size: int = 16
    player_hit_radius: float = 6.0
    patterns: tuple[str, ...] = PATTERNS
    hud: bool = True

    def __post_init__(self) -> None:
        if self.camera not in ("fixed", "follow"):
            raise ValueError(f"camera 只能是 fixed 或 follow，收到 {self.camera!r}")
        bad = [p for p in self.patterns if p not in PATTERNS]
        if bad:
            raise ValueError(f"不认识的弹幕类型: {bad}，可选: {PATTERNS}")
        self.patterns = tuple(self.patterns)


@dataclass
class Bullet:
    kind: str  # aimed / fan / homing / chaser
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    life: float = 20.0
    turn_rate: float = 0.0  # 追踪弹每秒最多转多少弧度
    speed: float = 0.0  # 追人的敌人用


@dataclass
class Beam:
    """激光（一条直线）或闪电（一条折线）。先预警，再生效。"""

    kind: str  # laser / lightning
    points: np.ndarray  # (N, 2) 折线顶点
    width: float
    warn_s: float
    active_s: float
    t: float = 0.0

    @property
    def active(self) -> bool:
        return self.warn_s <= self.t < self.warn_s + self.active_s

    @property
    def done(self) -> bool:
        return self.t >= self.warn_s + self.active_s


@dataclass
class Game:
    config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        self.rng = np.random.default_rng(cfg.seed)
        self.player = np.array([cfg.width / 2, cfg.height / 2], dtype=float)
        self.bullets: list[Bullet] = []
        self.beams: list[Beam] = []
        self.time = 0.0
        self.frame_index = 0
        self.alive = True
        self.death_cause: str | None = None
        # 每种弹幕下一次生成的时间，错开一点，免得开局一起来
        self._next_spawn = {
            p: cfg.grace_s + self.rng.uniform(0, BASE_INTERVAL[p]) for p in cfg.patterns
        }

    # ------------------------------------------------------------------ 公共接口
    @property
    def camera_offset(self) -> np.ndarray:
        """世界坐标减去它就是屏幕坐标。"""
        cfg = self.config
        if cfg.camera == "fixed":
            return np.zeros(2)
        return self.player - np.array([cfg.width / 2, cfg.height / 2])

    def player_screen_pos(self) -> np.ndarray:
        return self.player - self.camera_offset

    def step(self, move: tuple[float, float], dt: float) -> bool:
        """推进 dt 秒。move 是移动方向，长度会被截到 1 以内。返回是否还活着。"""
        if not self.alive:
            return False
        self._move_player(move, dt)
        self._spawn(dt)
        self._update_hazards(dt)
        self._check_hits()
        self.time += dt
        self.frame_index += 1
        return self.alive

    def render(self) -> np.ndarray:
        from .render import render

        return render(self)

    # ------------------------------------------------------------------ 内部
    def _view_rect(self) -> tuple[float, float, float, float]:
        off = self.camera_offset
        return off[0], off[1], off[0] + self.config.width, off[1] + self.config.height

    def _move_player(self, move: tuple[float, float], dt: float) -> None:
        cfg = self.config
        v = np.asarray(move, dtype=float)
        n = float(np.hypot(*v))
        if n > 1.0:
            v = v / n
        self.player = self.player + v * cfg.player_speed * dt
        if cfg.camera == "fixed":
            half = cfg.player_size / 2
            self.player[0] = min(max(self.player[0], half), cfg.width - half)
            self.player[1] = min(max(self.player[1], half), cfg.height - half)

    def _rate_scale(self) -> float:
        cfg = self.config
        return cfg.difficulty * (1.0 + self.time / cfg.ramp_s)

    def _spawn(self, dt: float) -> None:
        for p in self.config.patterns:
            if self.time >= self._next_spawn[p]:
                getattr(self, f"_spawn_{p}")()
                mean = BASE_INTERVAL[p] / self._rate_scale()
                self._next_spawn[p] = self.time + mean * self.rng.uniform(0.6, 1.4)

    def _edge_point(self, margin: float = 10.0) -> np.ndarray:
        """画面边缘外一点点的随机位置。"""
        x0, y0, x1, y1 = self._view_rect()
        w, h = x1 - x0, y1 - y0
        s = self.rng.uniform(0, 2 * (w + h))
        if s < w:
            return np.array([x0 + s, y0 - margin])
        s -= w
        if s < h:
            return np.array([x1 + margin, y0 + s])
        s -= h
        if s < w:
            return np.array([x1 - s, y1 + margin])
        s -= w
        return np.array([x0 - margin, y1 - s])

    def _aim(self, origin: np.ndarray, spread: float) -> float:
        d = self.player - origin
        return math.atan2(d[1], d[0]) + self.rng.normal(0, spread)

    def _speed_scale(self) -> float:
        return 0.85 + 0.15 * self.config.difficulty

    def _spawn_aimed(self) -> None:
        o = self._edge_point()
        a = self._aim(o, 0.08)
        sp = 260 * self._speed_scale() * self.rng.uniform(0.85, 1.15)
        self.bullets.append(Bullet("aimed", o, sp * np.array([math.cos(a), math.sin(a)]), 6))

    def _spawn_fan(self) -> None:
        o = self._edge_point()
        a0 = self._aim(o, 0.05)
        n = int(self.rng.integers(5, 10))
        spread = math.radians(self.rng.uniform(35, 60))
        sp = 190 * self._speed_scale()
        for i in range(n):
            a = a0 - spread / 2 + spread * i / (n - 1)
            self.bullets.append(Bullet("fan", o.copy(), sp * np.array([math.cos(a), math.sin(a)]), 5))

    def _spawn_homing(self) -> None:
        o = self._edge_point()
        a = self._aim(o, 0.3)
        sp = 165 * self._speed_scale()
        self.bullets.append(
            Bullet("homing", o, sp * np.array([math.cos(a), math.sin(a)]), 7, life=6.0, turn_rate=1.5)
        )

    def _spawn_chaser(self) -> None:
        o = self._edge_point(margin=16)
        sp = 105 * self._speed_scale()
        self.bullets.append(Bullet("chaser", o, np.zeros(2), 12, life=14.0, speed=sp))

    def _spawn_laser(self) -> None:
        cfg = self.config
        o = self._edge_point(margin=0)
        a = self._aim(o, 0.12)
        length = 2.5 * math.hypot(cfg.width, cfg.height)
        end = o + length * np.array([math.cos(a), math.sin(a)])
        self.beams.append(Beam("laser", np.array([o, end]), width=16, warn_s=0.9, active_s=0.5))

    def _spawn_lightning(self) -> None:
        cfg = self.config
        o = self._edge_point(margin=0)
        a = self._aim(o, 0.1)
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        length = 1.3 * math.hypot(cfg.width, cfg.height)
        seg = 55.0
        pts = [o]
        for i in range(1, int(length / seg) + 1):
            pts.append(o + d * seg * i + n * self.rng.uniform(-22, 22))
        self.beams.append(Beam("lightning", np.array(pts), width=10, warn_s=0.7, active_s=0.25))

    def _update_hazards(self, dt: float) -> None:
        for b in self.bullets:
            if b.kind == "homing":
                want = self.player - b.pos
                cur = math.atan2(b.vel[1], b.vel[0])
                diff = (math.atan2(want[1], want[0]) - cur + math.pi) % (2 * math.pi) - math.pi
                cur += max(-b.turn_rate * dt, min(b.turn_rate * dt, diff))
                sp = float(np.hypot(*b.vel))
                b.vel = sp * np.array([math.cos(cur), math.sin(cur)])
            elif b.kind == "chaser":
                want = self.player - b.pos
                n = float(np.hypot(*want))
                b.vel = b.speed * want / n if n > 1e-6 else np.zeros(2)
            b.pos = b.pos + b.vel * dt
            b.life -= dt
        x0, y0, x1, y1 = self._view_rect()
        m = 80.0
        self.bullets = [
            b
            for b in self.bullets
            if b.life > 0 and x0 - m < b.pos[0] < x1 + m and y0 - m < b.pos[1] < y1 + m
        ]
        for beam in self.beams:
            beam.t += dt
        self.beams = [b for b in self.beams if not b.done]

    def _check_hits(self) -> None:
        r = self.config.player_hit_radius
        p = self.player
        for b in self.bullets:
            if math.hypot(*(p - b.pos)) < r + b.radius:
                self.alive = False
                self.death_cause = b.kind
                return
        for beam in self.beams:
            if beam.active and polyline_distance(p, beam.points) < r + beam.width / 2:
                self.alive = False
                self.death_cause = beam.kind
                return


def polyline_distance(p: np.ndarray, pts: np.ndarray) -> float:
    """点到折线的最短距离。"""
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    closest = a + ab * t[:, None]
    return float(np.min(np.hypot(*(closest - p).T)))
