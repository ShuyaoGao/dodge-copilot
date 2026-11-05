"""预测和躲闪（需求 3.3）。

做法：
1. 候选走法 = 先往方向 A 走一小段（first_segment_s），再往方向 B 走完（horizon_s）。
   方向有"不动 + 8 个方向"（手柄时 16 个方向）。两段式让助手能想到"先往左闪，再停下"。
2. 每个危险物按当前速度往前推算；"会追人的"（一直转向玩家的敌人）假设它直奔玩家；
   激光、闪电等区域类危险物当成原地不动。
3. 对每个走法，沿途每隔 step_s 算一次"离危险物还有多远"，越近扣分越多，撞上扣大分。
   越往后越不确定，安全距离越大、权重越小。
4. 再加上：离画面边缘和角落远一点、往中间靠一点、少换方向（防抖）。
5. 选总分最低的走法，只执行它的第一段；下一帧重新算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Profile
from .perception import Observation
from .policy import Policy

COLLISION_COST = 100.0
SOFTNESS_PX = 10.0


def danger(clearance: np.ndarray) -> np.ndarray:
    """clearance：离"刚好擦到"还有几像素。<0 就是撞上了。"""
    return np.where(clearance < 0, COLLISION_COST - clearance, np.exp(-np.maximum(clearance, 0) / SOFTNESS_PX))


def direction_set(n: int) -> np.ndarray:
    a = np.arange(n) * 2 * math.pi / n
    return np.vstack([[0.0, 0.0], np.stack([np.cos(a), np.sin(a)], axis=1)])


@dataclass
class Plan:
    first: np.ndarray  # 执行的方向
    path: np.ndarray  # (T, 2) 预计的世界坐标路径
    cost: float
    costs: np.ndarray  # 所有候选的分数


class DodgePolicy(Policy):
    name = "assist"
    needs_frame = True

    def __init__(self, profile: Profile, n_dirs: Optional[int] = None) -> None:
        self.profile = profile
        pc = profile.predict
        self.speed = profile.player.speed_px_s
        self.r_player = profile.player.hit_radius
        self.half = np.array(profile.player.size, float) / 2
        self.follow = profile.hazards.motion_compensation or profile.player.mode == "fixed_center"
        if n_dirs is None:
            stick = profile.input.backend == "gamepad" or (profile.sim is not None and profile.sim.stick)
            n_dirs = 16 if stick else 8
        self.dirs = direction_set(n_dirs)
        self.ts = np.arange(1, int(round(pc.horizon_s / pc.step_s)) + 1) * pc.step_s
        t1 = min(pc.first_segment_s, pc.horizon_s)
        k = len(self.dirs)
        a1 = np.repeat(self.dirs, k, axis=0)  # (K*K, 2)
        a2 = np.tile(self.dirs, (k, 1))
        self.first_idx = np.repeat(np.arange(k), k)
        tt = self.ts[None, :, None]
        seg1 = np.minimum(tt, t1)
        seg2 = np.maximum(tt - t1, 0)
        self.disp = self.speed * (a1[:, None, :] * seg1 + a2[:, None, :] * seg2)  # (P, T, 2)
        self.latency = pc.latency_s
        # 预警区域：telegraph_delay_s 的一半之前几乎不算危险，到 telegraph_delay_s 时完全危险
        delay = profile.hazards.telegraph_delay_s
        self.warn_ramp = np.clip((self.ts + self.latency - 0.5 * delay) / (0.5 * delay + 1e-6), 0.05, 1.0)
        self.w_t = np.exp(-self.ts / max(pc.horizon_s, 1e-6))
        self.margin_t = pc.safety_margin_px + pc.uncertainty_px_per_s * self.ts
        self.reset()

    def reset(self) -> None:
        self.prev_first: Optional[int] = None
        self.last_plan: Optional[Plan] = None

    # ------------------------------------------------------------------ 打分
    def _point_cost(self, P: np.ndarray, obs: Observation) -> np.ndarray:
        """P: (N, T, 2) 世界坐标。返回 (N,)。"""
        if not obs.hazards:
            return np.zeros(P.shape[0])
        pos = np.array([h.pos for h in obs.hazards])
        vel = np.array([h.vel for h in obs.hazards])
        rad = np.array([h.radius for h in obs.hazards])
        # 刚出现、速度还不知道的：当成可能往任何方向飞，安全距离加大
        unknown = np.array([0.0 if h.velocity_known else 250.0 for h in obs.hazards])
        tt = self.ts + self.latency
        Hp = pos[None] + vel[None] * tt[:, None, None]  # (T, H, 2) 匀速直线
        d = np.linalg.norm(P[:, :, None, :] - Hp[None], axis=-1)  # (N, T, H)
        chase = np.array([h.pursuer for h in obs.hazards])
        if chase.any():
            # 会追人的：假设它每时每刻都直奔玩家（按这条走法）那一刻的位置
            cp, cs = pos[chase], np.linalg.norm(vel[chase], axis=1)
            to_p = P[:, :, None, :] - cp[None, None]  # (N, T, C, 2)
            dist0 = np.linalg.norm(to_p, axis=-1)
            d[:, :, chase] = np.maximum(dist0 - cs[None, None] * tt[None, :, None], 0)
        need = rad[None, None] + self.r_player + self.margin_t[None, :, None] \
            + unknown[None, None] * self.ts[None, :, None]
        return (danger(d - need).sum(axis=2) * self.w_t).sum(axis=1)

    def _region_cost(self, S: np.ndarray, obs: Observation) -> np.ndarray:
        """S: (N, T, 2) 画面坐标。已生效的区域一直危险；预警区域过一会儿才危险，在那之前可以穿过。"""
        total = np.zeros(S.shape[0])
        for dmap, w_t in ((obs.region_dist, self.w_t), (obs.warn_dist, self.w_t * self.warn_ramp)):
            if dmap is None:
                continue
            h, w = dmap.shape
            k = obs.region_dist_scale
            ix = np.clip((S[..., 0] / k).astype(int), 0, w - 1)
            iy = np.clip((S[..., 1] / k).astype(int), 0, h - 1)
            c = dmap[iy, ix] - self.r_player - self.profile.hazards.region_pad - self.margin_t[None, :]
            total += (danger(c) * w_t).sum(axis=1)
        return total

    def _screen_cost(self, S: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """镜头固定时：离边缘、角落远一点，往中间靠一点。"""
        if self.follow:
            return np.zeros(S.shape[0])
        pc = self.profile.predict
        w, h = size
        m = pc.edge_margin_px
        dx = np.minimum(S[..., 0] - self.half[0], w - self.half[0] - S[..., 0])
        dy = np.minimum(S[..., 1] - self.half[1], h - self.half[1] - S[..., 1])
        pen = np.clip((m - dx) / m, 0, None) ** 2 + np.clip((m - dy) / m, 0, None) ** 2
        edge = (pen * self.w_t).sum(axis=1) * 2.0
        center = np.array([w / 2, h / 2])
        far = np.linalg.norm(S[:, -1, :] - center, axis=1) / (0.5 * math.hypot(w, h))
        return edge + pc.center_weight * far

    def _clamp(self, S: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if self.follow:
            return S
        w, h = size
        S = S.copy()
        S[..., 0] = np.clip(S[..., 0], self.half[0], w - self.half[0])
        S[..., 1] = np.clip(S[..., 1], self.half[1], h - self.half[1])
        return S

    def _start(self, obs: Observation) -> np.ndarray:
        """画面是 latency 秒之前的；这段时间里玩家还在按上一个方向走。"""
        assert obs.player is not None
        if self.latency <= 0 or self.prev_first is None:
            return obs.player
        return obs.player + self.dirs[self.prev_first] * self.speed * self.latency

    def path_costs(self, obs: Observation) -> np.ndarray:
        assert obs.player is not None and obs.frame_size is not None
        start = self._start(obs)
        S = self._clamp(start[None, None, :] + self.disp, obs.frame_size)  # 画面坐标
        # 镜头跟随时，玩家在世界里移动，危险区域（画面坐标）相对玩家的位置也跟着变
        P = S + obs.cam_offset  # 世界坐标
        return self._point_cost(P, obs) + self._region_cost(S, obs) + self._screen_cost(S, obs.frame_size)

    # ------------------------------------------------------------------ 决策
    def decide(self, obs: Observation) -> tuple[float, float]:
        if obs.player is None or obs.frame_size is None:
            return (0.0, 0.0)
        start = self._start(obs)
        cost = self.path_costs(obs)
        if self.prev_first is not None:
            changed = self.first_idx != self.prev_first
            cost = cost + self.profile.predict.inertia * changed
        best = int(np.argmin(cost))
        fi = int(self.first_idx[best])
        self.prev_first = fi
        S = self._clamp(start[None, :] + self.disp[best], obs.frame_size)
        self.last_plan = Plan(self.dirs[fi].copy(), S + obs.cam_offset, float(cost[best]), cost)
        return float(self.dirs[fi][0]), float(self.dirs[fi][1])

    # ------------------------------------------------------------------ 调试用
    def danger_grid(self, obs: Observation, stride: int = 12) -> Optional[np.ndarray]:
        """站着不动时每个位置的危险程度（给调试窗口画热力图）。"""
        if obs.frame_size is None:
            return None
        w, h = obs.frame_size
        xs = np.arange(stride // 2, w, stride)
        ys = np.arange(stride // 2, h, stride)
        gx, gy = np.meshgrid(xs, ys)
        S = np.stack([gx.ravel(), gy.ravel()], axis=1)[:, None, :].repeat(len(self.ts), axis=1)
        c = self._point_cost(S + obs.cam_offset, obs) + self._region_cost(S, obs)
        return c.reshape(len(ys), len(xs))
