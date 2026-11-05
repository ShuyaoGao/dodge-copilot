"""调试画面（需求 3.5）：在截图上画出识别结果和助手的决策。

- 黄绿色圆圈：识别到的玩家和它的"被打中"半径
- 橙圈：判断为"会追人的"危险物
- 白圈 + 线：追踪到的危险物，线指向它 horizon 秒后预计到的位置（灰圈表示速度还不知道）
- 半透明红色：已生效的危险区域（激光、闪电）；半透明黄色：预警中的区域
- 热力图：站着不动时各处的危险程度（蓝 -> 红，越红越危险）
文字只用英文，OpenCV 在画面上写不了中文。
- 青色箭头和点线：助手这一帧选的方向和预计路径
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .perception import Observation

PLAYER_C = (0, 255, 200)
TRACK_C = (255, 255, 255)
TRACK_NEW_C = (150, 150, 150)
PLAN_C = (255, 255, 0)
PURSUER_C = (0, 140, 255)


def draw_debug(frame: np.ndarray, obs: Observation, policy=None, heatmap: bool = True,
               text: Optional[str] = None) -> np.ndarray:
    img = frame.copy()
    off = obs.cam_offset
    horizon = policy.profile.predict.horizon_s if hasattr(policy, "profile") else 0.5

    if heatmap and policy is not None and hasattr(policy, "danger_grid") and obs.player is not None:
        grid = policy.danger_grid(obs)
        if grid is not None:
            g = np.clip(np.log1p(grid) / np.log1p(300.0), 0, 1)
            g = cv2.resize((g * 255).astype(np.uint8), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            color = cv2.applyColorMap(g, cv2.COLORMAP_JET)
            alpha = (g.astype(np.float32) / 255 * 0.45)[..., None]
            img = (img * (1 - alpha) + color * alpha).astype(np.uint8)

    for mask, color in ((obs.warn_mask, (0, 200, 255)), (obs.region_mask, (0, 0, 255))):
        if mask is not None:
            m = mask > 0
            img[m] = (img[m] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

    for tr in obs.hazards:
        p = tr.pos - off
        c = (int(p[0]), int(p[1]))
        col = PURSUER_C if tr.pursuer else (TRACK_C if tr.velocity_known else TRACK_NEW_C)
        cv2.circle(img, c, int(tr.radius) + 3, col, 1, cv2.LINE_AA)
        if tr.velocity_known:
            q = p + tr.vel * horizon
            cv2.line(img, c, (int(q[0]), int(q[1])), col, 1, cv2.LINE_AA)

    if obs.player is not None:
        pc = (int(obs.player[0]), int(obs.player[1]))
        r = int(getattr(policy, "r_player", 6)) if policy is not None else 6
        cv2.circle(img, pc, r, PLAYER_C, 1, cv2.LINE_AA)
        plan = getattr(policy, "last_plan", None)
        if plan is not None:
            pts = (plan.path - off).round().astype(np.int32)
            for q in pts[::2]:
                cv2.circle(img, (int(q[0]), int(q[1])), 2, PLAN_C, -1)
            d = plan.first
            if np.hypot(*d) > 0:
                tip = (int(pc[0] + d[0] * 40), int(pc[1] + d[1] * 40))
                cv2.arrowedLine(img, pc, tip, PLAN_C, 2, cv2.LINE_AA, tipLength=0.3)
            else:
                cv2.circle(img, pc, 14, PLAN_C, 2, cv2.LINE_AA)

    if text:
        y = img.shape[0] - 10
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img
