"""测试模式的交互窗口：你自己玩测试游戏，按接管键让助手接手。

操作：方向键 / WASD 移动；F8 接管/交还；F12 急停；R 重开；F3 显示/隐藏耗时。
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .config import Profile
from .control import Mode, TakeoverController
from .io.sim import SimBridge, make_game
from .loop import Copilot
from .perception import Perception
from .policy import build_policy
from .safety import ReleaseGuard

CJK_FONTS = ["microsoftyahei", "simhei", "pingfangsc", "notosanscjksc", "wenquanyizenhei",
             "wenquanyimicrohei", "notosanscjk"]

STATUS_COLORS = {
    Mode.PLAYER: ((40, 140, 60), "玩家控制", "PLAYER"),  # 绿
    Mode.ASSIST: ((220, 90, 20), "助手接管中", "ASSIST"),  # 橙红
}


def _human_keymap(pygame) -> dict[int, str]:
    k = pygame
    return {
        k.K_UP: "up", k.K_w: "up",
        k.K_DOWN: "down", k.K_s: "down",
        k.K_LEFT: "left", k.K_a: "left",
        k.K_RIGHT: "right", k.K_d: "right",
    }


def _key_code(pygame, name: str) -> int:
    code = pygame.key.key_code(name.lower())
    if code == pygame.K_UNKNOWN:
        raise ValueError(f"不认识的热键名: {name}")
    return code


class StatusBadge:
    """屏幕角上的状态提示。"""

    def __init__(self, pygame) -> None:
        self.pg = pygame
        path = pygame.font.match_font(CJK_FONTS)
        self.cjk = path is not None
        self.font = pygame.font.Font(path, 20) if path else pygame.font.Font(None, 26)
        self.small = pygame.font.Font(path, 14) if path else pygame.font.Font(None, 18)

    def draw(self, surf, mode: Mode, extra: Optional[str] = None) -> None:
        color, zh, en = STATUS_COLORS[mode]
        text = self.font.render(zh if self.cjk else en, True, (255, 255, 255))
        pad = 8
        w, h = text.get_width() + 2 * pad, text.get_height() + 2 * pad
        x = surf.get_width() - w - 10
        self.pg.draw.rect(surf, color, (x, 10, w, h), border_radius=6)
        surf.blit(text, (x + pad, 10 + pad))
        if extra:
            t = self.small.render(extra, True, (220, 220, 220))
            surf.blit(t, (surf.get_width() - t.get_width() - 10, 10 + h + 4))


def run_sim(profile: Profile, policy_name: str = "random", seed: int = 0,
            max_frames: Optional[int] = None, headless: bool = False,
            script: Optional[dict[int, list[tuple[str, str]]]] = None) -> TakeoverController:
    """script：自动化测试用，{帧号: [(动作, 参数)]}，动作是 keydown/keyup，参数是热键名或方向。"""
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    assert profile.sim is not None
    pygame.display.init()  # 不用 pygame.init()，免得去初始化声卡
    pygame.font.init()
    screen = pygame.display.set_mode((profile.sim.width, profile.sim.height))
    pygame.display.set_caption("dodge-copilot 测试游戏")
    clock = pygame.time.Clock()
    badge = StatusBadge(pygame)
    human_map = _human_keymap(pygame)
    toggle_code = _key_code(pygame, profile.hotkeys.toggle)
    estop_code = _key_code(pygame, profile.hotkeys.estop)

    def on_change(mode: Mode, reason: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {mode.value}：{reason}", flush=True)

    game = make_game(profile, seed)
    bridge = SimBridge(game, profile)
    controller = TakeoverController(bridge.sink, profile.takeover.auto_resume_after_s,
                                    on_change=on_change)
    policy = build_policy(policy_name, seed)
    copilot = Copilot(bridge.grab, Perception(profile), policy, controller)
    show_timing = True
    dt = 1.0 / 60
    frame_no = 0

    def restart() -> None:
        nonlocal game
        controller.release("重开")
        game = make_game(profile, seed)
        bridge.game = game
        policy.reset()

    def handle_key(code: int, down: bool) -> None:
        nonlocal show_timing
        if down and code == estop_code:
            controller.estop()
        elif down and code == toggle_code:
            controller.toggle()
        elif code in human_map:
            d = human_map[code]
            (bridge.human_keys.add if down else bridge.human_keys.discard)(d)
            controller.user_direction(d, down)
        elif down and code == pygame.K_r:
            restart()
        elif down and code == pygame.K_F3:
            show_timing = not show_timing

    with ReleaseGuard(bridge.sink):
        try:
            while not controller.stopped:
                for action, arg in (script or {}).get(frame_no, []):
                    code = human_map_rev(pygame, arg) if arg in ("up", "down", "left", "right") \
                        else _key_code(pygame, arg)
                    handle_key(code, action == "keydown")
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        controller.estop()
                    elif ev.type in (pygame.KEYDOWN, pygame.KEYUP):
                        handle_key(ev.key, ev.type == pygame.KEYDOWN)
                if controller.stopped:
                    break
                if game.alive:
                    copilot.step(game.time)
                    game.step(bridge.game_move(), dt)
                img = game.render()
                surf = pygame.image.frombuffer(np.ascontiguousarray(img[:, :, ::-1]).tobytes(),
                                               (img.shape[1], img.shape[0]), "RGB")
                screen.blit(surf, (0, 0))
                badge.draw(screen, controller.mode,
                           copilot.timer.summary() if show_timing else None)
                if not game.alive:
                    msg = badge.font.render(
                        f"被 {game.death_cause} 打中，存活 {game.time:.1f} 秒，按 R 重开" if badge.cjk
                        else f"HIT by {game.death_cause}, {game.time:.1f}s - press R",
                        True, (255, 255, 255))
                    screen.blit(msg, ((screen.get_width() - msg.get_width()) // 2,
                                      screen.get_height() // 2))
                pygame.display.flip()
                frame_no += 1
                if max_frames is not None and frame_no >= max_frames:
                    break
                if not headless:
                    clock.tick(60)
        finally:
            bridge.sink.release_all()
            pygame.quit()
    return controller


def human_map_rev(pygame, direction: str) -> int:
    return {"up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT}[direction]
