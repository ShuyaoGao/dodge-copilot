"""命令行入口：python -m dodge_copilot <命令>"""
from __future__ import annotations

import argparse
import sys

from .config import ProfileError, load_profile


def _seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in text:
        return [int(x) for x in text.split(",")]
    return list(range(int(text)))


def apply_camera(profile, camera: str) -> None:
    """测试游戏切换镜头方式时，配置里跟着要改的几项。"""
    profile.sim.camera = camera
    if camera == "follow":
        profile.player.mode = "fixed_center"  # 玩家永远在画面中心
        profile.hazards.motion_compensation = True  # 要估计画面整体平移
    else:
        profile.hazards.motion_compensation = False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dodge_copilot", description="躲避副驾驶：单机肉鸽走位助手")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sim", help="测试模式：打开测试游戏窗口，自己玩，按 F8 让助手接管")
    p.add_argument("--profile", default="profiles/testgame.yaml")
    p.add_argument("--policy", default="assist", help="助手用哪个策略：assist（躲闪）/ random / idle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=None, help="跑这么多帧就退出（自动化测试用）")
    p.add_argument("--headless", action="store_true", help="不开真窗口（自动化测试用）")
    p.add_argument("--debug", action="store_true", help="一开始就显示调试画面（也可以按 F2 切换）")
    p.add_argument("--record", action="store_true", help="录制接管期间的画面和决策到 recordings/")
    p.add_argument("--camera", choices=["fixed", "follow"], default=None, help="测试游戏的镜头方式")

    b = sub.add_parser("bench", help="存活时间基准：同一组种子下各策略存活多久")
    b.add_argument("--profile", default="profiles/testgame.yaml")
    b.add_argument("--policies", default="idle,random")
    b.add_argument("--seeds", default="20", help="20 表示 0-19；也可以写 3-7 或 1,5,9")
    b.add_argument("--max-time", type=float, default=60.0)
    b.add_argument("--camera", choices=["fixed", "follow"], default=None, help="测试游戏的镜头方式")
    b.add_argument("--jobs", type=int, default=1, help="同时跑几局（多核机器上加快）")
    b.add_argument("--latency", type=int, default=None, help="覆盖配置里的 sim.latency_frames")

    r = sub.add_parser("run", help="真实模式：在真游戏上运行（M2 实现）")
    r.add_argument("--profile", required=True)

    args = ap.parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ProfileError as e:
        print(f"配置文件有问题：{e}", file=sys.stderr)
        return 2

    if args.cmd == "sim":
        from .app import run_sim

        if args.debug:
            profile.debug.show_window = True
        if args.camera:
            apply_camera(profile, args.camera)
        run_sim(profile, args.policy, args.seed, args.frames, args.headless,
                record=True if args.record else None)
    elif args.cmd == "bench":
        from .bench import format_table, run_bench

        seeds = _seeds(args.seeds)
        policies = [x.strip() for x in args.policies.split(",") if x.strip()]
        if args.camera:
            apply_camera(profile, args.camera)
        if args.latency is not None:
            profile.sim.latency_frames = args.latency
        res = run_bench(profile, policies, seeds, args.max_time, jobs=args.jobs)
        print()
        print(format_table(res, args.max_time))
    elif args.cmd == "run":
        print("真实模式（Windows 截屏 + 模拟输入）在 M2 阶段实现，现在请先用 sim 命令体验测试模式。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
