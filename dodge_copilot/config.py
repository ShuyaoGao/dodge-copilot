"""每个游戏一份配置文件（profiles/<名字>.yaml）。

读取时严格检查：写错字段名、类型不对都会直接报错，并指出是哪一项，
免得玩家调了半天参数其实没生效。
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args, get_origin, get_type_hints

import yaml


class ProfileError(ValueError):
    """配置文件有问题。"""


@dataclass
class ColorRange:
    """HSV 颜色范围，用 OpenCV 的取值：H 0-179，S 0-255，V 0-255。"""

    lower: list[int]
    upper: list[int]

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            v = getattr(self, name)
            if len(v) != 3:
                raise ProfileError(f"颜色范围的 {name} 要写 3 个数 [H, S, V]，收到 {v}")


@dataclass
class CaptureConfig:
    backend: Literal["sim", "dxcam", "mss"] = "sim"
    region: Optional[list[int]] = None  # [x, y, 宽, 高]；不写就是整个主屏
    target_fps: int = 60

    def __post_init__(self) -> None:
        _check_rect(self.region, "capture.region")


@dataclass
class PlayerConfig:
    # fixed_center：镜头跟随，玩家固定在画面中心，不用检测
    # color：按颜色找玩家
    # template：按玩家框出来的截图做模板匹配（M2 的标定工具生成）
    mode: Literal["fixed_center", "color", "template"] = "fixed_center"
    color: Optional[ColorRange] = None
    template_path: Optional[str] = None
    size: list[int] = field(default_factory=lambda: [16, 16])  # 角色在画面上的大小 [宽, 高]
    speed_px_s: float = 240.0  # 角色移动速度，像素/秒
    hit_radius: float = 6.0  # 角色"被打中"的半径，像素

    def __post_init__(self) -> None:
        if self.mode == "color" and self.color is None:
            raise ProfileError("player.mode 是 color 时，要写 player.color")
        if self.mode == "template" and not self.template_path:
            raise ProfileError("player.mode 是 template 时，要写 player.template_path")


@dataclass
class HazardConfig:
    # 通用方法：又亮又饱和的像素当成危险物候选
    min_brightness: int = 170
    min_saturation: int = 90
    # 按游戏调：额外指定危险物颜色（可不写）
    colors: list[ColorRange] = field(default_factory=list)
    min_area: int = 6  # 比这还小的色块当成噪点
    max_area: int = 4000  # 比这还大的色块按"一片区域"处理，不按点处理
    # 激光、闪电的预警线通常暗一些，只要是细长的线就算危险区域
    line_min_brightness: int = 100
    line_min_saturation: int = 70
    line_min_length: int = 60
    # 预警线（较暗的线）多久之后才会变成真的危险：在这之前可以穿过去
    telegraph_delay_s: float = 0.3
    region_pad: float = 8.0  # 区域类危险物（激光、闪电）往外多算几像素，预警线变成真激光会变粗
    # 镜头跟着玩家走时打开：先估计画面整体平移，再追踪危险物
    motion_compensation: bool = False
    track_gate_px: float = 30.0  # 相邻两帧同一个危险物最多挪多远
    track_max_misses: int = 3  # 连续几帧没看到就不再追踪


@dataclass
class InputConfig:
    backend: Literal["sim", "keyboard", "gamepad"] = "sim"
    # 逻辑方向 -> 游戏里的键（真实模式用扫描码名，如 W、A、S、D、Up、Left）
    keys: dict[str, str] = field(
        default_factory=lambda: {"up": "up", "down": "down", "left": "left", "right": "right"}
    )

    def __post_init__(self) -> None:
        missing = {"up", "down", "left", "right"} - set(self.keys)
        if missing:
            raise ProfileError(f"input.keys 缺少方向: {sorted(missing)}")


@dataclass
class HotkeyConfig:
    toggle: str = "F8"  # 接管 / 交还
    estop: str = "F12"  # 急停：松开所有键并退出


@dataclass
class TakeoverConfig:
    # 玩家按方向键抢回控制后，松手多少秒自动恢复接管。
    # 默认 null：不自动恢复，要再按一次接管键。
    auto_resume_after_s: Optional[float] = None


@dataclass
class PredictConfig:
    horizon_s: float = 0.6  # 往前推算多少秒（0.2-1）
    step_s: float = 1 / 30  # 推算的时间步长
    first_segment_s: float = 0.2  # 每个候选走法 = 先往一个方向走这么久，再换一个方向走完
    safety_margin_px: float = 4.0  # 和危险物至少再隔开这么多像素
    uncertainty_px_per_s: float = 40.0  # 越往后越不确定，安全距离每秒多加这么多
    edge_margin_px: float = 80.0  # 离画面边缘小于这个距离就开始扣分（镜头固定时）
    center_weight: float = 0.4  # 往画面中间靠的倾向（镜头固定时）
    inertia: float = 0.6  # 换方向的代价，越大越不抖
    latency_s: float = 0.0  # 从截屏到游戏收到按键大约多久；助手会往前多推算这么久


@dataclass
class DebugConfig:
    show_window: bool = False
    record: bool = False


@dataclass
class SimConfig:
    """只在测试模式下用：测试游戏的参数。"""

    camera: Literal["fixed", "follow"] = "fixed"
    difficulty: float = 1.0
    width: int = 960
    height: int = 720
    patterns: Optional[list[str]] = None  # 不写就是全部弹幕类型
    stick: bool = False  # true：模拟手柄摇杆（连续方向）；false：模拟方向键（8 方向）
    latency_frames: int = 0  # 模拟截屏延迟：助手看到的是几帧之前的画面


@dataclass
class Profile:
    name: str
    genre: Literal["P1", "P2", "P3"] = "P1"
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)
    ignore_regions: list[list[int]] = field(default_factory=list)  # 血条、分数等，[x, y, 宽, 高]
    input: InputConfig = field(default_factory=InputConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    takeover: TakeoverConfig = field(default_factory=TakeoverConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    sim: Optional[SimConfig] = None

    def __post_init__(self) -> None:
        for i, r in enumerate(self.ignore_regions):
            _check_rect(r, f"ignore_regions[{i}]")
        if self.capture.backend == "sim" and self.sim is None:
            self.sim = SimConfig()


def _check_rect(r: Optional[list[int]], where: str) -> None:
    if r is None:
        return
    if len(r) != 4 or r[2] <= 0 or r[3] <= 0:
        raise ProfileError(f"{where} 要写 [x, y, 宽, 高]，宽高大于 0，收到 {r}")


# ---------------------------------------------------------------------- 读取
def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProfileError(f"找不到配置文件: {path}") from None
    except yaml.YAMLError as e:
        raise ProfileError(f"{path} 不是合法的 YAML: {e}") from None
    return profile_from_dict(data, where=path.name)


def profile_from_dict(data: Any, where: str = "profile") -> Profile:
    try:
        return _build(Profile, data, where)
    except ProfileError:
        raise
    except (TypeError, ValueError) as e:
        raise ProfileError(f"{where}: {e}") from None


def _build(cls: type, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ProfileError(f"{where} 应该是一组 键: 值，收到 {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ProfileError(f"{where} 里有不认识的字段 {unknown}，可用的字段: {sorted(names)}")
    kwargs = {k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ProfileError(f"{where}: {e}") from None


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value, where)
    if value is None:
        raise ProfileError(f"{where} 不能为空")
    if is_dataclass(tp):
        return _build(tp, value, where)
    if origin is Literal:
        if value not in get_args(tp):
            raise ProfileError(f"{where} 只能是 {list(get_args(tp))} 之一，收到 {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ProfileError(f"{where} 应该是列表，收到 {value!r}")
        (et,) = get_args(tp)
        return [_convert(et, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ProfileError(f"{where} 应该是一组 键: 值，收到 {value!r}")
        kt, vt = get_args(tp)
        return {_convert(kt, k, where): _convert(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ProfileError(f"{where} 应该是 true 或 false，收到 {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError(f"{where} 应该是整数，收到 {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{where} 应该是数字，收到 {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ProfileError(f"{where} 应该是文字，收到 {value!r}")
        return str(value)
    return value
