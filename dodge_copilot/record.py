"""录制（需求 3.5）：把接管期间的画面和助手的决策存到本地 recordings/，用来复盘。

每次录制一个文件夹 recordings/<日期-时间>/：
- 000123.jpg：第 123 帧的调试画面（带识别结果和决策）
- decisions.jsonl：每帧一行，时间、玩家位置、选的方向、危险物个数、各环节耗时
recordings/ 不进 git（可能含游戏画面）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Recorder:
    def __init__(self, root: str | Path = "recordings", every: int = 1, jpeg_quality: int = 80) -> None:
        self.root = Path(root)
        self.every = max(1, every)
        self.quality = jpeg_quality
        self.dir: Optional[Path] = None
        self._log = None
        self._n = 0

    @property
    def active(self) -> bool:
        return self._log is not None

    def start(self) -> Path:
        if self.active:
            assert self.dir is not None
            return self.dir
        stamp = time.strftime("%Y%m%d-%H%M%S")
        d = self.root / stamp
        i = 1
        while d.exists():
            d = self.root / f"{stamp}-{i}"
            i += 1
        d.mkdir(parents=True)
        self.dir = d
        self._log = open(d / "decisions.jsonl", "w", encoding="utf-8")
        self._n = 0
        return d

    def write(self, image: Optional[np.ndarray], record: dict) -> None:
        if not self.active:
            return
        assert self.dir is not None and self._log is not None
        idx = self._n
        self._n += 1
        if idx % self.every:
            return
        if image is not None:
            cv2.imwrite(str(self.dir / f"{idx:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self._log.write(json.dumps({"frame": idx, **record}, ensure_ascii=False, default=_jsonable) + "\n")

    def stop(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __del__(self) -> None:
        self.stop()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 2) for x in o.ravel()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
