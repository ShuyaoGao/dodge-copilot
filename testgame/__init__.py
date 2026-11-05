"""测试用弹幕小游戏。助手在这里做闭环测试，不需要真屏幕和真键盘。"""
from .game import PATTERNS, Beam, Bullet, Game, GameConfig, polyline_distance

__all__ = ["PATTERNS", "Beam", "Bullet", "Game", "GameConfig", "polyline_distance"]
