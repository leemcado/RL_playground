"""다이아 봇 — 항상 그리디 (100%). 설정·구성에서 이름 'diamond' 으로 쓴다"""

from __future__ import annotations

from tiered import TieredBot


class DiamondBot(TieredBot):
    name = "diamond"
    color = (110, 230, 255)
    greedy_frac = 1.0
