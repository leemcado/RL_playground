"""브론즈 봇 — 항상 랜덤워크 (그리디 0%). 설정·구성에서 이름 'bronze' 으로 쓴다"""

from __future__ import annotations

from tiered import TieredBot


class BronzeBot(TieredBot):
    name = "bronze"
    color = (205, 127, 50)
    greedy_frac = 0.0
