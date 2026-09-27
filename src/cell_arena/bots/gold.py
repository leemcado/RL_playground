"""골드 봇 — 시간의 75% 를 그리디, 25% 를 랜덤워크로 (모드는 가끔 확률적으로 전환). 설정·구성에서 이름 'gold' 으로 쓴다"""

from __future__ import annotations

from tiered import TieredBot


class GoldBot(TieredBot):
    name = "gold"
    color = (240, 190, 0)
    greedy_frac = 0.75
