"""실버 봇 — 시간의 20% 를 그리디, 80% 를 랜덤워크로 (모드는 가끔 확률적으로 전환). 설정·구성에서 이름 'silver' 으로 쓴다"""

from __future__ import annotations

from tiered import TieredBot


class SilverBot(TieredBot):
    name = "silver"
    color = (200, 212, 230)
    greedy_frac = 0.2
