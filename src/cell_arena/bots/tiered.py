"""티어 봇 공통 메커니즘 — 랜덤워크 모드와 그리디 모드를 확률적으로 오간다.

매 스텝 ε-greedy 로 섞으면 행동이 떨리므로, 모드 자체가 2상태 마르코프 연쇄로 가끔만 바뀐다.
- 그리디 모드 평균 지속: greedy_frac × mode_cycle 스텝
- 랜덤워크 모드 평균 지속: (1 − greedy_frac) × mode_cycle 스텝
- 장기적으로 전체 시간의 greedy_frac 만큼 그리디 (정상 분포)
greedy_frac = 0 이면 항상 랜덤워크(브론즈), 1 이면 항상 그리디(다이아).

부품: random_walk.py (RandomWalkBot), greedy.py (GreedyBot). 둘 다 자기 관측만 쓴다.
"""

from __future__ import annotations

import numpy as np
from greedy import GreedyBot
from random_walk import RandomWalkBot

from cell_arena import ActionSpec, Agent, Observation

# 랜덤워크 이산 이동 1~8 (상·하·좌·우·좌상·우상·좌하·우하) → 극좌표 θ (y-down: 위가 -π/2)
_MOVE_ANGLES = np.array(
    [0.0, -np.pi / 2, np.pi / 2, np.pi, 0.0, -3 * np.pi / 4, -np.pi / 4, 3 * np.pi / 4, np.pi / 4]
)


class TieredBot(Agent):
    """랜덤워크 ↔ 그리디 모드 전환 봇. 하위 클래스는 name 과 greedy_frac 만 정한다.

    Args:
        greedy_frac: 그리디 모드로 지내는 시간 비율 (None 이면 클래스 값)
        mode_cycle: 두 모드를 한 번씩 거치는 평균 스텝 수
        seed: 난수 시드
    """

    name = "tiered"
    greedy_frac: float = 0.5

    # 1) 입출력 형태 — 그리디 부품이 쓰는 관측, 두 모드 모두 연속 액션으로 낸다
    obs_spec = GreedyBot.obs_spec
    action_spec = ActionSpec(mode="continuous")  # [θ, move, dash]

    def __init__(self, greedy_frac: float | None = None, mode_cycle: float = 200.0, seed: int | None = None) -> None:
        if greedy_frac is not None:
            self.greedy_frac = greedy_frac
        p = self.greedy_frac
        self.mode_cycle = mode_cycle
        self.rng = np.random.default_rng(seed)
        self.walker = RandomWalkBot(seed=None if seed is None else seed + 1)
        self.greedy = GreedyBot(seed=None if seed is None else seed + 2)
        self._leave_greedy = 1.0 / (p * mode_cycle) if 0.0 < p < 1.0 else 0.0  # 그리디 → 랜덤워크 확률/스텝
        self._enter_greedy = 1.0 / ((1.0 - p) * mode_cycle) if 0.0 < p < 1.0 else 0.0
        self.greedy_mode = np.zeros(0, dtype=bool)

    # 3) 에피소드 시작 — 모드를 정상 분포에서 새로 뽑는다
    def reset(self, done: np.ndarray) -> None:
        self.walker.reset(done)
        self.greedy.reset(done)
        if self.greedy_mode.shape[0] != done.shape[0]:
            self.greedy_mode = self.rng.random(done.shape[0]) < self.greedy_frac
        else:
            self.greedy_mode[done] = self.rng.random(int(done.sum())) < self.greedy_frac

    # 5) 행동 (배치 연산)
    def act(self, obs: Observation) -> np.ndarray:
        b = obs.batch_size
        if self.greedy_mode.shape[0] != b:
            self.reset(np.ones(b, dtype=bool))
        u = self.rng.random(b)
        self.greedy_mode = self.greedy_mode ^ np.where(self.greedy_mode, u < self._leave_greedy, u < self._enter_greedy)

        walk = self.walker.act(obs)
        action = np.stack([_MOVE_ANGLES[walk], np.ones(b), np.zeros(b)], axis=-1)
        if self.greedy_mode.any():
            action = np.where(self.greedy_mode[:, None], self.greedy.act(obs), action)
        return action
