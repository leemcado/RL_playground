"""랜덤워크 봇 — 평균 20스텝마다 8방향 중 하나로 방향을 바꾸며 돌아다닌다 (티어 봇 부품, 단독으로는 브론즈와 같다)."""

from __future__ import annotations

import numpy as np

from cell_arena import ActionSpec, Agent, Observation, ObsSpec


class RandomWalkBot(Agent):
    """스텝당 turn_prob 확률로 방향 전환, 돌진 없음.

    Args:
        turn_prob: 스텝당 방향 전환 확률
        seed: 난수 시드
    """

    name = "random_walk"

    # 1) 입출력 형태 — 관측을 쓰지 않으므로 최소 크기
    obs_spec = ObsSpec(mode="state", max_objects=1)
    action_spec = ActionSpec(mode="discrete")  # 1~8 = 상·하·좌·우·좌상·우상·좌하·우하

    def __init__(self, turn_prob: float = 0.05, seed: int | None = None) -> None:
        self.turn_prob = turn_prob
        self.rng = np.random.default_rng(seed)
        self.move = np.zeros(0, dtype=np.int64)

    # 3) 에피소드 시작 — 새로 시작한 배치 원소는 방향을 새로 뽑는다
    def reset(self, done: np.ndarray) -> None:
        if self.move.shape[0] != done.shape[0]:
            self.move = self.rng.integers(1, 9, done.shape[0])
        else:
            self.move[done] = self.rng.integers(1, 9, int(done.sum()))

    # 5) 행동 (배치 연산)
    def act(self, obs: Observation) -> np.ndarray:
        b = obs.batch_size
        if self.move.shape[0] != b:
            self.reset(np.ones(b, dtype=bool))
        turn = self.rng.random(b) < self.turn_prob
        self.move[turn] = self.rng.integers(1, 9, int(turn.sum()))
        return self.move.copy()
