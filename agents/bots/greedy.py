"""그리디 봇 — 관측만 보고 도망·추격·채집하는 휴리스틱 (티어 봇 부품, 단독으로는 다이아와 같다).

학습 env 에서는 env 수만큼의 배치로 매 스텝 호출되므로, 배치 원소별 파이썬 루프 없이 배열 연산으로 짰다.
"""

from __future__ import annotations

import numpy as np

from cell_arena import ActionSpec, Agent, Observation, ObsSpec

DIAMETER_SCALE = 2.0  # 규칙: 직경 d(s) = 2 × √(s/100)


class GreedyBot(Agent):
    """우선순위: 위협 도주 > 블랙홀 회피(크기 500 초과) > 먹이 세포 추격 > 화이트홀(500 미만)·밥 > 배회.

    자기 관측(자기 상태 + 시야 안 객체)만 쓴다. 다른 세포와의 크기 비는 직경으로 계산한다:
    s_other / s_me = (d_other / d_me)². 닿으면 큰 쪽이 먹으므로 나보다 크면 위협, 작으면 먹이.

    Args:
        flee_gap: 위협과의 경계면 거리가 이보다 가까우면 도주 (돌진 가능하면 돌진)
        dash_gap: 먹이 세포가 이 거리 안이면 돌진해서 추격
        black_gap: 크기 500 초과일 때 블랙홀이 이 거리 안이면 회피
        seed: 난수 시드 (배회 방향)
    """

    name = "greedy"

    # 1) 입출력 형태
    obs_spec = ObsSpec(mode="state", max_objects=32)
    action_spec = ActionSpec(mode="continuous")  # [θ, move, dash]

    def __init__(self, flee_gap: float = 6.0, dash_gap: float = 10.0, black_gap: float = 2.0,
                 seed: int | None = None) -> None:
        self.flee_gap = flee_gap
        self.dash_gap = dash_gap
        self.black_gap = black_gap
        self.rng = np.random.default_rng(seed)
        self.heading = np.zeros(0)

    # 3) 에피소드 시작 — 새로 시작한 배치 원소는 배회 방향을 새로 뽑는다
    def reset(self, done: np.ndarray) -> None:
        if self.heading.shape[0] != done.shape[0]:
            self.heading = self.rng.uniform(-np.pi, np.pi, done.shape[0])
        else:
            self.heading[done] = self.rng.uniform(-np.pi, np.pi, int(done.sum()))

    # 5) 행동 (배치 연산)
    def act(self, obs: Observation) -> np.ndarray:
        b = obs.batch_size
        if self.heading.shape[0] != b:
            self.reset(np.ones(b, dtype=bool))
        rows = np.arange(b)
        s_me = obs.self_state[:, 0]
        d_me = DIAMETER_SCALE * np.sqrt(s_me / 100.0)[:, None]  # 직경 d = 2√(s/100)
        dx, dy, d = obs.objects[..., 0], obs.objects[..., 1], obs.objects[..., 2]
        r, theta = np.hypot(dx, dy), np.arctan2(dy, dx)
        kind = np.argmax(obs.objects[..., 3:7], axis=-1)  # 0 밥, 1 블랙홀, 2 화이트홀, 3 세포
        gap = np.where(obs.mask, r - 0.5 * (d + d_me), np.inf)  # 경계면 사이 거리
        ratio = (d / d_me) ** 2  # s_other / s_me

        def nearest(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            g = np.where(m, gap, np.inf)
            j = np.argmin(g, axis=1)
            return theta[rows, j], g[rows, j]

        th_threat, g_threat = nearest((kind == 3) & (ratio > 1.0))
        th_black, g_black = nearest(kind == 1)
        th_prey, g_prey = nearest((kind == 3) & (ratio < 1.0))
        th_food, g_food = nearest((kind == 0) | ((kind == 2) & (s_me[:, None] < 500)))

        turn = self.rng.random(b) < 0.05
        self.heading = self.heading + np.where(turn, self.rng.normal(0.0, 1.0, b), 0.0)

        flee = g_threat < self.flee_gap
        avoid = (s_me > 500) & (g_black < self.black_gap)
        chase = np.isfinite(g_prey)
        feed = np.isfinite(g_food)
        out_theta = np.select(
            [flee, avoid, chase, feed], [th_threat + np.pi, th_black + np.pi, th_prey, th_food], self.heading
        )
        # 돌진은 도주·근거리 추격 때만 요청 (크기 100 이하면 엔진이 알아서 무시)
        out_dash = np.select([flee, avoid, chase], [True, False, g_prey < self.dash_gap], False)
        return np.stack([out_theta, np.ones(b), out_dash.astype(np.float64)], axis=-1)
