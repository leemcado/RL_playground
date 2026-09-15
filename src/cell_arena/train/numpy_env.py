"""학습용 배치 환경 — NumPy 백엔드 (JAX 를 쓸 수 없을 때). JaxTrainEnv 와 입출력·규칙이 같다.

NumpyArenaEngine B개를 파이썬 루프로 돌리므로 느리다.

각 env = 학습 세포 1개(인덱스 0) + 상대 len(opponents)개.
- 에피소드 종료는 두 가지뿐이고, 끝난 env 는 자동 리셋된다:
  어떤 세포든 승리 크기 도달 → terminated / 경과 스텝 ≥ max_steps → truncated
- 죽음은 에피소드를 끝내지 않는다: 학습 세포든 상대든 모든 시야 밖에서 초기 상태(크기 100, 정지)로 즉시 리스폰.
  학습 세포의 사망은 events.died 로 알 수 있다
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cell_arena.agent import Agent
from cell_arena.core.actions import check_action, decode_action
from cell_arena.core.api import ActionSpec, Events, Observation, ObsSpec, StepOutput
from cell_arena.core.config import ArenaConfig
from cell_arena.core.observation import build_observation
from cell_arena.engine.numpy_engine import NumpyArenaEngine


class NumpyTrainEnv:
    """학습용 배치 환경 (NumPy).

    Args:
        num_envs: 병렬 env 수 B
        obs_spec: 학습 에이전트의 관측 형태
        action_spec: 학습 에이전트의 액션 형태
        opponents: 상대 Agent 인스턴스 목록
        max_steps: 에피소드 최대 길이. None 이면 무한
        seed: 난수 시드
        cfg: 게임 규칙
    """

    def __init__(
        self,
        num_envs: int,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        opponents: Sequence[Agent] = (),
        max_steps: int | None = 1000,
        seed: int | None = None,
        cfg: ArenaConfig | None = None,
    ) -> None:
        self.cfg = cfg or ArenaConfig()
        self.num_envs = num_envs
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.opponents = list(opponents)
        self.max_steps = max_steps
        self.num_cells = 1 + len(self.opponents)
        rng = np.random.default_rng(seed)
        self._engines = [
            NumpyArenaEngine(self.cfg, self.num_cells, seed=int(s)) for s in rng.integers(2**31, size=num_envs)
        ]
        self._t = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> Observation:
        """모든 env 리셋 후 학습 세포의 관측 반환."""
        for engine in self._engines:
            engine.reset()
        self._t[:] = 0
        for opp in self.opponents:
            opp.reset(np.ones(self.num_envs, dtype=bool))
        return Observation.stack(self._rows(0, self.obs_spec))

    def step(self, action: np.ndarray) -> StepOutput:
        """학습 에이전트 액션 (B, ...) 으로 모든 env 한 스텝 진행. 보상은 없고 사건(events)을 준다."""
        b, n = self.num_envs, self.num_cells
        direction = np.zeros((b, n, 2))
        moving = np.zeros((b, n), dtype=bool)
        dash = np.zeros((b, n), dtype=bool)
        action = check_action(action, self.action_spec, b)
        direction[:, 0], moving[:, 0], dash[:, 0] = decode_action(action, self.action_spec)
        for k, opp in enumerate(self.opponents, start=1):
            opp_action = opp.act(Observation.stack(self._rows(k, opp.obs_spec)))
            direction[:, k], moving[:, k], dash[:, k] = decode_action(opp_action, opp.action_spec)

        event_rows = []
        terminated = np.zeros(b, dtype=bool)
        respawned = np.zeros((b, n), dtype=bool)
        for e, engine in enumerate(self._engines):
            ev = engine.step(direction[e], moving[e], dash[e])
            self._t[e] += 1
            event_rows.append({key: val[0] for key, val in ev.items()} | {"t": self._t[e]})
            terminated[e] = bool(ev["won"].any())  # 어떤 세포든 승리 크기 도달 → 게임 끝
            for k in np.flatnonzero(ev["died"]):  # 학습 세포 포함 모두 시야 밖에서 초기 상태로 리스폰
                engine.respawn(int(k))
                respawned[e, k] = True

        truncated = ~terminated & (self._t >= self.max_steps) if self.max_steps is not None else np.zeros(b, bool)
        done = terminated | truncated
        final_rows = self._rows(0, self.obs_spec)
        next_rows = list(final_rows)
        for e in np.flatnonzero(done):
            self._engines[e].reset()
            self._t[e] = 0
            next_rows[e] = build_observation(self._engines[e].state, 0, self.obs_spec, self.cfg)
        for k, opp in enumerate(self.opponents, start=1):
            mask = done | respawned[:, k]
            if mask.any():
                opp.reset(mask)

        return StepOutput(
            obs=Observation.stack(next_rows),
            final_obs=Observation.stack(final_rows),
            events=Events.stack(event_rows),
            terminated=terminated,
            truncated=truncated,
        )

    def _rows(self, cell: int, spec: ObsSpec) -> list[dict[str, np.ndarray]]:
        """모든 env 에서 세포 하나의 관측 (배치 전 행 리스트)."""
        return [build_observation(engine.state, cell, spec, self.cfg) for engine in self._engines]
