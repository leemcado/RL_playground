"""실전 대결장 (play env).

- 시간 제한 없음이 기본 (max_steps=None). 필요하면 정할 수 있다.
- 기본은 탈락제: 먹힌 세포는 리스폰하지 않는다 (청사진 6절 토너먼트).
  ``respawn`` 으로 지정한 에이전트는 죽으면 모든 시야 밖에서 초기 상태(크기 100, 정지)로 되살아난다.
- 종료: 누군가 승리 크기(4000) 도달 / 생존자 1명 이하 (재스폰 없는 경기) /
  재스폰하지 않는 에이전트가 모두 탈락 (재스폰이 섞인 경기) / max_steps 도달.
- 에이전트마다 자기 ObsSpec 으로 관측을 받고 자기 ActionSpec 으로 행동한다 (이산·연속 혼합 가능).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence

import numpy as np

from cell_arena.agent import Agent
from cell_arena.core.actions import decode_action
from cell_arena.core.api import Observation
from cell_arena.core.config import ArenaConfig
from cell_arena.core.observation import build_observation
from cell_arena.core.state import WorldState
from cell_arena.engine.numpy_engine import NumpyArenaEngine


class BattleArena:
    """에이전트 N명의 대결 (세포 i = agents[i]).

    Args:
        agents: 참가 에이전트 (최대 8명 권장 — 팔레트 색 8개)
        cfg: 게임 규칙
        seed: 난수 시드
        max_steps: 시간 제한. None 이면 무한
        respawn: 죽으면 되살릴지 — 전체에 bool 하나, 또는 에이전트별 목록 (예: 사람 False, 봇 True)
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        cfg: ArenaConfig | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        respawn: bool | Sequence[bool] = False,
    ) -> None:
        self.agents = list(agents)
        self.cfg = cfg or ArenaConfig()
        self.max_steps = max_steps
        self.respawn = np.broadcast_to(np.asarray(respawn, dtype=bool), (len(self.agents),)).copy()
        self.engine = NumpyArenaEngine(self.cfg, len(self.agents), seed=seed)
        self.names = self._unique_names([a.name for a in self.agents])
        self.reset()

    @property
    def state(self) -> WorldState:
        return self.engine.state

    @property
    def done(self) -> bool:
        return bool(self.status)

    @property
    def colors(self) -> list:
        """에이전트 파일에서 정한 표시 색 목록 (없으면 None → 화면 기본 팔레트)."""
        return [a.color for a in self.agents]

    def reset(self) -> None:
        """새 경기."""
        self.engine.reset()
        self.t = 0
        self.status = ""
        n = len(self.agents)
        self.eliminated_at = np.full(n, -1, dtype=np.int64)
        self.deaths = np.zeros(n, dtype=np.int64)
        self.act_ms_max = np.zeros(n)
        for agent in self.agents:
            agent.reset(np.ones(1, dtype=bool))

    def step(self) -> dict[str, np.ndarray]:
        """살아있는 에이전트 행동 수집 → 한 스텝 진행 → 재스폰 대상 되살리기. 세포별 사건 (N,) 반환."""
        if self.done:
            raise RuntimeError(f"경기가 끝났다: {self.status}")
        n = len(self.agents)
        direction = np.zeros((n, 2))
        moving = np.zeros(n, dtype=bool)
        dash = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(self.state.cells.alive):
            agent = self.agents[i]
            obs = Observation.stack([build_observation(self.state, int(i), agent.obs_spec, self.cfg)])
            start = time.perf_counter()
            action = agent.act(obs)
            self.act_ms_max[i] = max(self.act_ms_max[i], (time.perf_counter() - start) * 1000)
            d, m, s = decode_action(action, agent.action_spec)
            direction[i], moving[i], dash[i] = d[0], m[0], s[0]

        ev = self.engine.step(direction, moving, dash)
        self.t += 1
        died = ev["died"]
        self.deaths += died
        self.eliminated_at[died & ~self.respawn] = self.t
        for i in np.flatnonzero(died & self.respawn):
            self.engine.respawn(int(i))  # 모든 시야 밖, 크기 100, 정지
            self.agents[i].reset(np.ones(1, dtype=bool))
        self.status = self._end_status(ev)
        return ev

    def results(self) -> list[dict[str, object]]:
        """순위표: 생존자(크기순) → 탈락자(늦게 탈락한 순)."""
        cells = self.state.cells
        order = sorted(
            range(len(self.agents)),
            key=lambda i: (not cells.alive[i], -cells.size[i] if cells.alive[i] else -self.eliminated_at[i]),
        )
        return [
            {
                "rank": rank,
                "name": self.names[i],
                "size": float(cells.size[i]),
                "alive": bool(cells.alive[i]),
                "deaths": int(self.deaths[i]),
                "eliminated_at": int(self.eliminated_at[i]),
                "act_ms_max": float(self.act_ms_max[i]),
            }
            for rank, i in enumerate(order, start=1)
        ]

    def _end_status(self, ev: dict[str, np.ndarray]) -> str:
        cells = self.state.cells
        if ev["won"].any():
            i = int(np.argmax(np.where(ev["won"], cells.size, -np.inf)))
            return f"WINNER {self.names[i]}"
        fixed = ~self.respawn
        if self.respawn.any():
            if fixed.any() and not (cells.alive & fixed).any():
                return "ELIMINATED"  # 재스폰하지 않는 참가자(예: 사람)가 모두 탈락
        elif len(self.agents) > 1 and cells.alive.sum() <= 1:
            return f"LAST SURVIVOR {self.names[int(np.argmax(cells.alive))]}" if cells.alive.any() else "NO SURVIVORS"
        if self.max_steps is not None and self.t >= self.max_steps:
            return f"TIME UP  {self.max_steps} steps"
        return ""

    @staticmethod
    def _unique_names(names: list[str]) -> list[str]:
        counts = Counter(names)
        seen: Counter[str] = Counter()
        out = []
        for name in names:
            seen[name] += 1
            out.append(f"{name}#{seen[name]}" if counts[name] > 1 else name)
        return out
