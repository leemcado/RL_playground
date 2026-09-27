"""JAX 학습 env — NumpyTrainEnv 와 같은 입출력·규칙, 월드 진행과 관측 생성은 jit(vmap(step)).

- 모든 env 의 규칙 진행·관측 생성은 jit 한 번으로 동시에 한다 (CPU / GPU).
- 상대는 학생 에이전트와 같은 Agent 인스턴스다 (cell_arena.bots 의 봇, 다른 에이전트 파일).
  env 가 상대마다 그 에이전트의 ObsSpec 으로 관측을 만들어 주고, 호스트에서 상대의 act 를 배치로 호출한다.
- 관측은 state / image 모두 (에이전트마다 자기 ObsSpec).
- 에피소드 규칙은 NumpyTrainEnv 와 같다: 어떤 세포든 승리 크기 도달 → terminated, max_steps → truncated,
  끝난 env 는 자동 리셋. 죽음은 종료가 아니다 — 학습 세포·상대 모두 시야 밖에서 즉시 리스폰.
- 리셋용 초기 월드는 env 마다 한 벌씩 미리 만들어 두고(spare), 쓴 env 가 있는 스텝에만 새로 만든다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cell_arena.agent import Agent
from cell_arena.core.actions import check_action, decode_action
from cell_arena.core.api import ActionSpec, Events, Observation, ObsSpec, StepOutput
from cell_arena.core.config import ArenaConfig
from cell_arena.engine.jax_engine import World, init_world, respawn_cells, world_step
from cell_arena.engine.jax_observation import observe


class EnvState(NamedTuple):
    """env 하나의 상태 (vmap 하면 모든 필드에 배치 차원이 붙는다)."""

    world: World
    t: jax.Array
    key: jax.Array
    spare: World  # 다음 리셋에 쓸 초기 월드


class JaxTrainEnv:
    """학습용 배치 환경 (JAX 백엔드). 인자·반환은 NumpyTrainEnv 와 같다.

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
        self._specs = [obs_spec] + [o.obs_spec for o in opponents]
        self.cfg = cfg or ArenaConfig()
        self.num_envs = num_envs
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.opponents = list(opponents)
        self.max_steps = max_steps
        self.num_cells = 1 + len(self.opponents)
        self._key = jax.random.PRNGKey(seed if seed is not None else int(np.random.default_rng().integers(2**31)))
        self._reset_batch = jax.jit(jax.vmap(self._reset_one))
        self._step_batch = jax.jit(jax.vmap(self._step_one))
        self._refill_batch = jax.jit(jax.vmap(self._refill_one))
        self._state: EnvState | None = None
        self._opp_obs: list[Observation] = []

    def reset(self) -> Observation:
        """모든 env 리셋 후 학습 세포의 관측 반환."""
        self._state, obs = self._reset_batch(self._keys())
        # 초기 월드 채우기를 미리 컴파일해 둔다 (학습 도중 첫 리셋에서 멈칫하지 않게)
        self._state = self._refill_batch(self._state, jnp.zeros(self.num_envs, dtype=bool), self._keys())
        obs = [_to_observation(o) for o in jax.device_get(obs)]
        self._opp_obs = obs[1:]
        for opp in self.opponents:
            opp.reset(np.ones(self.num_envs, dtype=bool))
        return obs[0]

    def step(self, action: np.ndarray) -> StepOutput:
        """학습 에이전트 액션 (B, ...) 으로 모든 env 한 스텝 진행 (상대 행동은 여기서 수집)."""
        if self._state is None:
            raise RuntimeError("reset() 을 먼저 호출해야 한다")
        b, n = self.num_envs, self.num_cells
        direction = np.zeros((b, n, 2), dtype=np.float32)
        moving = np.zeros((b, n), dtype=bool)
        dash = np.zeros((b, n), dtype=bool)
        action = check_action(action, self.action_spec, b)
        direction[:, 0], moving[:, 0], dash[:, 0] = decode_action(action, self.action_spec)
        for k, (opp, obs) in enumerate(zip(self.opponents, self._opp_obs), start=1):
            direction[:, k], moving[:, k], dash[:, k] = decode_action(opp.act(obs), opp.action_spec)

        self._state, out = self._step_batch(self._state, jnp.asarray(direction), jnp.asarray(moving), jnp.asarray(dash))
        next_obs, final_obs, events, terminated, truncated, opp_reset = jax.device_get(out)
        done = terminated | truncated
        if done.any():  # 초기 월드를 쓴 env 만 새 초기 월드를 채운다
            self._state = self._refill_batch(self._state, jnp.asarray(done), self._keys())
        next_obs = [_to_observation(o) for o in next_obs]
        self._opp_obs = next_obs[1:]
        for k, opp in enumerate(self.opponents, start=1):
            if opp_reset[:, k].any():
                opp.reset(np.array(opp_reset[:, k]))
        return StepOutput(
            obs=next_obs[0],
            final_obs=_to_observation(final_obs),
            events=Events(**{k: np.array(v) for k, v in events.items()}),
            terminated=np.array(terminated),
            truncated=np.array(truncated),
        )

    def _keys(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return jax.random.split(sub, self.num_envs)

    # ------------------------------------------------------------------ jit 대상 (env 하나)

    def _new_world(self, key: jax.Array) -> World:
        return init_world(key, self.cfg, self.num_cells)

    def _observe_all(self, world: World) -> tuple[dict[str, jax.Array], ...]:
        """세포마다 그 에이전트의 ObsSpec 으로 관측."""
        return tuple(observe(world, k, spec, self.cfg) for k, spec in enumerate(self._specs))

    def _reset_one(self, key: jax.Array) -> tuple[EnvState, tuple[dict[str, jax.Array], ...]]:
        k_world, k_spare, k_env = jax.random.split(key, 3)
        state = EnvState(self._new_world(k_world), jnp.int32(0), k_env, self._new_world(k_spare))
        return state, self._observe_all(state.world)

    def _refill_one(self, state: EnvState, used: jax.Array, key: jax.Array) -> EnvState:
        spare = jax.tree.map(lambda a, b: jnp.where(used, a, b), self._new_world(key), state.spare)
        return state._replace(spare=spare)

    def _step_one(self, state: EnvState, direction: jax.Array, moving: jax.Array, dash: jax.Array):
        key, k_world, k_resp = jax.random.split(state.key, 3)
        world, ev = world_step(state.world, k_world, direction, moving, dash, self.cfg)
        t = state.t + 1
        world = respawn_cells(world, k_resp, ev["died"], self.cfg)  # 학습 세포 포함 모두 리스폰 (죽음은 종료 아님)

        terminated = jnp.any(ev["won"])  # 어떤 세포든 승리 크기 도달 → 게임 끝
        truncated = (~terminated & (t >= self.max_steps)) if self.max_steps is not None else jnp.bool_(False)
        done = terminated | truncated
        final_obs = observe(world, 0, self._specs[0], self.cfg)

        next_world = jax.tree.map(lambda a, b: jnp.where(done, a, b), state.spare, world)
        new_state = EnvState(next_world, jnp.where(done, 0, t), key, state.spare)
        events = {k: v[0] for k, v in ev.items()} | {"t": t}
        opp_reset = ev["died"] | done  # 상대 에이전트 기억 초기화 대상 (리스폰 또는 env 리셋)
        return new_state, (self._observe_all(next_world), final_obs, events, terminated, truncated, opp_reset)


def _to_observation(obs: dict[str, np.ndarray]) -> Observation:
    return Observation(**{k: np.array(v) for k, v in obs.items()})  # 쓰기 가능한 복사본 (torch.as_tensor 경고 방지)
