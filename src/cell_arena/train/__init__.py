"""학습 env 만들기 — ``make_env(cfg, agent)`` 하나로 설정·하드웨어에 맞는 배치 환경을 준다.

- ``JaxTrainEnv`` (train/jax_env.py): 기본. 월드 진행·관측 생성을 jit(vmap) 으로 (CPU / GPU)
- ``NumpyTrainEnv`` (train/numpy_env.py): JAX 를 쓸 수 없을 때 자동으로 대신 쓴다. 입출력·규칙이 같다
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from cell_arena.train.config import Config, load_config

if TYPE_CHECKING:
    from cell_arena.agent import Agent

__all__ = ["Config", "load_config", "make_env"]


def make_env(cfg: Config, agent: Agent, backend: str = "auto", verbose: bool = True) -> Any:
    """설정의 상대 구성·env 수·에피소드 길이·시드로 학습 env 를 만든다.

    Args:
        cfg: opponents (봇 이름 또는 에이전트 파일 경로 목록), num_envs, max_steps (null = 무한), seed
        agent: 학습할 에이전트 (obs_spec / action_spec 을 읽는다)
        backend: "auto" (JAX 를 쓸 수 있으면 JAX, 아니면 NumPy) / "jax" / "numpy"
        verbose: 만든 env 요약 한 줄 출력

    Returns:
        reset() -> Observation, step(action) -> StepOutput 을 가진 env

    Raises:
        ValueError: 설정 키가 없거나 형식이 틀림
    """
    from cell_arena.hardware import jax_status
    from cell_arena.play.loader import load_agent

    _check(cfg)
    opponents = [load_agent(spec) for spec in cfg.opponents]
    kwargs = dict(
        num_envs=cfg.num_envs, obs_spec=agent.obs_spec, action_spec=agent.action_spec,
        opponents=opponents, max_steps=cfg.max_steps, seed=cfg.seed,
    )
    jax_ok, jax_info = jax_status() if backend in ("auto", "jax") else (False, "")
    if backend == "jax" and not jax_ok:
        raise RuntimeError(f"JAX 를 쓸 수 없다: {jax_info}")
    if jax_ok:
        from cell_arena.train.jax_env import JaxTrainEnv

        env, info = JaxTrainEnv(**kwargs), jax_info
    else:
        from cell_arena.train.numpy_env import NumpyTrainEnv

        env = NumpyTrainEnv(**kwargs)
        info = "numpy" if backend == "numpy" else f"numpy — JAX 를 쓸 수 없어 느린 NumPy env 로 대신한다 ({jax_info})"
    if verbose:
        counts = ", ".join(f"{name}×{c}" for name, c in Counter(o.name for o in opponents).items())
        print(f"[cell-arena] env {info} | {cfg.num_envs} envs | T={cfg.max_steps} | 상대 {len(opponents)}: {counts} "
              f"| obs {agent.obs_spec.mode}, action {agent.action_spec.mode}", flush=True)
    return env


def _check(cfg: Config) -> None:
    for key in ("opponents", "num_envs", "max_steps", "seed"):
        if key not in cfg:
            raise ValueError(f"설정에 '{key}' 가 필요하다")
    if not isinstance(cfg.opponents, list) or not all(isinstance(s, str) for s in cfg.opponents):
        raise ValueError(f"opponents 는 문자열 목록이어야 한다 (예: [bronze, silver, gold]): {cfg.opponents}")
    if not isinstance(cfg.num_envs, int) or cfg.num_envs < 1:
        raise ValueError(f"num_envs 는 1 이상 정수: {cfg.num_envs}")
    if cfg.max_steps is not None and (not isinstance(cfg.max_steps, int) or cfg.max_steps < 1):
        raise ValueError(f"max_steps 는 1 이상 정수 또는 null: {cfg.max_steps}")
    if not isinstance(cfg.seed, int):
        raise ValueError(f"seed 는 정수: {cfg.seed}")
