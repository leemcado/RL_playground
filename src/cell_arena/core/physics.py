"""크기 의존 물리 공식과 접촉·포식 규칙 (청사진 1.1~1.3, 2, 3.1절).

모든 함수는 분기 없는 요소별 연산이고 ``xp`` 인자로 배열 모듈을 받는다.
NumPy 엔진은 ``xp=np``(기본), JAX 엔진은 ``xp=jnp`` 로 같은 코드를 쓴다 — 규칙의 단일 출처.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from cell_arena.core.config import ArenaConfig

Array = Any  # np.ndarray 또는 jax.Array


def diameter(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """직경 d(s) = radius_scale × sqrt(s / 100). 충돌 판정·관측·표시에 모두 쓰인다."""
    return cfg.radius_scale * xp.sqrt(xp.asarray(size) / cfg.base_size)


def radius(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """반지름 r(s) = d(s) / 2."""
    return 0.5 * diameter(size, cfg, xp)


def base_speed(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """평상 속도 v_base(s) = (s / 100) ^ -0.19."""
    return (xp.asarray(size) / cfg.base_size) ** cfg.speed_exponent


def dash_speed(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """돌진 속도 v_dash(s) = 1.1 × (s / 100) ^ 0.2 — 클수록 빨라서 쫓는 큰 쪽이 따라잡을 수 있다.

    청사진 원안 v_base(s) × (1 + s/1000) 은 200~500 구간이 1.05~1.11 로 거의 평평해 추격이 성립하지 않았다.
    """
    return cfg.dash_speed_base * (xp.asarray(size) / cfg.base_size) ** cfg.dash_speed_exponent


def dash_emit_frac(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """돌진 방출 비율 f(s) = 1% × (s / 100) ^ 0.5 (4스텝마다) — 클수록 더 큰 비율로 잃고 더 큰 세포밥을 흘린다."""
    return cfg.dash_emit_frac * (xp.asarray(size) / cfg.base_size) ** cfg.dash_emit_exponent


def kappa(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """조향 계수 κ(s) = 0.35 × (s / 100) ^ -0.25 (평상시)."""
    return cfg.kappa_base * (xp.asarray(size) / cfg.base_size) ** cfg.kappa_exponent


def vision_radius(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """시야 반경 h(s) = min(8 × sqrt(s / 100), 25). 시야는 한 변 2h 인 정사각형.

    radius_scale 과 무관하게 청사진 원래 직경 기준으로 계산한다 (반경을 키워도 시야는 그대로).
    """
    return xp.minimum(cfg.vision_coef * xp.sqrt(xp.asarray(size) / cfg.base_size), cfg.vision_cap)


def can_dash(size: Array, cfg: ArenaConfig, xp: ModuleType = np) -> Array:
    """돌진 가능 여부.

    다음 방출 후에도 크기가 최소 돌진 크기(100) 이상이어야 한다.
    "s ≤ 100 이면 돌진 불가"와 "방출로 100 아래가 될 상황이면 자동 해제"를 한 식으로 표현.
    """
    s = xp.asarray(size)
    return s * (1.0 - dash_emit_frac(s, cfg, xp)) >= cfg.min_dash_size


def contact_reach(r_a: Array, r_b: Array, xp: ModuleType = np) -> Array:
    """접촉 거리: 두 원의 경계면이 닿으면 접촉 (dist < r_a + r_b).

    청사진 1.3절(작은 쪽 중심이 큰 쪽 원 안)에서 변경. 세포-객체, 세포-세포 모두 이 규칙을 쓴다.
    """
    return xp.asarray(r_a) + xp.asarray(r_b)


def torus_delta(a: Array, b: Array, map_size: float, xp: ModuleType = np) -> Array:
    """토러스 최소상 변위 a - b (각 성분이 [-L/2, L/2) 범위)."""
    return xp.mod(a - b + map_size / 2, map_size) - map_size / 2


def can_eat_matrix(
    pos: Array, size: Array, alive: Array, priority: Array, cfg: ArenaConfig, xp: ModuleType = np
) -> Array:
    """can_eat[a, b]: 세포 a 가 b 를 먹는 조건.

    경계면 접촉 AND s_a > predation_ratio × s_b. predation_ratio=1 (기본) 이면 닿기만 하면
    큰 쪽이 무조건 먹고, 크기가 정확히 같으면 이번 스텝의 무작위 priority 가 큰 쪽이 먹는다.

    Args:
        pos: (N, 2), size: (N,), alive: (N,)
        priority: (N,) 스텝마다 새로 뽑는 난수 (같은 크기 동률 해소용)
        cfg: 게임 규칙

    Returns:
        (N, N) bool
    """
    n = size.shape[0]
    dist = xp.linalg.norm(torus_delta(pos[None], pos[:, None], cfg.map_size, xp), axis=-1)
    r = radius(size, cfg, xp)
    beats = size[:, None] > cfg.predation_ratio * size[None, :]
    if cfg.predation_ratio <= 1.0:
        beats = beats | ((size[:, None] == size[None, :]) & (priority[:, None] > priority[None, :]))
    touching = dist < contact_reach(r[:, None], r[None], xp)
    return alive[:, None] & alive[None, :] & touching & beats & ~xp.eye(n, dtype=bool)


def resolve_predation(can_eat: Array, alive: Array, size: Array, xp: ModuleType = np) -> tuple[Array, Array]:
    """세포 간 포식 동시 판정.

    먹히는 세포는 같은 스텝에 다른 세포를 먹을 수 없다. 큰 세포부터 확정되도록 N 번 반복한다
    (가장 큰 세포는 절대 먹히지 않고, k 번째 세포의 운명은 k-1 번 반복 뒤 확정).
    여러 포식자가 같은 먹이에 닿으면 가장 큰 포식자가 가져간다.

    Args:
        can_eat: (N, N) bool — can_eat_matrix 결과
        alive: (N,) 생존 여부
        size: (N,) 크기

    Returns:
        eaten (N,) bool, predator (N,) int — eaten 인 b 를 먹은 세포 인덱스 (나머지는 의미 없음)
    """
    n = alive.shape[0]
    eaten = xp.zeros(n, dtype=bool)
    for _ in range(n):
        valid = alive & ~eaten
        eaten = (can_eat & valid[:, None]).any(axis=0)
    valid = alive & ~eaten
    predator = xp.argmax(xp.where(can_eat & valid[:, None], size[:, None], -xp.inf), axis=0)
    return eaten, predator
