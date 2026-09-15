"""JAX 관측 생성 (core/observation.py 의 jnp 판). 결과는 NumPy 판과 같다 (scripts/doctor.py 가 검증).

고정 shape 를 위해 가시 객체를 골라내는 대신 전체 객체에 가시 마스크를 씌우고 ``lax.top_k`` 로 가까운 순으로 고른다.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cell_arena.core import physics
from cell_arena.core.api import IMAGE_CHANNEL_CAPS, ObsSpec
from cell_arena.core.config import ArenaConfig
from cell_arena.core.observation import KIND_CELL, KIND_SELF, disk_cover, object_kind
from cell_arena.engine.jax_engine import F32, World


def self_state(world: World, idx: int) -> jax.Array:
    """자기 상태 ``[size, x, y, v_x, v_y]`` (5,)."""
    cells = world.cells
    return jnp.stack(
        [cells.size[idx], cells.pos[idx, 0], cells.pos[idx, 1], cells.vel[idx, 0], cells.vel[idx, 1]]
    ).astype(F32)


def _candidates(world: World, idx: int, cfg: ArenaConfig) -> tuple[jax.Array, ...]:
    """모든 객체·세포의 변위 (K, 2), 반지름, 종류, 거리, 가시 여부, 시야 반경 h."""
    cells, obj = world.cells, world.objects
    n, m = cells.size.shape[0], obj.size.shape[0]
    center = cells.pos[idx]
    h = physics.vision_radius(cells.size[idx], cfg, jnp)
    delta = physics.torus_delta(jnp.concatenate([obj.pos, cells.pos]), center, cfg.map_size, jnp)
    radius = physics.radius(jnp.concatenate([obj.size, cells.size]), cfg, jnp)
    kind = jnp.concatenate([object_kind(obj.type, jnp), jnp.full(n, KIND_CELL)]).at[m + idx].set(KIND_SELF)
    alive = jnp.concatenate([obj.alive, cells.alive])
    visible = alive & jnp.all(jnp.abs(delta) < (h + radius)[:, None], axis=-1)
    dist = jnp.linalg.norm(delta, axis=-1)
    return delta, radius, kind, dist, visible, h


def _nearest(dist: jax.Array, select: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """select 중 가까운 순 k 개의 인덱스와 유효 여부 (동률은 앞 인덱스 먼저 — NumPy stable argsort 와 같음)."""
    top, order = jax.lax.top_k(jnp.where(select, -dist, -jnp.inf), k)
    return order, top > -jnp.inf


def object_list(world: World, idx: int, cfg: ArenaConfig, max_objects: int) -> tuple[jax.Array, jax.Array]:
    """시야 안 객체 가까운 순 최대 M개. 반환 (M, 7) 피처, (M,) mask."""
    delta, radius, kind, dist, visible, _ = _candidates(world, idx, cfg)
    order, valid = _nearest(dist, visible & (kind != KIND_SELF), max_objects)
    feats = jnp.concatenate(
        [delta[order], 2.0 * radius[order][:, None], jax.nn.one_hot(kind[order], 4, dtype=F32)], axis=-1
    )
    return jnp.where(valid[:, None], feats, 0.0).astype(F32), valid


def semantic_image(world: World, idx: int, cfg: ArenaConfig, resolution: int) -> jax.Array:
    """타입별 5채널 이미지 (5, R, R) uint8 — 채널마다 가까운 순 IMAGE_CHANNEL_CAPS 개."""
    delta, radius, kind, dist, visible, h = _candidates(world, idx, cfg)
    channels = []
    for c, cap in enumerate(IMAGE_CHANNEL_CAPS):
        order, valid = _nearest(dist, visible & (kind == c), min(cap, dist.shape[0]))
        channels.append(disk_cover(delta[order], radius[order], valid, h, resolution, jnp))
    return jnp.stack(channels).astype(jnp.uint8)


def observe(world: World, idx: int, spec: ObsSpec, cfg: ArenaConfig) -> dict[str, jax.Array]:
    """세포 하나의 관측 dict (배치 차원 없음)."""
    out = {"self_state": self_state(world, idx)}
    if spec.mode == "state":
        out["objects"], out["mask"] = object_list(world, idx, cfg, spec.max_objects)
    else:
        out["image"] = semantic_image(world, idx, cfg, spec.resolution)
    return out
