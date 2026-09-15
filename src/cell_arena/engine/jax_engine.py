"""JAX 규칙 엔진 (학습용). 순수 함수 + 고정 shape → ``jit`` / ``vmap`` / ``scan`` 가능.

규칙은 NumpyArenaEngine 과 같다 (``core.physics`` 를 ``xp=jnp`` 로 공유,
scripts/doctor.py 가 동등성을 검증). NumPy 엔진과의 차이:
- float32 연산, 난수는 jax.random (재스폰 위치·밥 크기는 엔진마다 다름)
- 한 스텝에 재스폰하는 밥·홀은 최대 RESPAWN_SLOTS 개. 넘치면 다음 스텝으로 밀린다 (실제로는 드묾)

관측 생성은 jax_observation.py.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cell_arena.core import physics
from cell_arena.core.config import BLACK_HOLE, CELL_FOOD, FOOD, WHITE_HOLE, ArenaConfig
from cell_arena.core.state import WorldState

RESPAWN_SLOTS = 16
F32 = jnp.float32


class Cells(NamedTuple):
    """세포 배열 (N,) — core.state.CellArrays 와 같은 의미 (color 제외)."""

    pos: jax.Array
    vel: jax.Array
    size: jax.Array
    alive: jax.Array
    dash_steps: jax.Array
    dashing: jax.Array


class Objects(NamedTuple):
    """객체 배열 (M,) — core.state.ObjectArrays 와 같은 레이아웃."""

    pos: jax.Array
    size: jax.Array
    type: jax.Array
    alive: jax.Array
    owner: jax.Array


class World(NamedTuple):
    """월드 상태 pytree."""

    cells: Cells
    objects: Objects
    cursor: jax.Array  # 세포밥 링버퍼 다음 쓰기 위치
    step: jax.Array


def _object_types(cfg: ArenaConfig, num_cells: int) -> np.ndarray:
    types = np.full(cfg.num_objects(num_cells), CELL_FOOD, dtype=np.int32)
    types[: cfg.num_food] = FOOD
    types[cfg.num_food : cfg.num_food + cfg.num_black_holes] = BLACK_HOLE
    types[cfg.num_food + cfg.num_black_holes : cfg.cell_food_start] = WHITE_HOLE
    return types


def _sum_to(owner: jax.Array, values: jax.Array, n: int) -> jax.Array:
    """owner[j] == i 인 values[j] 의 합 (n,). 작은 n 으로 모으는 scatter-add 를 (n, M) 밀집 합으로 대신한다."""
    return jnp.sum(jnp.where(owner[None, :] == jnp.arange(n)[:, None], values[None, :], 0), axis=1)


def _food_sizes(key: jax.Array, n: int, cfg: ArenaConfig) -> jax.Array:
    """밥 크기 ~ Exp(평균 8), 상한 40."""
    return jnp.minimum(jax.random.exponential(key, (n,), F32) * cfg.food_mean, cfg.food_max)


def spawn_positions(key: jax.Array, cells: Cells, obj_radius: jax.Array, cfg: ArenaConfig) -> jax.Array:
    """스폰 위치: 모든 세포 시야 밖 (NumpyArenaEngine._spawn_positions 와 같음). 반환 (R, 2)."""
    count, k = obj_radius.shape[0], cfg.spawn_candidates
    k_cand, k_pick = jax.random.split(key)
    cand = jax.random.uniform(k_cand, (count, k, 2), F32, 0.0, cfg.map_size)
    h = physics.vision_radius(cells.size, cfg, jnp)
    delta = physics.torus_delta(cand[:, :, None, :], cells.pos[None, None], cfg.map_size, jnp)
    reach = h[None, None, :] + obj_radius[:, None, None]
    in_view = jnp.all(jnp.abs(delta) < reach[..., None], axis=-1) & cells.alive[None, None, :]
    outside = ~jnp.any(in_view, axis=-1)
    dist = jnp.where(cells.alive[None, None, :], jnp.linalg.norm(delta, axis=-1), jnp.inf)
    nearest = jnp.min(dist, axis=-1)
    score = jnp.where(outside, jax.random.uniform(k_pick, (count, k)), -1.0)
    pick = jnp.where(jnp.any(outside, axis=-1), jnp.argmax(score, axis=-1), jnp.argmax(nearest, axis=-1))
    return cand[jnp.arange(count), pick]


def init_world(key: jax.Array, cfg: ArenaConfig, num_cells: int) -> World:
    """초기 월드: 세포는 기본 크기·정지, 밥은 균일 배치, 홀은 시야 밖 스폰."""
    n = num_cells
    types = jnp.asarray(_object_types(cfg, n))
    m, nf, start = types.shape[0], cfg.num_food, cfg.cell_food_start
    k_cell, k_fpos, k_fsize, k_hole = jax.random.split(key, 4)
    cells = Cells(
        pos=jax.random.uniform(k_cell, (n, 2), F32, 0.0, cfg.map_size),
        vel=jnp.zeros((n, 2), F32),
        size=jnp.full(n, cfg.base_size, F32),
        alive=jnp.ones(n, bool),
        dash_steps=jnp.zeros(n, jnp.int32),
        dashing=jnp.zeros(n, bool),
    )
    sizes = jnp.zeros(m, F32).at[:nf].set(_food_sizes(k_fsize, nf, cfg)).at[nf:start].set(cfg.hole_size)
    pos = jnp.zeros((m, 2), F32).at[:nf].set(jax.random.uniform(k_fpos, (nf, 2), F32, 0.0, cfg.map_size))
    pos = pos.at[nf:start].set(spawn_positions(k_hole, cells, physics.radius(sizes[nf:start], cfg, jnp), cfg))
    objects = Objects(pos=pos, size=sizes, type=types, alive=jnp.arange(m) < start, owner=jnp.full(m, -1, jnp.int32))
    return World(cells=cells, objects=objects, cursor=jnp.int32(0), step=jnp.int32(0))


def world_step(
    world: World, key: jax.Array, direction: jax.Array, moving: jax.Array, dash: jax.Array, cfg: ArenaConfig
) -> tuple[World, dict[str, jax.Array]]:
    """한 스텝: 조향 → 이동 → 돌진 방출 → 세포 간 포식 → 객체 접촉 → 밥·홀 재스폰.

    Args:
        world: 월드 상태
        key: 난수 키
        direction: (N, 2) 이동 방향 단위벡터
        moving: (N,) 이동 입력
        dash: (N,) 돌진 입력
        cfg: 게임 규칙 (정적)

    Returns:
        새 월드, 세포별 사건 (N,) — engine.base.EVENT_KEYS
    """
    cells, obj = world.cells, world.objects
    n, m = cells.size.shape[0], obj.size.shape[0]
    start, L = cfg.cell_food_start, cfg.map_size
    cap = m - start
    k_food, k_spawn, k_prio = jax.random.split(key, 3)
    alive0, size_before = cells.alive, cells.size

    # 운동
    dashing = dash & moving & physics.can_dash(cells.size, cfg, jnp) & alive0
    speed = jnp.where(dashing, physics.dash_speed(cells.size, cfg, jnp), physics.base_speed(cells.size, cfg, jnp))
    kap = physics.kappa(cells.size, cfg, jnp) * jnp.where(dashing, cfg.dash_kappa_scale, 1.0)
    v_target = direction * (speed * moving)[:, None]
    vel = jnp.where(alive0[:, None], cells.vel + (v_target - cells.vel) * kap[:, None], 0.0)
    pos = jnp.where(alive0[:, None], jnp.mod(cells.pos + vel, L), cells.pos)

    # 돌진 방출 → 세포밥 링버퍼 (방출하지 않는 세포는 범위 밖 인덱스로 보내 drop)
    dash_steps = cells.dash_steps + dashing.astype(jnp.int32)
    emit = dashing & (dash_steps % cfg.dash_emit_interval == 0)
    amount = jnp.where(emit, cells.size * physics.dash_emit_frac(cells.size, cfg, jnp), 0.0)
    size = cells.size - amount
    speed_now = jnp.linalg.norm(vel, axis=-1)
    heading = jnp.where((speed_now > 0)[:, None], vel / jnp.maximum(speed_now, 1e-12)[:, None], jnp.array([1.0, 0.0], F32))
    gap = physics.radius(size, cfg, jnp) + physics.radius(amount, cfg, jnp) + 0.2
    slot = jnp.where(emit, start + (world.cursor + jnp.cumsum(emit) - 1) % cap, m)
    o_pos = obj.pos.at[slot].set(jnp.mod(pos - heading * gap[:, None], L), mode="drop")
    o_size = obj.size.at[slot].set(amount, mode="drop")
    o_alive = obj.alive.at[slot].set(True, mode="drop")
    o_owner = obj.owner.at[slot].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    cursor = (world.cursor + jnp.sum(emit)) % cap

    # 세포 간 포식 (닿으면 큰 쪽이 먹음, 같은 크기면 무작위, 동시 판정)
    can_eat = physics.can_eat_matrix(pos, size, alive0, jax.random.uniform(k_prio, (n,)), cfg, jnp)
    eaten, predator = physics.resolve_predation(can_eat, alive0, size, jnp)
    kill_mass = _sum_to(predator, jnp.where(eaten, size, 0.0), n)
    kills = _sum_to(predator, eaten.astype(jnp.int32), n)
    size = size + kill_mass
    alive = alive0 & ~eaten
    vel = jnp.where(eaten[:, None], 0.0, vel)
    dashing = dashing & ~eaten

    # 세포-객체 접촉 (경계면 접촉, 겹치면 큰 세포가 가져감)
    t = obj.type
    dist_co = jnp.linalg.norm(physics.torus_delta(o_pos[None], pos[:, None], L, jnp), axis=-1)
    reach = physics.contact_reach(physics.radius(size, cfg, jnp)[:, None], physics.radius(o_size, cfg, jnp)[None], jnp)
    contact = (dist_co < reach) & alive[:, None] & o_alive[None]
    s = size[:, None]
    edible = (t == FOOD) | (t == CELL_FOOD)
    eligible = contact & (
        edible[None] | ((t == BLACK_HOLE)[None] & (s > cfg.hole_size)) | ((t == WHITE_HOLE)[None] & (s < cfg.hole_size))
    )
    consumed = jnp.any(eligible, axis=0)
    winner = jnp.argmax(jnp.where(eligible, s, -jnp.inf), axis=0)
    gain = jnp.where(t == BLACK_HOLE, -cfg.black_hole_penalty, jnp.where(t == WHITE_HOLE, cfg.white_hole_bonus, cfg.food_gain * o_size))
    gain = jnp.where(consumed, gain, 0.0)
    raw_size = size + _sum_to(winner, gain, n)
    crushed = alive & (raw_size <= 0)  # 블랙홀 여러 개를 한 번에 밟아 0 이하 → 사망
    size = jnp.where(crushed, size, raw_size)  # 죽은 세포는 직전 크기 유지 (관측·렌더링용)
    alive = alive & ~crushed
    vel = jnp.where(crushed[:, None], 0.0, vel)
    dashing = dashing & ~crushed
    food_mass = _sum_to(winner, jnp.where(edible, gain, 0.0), n)
    white = _sum_to(winner, (consumed & (t == WHITE_HOLE)).astype(jnp.int32), n)
    black = _sum_to(winner, (consumed & (t == BLACK_HOLE)).astype(jnp.int32), n)
    o_alive = o_alive & ~consumed

    # 먹힌 밥·홀 재스폰 (고정 슬롯 RESPAWN_SLOTS 개)
    new_cells = Cells(pos, vel, size, alive, dash_steps, dashing)
    pending = ~o_alive & (t != CELL_FOOD)
    idx = jnp.nonzero(pending, size=RESPAWN_SLOTS, fill_value=m)[0]
    safe = jnp.minimum(idx, m - 1)
    new_size = jnp.where(t[safe] == FOOD, _food_sizes(k_food, RESPAWN_SLOTS, cfg), o_size[safe])
    new_pos = spawn_positions(k_spawn, new_cells, physics.radius(new_size, cfg, jnp), cfg)
    o_pos = o_pos.at[idx].set(new_pos, mode="drop")
    o_size = o_size.at[idx].set(new_size, mode="drop")
    o_alive = o_alive.at[idx].set(True, mode="drop")

    died = eaten | crushed
    won = alive0 & ~died & (size_before < cfg.win_size) & (size >= cfg.win_size)
    events = {
        "size_before": size_before,
        "size_after": jnp.where(crushed, raw_size, size),
        "food_mass": food_mass,
        "white_hole": white,
        "black_hole": black,
        "dash_cost": amount,
        "kills": kills,
        "kill_mass": kill_mass,
        "died": died,
        "won": won,
    }
    new_world = World(new_cells, Objects(o_pos, o_size, t, o_alive, o_owner), cursor, world.step + 1)
    return new_world, events


def respawn_cells(world: World, key: jax.Array, mask: jax.Array, cfg: ArenaConfig) -> World:
    """mask 인 세포를 기본 크기·정지 상태로 모든 시야 밖에 되살린다."""
    cells = world.cells
    n = cells.size.shape[0]
    others = cells._replace(alive=cells.alive & ~mask)
    r = jnp.full(n, float(physics.radius(cfg.base_size, cfg)), F32)
    new_pos = spawn_positions(key, others, r, cfg)
    return world._replace(
        cells=Cells(
            pos=jnp.where(mask[:, None], new_pos, cells.pos),
            vel=jnp.where(mask[:, None], 0.0, cells.vel),
            size=jnp.where(mask, cfg.base_size, cells.size),
            alive=cells.alive | mask,
            dash_steps=jnp.where(mask, 0, cells.dash_steps),
            dashing=cells.dashing & ~mask,
        )
    )


def from_numpy_state(state: WorldState) -> World:
    """NumPy WorldState → JAX 월드 (동등성 검증용)."""
    c, o = state.cells, state.objects
    return World(
        cells=Cells(
            pos=jnp.asarray(c.pos, F32),
            vel=jnp.asarray(c.vel, F32),
            size=jnp.asarray(c.size, F32),
            alive=jnp.asarray(c.alive),
            dash_steps=jnp.asarray(c.dash_steps, jnp.int32),
            dashing=jnp.asarray(c.dashing),
        ),
        objects=Objects(
            pos=jnp.asarray(o.pos, F32),
            size=jnp.asarray(o.size, F32),
            type=jnp.asarray(o.type, jnp.int32),
            alive=jnp.asarray(o.alive),
            owner=jnp.asarray(o.owner, jnp.int32),
        ),
        cursor=jnp.int32(state.cell_food_cursor),
        step=jnp.int32(state.step),
    )
