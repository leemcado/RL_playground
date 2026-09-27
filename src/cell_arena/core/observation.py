"""관측 생성 (NumPy). JAX 판은 engine/jax_observation.py — 같은 규칙, 같은 결과.

- ``self_state``: 자기 상태 [크기, x, y, v_x, v_y]
- ``object_list``: 시야 안 객체를 가까운 순으로 M개 + mask (state 모드). 크기는 직경이 아니라 원시 크기로 준다
- ``semantic_image``: 타입별 5채널 0/1 이미지 (image 모드). ``disk_cover`` 는 np / jnp 공용

다른 세포의 속도는 넣지 않는다 (움직임은 여러 프레임을 보고 추론).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from cell_arena.core import physics
from cell_arena.core.api import IMAGE_CHANNEL_CAPS, ObsSpec
from cell_arena.core.config import BLACK_HOLE, WHITE_HOLE, ArenaConfig
from cell_arena.core.state import CellArrays, WorldState

# 객체 타입 → 관측 종류: 0 밥(세포밥 포함), 1 블랙홀, 2 화이트홀, 3 다른 세포, 4 자기 (이미지 채널 번호와 같다)
KIND_CELL, KIND_SELF = 3, 4


def object_kind(obj_type: Any, xp: ModuleType = np) -> Any:
    """객체 타입 배열 → 관측 종류 (세포밥은 밥과 규칙이 같아 밥으로 합친다)."""
    return xp.where(obj_type == BLACK_HOLE, 1, xp.where(obj_type == WHITE_HOLE, 2, 0))


def self_state(cells: CellArrays, idx: int) -> np.ndarray:
    """자기 상태 ``[size, x, y, v_x, v_y]`` (5,) float32 원시 값."""
    return np.array(
        [cells.size[idx], cells.pos[idx, 0], cells.pos[idx, 1], cells.vel[idx, 0], cells.vel[idx, 1]],
        dtype=np.float32,
    )


def _candidates(state: WorldState, idx: int, cfg: ArenaConfig) -> tuple[np.ndarray, ...]:
    """시야에 일부라도 들어오는 모든 객체·세포: 변위 (K, 2), 반지름, 크기, 종류, 거리, 시야 반경 h."""
    cells, obj = state.cells, state.objects
    center = cells.pos[idx]
    h = float(physics.vision_radius(cells.size[idx], cfg))
    d_obj = physics.torus_delta(obj.pos, center, cfg.map_size)
    d_cell = physics.torus_delta(cells.pos, center, cfg.map_size)
    delta = np.concatenate([d_obj, d_cell])
    size = np.concatenate([obj.size, cells.size])
    radius = physics.radius(size, cfg)
    kind = np.concatenate([object_kind(obj.type), np.full(cells.size.shape[0], KIND_CELL)])
    kind[obj.size.shape[0] + idx] = KIND_SELF
    alive = np.concatenate([obj.alive, cells.alive])
    visible = alive & (np.abs(delta) < (h + radius)[:, None]).all(axis=-1)
    dist = np.linalg.norm(delta, axis=-1)
    return delta[visible], radius[visible], size[visible], kind[visible], dist[visible], h


def object_list(state: WorldState, idx: int, cfg: ArenaConfig, max_objects: int) -> tuple[np.ndarray, np.ndarray]:
    """시야 안 객체(밥·홀·다른 세포)를 가까운 순으로 최대 M개.

    Returns:
        objects (M, 7) float32 [dx, dy, 크기, is_food, is_black_hole, is_white_hole, is_cell], mask (M,) bool
    """
    delta, _, size, kind, dist, _ = _candidates(state, idx, cfg)
    others = kind != KIND_SELF
    delta, size, kind, dist = delta[others], size[others], kind[others], dist[others]
    order = np.argsort(dist, kind="stable")[:max_objects]
    k = order.size
    feats = np.zeros((max_objects, 7), dtype=np.float32)
    feats[:k, 0:2] = delta[order]
    feats[:k, 2] = size[order]
    feats[np.arange(k), 3 + kind[order]] = 1.0
    return feats, np.arange(max_objects) < k


def disk_cover(delta: Any, radius: Any, valid: Any, h: Any, res: int, xp: ModuleType = np) -> Any:
    """원 K개가 덮는 픽셀 (R, R) bool, [y, x].

    시야 정사각형 [-h, h]² 를 R×R 픽셀로 나누고, 픽셀 중심이 원 안(거리 < 반지름)이면 1.
    원이 픽셀 중심을 하나도 덮지 못할 만큼 작아도 원 중심이 있는 픽셀은 1 (최소 1픽셀).
    (K, R, R) 비교를 원 축으로 줄이는 형태라 XLA 가 한 번에 합쳐 계산한다 (scatter 없음 — 옛 jax·GPU 에서도 빠름).

    Args:
        delta: (K, 2) 시야 중심 기준 변위, radius: (K,), valid: (K,) 그릴지
        h: 시야 반경, res: 해상도 R, xp: np 또는 jnp
    """
    pix = 2.0 * h / res
    centers = (xp.arange(res) + 0.5) * pix - h  # 픽셀 중심 오프셋 (R,)
    d2y = (centers[None, :] - delta[:, 1:2]) ** 2  # (K, R)
    d2x = (centers[None, :] - delta[:, 0:1]) ** 2
    inside = d2y[:, :, None] + d2x[:, None, :] < (radius**2)[:, None, None]
    ix = xp.floor((delta[:, 0] + h) / pix).astype(xp.int32)
    iy = xp.floor((delta[:, 1] + h) / pix).astype(xp.int32)
    pixel = xp.arange(res)
    point = (pixel[None, :, None] == iy[:, None, None]) & (pixel[None, None, :] == ix[:, None, None])
    return xp.any(valid[:, None, None] & (inside | point), axis=0)


def semantic_image(state: WorldState, idx: int, cfg: ArenaConfig, resolution: int) -> np.ndarray:
    """자기 시야의 타입별 5채널 이미지 (5, R, R) uint8 0/1 — api.IMAGE_CHANNELS 순서.

    채널마다 가까운 순으로 IMAGE_CHANNEL_CAPS 개까지 그린다 (JAX 판과 같은 규칙).
    """
    delta, radius, _, kind, dist, h = _candidates(state, idx, cfg)
    image = np.zeros((5, resolution, resolution), dtype=np.uint8)
    for c, cap in enumerate(IMAGE_CHANNEL_CAPS):
        sel = np.flatnonzero(kind == c)
        sel = sel[np.argsort(dist[sel], kind="stable")[:cap]]
        image[c] = disk_cover(delta[sel], radius[sel], np.ones(sel.size, dtype=bool), h, resolution)
    return image


def build_observation(state: WorldState, idx: int, spec: ObsSpec, cfg: ArenaConfig) -> dict[str, np.ndarray]:
    """세포 하나의 관측 (배치 차원 없음). Observation.stack 으로 배치를 만든다."""
    out = {"self_state": self_state(state.cells, idx)}
    if spec.mode == "state":
        out["objects"], out["mask"] = object_list(state, idx, cfg, spec.max_objects)
    else:
        out["image"] = semantic_image(state, idx, cfg, spec.resolution)
    return out
