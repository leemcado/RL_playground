"""액션 디코딩: 에이전트 액션 (ActionSpec 별) → 엔진 내부 액션 (방향, 이동, 돌진).

모든 모드가 같은 내부 표현으로 바뀌므로 이산·연속 에이전트가 한 경기장에서 대결할 수 있다.
"""

from __future__ import annotations

import numpy as np

from cell_arena.core.api import NUM_DISCRETE_ACTIONS, ActionSpec

# Discrete(18) → MultiBinary(5). 인덱스 = 9 × 돌진 + 이동
_MOVES = np.array(
    [
        [0, 0, 0, 0],  # 정지
        [1, 0, 0, 0],  # 상
        [0, 1, 0, 0],  # 하
        [0, 0, 1, 0],  # 좌
        [0, 0, 0, 1],  # 우
        [1, 0, 1, 0],  # 좌상
        [1, 0, 0, 1],  # 우상
        [0, 1, 1, 0],  # 좌하
        [0, 1, 0, 1],  # 우하
    ],
    dtype=bool,
)
DISCRETE_TO_MULTIBINARY = np.concatenate(
    [
        np.concatenate([_MOVES, np.zeros((9, 1), bool)], axis=1),
        np.concatenate([_MOVES, np.ones((9, 1), bool)], axis=1),
    ]
)
_SHAPES = {"discrete": (), "multibinary": (5,), "continuous": (3,)}


def check_action(action: np.ndarray, spec: ActionSpec, batch_size: int) -> np.ndarray:
    """액션 형태 검사 후 NumPy 배열로 돌려준다.

    Raises:
        ValueError: 형태가 ActionSpec 과 다르거나 discrete 인덱스가 범위를 벗어남
    """
    a = np.asarray(action)
    expected = (batch_size, *_SHAPES[spec.mode])
    if a.shape != expected:
        raise ValueError(f"{spec.mode} 액션은 {expected} 이어야 한다 (배치 B={batch_size}): {a.shape}")
    if spec.mode == "discrete":
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"discrete 액션은 정수여야 한다: {a.dtype}")
        if ((a < 0) | (a >= NUM_DISCRETE_ACTIONS)).any():
            raise ValueError(f"discrete 액션은 0~{NUM_DISCRETE_ACTIONS - 1}: {a}")
    elif spec.mode == "continuous" and not np.isfinite(a).all():
        raise ValueError(f"continuous 액션에 NaN/inf 가 있다: {a}")
    return a


def _multibinary_to_internal(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MultiBinary(5) → (방향, 이동, 돌진). y-down 이라 '상' 이 -y, 대각선은 1/√2, 대치 방향은 상쇄."""
    a = np.asarray(action, dtype=bool).reshape(-1, 5).astype(np.float64)
    vec = np.stack([a[:, 3] - a[:, 2], a[:, 1] - a[:, 0]], axis=-1)
    norm = np.linalg.norm(vec, axis=-1)
    moving = norm > 0
    direction = vec / np.where(moving, norm, 1.0)[:, None]
    return direction, moving, a[:, 4] > 0


def decode_action(action: np.ndarray, spec: ActionSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """스펙별 에이전트 액션 → 내부 액션.

    Returns:
        direction (B, 2) 단위벡터 (정지면 0), moving (B,), dash (B,)
    """
    if spec.mode == "multibinary":
        return _multibinary_to_internal(action)
    if spec.mode == "discrete":
        idx = np.asarray(action).astype(np.int64).reshape(-1)
        return _multibinary_to_internal(DISCRETE_TO_MULTIBINARY[idx])
    a = np.asarray(action, dtype=np.float64).reshape(-1, 3)
    moving = a[:, 1] > 0.5
    direction = np.stack([np.cos(a[:, 0]), np.sin(a[:, 0])], axis=-1) * moving[:, None]
    return direction, moving, a[:, 2] > 0.5
