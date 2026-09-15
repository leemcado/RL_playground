"""월드 상태 배열 (청사진 8절 레이아웃).

모든 배열은 고정 크기이며 ``alive`` 마스크로 활성 여부를 표시한다.
동적 할당이 없으므로 JAX pytree 로 그대로 옮길 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellArrays:
    """세포 배열. N = 세포 수.

    Attributes:
        pos: 위치 (N, 2)
        vel: 속도 (N, 2)
        size: 크기 (N,). 죽은 세포는 먹히기 직전 크기를 유지한다.
        alive: 생존 여부 (N,)
        dash_steps: 누적 돌진 스텝 수, 방출 주기 계산용 (N,)
        dashing: 직전 스텝에 돌진이 실제로 적용됐는지 (N,)
    """

    pos: np.ndarray
    vel: np.ndarray
    size: np.ndarray
    alive: np.ndarray
    dash_steps: np.ndarray
    dashing: np.ndarray


@dataclass
class ObjectArrays:
    """비세포 객체 배열. M = ArenaConfig.num_objects(N).

    인덱스 구간: [밥 | 블랙홀 | 화이트홀 | 세포밥 링버퍼].
    밥·홀은 먹히면 즉시 재스폰되어 항상 alive, 세포밥만 alive 가 바뀐다.

    Attributes:
        pos: 위치 (M, 2)
        size: 크기 (M,)
        type: 객체 타입 (M,) — config.FOOD 등
        alive: 활성 여부 (M,)
        owner: 세포밥을 방출한 세포 인덱스, 그 외 -1 (M,)
    """

    pos: np.ndarray
    size: np.ndarray
    type: np.ndarray
    alive: np.ndarray
    owner: np.ndarray


@dataclass
class WorldState:
    """전체 월드 상태."""

    cells: CellArrays
    objects: ObjectArrays
    step: int
    cell_food_cursor: int  # 링버퍼 다음 쓰기 위치
