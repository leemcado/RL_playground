"""에이전트 ↔ 환경 고정 계약 (학습 env / 대결장 공통).

- 모든 배열은 배치 우선 ``(B, ...)`` NumPy 배열이다. 대결장에서는 B=1.
- 관측은 자기 상태와 자기 시야 안의 객체만 담는다 (POMDP). 전역 맵·시야 밖 정보·다른 세포의 속도는 없다.
  봇도 같은 관측만 받는다.
- 환경은 보상을 주지 않는다. 자기 세포에게 일어난 사건 ``Events`` 만 주고, 보상은 에이전트가 설계한다.

좌표계: 자기 세포 중심 기준 상대좌표, 화면과 같은 y-down (위가 -y). 연속 액션의 θ = atan2(dy, dx).
크기에서 정해지는 값 (관측에 따로 넣지 않음): 직경 d = 2·√(size/100), 시야 반경 h = min(8·√(size/100), 25)
(한 변 2h 정사각형). 세포끼리 경계면이 닿으면 큰 쪽이 먹는다 (같은 크기면 무작위).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

ObsMode = Literal["state", "image"]
ActionMode = Literal["discrete", "multibinary", "continuous"]

SELF_FEATURES = ("size", "x", "y", "v_x", "v_y")
OBJECT_FEATURES = ("dx", "dy", "size", "is_food", "is_black_hole", "is_white_hole", "is_cell")
IMAGE_CHANNELS = ("food", "black_hole", "white_hole", "other_cell", "self")
NUM_DISCRETE_ACTIONS = 18
# 이미지 채널별로 그리는 객체 수 상한 (가까운 순). JAX 판의 고정 shape 를 위한 값으로 두 엔진이 같은 규칙을 쓴다.
# 밥은 시야가 가장 넓을 때(한 변 50) 보통 40~120개, 홀은 종류별 15개가 전부, 세포는 8명 판 기준
IMAGE_CHANNEL_CAPS = (128, 16, 16, 16, 1)


@dataclass(frozen=True)
class ObsSpec:
    """에이전트가 받을 관측 형태. 에이전트가 선언하고 env 가 맞춰서 만든다.

    Attributes:
        mode: "state" = 거리순 객체 목록, "image" = 타입별 5채널 0/1 이미지. 둘 다 자기 상태 벡터를 함께 준다
        max_objects: state 모드 객체 수 M (가까운 순, 부족분은 0 패딩 + mask)
        resolution: image 모드 해상도 R (시야 정사각형을 R×R 로)
    """

    mode: ObsMode = "state"
    max_objects: int = 32
    resolution: int = 64

    def __post_init__(self) -> None:
        if self.mode not in ("state", "image"):
            raise ValueError(f"ObsSpec.mode 는 'state' 또는 'image': {self.mode!r}")
        if self.max_objects < 1 or self.resolution < 8:
            raise ValueError(f"max_objects ≥ 1, resolution ≥ 8 이어야 한다: {self}")


@dataclass(frozen=True)
class ActionSpec:
    """에이전트 출력 형태.

    - "discrete": (B,) int ∈ [0, 18). 0~8 = 정지·상·하·좌·우·좌상·우상·좌하·우하, 9~17 = 같은 순서 + 돌진
    - "multibinary": (B, 5) bool ``[상, 하, 좌, 우, 돌진]``
    - "continuous": (B, 3) float ``[θ, move, dash]`` — move·dash 는 0.5 초과면 켜짐
    """

    mode: ActionMode = "discrete"

    def __post_init__(self) -> None:
        if self.mode not in ("discrete", "multibinary", "continuous"):
            raise ValueError(f"ActionSpec.mode 는 discrete/multibinary/continuous: {self.mode!r}")


def _stack(cls: type, rows: list[dict[str, np.ndarray]]) -> object:
    """행(dict) 리스트를 배치 dataclass 로 쌓는다. 행에 없는 필드는 None."""
    return cls(**{f.name: np.stack([r[f.name] for r in rows]) for f in fields(cls) if f.name in rows[0]})


@dataclass
class Observation:
    """배치 관측.

    Attributes:
        self_state: (B, 5) float32 — SELF_FEATURES [크기, x, y, v_x, v_y] 원시 값 (x, y 는 맵 좌표 0~100)
        objects: (B, M, 7) float32 — state 모드. OBJECT_FEATURES [dx, dy, 크기, 타입 원핫 4]
        mask: (B, M) bool — state 모드. 실제 객체 자리 True
        image: (B, 5, R, R) uint8 0/1 — image 모드. IMAGE_CHANNELS 순서, [채널, y, x]
    """

    self_state: np.ndarray
    objects: np.ndarray | None = None
    mask: np.ndarray | None = None
    image: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.self_state.shape[0])

    @classmethod
    def stack(cls, rows: list[dict[str, np.ndarray]]) -> Observation:
        """관측 하나씩의 dict 리스트 → 배치 관측."""
        return _stack(cls, rows)  # type: ignore[return-value]


@dataclass
class Events:
    """자기 세포에게 이번 스텝 일어난 사건 (각 (B,)). 보상 설계의 재료.

    Attributes:
        size_before: 스텝 전 크기
        size_after: 스텝 후 크기. 먹혀 죽으면 먹히기 직전 크기, 블랙홀로 0 이하가 되어 죽으면 그 값 (음수 가능)
        food_mass: 밥·세포밥으로 얻은 질량
        white_hole: 화이트홀 발동 횟수 (+100 씩)
        black_hole: 블랙홀 발동 횟수 (-250 씩)
        dash_cost: 돌진 방출로 잃은 질량
        kills: 잡아먹은 세포 수
        kill_mass: 잡아먹어 얻은 질량
        died: 죽었는지 (먹힘 또는 블랙홀로 크기 0 이하). 학습 env 에서는 곧바로 리스폰하고 에피소드는 계속된다
        won: 이번 스텝에 승리 크기(4000)를 넘었는지
        t: 에피소드 경과 스텝 (이번 스텝 포함)
    """

    size_before: np.ndarray
    size_after: np.ndarray
    food_mass: np.ndarray
    white_hole: np.ndarray
    black_hole: np.ndarray
    dash_cost: np.ndarray
    kills: np.ndarray
    kill_mass: np.ndarray
    died: np.ndarray
    won: np.ndarray
    t: np.ndarray

    @classmethod
    def stack(cls, rows: list[dict[str, np.ndarray]]) -> Events:
        """사건 하나씩의 dict 리스트 → 배치 사건."""
        return _stack(cls, rows)  # type: ignore[return-value]


# 엔진 step() 이 세포별로 돌려주는 사건 키 (Events 에서 t 를 뺀 것)
EVENT_KEYS = tuple(f.name for f in fields(Events) if f.name != "t")


@dataclass
class StepOutput:
    """학습 env 한 스텝 결과.

    Attributes:
        obs: 다음 행동에 쓸 관측. 끝난 env 는 자동 리셋된 새 에피소드의 첫 관측
        final_obs: 이번 스텝 직후 관측 (끝난 env 도 리셋 전). 보상 계산·부트스트랩용
        events: 학습 세포의 사건
        terminated: (B,) 어떤 세포든 승리 크기에 도달해 게임이 끝남 — 부트스트랩하지 않는다
        truncated: (B,) max_steps 도달로 잘림 — final_obs 로 부트스트랩한다
    """

    obs: Observation
    final_obs: Observation
    events: Events
    terminated: np.ndarray
    truncated: np.ndarray
