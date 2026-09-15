"""Cell Arena 게임 규칙 (청사진 v1.0 + README 의 변경 사항).

에피소드 길이와 보상은 여기 없다. 길이는 env(학습: 하이퍼파라미터, 실전: 무한)가,
보상은 학생 에이전트가 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

# 객체 타입 (objects.type 배열 값). 정수로 두어 JAX 배열에 그대로 담기게 한다.
FOOD = 0
BLACK_HOLE = 1
WHITE_HOLE = 2
CELL_FOOD = 3


@dataclass(frozen=True)
class ArenaConfig:
    """게임 규칙."""

    # 1.1 세계
    map_size: float = 100.0
    base_size: float = 100.0  # 시작·리스폰 크기
    win_size: float = 4000.0  # 승리 크기 (청사진 1000 에서 변경)
    radius_scale: float = 2.0  # 청사진 d(s) 대비 객체 반경 배율 (충돌·관측·표시). 시야에는 미적용

    # 1.2 객체
    num_food: int = 300
    food_mean: float = 8.0
    food_max: float = 40.0
    food_gain: float = 0.5
    num_black_holes: int = 15
    num_white_holes: int = 15
    hole_size: float = 500.0
    black_hole_penalty: float = 250.0
    white_hole_bonus: float = 100.0
    cell_food_per_cell: int = 70  # 세포밥 링버퍼 칸 수 = 세포 수 × 이 값

    # 1.3 포식: 닿으면 s_a > ratio × s_b 인 쪽이 먹는다. 1.0 = 큰 쪽이 무조건 (같은 크기면 무작위).
    # 청사진 원안은 1.1 (비슷한 크기끼리는 아무 일도 없는 교착 대역).
    predation_ratio: float = 1.0
    # 크기 하한 없음: 블랙홀 여러 개를 한 번에 밟아 크기가 0 이하가 되면 사망

    # 2 운동 모델
    speed_exponent: float = -0.19
    dash_speed_base: float = 1.1  # 돌진 속도 v_dash(s) = 1.1 × (s/100)^0.2 — 클수록 빠름 (추격 성립)
    dash_speed_exponent: float = 0.2
    kappa_base: float = 0.35
    kappa_exponent: float = -0.25
    dash_kappa_scale: float = 0.5
    dash_emit_interval: int = 4
    dash_emit_frac: float = 0.01  # 방출 비율 f(s) = 1% × (s/100)^0.5 — 클수록 빨리 잃고 큰 세포밥
    dash_emit_exponent: float = 0.5
    min_dash_size: float = 100.0

    # 3.1 시야
    vision_coef: float = 8.0
    vision_cap: float = 25.0

    # 1.4 스폰
    spawn_candidates: int = 32

    @property
    def cell_food_start(self) -> int:
        """세포밥 링버퍼가 시작되는 objects 인덱스."""
        return self.num_food + self.num_black_holes + self.num_white_holes

    def num_objects(self, num_cells: int) -> int:
        """objects 배열 총 길이 (세포 수에 따라 세포밥 칸이 늘어난다)."""
        return self.cell_food_start + self.cell_food_per_cell * num_cells
