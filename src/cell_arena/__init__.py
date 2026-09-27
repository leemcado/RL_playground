"""Cell Arena — 강화학습 실습용 토러스 세포 대전 환경.

- ``core``: 규칙 수치·물리 공식·입출력 계약·관측 생성 (학습 env 와 대결장 공통)
- ``engine``: 규칙 엔진 — NumPy (대결장) / JAX (학습). 같은 규칙
- ``train``: ``make_env`` 와 학습용 배치 env, YAML 설정
- ``play``: 대결장·관전 화면·사람 조작·참가자 불러오기
- ``StudentAgent``: 학생 에이전트 기반 클래스 (대결·저장 메소드는 잠겨 있다)

환경 자체는 특정 정책(봇 등)에 의존하지 않는다. 연습용 봇은 ``cell_arena.bots`` 에 학생 에이전트와 같은 형식으로 들어 있다.
"""

from cell_arena.hardware import configure_env

configure_env()

from cell_arena.agent import Agent, StudentAgent  # noqa: E402
from cell_arena.core.api import (  # noqa: E402
    IMAGE_CHANNELS,
    OBJECT_FEATURES,
    ActionSpec,
    Events,
    Observation,
    ObsSpec,
    StepOutput,
)
from cell_arena.train import Config, load_config, make_env  # noqa: E402

__all__ = [
    "IMAGE_CHANNELS",
    "OBJECT_FEATURES",
    "ActionSpec",
    "Agent",
    "Config",
    "Events",
    "Observation",
    "ObsSpec",
    "StepOutput",
    "StudentAgent",
    "load_config",
    "make_env",
]
