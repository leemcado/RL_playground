"""YAML 설정 — 학습 env 에 들어갈 상대 구성과 학습 하이퍼파라미터.

make_env 가 읽는 키 (필수): opponents, num_envs, max_steps, seed. 나머지 키는 학생이 자유롭게 추가해
코드에서 ``cfg.<키>`` 로 읽는다. 게임 규칙은 설정할 수 없다 (모두가 같은 규칙에서 대결).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """PyYAML 은 '3e-4' 처럼 소수점 없는 지수 표기를 문자열로 읽는다 → 실수로 읽게 한다."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d[\d_]*)(?:\.\d*)?[eE][-+]?\d+$"),
    list("-+0123456789"),
)


class Config:
    """설정 값 묶음. ``cfg.lr`` 처럼 속성으로 읽고, 중첩된 dict 도 속성으로 읽는다.

    Args:
        data: 설정 dict
    """

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):  # _data 가 없는 상태(복사·역직렬화 중)에서 무한 재귀 방지
            raise AttributeError(key)
        if key not in self._data:
            raise AttributeError(f"설정에 '{key}' 키가 없다 (config.yaml 에 추가). 있는 키: {sorted(self._data)}")
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """없으면 default."""
        return getattr(self, key) if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        """wandb config·가중치 파일 기록용 dict (복사본)."""
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self._data.items()}

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(path: str | Path) -> Config:
    """YAML 파일 → Config.

    Raises:
        FileNotFoundError: 파일 없음
        ValueError: 최상위가 key: value 형식이 아님
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.load(f, Loader=_Loader) or {}  # noqa: S506 — SafeLoader 하위 클래스
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 최상위는 'key: value' 형식이어야 한다")
    return Config(data)
