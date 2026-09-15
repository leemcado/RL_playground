"""에이전트 기반 클래스.

- ``Agent``: 봇·사람 조작 등 모든 참가자가 따르는 최소 API (``obs_spec``, ``action_spec``, ``reset``, ``act``).
- ``StudentAgent``: 학생 에이전트용. 대결·저장에 쓰이는 메소드(``__init__``, ``act``, ``save``, ``load``)는
  이 클래스가 정하고 하위 클래스에서 재정의할 수 없다. 학생은 ``setup`` / ``preprocess`` / ``policy`` /
  ``reward`` (+ 선택 ``reset``) 를 채운다.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cell_arena.core.actions import check_action
from cell_arena.core.api import ActionSpec, Events, Observation, ObsSpec
from cell_arena.hardware import torch_device
from cell_arena.train.config import Config

PROTECTED = ("__init__", "act", "save", "load")
CHECKPOINT_FORMAT = 1


class Agent(ABC):
    """참가자 공통 틀. ``act`` 는 학습 env (B = env 수) 와 대결장 (B = 1) 에서 똑같이 호출된다."""

    name: str = "agent"  # 대결·화면에 표시되는 이름
    color: tuple[int, int, int] | str | None = None  # 표시 색 (R, G, B) 또는 "#RRGGBB". None 이면 기본 팔레트
    obs_spec: ObsSpec = ObsSpec()
    action_spec: ActionSpec = ActionSpec()

    def reset(self, done: np.ndarray) -> None:
        """done (B,) 인 배치 원소의 기억(프레임 스택·RNN 등)을 지운다. 에피소드 시작·리스폰 때 호출된다."""

    @abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """관측 → 행동 (ActionSpec 형태, 배치)."""


class StudentAgent(Agent):
    """학생 에이전트 기반 클래스.

    학생이 채우는 것:
        setup()                     모델 생성 (self.cfg, self.device 사용 가능)
        preprocess(obs) -> x        관측 → 신경망 입력 (전처리·피처 엔지니어링)
        policy(x, explore) -> a     신경망 입력 → 행동 (explore=False 가 대결 때 쓰는 행동)
        reward(events, obs) -> r    보상 설계 (학습 루프에서 직접 호출)
        reset(done)                 (선택) 기억 초기화

    프레임워크가 정하는 것 (재정의 금지):
        __init__(cfg, device)       self.cfg, self.device 를 준비하고 setup() 호출
        act(obs, explore)           preprocess → policy (그래디언트 없이), 액션 형태 검사
        save(path) / load(path)     가중치 파일 형식 — 에이전트 속성 중 torch.nn.Module 을 모두 저장·복원

    Args:
        cfg: 설정 (config.yaml 을 load_config 로 읽은 것, 또는 dict)
        device: torch 장치. None 이면 자동 (cuda > mps > cpu)
    """

    weights: str | None = None  # 가중치 파일 이름 (이 에이전트 파일과 같은 폴더). 대결 때 이 파일로 load

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        broken = [m for m in PROTECTED if m in vars(cls)]
        if broken:
            hint = " (준비 코드는 setup() 에 쓴다)" if "__init__" in broken else ""
            raise TypeError(f"{cls.__name__}: {', '.join(broken)} 은(는) 재정의할 수 없다{hint}")
        if not isinstance(cls.obs_spec, ObsSpec) or not isinstance(cls.action_spec, ActionSpec):
            raise TypeError(f"{cls.__name__}: obs_spec 은 ObsSpec(...), action_spec 은 ActionSpec(...) 이어야 한다")

    def __init__(self, cfg: Config | dict[str, Any] | None = None, device: str | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, Config) else Config(cfg or {})
        self.device = torch_device(device)
        self.setup()

    # ------------------------------------------------------------------ 학생이 채우는 곳

    @abstractmethod
    def setup(self) -> None:
        """모델(torch.nn.Module)을 만들어 속성으로 둔다. 속성인 nn.Module 은 모두 가중치 파일에 저장된다."""

    @abstractmethod
    def preprocess(self, obs: Observation) -> Any:
        """관측 → 신경망 입력."""

    @abstractmethod
    def policy(self, x: Any, explore: bool) -> Any:
        """신경망 입력 → 행동 (ActionSpec 형태, np.ndarray 또는 torch.Tensor)."""

    @abstractmethod
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        """사건·관측 → 보상 (B,)."""

    # ------------------------------------------------------------------ 프레임워크 (재정의 금지)

    def act(self, obs: Observation, explore: bool = False) -> np.ndarray:
        """관측 → 행동. 대결장은 explore=False 로 부른다."""
        import torch

        with torch.no_grad():
            action = self.policy(self.preprocess(obs), explore)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return check_action(action, self.action_spec, obs.batch_size)

    def save(self, path: str | Path | None = None) -> Path:
        """가중치 저장: 속성인 torch.nn.Module 전부 + 설정 + 입출력 형태.

        Args:
            path: 저장 경로. None 이면 이 에이전트 파일 폴더의 ``weights`` (없으면 ``<name>.pt``)

        Returns:
            저장한 경로
        """
        import torch

        modules = self._torch_modules()
        if not modules:
            raise RuntimeError(f"{type(self).__name__}: 저장할 torch.nn.Module 속성이 없다 (setup() 에서 self.<이름> = 모델)")
        path = Path(path) if path is not None else self._home() / (self.weights or f"{self.name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "agent": type(self).__name__,
                "name": self.name,
                "config": self.cfg.to_dict(),
                "obs_spec": asdict(self.obs_spec),
                "action_spec": asdict(self.action_spec),
                "state_dicts": {key: m.state_dict() for key, m in modules.items()},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = "cpu") -> StudentAgent:
        """가중치 파일로 에이전트를 만든다 (저장할 때의 설정으로 setup → 가중치 복원 → eval 모드).

        Raises:
            ValueError: 입출력 형태나 모델 속성이 저장할 때와 다름
        """
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: StudentAgent.save 로 만든 가중치 파일이 아니다")
        for key, spec in (("obs_spec", cls.obs_spec), ("action_spec", cls.action_spec)):
            if ckpt[key] != asdict(spec):
                raise ValueError(f"{path}: 저장할 때 {key}={ckpt[key]} 인데 지금 클래스는 {asdict(spec)}")
        agent = cls(ckpt["config"], device=device)
        modules = agent._torch_modules()
        if set(modules) != set(ckpt["state_dicts"]):
            raise ValueError(f"{path}: 모델 속성이 다르다 — 파일 {sorted(ckpt['state_dicts'])}, setup() {sorted(modules)}")
        for key, module in modules.items():
            module.load_state_dict(ckpt["state_dicts"][key])
            module.eval()
        return agent

    def _torch_modules(self) -> dict[str, Any]:
        import torch

        return {k: v for k, v in vars(self).items() if isinstance(v, torch.nn.Module)}

    def _home(self) -> Path:
        return Path(inspect.getfile(type(self))).resolve().parent
