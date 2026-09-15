"""사람 조작 에이전트 (키보드). ArenaViewer 가 pygame 을 초기화한 뒤에 써야 한다."""

from __future__ import annotations

import numpy as np
import pygame

from cell_arena.agent import Agent
from cell_arena.core.api import ActionSpec, Observation, ObsSpec


class HumanAgent(Agent):
    """W/A/S/D 이동 + SPACE 돌진 → MultiBinary(5)."""

    name = "human"
    color = (80, 220, 160)
    obs_spec = ObsSpec(mode="state", max_objects=1)  # 사람은 관측을 쓰지 않는다 (계산 최소화)
    action_spec = ActionSpec("multibinary")

    def __init__(self) -> None:
        self.last_action = np.zeros(5, dtype=bool)

    def act(self, obs: Observation) -> np.ndarray:
        keys = pygame.key.get_pressed()
        self.last_action = np.array(
            [keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d], keys[pygame.K_SPACE]],
            dtype=bool,
        )
        return self.last_action[None]
