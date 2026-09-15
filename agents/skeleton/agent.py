"""Cell Arena 학생 에이전트 — 이 파일 하나가 학습 코드이자 제출물이다.

    학습   python agents/skeleton/agent.py                     (같은 폴더의 config.yaml)
           python agents/skeleton/agent.py --config dqn.yaml
    점검   python scripts/check_agent.py agents/skeleton/agent.py
    제출   이 파일(<이름>.py) + 가중치 파일(weights)

채울 곳 (TODO)
    ① 입출력 형태     obs_spec / action_spec
    ② 모델           setup()       인코더 + 알고리즘별 머리 (DQN / Dueling DQN / PPO)
    ③ 전처리·피처     preprocess()
    ④ 행동 선택       policy()
    ⑤ 보상           reward()
    ⑥ 학습 루프       train()       데이터 흐름(버퍼·업데이트 주기)과 로그(wandb)
고칠 수 없는 것: __init__, act, save, load — StudentAgent 가 정한다 (재정의하면 불러올 때 오류)
API 설명: agents/skeleton/README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from cell_arena import ActionSpec, Config, Events, Observation, ObsSpec, StudentAgent, load_config, make_env


class MyAgent(StudentAgent):
    # ⓪ 대결에 나갈 이름·색·가중치 파일 (가중치는 이 파일과 같은 폴더에 저장되고 대결 때 여기서 불러온다)
    name = "my_agent"
    color = (90, 160, 250)  # (R, G, B) 또는 "#5aa0fa"
    weights = "my_agent.pt"

    # ① 입출력 형태
    #   관측  "state": self_state (B, 5) + objects (B, M, 7) + mask (B, M)
    #         "image": self_state (B, 5) + image (B, 5, R, R) uint8 0/1 — [food, black_hole, white_hole, other_cell, self]
    #   액션  "discrete": (B,) ∈ [0, 18)  /  "multibinary": (B, 5)  /  "continuous": (B, 3) [θ, move, dash]
    obs_spec = ObsSpec(mode="state", max_objects=32)
    action_spec = ActionSpec(mode="discrete")

    # ② 모델 — self.cfg (config.yaml), self.device (cuda > mps > cpu) 가 준비된 뒤 호출된다.
    #    속성으로 둔 torch.nn.Module 은 모두 save() 에 저장되고, load() 는 여기서 모델을 다시 만든 뒤 가중치를 채운다.
    def setup(self) -> None:
        # TODO: 인코더 — preprocess() 의 출력 → 특징 벡터 (MLP, CNN, ...)
        # TODO: 머리 — DQN: Q(s, ·) / Dueling DQN: V(s) + A(s, ·) / PPO: 정책 π(·|s) + 가치 V(s)
        #   예) self.net = MyNetwork(...).to(self.device)
        raise NotImplementedError("setup(): 인코더와 머리를 만든다")

    # ③ 전처리·피처 엔지니어링 — 관측 (NumPy, 배치 우선) → 신경망 입력 (torch.Tensor, self.device)
    #    - 원시 값이라 스케일이 제각각이다 (크기 100~4000, 좌표 0~100, 속도 약 ±2, 객체 dx·dy 는 시야 ±25)
    #    - 다른 세포의 속도는 관측에 없다 → 여러 프레임을 쌓거나 RNN (기억은 reset 에서 지운다)
    #    - objects 는 거리순이라 가까운 순위가 바뀌면 같은 객체가 다른 자리로 옮겨 간다
    def preprocess(self, obs: Observation) -> torch.Tensor:
        raise NotImplementedError("preprocess(): 관측 → 신경망 입력")

    # ④ 행동 선택 — explore=True: 학습 중 탐색 (ε-greedy, 확률적 샘플링 ...), False: 대결에서 쓰는 행동
    def policy(self, x: torch.Tensor, explore: bool) -> np.ndarray:
        raise NotImplementedError("policy(): 신경망 입력 → 행동")

    # (선택) 기억 초기화 — 에피소드 시작·리스폰·대결 시작 때 done=True 인 배치 원소
    def reset(self, done: np.ndarray) -> None:
        pass

    # ⑤ 보상 — 내 세포의 사건(events)과 관측(obs)만 쓸 수 있다. 각 (B,)
    #    size_before, size_after, food_mass, white_hole, black_hole, dash_cost, kills, kill_mass, died, won, t
    def reward(self, events: Events, obs: Observation) -> np.ndarray:
        # TODO: 설계. 아래는 출발점 (크기 변화/100 − 사망 + 5·승리)
        return (events.size_after - events.size_before) / 100.0 - 1.0 * events.died + 5.0 * events.won


# ⑥ 학습 루프 — 데이터 흐름과 로그를 직접 짠다
def train(cfg: Config) -> None:
    agent = MyAgent(cfg)  # 장치를 고르려면 MyAgent(cfg, device="cpu")
    env = make_env(cfg, agent)  # config.yaml 의 opponents / num_envs / max_steps / seed, 하드웨어는 자동
    run = wandb.init(project="cell-arena", name=agent.name, config=cfg.to_dict())
    # TODO: 옵티마이저, 버퍼 (DQN: replay buffer / PPO: rollout buffer)

    obs = env.reset()
    agent.reset(np.ones(cfg.num_envs, dtype=bool))
    samples = 0
    while samples < cfg.total_samples:
        action = agent.act(obs, explore=True)  # PPO 처럼 log π(a|s)·V(s) 도 필요하면 신경망을 직접 부른다
        out = env.step(action)
        reward = agent.reward(out.events, out.final_obs)  # noqa: F841
        samples += cfg.num_envs

        # TODO: 전이 저장 — (obs, action, reward, out.final_obs, out.terminated, out.truncated)
        #   terminated : 누군가 승리 크기(4000) 도달 → 게임 끝. 다음 상태 가치로 부트스트랩하지 않는다
        #   truncated  : max_steps 도달 → 잘렸을 뿐이다. out.final_obs 로 부트스트랩한다
        #   events.died: 내가 먹혀도 에피소드는 계속된다 (곧바로 시야 밖에서 크기 100 으로 리스폰)
        #   out.obs 는 다음 행동용 — 끝난 env 는 이미 새 에피소드의 첫 관측이다
        # TODO: 업데이트 — DQN: 미니배치 샘플·타깃 네트워크 동기화·ε 스케줄 / PPO: 롤아웃이 차면 GAE → 여러 에폭
        # TODO: 로그 — 무엇을 볼지 정해서 run.log({...}, step=samples)
        #   예) 에피소드 보상·길이, 최종 크기, 사망 횟수, 손실, Q 값, 엔트로피, 초당 샘플 수 ...

        agent.reset(out.terminated | out.truncated | out.events.died)  # 새 에피소드·리스폰 → 기억 초기화
        obs = out.obs

    agent.save()  # 이 파일 폴더의 weights 로 저장 (중간 저장은 agent.save("경로.pt"))
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Arena 학생 에이전트 학습")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"), help="YAML 설정 파일")
    train(load_config(parser.parse_args().config))
