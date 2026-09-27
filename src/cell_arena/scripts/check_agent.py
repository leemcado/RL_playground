"""학생 에이전트 점검 — 받은 파일이 대결에 바로 들어갈 수 있는지 확인한다 (학생도 제출 전에 돌린다).

    cell-arena-check kim.py [lee.py ...]
    cell-arena-check agent.py --steps 300

1. 불러오기   StudentAgent 하위 클래스 하나, 잠긴 메소드(__init__ / act / save / load) 재정의 없음
2. 가중치     weights 파일로 load (없으면 같은 폴더 config.yaml 로 새 모델을 만들어 형식만 점검)
3. 학습 env   배치 4 로 act(explore=True/False)·reward 형태, 저장 → 불러오기
4. 대결장     기본 연습 상대들과 짧은 경기 (B=1, act 최대 시간)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from cell_arena import Config, StudentAgent, load_config, make_env
from cell_arena.bots import DEFAULT_OPPONENTS
from cell_arena.play import BattleArena, load_agent
from cell_arena.play.loader import agent_class


def check(path: Path, steps: int) -> bool:
    """에이전트 파일 하나 점검. 통과하면 True."""
    stage = "불러오기"
    try:
        cls = agent_class(path.resolve())
        if not issubclass(cls, StudentAgent):
            raise TypeError(f"{cls.__name__} 는 StudentAgent 를 상속해야 한다")

        stage = "모델 만들기"
        start = time.perf_counter()
        weights = path.parent / cls.weights if cls.weights else None
        if weights is not None and weights.exists():
            agent, note = cls.load(weights), f"가중치 {weights.name}"
        else:
            cfg_file = path.with_name("config.yaml")
            agent = cls(load_config(cfg_file) if cfg_file.exists() else None, device="cpu")
            note = f"가중치 없음 ({cls.weights}) — 학습 전 모델로 형식만 점검"
        load_s = time.perf_counter() - start

        stage = "학습 env"
        cfg = Config({"opponents": DEFAULT_OPPONENTS, "num_envs": 4, "max_steps": 50, "seed": 0})
        env = make_env(cfg, agent, backend="numpy", verbose=False)
        obs = env.reset()
        agent.reset(np.ones(4, dtype=bool))
        agent.act(obs, explore=True)
        out = env.step(agent.act(obs, explore=False))
        reward = np.asarray(agent.reward(out.events, out.final_obs))
        if reward.shape != (4,) or not np.isfinite(reward).all():
            raise ValueError(f"reward 는 유한한 (B,) 배열이어야 한다: shape {reward.shape}")

        stage = "저장·불러오기"
        with tempfile.TemporaryDirectory() as tmp:
            cls.load(agent.save(Path(tmp) / "check.pt"))

        stage = "대결장"
        if weights is not None and weights.exists():
            agent = load_agent(str(path))  # 대결장이 쓰는 경로 그대로
        arena = BattleArena([agent] + [load_agent(s) for s in DEFAULT_OPPONENTS], seed=0, max_steps=steps, respawn=True)
        while not arena.done:
            arena.step()
        print(f"[통과] {path}  이름 {agent.name}  색 {agent.color}  관측 {agent.obs_spec.mode}  "
              f"액션 {agent.action_spec.mode}  | {note} | 불러오기 {load_s:.1f}s  act 최대 {arena.act_ms_max[0]:.1f}ms "
              f"({arena.t}스텝, 크기 {arena.state.cells.size[0]:.0f}, 사망 {arena.deaths[0]})")
        return True
    except Exception as err:  # noqa: BLE001 — 어떤 오류든 점검 결과로 보고
        print(f"[실패] {path}  단계 '{stage}': {type(err).__name__}: {err}")
        traceback.print_exc(limit=-3)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="학생 에이전트 점검")
    parser.add_argument("paths", nargs="+", type=Path, help="에이전트 파일 (.py)")
    parser.add_argument("--steps", type=int, default=200, help="점검용 대결 길이")
    args = parser.parse_args()
    results = [check(p, args.steps) for p in args.paths]
    print(f"\n{sum(results)}/{len(results)} 통과")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
