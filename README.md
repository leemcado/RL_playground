# Cell Arena

강화학습(DQN · Dueling DQN · PPO) 실습용 세포 대전 환경.
`agents/skeleton/agent.py` 하나에 에이전트와 학습 루프를 구현해 학습하고, 서로 대결한다.

## 설치

```bash
pip install -e .            # Linux + NVIDIA GPU 는 pip install -e ".[cuda]"
python scripts/doctor.py    # 설치 점검
wandb login
```

## 할 일

`agents/skeleton/agent.py` 와 `config.yaml` 만 고친다. API 는 [`agents/skeleton/README.md`](agents/skeleton/README.md).

- `obs_spec` / `action_spec` — 관측(`state` / `image`)과 액션 형태
- `setup` — 모델 (DQN / Dueling DQN / PPO)
- `preprocess` — 관측 → 신경망 입력
- `policy` — 행동 선택
- `reward` — 보상
- `train` — 학습 루프 (버퍼, 업데이트, wandb 로그)

`__init__` · `act` · `save` · `load` 는 고칠 수 없다. `config.yaml` 에는 상대 봇 목록과 하이퍼파라미터를 적는다.

```bash
python agents/skeleton/agent.py                         # 학습
python scripts/check_agent.py agents/skeleton/agent.py  # 제출 전 점검
python scripts/play.py                                  # 직접 플레이 (WASD 이동, SPACE 돌진)
```

제출: `agent.py` 를 `<이름>.py` 로 바꿔 가중치 `<이름>.pt` 와 함께 낸다.

## 게임 규칙

`s` = 세포 크기.

| | |
|---|---|
| 맵 | 100 × 100 토러스. 크기 100 에서 시작 |
| 밥 | 먹으면 밥 크기의 절반만큼 커진다. 돌진할 때 흘리는 세포밥도 같다 |
| 블랙홀 | 크기 500 초과일 때 닿으면 −250 |
| 화이트홀 | 크기 500 미만일 때 닿으면 +100 |
| 포식 | 세포끼리 경계면이 닿으면 큰 쪽이 작은 쪽을 통째로 먹는다 (같은 크기면 무작위) |
| 이동 | 클수록 느리고 방향 전환이 둔하다 |
| 돌진 | 클수록 빠르지만 4스텝마다 크기의 `1%·√(s/100)` 를 흘린다. 크기 100 이하면 못 한다 |
| 시야 | 한 변 `2·min(8·√(s/100), 25)` 인 정사각형. 그 밖은 보이지 않는다 |
| 승리 | 크기 4000 |
| 학습 중 | 죽으면 크기 100 으로 바로 리스폰하고 에피소드는 계속된다. 누군가 4000 에 도달하면 `terminated`, `max_steps` 에 닿으면 `truncated` |
| 대결 | 먹히면 탈락. 누군가 4000 에 도달하거나 한 명만 남으면 끝 |

관측에는 내 상태 `[크기, x, y, v_x, v_y]` 와 시야 안 객체만 있고, 다른 세포의 속도는 없다.
환경은 보상을 주지 않고 내 세포에게 일어난 사건(`events`)만 준다.
