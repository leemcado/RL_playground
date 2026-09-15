# Cell Arena

강화학습 실습(DQN · Dueling DQN · PPO)용 토러스 세포 대전 환경.
에이전트 파일 하나에 모델·전처리·보상·학습 루프를 구현해 학습하고, 같은 규칙의 대결장에서 겨룬다.

## 설치

```bash
pip install -e .            # macOS (Apple Silicon / Intel), Windows, Linux CPU
pip install -e ".[cuda]"    # Linux + NVIDIA GPU (JAX 시뮬레이션을 GPU 에서)
python scripts/doctor.py    # 패키지·장치·JAX 정합성·처리량 점검
wandb login
```

- 시뮬레이션은 JAX, 학습은 PyTorch. 장치는 자동으로 고른다 (torch: cuda > mps > cpu).
  JAX 를 불러올 수 없으면 입출력·규칙이 같은 NumPy env 로 대신한다.
- Intel Mac 은 jax 0.4.38 (지원 마지막 판) 과 NumPy 1.x 로 자동 고정된다. Windows 의 JAX GPU 는 WSL2 에서만 된다.

## 구조

```
src/cell_arena/     환경 패키지 — 규칙, NumPy·JAX 엔진, make_env, 대결장, StudentAgent
agents/bots/        상대 봇: bronze / silver / gold / diamond
agents/skeleton/    agent.py (학습 코드 = 제출물), config.yaml, README.md (API 가이드)
scripts/            doctor.py (설치 점검) · check_agent.py (제출 전 점검) · play.py (직접 플레이)
```

## 에이전트 만들기

고칠 수 있는 파일은 `agents/skeleton/agent.py` 와 `config.yaml` 뿐이다. API 는 [`agents/skeleton/README.md`](agents/skeleton/README.md).

| 직접 구현 | 고정 (재정의하면 오류) |
|---|---|
| 관측(`state` / `image`)·액션 형태 선언 | 게임 규칙, 관측·사건 형식 |
| `setup` 모델, `preprocess` 전처리·피처, `policy` 행동 선택, `reward` 보상 | `make_env` — 설정·장치에 맞는 학습 env |
| `train` 학습 루프 — 버퍼, 업데이트, 부트스트랩, wandb 로그 | `__init__` / `act` / `save` / `load` |

`config.yaml` 에는 학습 상대 봇 목록과 학습 하이퍼파라미터만 적는다.

```bash
python agents/skeleton/agent.py                         # 학습 → 같은 폴더에 가중치 저장
python scripts/check_agent.py agents/skeleton/agent.py  # 제출 전 점검
python scripts/play.py                                  # 봇 7명과 직접 플레이 (WASD 이동, SPACE 돌진)
```

제출: `agent.py` 를 `<이름>.py` 로 바꿔 가중치 `<이름>.pt` 와 함께 낸다.

## 게임 규칙

`s` 는 세포 크기, 거리 단위는 게임유닛, 시간 단위는 스텝. 수치는 `src/cell_arena/core/config.py`.

| 항목 | 규칙 |
|---|---|
| 맵 | 100 × 100 토러스. 세포는 크기 100, 정지 상태로 시작 |
| 밥 | 300개, 크기 ~ Exp(평균 8), 최대 40. 먹으면 밥 크기의 0.5 배만큼 커진다 |
| 블랙홀 | 15개 (크기 500). 크기 500 초과인 세포가 닿으면 −250 |
| 화이트홀 | 15개 (크기 500). 크기 500 미만인 세포가 닿으면 +100 |
| 세포밥 | 돌진할 때 흘리는 밥. 밥과 같은 규칙으로 먹는다 |
| 재배치 | 먹힌 밥·홀은 곧바로 모든 세포의 시야 밖에 다시 생긴다 (개수 유지) |
| 크기·시야 | 직경 `2·√(s/100)`. 시야는 한 변 `2h` 인 정사각형, `h = min(8·√(s/100), 25)` |
| 이동 | 목표 속도 `(s/100)^-0.19` 로 조향: `v ← v + κ·(v_목표 − v)`, `κ = 0.35·(s/100)^-0.25` |
| 돌진 | 속도 `1.1·(s/100)^0.2`, κ 절반. 누적 돌진 4스텝마다 크기의 `1%·√(s/100)` 를 세포밥으로 흘린다. 흘린 뒤에도 크기 100 이상일 때만 된다 |
| 접촉 | 두 원의 경계면이 닿으면 (`거리 < r_a + r_b`). 한 객체에 여러 세포가 닿으면 가장 큰 세포가 가져간다 |
| 포식 | 세포끼리 닿으면 큰 쪽이 작은 쪽을 통째로 먹는다 (같은 크기면 무작위). 먹히는 세포는 같은 스텝에 먹을 수 없다 |
| 사망 | 먹히거나, 블랙홀을 한 스텝에 여러 개 밟아 크기가 0 이하가 될 때 |
| 승리 | 크기 4000 도달 |
| 학습 env | 죽으면 모든 시야 밖에서 크기 100 으로 리스폰하고 에피소드는 계속된다. 종료는 누군가 4000 도달(`terminated`) 또는 `max_steps` 도달(`truncated`) |
| 대결장 | 먹히면 탈락, 시간 무한. 누군가 4000 에 도달하거나 한 명만 남으면 끝난다 |

관측은 자기 상태 `[크기, x, y, v_x, v_y]` 와 시야 안 객체뿐이다 (다른 세포의 속도·시야 밖 정보 없음).
`state`(가까운 객체 목록) 또는 `image`(타입별 5채널 0/1) 로 받는다. 환경은 보상을 주지 않고 내 세포의 사건(`Events`)만 준다.

## 엔진

| 엔진 | 쓰는 곳 |
|---|---|
| NumPy (`engine/numpy_engine.py`) | 대결장, JAX 가 없을 때의 학습 — 참조 구현 |
| JAX (`engine/jax_engine.py`) | 학습 — `jit(vmap)` 으로 env 여러 개를 동시에 |

두 엔진은 규칙 공식과 이미지 래스터라이저를 `np` / `jnp` 로 공유하고, `doctor.py` 가 그 기기에서 결과가 같은지 확인한다.
처리량 (Apple M5 CPU, env 64개, 봇 7명 상대): state 약 7,900 · image 약 5,300 샘플/s.
