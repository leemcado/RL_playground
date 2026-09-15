# Cell Arena

강화학습 실습(DQN · Dueling DQN · PPO)용 토러스 세포 대전 환경.
학생은 에이전트 파일 하나에 모델·전처리·보상·학습 루프를 구현해 학습하고, 같은 규칙의 대결장에서 서로 겨룬다.
규칙 원본은 `read/cell-arena-blueprint.md`, 바꾼 점은 [아래](#청사진에서-바꾼-규칙).

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
- `doctor.py` 는 그 기기의 JAX 결과가 NumPy 참조 엔진과 같은지 확인한다.

## 구조

```
src/cell_arena/       환경 패키지 — 규칙, NumPy·JAX 엔진, make_env, 대결장, StudentAgent
agents/bots/          상대 봇: bronze / silver / gold / diamond
agents/skeleton/      학생 배포본: agent.py (학습 코드 = 제출물), config.yaml, README.md (API 가이드)
configs/lineups.yaml  대결장 참가자 구성
submissions/          학생 제출물 (<이름>.py + <이름>.pt)
scripts/              doctor · check_agent · play · battle
```

## 학생

고칠 수 있는 파일은 `agents/skeleton/agent.py` 와 `config.yaml` 뿐이다. API 는 [`agents/skeleton/README.md`](agents/skeleton/README.md).

| 학생이 구현 | 프레임워크가 고정 |
|---|---|
| 관측(`state` / `image`)·액션 형태 선언 | 게임 규칙, 관측·사건 형식 |
| `setup` 모델, `preprocess` 전처리·피처, `policy` 행동 선택, `reward` 보상 | `make_env` — 설정·장치에 맞는 학습 env |
| `train` 학습 루프 — 버퍼, 업데이트, 부트스트랩, wandb 로그 | `__init__` / `act` / `save` / `load` (재정의하면 오류) |

`config.yaml` 에는 학습 상대 봇 목록과 학습 하이퍼파라미터만 적는다.

```bash
python agents/skeleton/agent.py                         # 학습 → 같은 폴더에 가중치 저장
python scripts/check_agent.py agents/skeleton/agent.py  # 제출 전 점검
```

제출: `agent.py` 를 `<이름>.py` 로 바꿔 가중치 `<이름>.pt` 와 함께 낸다.

## 강사

1. 제출물을 `submissions/` 에 넣고 `python scripts/check_agent.py submissions/kim.py`
2. `configs/lineups.yaml` 의 `tournament` 에 `submissions/kim.py` 추가 (봇은 이름으로: `diamond`, `gold`, ...)
3. 대결

```bash
python scripts/battle.py --render                      # 관전 (TAB 포커스, R 리셋, ESC 종료)
python scripts/battle.py --games 20                    # 20판 성적표
python scripts/battle.py --record outputs/final.mp4    # 녹화
python scripts/play.py                                 # 사람이 직접 (WASD 이동, SPACE 돌진)
```

대결장은 탈락제 + 시간 무한이 기본이고, 누군가 크기 4000 에 도달하거나 한 명만 남으면 끝난다.
`--respawn` 이면 학습 env 처럼 전원 리스폰한다.

## 엔진

| 엔진 | 쓰는 곳 |
|---|---|
| NumPy (`engine/numpy_engine.py`) | 대결장, JAX 가 없을 때의 학습 — 참조 구현 |
| JAX (`engine/jax_engine.py`) | 학습 — `jit(vmap)` 으로 env 여러 개를 동시에 |

두 엔진은 규칙 공식(`core/physics.py`)과 이미지 래스터라이저를 `np` / `jnp` 로 공유한다.
처리량 (Apple M5 CPU, env 64개, 봇 7명 상대): state 약 7,900 · image 약 5,300 샘플/s.

## 청사진에서 바꾼 규칙

| 항목 | 결정 |
|---|---|
| 승리 크기 | 4000 (청사진 1000) |
| 자기 상태 | `[크기, x, y, v_x, v_y]` 원시 값 (직경·시야·돌진 가능 여부 제외) |
| 관측 | `state`: 가까운 객체 M개 `[dx, dy, 직경, 타입 원핫 4]` + mask / `image`: 타입별 5채널 0/1 `(5, R, R)` (밥·블랙홀·화이트홀·다른 세포·자기). 둘 다 자기 상태 포함, 다른 세포의 속도는 없음 |
| 객체 반경 | 청사진 직경 × 2 (충돌·관측·표시). 시야는 그대로 |
| 접촉 | 경계면이 닿으면 접촉 (`dist < r_a + r_b`) |
| 포식 | 닿으면 큰 쪽이 먹는다 (같은 크기면 무작위). 동시 판정 |
| 돌진 | 속도 `1.1·(s/100)^0.2` (클수록 빠름). 누적 4스텝마다 크기의 `1%·√(s/100)` 를 세포밥으로 방출 |
| 블랙홀 | 크기 하한 없음 — 한 스텝에 여러 개를 밟아 0 이하가 되면 사망 |
| 학습 env | 죽으면 모든 시야 밖에서 크기 100 으로 리스폰, 에피소드는 계속. 종료는 누군가 4000 도달(terminated) 또는 `max_steps`(truncated) |
| 좌표계 | 화면과 같은 y-down, θ = atan2(dy, dx) |
