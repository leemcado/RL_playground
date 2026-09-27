# 스켈레톤 API

`cp -r agents/skeleton agents/<이름>` 으로 복사해 채움. 할 일과 게임 규칙은 [최상위 README](../../README.md).

## 관측

첫 차원은 배치 B (학습 = `num_envs`, 대결 = 1). 좌표는 내 세포 기준, y 는 아래가 +.

| 필드 | 형태 | 내용 |
|---|---|---|
| `self_state` | `(B, 5)` | `[크기, x, y, v_x, v_y]` |
| `objects` | `(B, M, 8)` | state 모드. 시야 안 객체 가까운 순 `[dx, dy, 크기, is_food, is_cell_food, is_black_hole, is_white_hole, is_cell]` |
| `mask` | `(B, M)` | state 모드. 실제 객체면 True, 나머지는 0 패딩 |
| `image` | `(B, 6, R, R)` | image 모드. 채널 `[밥, 세포밥, 블랙홀, 화이트홀, 다른 세포, 나]`, 값 0/1 |

- `ObsSpec(mode="image", resolution=64)` (기본) 또는 `ObsSpec(mode="state", max_objects=32)`
- 이미지는 크기와 상관없이 R x R 이라 클수록 넓고 거칠게 보임
- 밥과 세포밥은 따로 구분됨 (`is_food` / `is_cell_food`, 이미지 채널 0 / 1). 순서는 `cell_arena.OBJECT_FEATURES`, `cell_arena.IMAGE_CHANNELS`
- 다른 세포의 속도는 주지 않음

## 액션

| 모드 | 형태 | 내용 |
|---|---|---|
| `discrete` | `(B,)` 정수 0~17 | 0~8 = 정지, 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하 / 9~17 = 같은 순서 + 돌진 |
| `continuous` | `(B, 3)` | `[theta, move, dash]`. theta 는 오른쪽 0, 아래 +pi/2 (라디안). move, dash 는 0.5 초과면 켜짐 |

## 사건 (`events`)

환경은 보상을 주지 않고 내 세포에게 일어난 사건만 줌. 각 `(B,)`.

| 필드 | 내용 |
|---|---|
| `size_before`, `size_after` | 스텝 전후 크기 |
| `food_mass` | 밥, 세포밥으로 얻은 질량 |
| `white_hole`, `black_hole` | 발동 횟수 |
| `dash_cost` | 돌진으로 흘린 질량 |
| `kills`, `kill_mass` | 잡아먹은 세포 수와 얻은 질량 |
| `died`, `won` | 죽었는지, 4000 에 도달했는지 |
| `t` | 에피소드 경과 스텝 |

## 학습 env

`env.step(action)` 은 `obs, final_obs, events, terminated, truncated` 를 반환.

- `terminated`: 누군가 4000 도달. 부트스트랩하지 않음
- `truncated`: `max_steps` 도달. `final_obs` 로 부트스트랩함
- 끝난 env 는 자동 리셋됨. `obs` 는 새 에피소드의 첫 관측, `final_obs` 는 리셋 전 마지막 관측
- 죽어도 에피소드는 끝나지 않음 (`events.died`)
