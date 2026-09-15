"""NumPy 참조 엔진 (play env 백엔드).

환경 1개를 파이썬에서 순차 진행한다. play env(실시간 대결·관전)에는 충분하지만 학습 처리량은
낼 수 없으므로, 학습에는 같은 규칙의 JAX 엔진(jax_engine.py → JaxTrainEnv)을 쓴다.
두 엔진의 동등성은 scripts/doctor.py 가 검증한다. ``TODO(JAX)`` 는 JAX 판에서 구조가
달라진 곳 (링버퍼 scatter, 고정 슬롯 재스폰 등) 표시.

엔진은 규칙만 진행한다. 관측·보상·에피소드 종료·리스폰 시점은 env(play/train)가 정한다.
"""

from __future__ import annotations

import numpy as np

from cell_arena.core import physics
from cell_arena.core.config import BLACK_HOLE, CELL_FOOD, FOOD, WHITE_HOLE, ArenaConfig
from cell_arena.core.state import CellArrays, ObjectArrays, WorldState


class NumpyArenaEngine:
    """NumPy 참조 엔진 (단일 월드, 상태 in-place 갱신).

    Args:
        cfg: 게임 규칙
        num_cells: 세포 수 N
        seed: 난수 시드
    """

    def __init__(self, cfg: ArenaConfig | None = None, num_cells: int = 1, seed: int | None = None) -> None:
        self.cfg = cfg or ArenaConfig()
        self.num_cells = num_cells
        self._rng = np.random.default_rng(seed)
        self._state = self._initial_state()

    @property
    def state(self) -> WorldState:
        """현재 월드 상태."""
        return self._state

    def reset(self, seed: int | None = None) -> None:
        """새 월드 생성. seed 가 주어지면 난수 생성기를 다시 시드한다."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._initial_state()

    def step(self, direction: np.ndarray, moving: np.ndarray, dash: np.ndarray) -> dict[str, np.ndarray]:
        """한 스텝: 조향 → 이동 → 돌진 방출 → 세포 간 포식 → 객체 접촉.

        Args:
            direction: (N, 2) 이동 방향 단위벡터
            moving: (N,) 이동 입력 여부
            dash: (N,) 돌진 입력 여부

        Returns:
            세포별 사건 (N,) — engine.base.EVENT_KEYS
        """
        cfg, st = self.cfg, self._state
        cells, n = st.cells, self.num_cells
        direction = np.asarray(direction, dtype=np.float64).reshape(n, 2)
        moving = np.asarray(moving, dtype=bool).reshape(n)
        dash = np.asarray(dash, dtype=bool).reshape(n)
        alive = cells.alive.copy()
        size_before = cells.size.copy()

        # 운동: 돌진은 이동 입력이 있고 크기 여유가 있을 때만 적용
        dashing = dash & moving & physics.can_dash(cells.size, cfg) & alive
        speed = np.where(dashing, physics.dash_speed(cells.size, cfg), physics.base_speed(cells.size, cfg))
        k = physics.kappa(cells.size, cfg) * np.where(dashing, cfg.dash_kappa_scale, 1.0)
        v_target = direction * (speed * moving)[:, None]
        new_vel = cells.vel + (v_target - cells.vel) * k[:, None]
        cells.vel = np.where(alive[:, None], new_vel, 0.0)
        cells.pos = np.where(alive[:, None], np.mod(cells.pos + cells.vel, cfg.map_size), cells.pos)
        cells.dashing = dashing

        dash_cost = self._emit_cell_food(dashing)
        kills, kill_mass, eaten = self._resolve_predation()
        food_mass, white, black, crushed, size_after = self._resolve_contacts()
        died = eaten | crushed
        st.step += 1

        won = alive & ~died & (size_before < cfg.win_size) & (size_after >= cfg.win_size)
        return {
            "size_before": size_before,
            "size_after": size_after,
            "food_mass": food_mass,
            "white_hole": white,
            "black_hole": black,
            "dash_cost": dash_cost,
            "kills": kills,
            "kill_mass": kill_mass,
            "died": died,
            "won": won,
        }

    def respawn(self, idx: int) -> None:
        """죽은 세포를 기본 크기·정지 상태로 모든 시야 밖에 되살린다 (학습 env 리스폰)."""
        cells = self._state.cells
        cells.alive[idx] = False  # 스폰 판정에서 자기 시야 제외
        r = np.atleast_1d(physics.radius(self.cfg.base_size, self.cfg))
        cells.pos[idx] = self._spawn_positions(cells, r)[0]
        cells.vel[idx] = 0.0
        cells.size[idx] = self.cfg.base_size
        cells.dash_steps[idx] = 0
        cells.dashing[idx] = False
        cells.alive[idx] = True

    # ------------------------------------------------------------------ 내부

    def _initial_state(self) -> WorldState:
        """초기 월드: 세포는 기본 크기·정지, 밥은 균일 배치, 홀은 시야 밖 스폰."""
        cfg = self.cfg
        n, m = self.num_cells, cfg.num_objects(self.num_cells)
        nf, nb, start = cfg.num_food, cfg.num_black_holes, cfg.cell_food_start

        cells = CellArrays(
            pos=self._rng.uniform(0.0, cfg.map_size, (n, 2)),
            vel=np.zeros((n, 2)),
            size=np.full(n, cfg.base_size),
            alive=np.ones(n, dtype=bool),
            dash_steps=np.zeros(n, dtype=np.int64),
            dashing=np.zeros(n, dtype=bool),
        )
        types = np.full(m, CELL_FOOD, dtype=np.int64)
        types[:nf] = FOOD
        types[nf : nf + nb] = BLACK_HOLE
        types[nf + nb : start] = WHITE_HOLE
        sizes = np.zeros(m)
        sizes[:nf] = self._sample_food_size(nf)
        sizes[nf:start] = cfg.hole_size
        pos = np.zeros((m, 2))
        pos[:nf] = self._rng.uniform(0.0, cfg.map_size, (nf, 2))
        pos[nf:start] = self._spawn_positions(cells, physics.radius(sizes[nf:start], cfg))

        objects = ObjectArrays(
            pos=pos,
            size=sizes,
            type=types,
            alive=np.arange(m) < start,
            owner=np.full(m, -1, dtype=np.int64),
        )
        return WorldState(cells=cells, objects=objects, step=0, cell_food_cursor=0)

    def _sample_food_size(self, n: int) -> np.ndarray:
        """밥 크기 ~ Exp(평균 8), 상한 40."""
        return np.minimum(self._rng.exponential(self.cfg.food_mean, n), self.cfg.food_max)

    def _spawn_positions(self, cells: CellArrays, obj_radius: np.ndarray) -> np.ndarray:
        """스폰 규칙.

        후보 K개를 균등 샘플 → 살아있는 모든 세포 시야(정사각형, 객체 반지름만큼 여유) 밖인 후보 중
        랜덤 선택. 하나도 없으면 가장 가까운 세포까지 거리가 최대인 후보.

        Returns:
            스폰 위치 (n, 2)
        """
        cfg = self.cfg
        n, k = obj_radius.shape[0], cfg.spawn_candidates
        cand = self._rng.uniform(0.0, cfg.map_size, (n, k, 2))
        live = cells.alive
        if not live.any():
            return cand[:, 0]

        cpos = cells.pos[live]
        h = physics.vision_radius(cells.size[live], cfg)
        delta = physics.torus_delta(cand[:, :, None, :], cpos[None, None], cfg.map_size)
        reach = h[None, None, :] + obj_radius[:, None, None]
        in_view = (np.abs(delta) < reach[..., None]).all(axis=-1)
        outside = ~in_view.any(axis=-1)
        nearest = np.linalg.norm(delta, axis=-1).min(axis=-1)

        random_pick = np.argmax(np.where(outside, self._rng.random((n, k)), -1.0), axis=-1)
        fallback = np.argmax(nearest, axis=-1)
        choice = np.where(outside.any(axis=-1), random_pick, fallback)
        return cand[np.arange(n), choice]

    def _emit_cell_food(self, dashing: np.ndarray) -> np.ndarray:
        """돌진 중 4스텝마다 크기의 f(s) = 1% × √(s/100) 를 세포밥으로 방출 (클수록 큰 비율·큰 세포밥).

        방출 주기는 누적 돌진 스텝으로 센다 (3스텝씩 끊어 비용 회피 방지).
        세포밥은 이동 반대쪽 세포 원 바깥에 놓아 곧바로 재흡수되지 않게 한다.

        Returns:
            세포별 방출 질량 (N,)
        """
        cfg, st = self.cfg, self._state
        cells, obj = st.cells, st.objects
        capacity = obj.size.shape[0] - cfg.cell_food_start
        cells.dash_steps = cells.dash_steps + dashing
        emit = dashing & (cells.dash_steps % cfg.dash_emit_interval == 0)
        cost = np.zeros(self.num_cells)

        # TODO(JAX): 파이썬 루프 대신 고정 슬롯에 .at[].set 으로 링버퍼 쓰기
        for i in np.flatnonzero(emit):
            amount = cells.size[i] * physics.dash_emit_frac(cells.size[i], cfg)
            cells.size[i] -= amount
            cost[i] = amount
            speed = np.linalg.norm(cells.vel[i])
            heading = cells.vel[i] / speed if speed > 0 else np.array([1.0, 0.0])
            gap = physics.radius(cells.size[i], cfg) + physics.radius(amount, cfg) + 0.2
            slot = cfg.cell_food_start + st.cell_food_cursor
            obj.pos[slot] = np.mod(cells.pos[i] - heading * gap, cfg.map_size)
            obj.size[slot] = amount
            obj.alive[slot] = True
            obj.owner[slot] = i
            st.cell_food_cursor = (st.cell_food_cursor + 1) % capacity
        return cost

    def _resolve_predation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """세포 간 포식: 경계면이 닿으면 큰 쪽이 작은 쪽을 통째로 흡수 (같은 크기면 무작위로 한쪽).

        판정은 ``physics.can_eat_matrix`` + ``physics.resolve_predation`` (JAX 엔진과 같은 규칙):
        먹히는 세포는 같은 스텝에 먹을 수 없고, 여러 포식자가 겹치면 가장 큰 쪽이 가져간다.
        먹힌 세포는 alive=False 가 되고 크기는 먹히기 직전 값을 유지한다.

        Returns:
            kills (N,), kill_mass (N,), died (N,)
        """
        cfg, cells, n = self.cfg, self._state.cells, self.num_cells
        kills = np.zeros(n, dtype=np.int64)
        kill_mass = np.zeros(n)
        if n < 2:
            return kills, kill_mass, np.zeros(n, dtype=bool)

        can_eat = physics.can_eat_matrix(cells.pos, cells.size, cells.alive, self._rng.random(n), cfg)
        died, predator = physics.resolve_predation(can_eat, cells.alive, cells.size)

        np.add.at(kill_mass, predator[died], cells.size[died])
        np.add.at(kills, predator[died], 1)
        cells.size = cells.size + kill_mass
        cells.alive = cells.alive & ~died
        cells.vel[died] = 0.0
        cells.dashing = cells.dashing & ~died
        return kills, kill_mass, died

    def _resolve_contacts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """세포-객체 접촉 (경계면 접촉).

        - 밥·세포밥: 무조건 흡수, +0.5 × 크기
        - 블랙홀: s > 500 이면 소멸시키고 -250, 아니면 무해
        - 화이트홀: s < 500 이면 소멸시키고 +100, 아니면 무해
        여러 세포가 같은 객체에 닿으면 가장 큰 세포가 가져간다.
        블랙홀 여러 개를 한 번에 밟아 크기가 0 이하가 되면 사망한다 (하한 없음).

        Returns:
            food_mass (N,), white_hole (N,), black_hole (N,), crushed (N,) 블랙홀 사망,
            size_after (N,) — 블랙홀 사망 세포는 실제 값(0 이하), 나머지는 현재 크기
        """
        cfg = self.cfg
        cells, obj = self._state.cells, self._state.objects
        n = self.num_cells

        delta = physics.torus_delta(obj.pos[None], cells.pos[:, None], cfg.map_size)
        dist = np.linalg.norm(delta, axis=-1)
        reach = physics.contact_reach(physics.radius(cells.size, cfg)[:, None], physics.radius(obj.size, cfg)[None])
        contact = (dist < reach) & cells.alive[:, None] & obj.alive[None]

        s = cells.size[:, None]
        t = obj.type[None]
        edible_t = (t == FOOD) | (t == CELL_FOOD)
        eligible = contact & (edible_t | ((t == BLACK_HOLE) & (s > cfg.hole_size)) | ((t == WHITE_HOLE) & (s < cfg.hole_size)))

        consumed = eligible.any(axis=0)
        winner = np.argmax(np.where(eligible, s, -np.inf), axis=0)
        gain = np.select(
            [obj.type == BLACK_HOLE, obj.type == WHITE_HOLE],
            [-cfg.black_hole_penalty, cfg.white_hole_bonus],
            default=cfg.food_gain * obj.size,
        )
        size_delta = np.zeros(n)
        np.add.at(size_delta, winner[consumed], gain[consumed])
        new_size = cells.size + size_delta
        crushed = cells.alive & (new_size <= 0)  # 블랙홀 여러 개를 한 번에 밟아 0 이하 → 사망
        cells.size = np.where(crushed, cells.size, new_size)  # 죽은 세포는 직전 크기 유지 (관측·렌더링용)
        cells.alive = cells.alive & ~crushed
        cells.vel[crushed] = 0.0
        cells.dashing = cells.dashing & ~crushed

        edible = (obj.type == FOOD) | (obj.type == CELL_FOOD)
        food_mass = np.zeros(n)
        np.add.at(food_mass, winner[consumed & edible], gain[consumed & edible])
        white = np.zeros(n, dtype=np.int64)
        np.add.at(white, winner[consumed & (obj.type == WHITE_HOLE)], 1)
        black = np.zeros(n, dtype=np.int64)
        np.add.at(black, winner[consumed & (obj.type == BLACK_HOLE)], 1)

        obj.alive[consumed & (obj.type == CELL_FOOD)] = False
        self._respawn_objects(np.flatnonzero(consumed & (obj.type != CELL_FOOD)))
        return food_mass, white, black, crushed, np.where(crushed, new_size, cells.size)

    def _respawn_objects(self, idx: np.ndarray) -> None:
        """먹힌 밥·홀을 시야 밖에 재배치 (개수 유지).

        TODO(JAX): 가변 길이 idx 대신 전체 M 개에 대해 후보를 계산하고 jnp.where 로 선택.
        """
        if idx.size == 0:
            return
        obj = self._state.objects
        is_food = obj.type[idx] == FOOD
        obj.size[idx[is_food]] = self._sample_food_size(int(is_food.sum()))
        obj.pos[idx] = self._spawn_positions(self._state.cells, physics.radius(obj.size[idx], self.cfg))
