"""대결 — 참가자는 configs/lineups.yaml 의 구성 또는 명령줄 목록.

    python scripts/battle.py                                 # 'tournament' 1판, 순위표
    python scripts/battle.py --render                        # 관전 창 (TAB 포커스, R 리셋, ESC 종료)
    python scripts/battle.py --games 20                      # 20판 성적표 (시드 0~19)
    python scripts/battle.py --record outputs/final.mp4      # mp4 녹화 (창 없이)
    python scripts/battle.py submissions/kim.py diamond gold # 참가자 직접 지정

기본 규칙은 탈락제 (먹히면 끝) + 시간 무한: 누군가 4000 도달 또는 생존자 1명이면 종료.
--respawn 이면 학습과 같이 전원 리스폰 (창 없이 돌릴 때는 시간 상한이 없으면 3000 으로 둔다).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cell_arena.play import BattleArena, lineup, load_agent  # noqa: E402


def play_headless(arena: BattleArena) -> dict[str, dict[str, float]]:
    """끝날 때까지 진행하고 참가자별 결과 (순위·승리·평균 크기·사망)."""
    size_sum = np.zeros(len(arena.agents))
    while not arena.done:
        arena.step()
        size_sum += np.where(arena.state.cells.alive, arena.state.cells.size, 0.0)
    winner = arena.status.removeprefix("WINNER ") if arena.status.startswith("WINNER") else None
    return {
        r["name"]: {"rank": r["rank"], "win": float(r["name"] == winner), "final_size": r["size"] if r["alive"] else 0.0,
                    "avg_size": size_sum[arena.names.index(r["name"])] / max(arena.t, 1), "deaths": r["deaths"],
                    "act_ms_max": r["act_ms_max"]}
        for r in arena.results()
    }


def print_results(arena: BattleArena) -> None:
    print(f"\n{arena.status or 'STOPPED'}  (step {arena.t})")
    print(f"{'rank':>4}  {'name':<24} {'size':>8}  {'status':<14} {'deaths':>6} {'max act ms':>10}")
    for r in arena.results():
        status = "alive" if r["alive"] else f"out @ {r['eliminated_at']}"
        print(f"{r['rank']:>4}  {r['name']:<24} {r['size']:>8.1f}  {status:<14} {r['deaths']:>6} {r['act_ms_max']:>10.2f}")


def run_games(specs: list[str], games: int, max_steps: int | None, respawn: bool) -> None:
    """시드 0 ~ games-1 로 여러 판, 참가자별 평균 성적표."""
    table: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seed in range(games):
        arena = BattleArena([load_agent(s) for s in specs], seed=seed, max_steps=max_steps, respawn=respawn)
        for name, row in play_headless(arena).items():
            table[name].append(row)
        print(f"  판 {seed + 1}/{games}: {arena.status} (step {arena.t})", flush=True)
    print(f"\n{games}판 성적표 ({'전원 리스폰' if respawn else '탈락제'}, 시간 상한 {max_steps})")
    print(f"{'name':<24} {'avg rank':>8} {'wins':>5} {'avg size':>9} {'final size':>10} {'deaths':>7}")
    means = {name: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for name, rows in table.items()}
    for name, m in sorted(means.items(), key=lambda kv: kv[1]["rank"]):
        print(f"{name:<24} {m['rank']:>8.2f} {m['win'] * games:>5.0f} {m['avg_size']:>9.0f} "
              f"{m['final_size']:>10.0f} {m['deaths']:>7.2f}")


def run_render(arena: BattleArena, fps: int, focus: int) -> None:
    """관전 창에서 진행."""
    import pygame

    from cell_arena.play.render import ArenaViewer, HudInfo

    viewer = ArenaViewer(arena.cfg)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                arena.reset()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                focus = (focus + 1) % len(arena.agents)
        if not arena.done:
            arena.step()
        viewer.draw(arena.state, HudInfo(arena.t, arena.max_steps, focus, arena.names, arena.status, None,
                                         arena.colors))
        clock.tick(fps)
    pygame.quit()


def run_record(arena: BattleArena, fps: int, focus: int, out: Path) -> None:
    """창 없이 렌더해 mp4 로 저장 (끝난 뒤 결과 화면 3초 유지)."""
    import pygame

    from cell_arena.play.render import ArenaViewer, HudInfo

    viewer = ArenaViewer(arena.cfg)
    width, height = viewer.screen.get_size()
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
        stdin=subprocess.PIPE,
    )

    def frame() -> None:
        viewer.draw(arena.state, HudInfo(arena.t, arena.max_steps, focus, arena.names, arena.status, None,
                                         arena.colors))
        encoder.stdin.write(pygame.surfarray.array3d(viewer.screen).swapaxes(0, 1).tobytes())

    frame()
    while not arena.done:
        arena.step()
        frame()
    for _ in range(fps * 3):
        frame()
    encoder.stdin.close()
    encoder.wait()
    pygame.quit()
    print(f"녹화: {out} ({out.stat().st_size / 1e6:.1f} MB, {arena.t / fps:.0f}초)")


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise SystemExit("녹화에는 ffmpeg 가 필요하다: pip install imageio-ffmpeg") from None
        return exe


def main() -> None:
    parser = argparse.ArgumentParser(description="Cell Arena 대결")
    parser.add_argument("agents", nargs="*", help="참가자 직접 지정: 봇 이름 | 파일.py[@가중치] (없으면 --lineup)")
    parser.add_argument("--lineup", default="tournament", help="참가자 구성 (configs/lineups.yaml)")
    parser.add_argument("--games", type=int, default=1, help="판 수 (2 이상이면 성적표)")
    parser.add_argument("--render", action="store_true", help="관전 창")
    parser.add_argument("--record", type=Path, default=None, help="mp4 로 녹화할 경로")
    parser.add_argument("--focus", default=None, help="패널에 보여줄 참가자 이름 (기본: 첫 참가자)")
    parser.add_argument("--respawn", action="store_true", help="죽으면 시야 밖에서 리스폰 (기본: 탈락제)")
    parser.add_argument("--max-steps", type=int, default=None, help="시간 상한 (기본 무한)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    specs = args.agents or lineup(args.lineup)
    max_steps = args.max_steps
    if args.respawn and max_steps is None and not args.render:
        max_steps = 3000  # 전원 리스폰 + 무한 시간이면 아무도 4000 에 못 가는 판이 끝나지 않는다
    if args.games > 1:
        run_games(specs, args.games, max_steps, args.respawn)
        return

    if args.record is not None and not args.render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # 창 없이 렌더
    arena = BattleArena([load_agent(s) for s in specs], seed=args.seed, max_steps=max_steps, respawn=args.respawn)
    print("참가자:", ", ".join(arena.names))
    focus = 0
    if args.focus is not None:
        matches = [i for i, n in enumerate(arena.names) if n == args.focus or n.startswith(f"{args.focus}#")]
        if not matches:
            raise SystemExit(f"'{args.focus}' 참가자가 없다: {arena.names}")
        focus = matches[0]
    if args.record is not None:
        run_record(arena, args.fps, focus, args.record)
    elif args.render:
        run_render(arena, args.fps, focus)
    else:
        play_headless(arena)
    print_results(arena)


if __name__ == "__main__":
    main()
