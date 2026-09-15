"""사람 플레이: 사람(세포 0) + 상대들. 시간 무한, 모두 죽으면 시야 밖에서 리스폰, 누군가 승리 크기에 도달하면 끝.

조작: W/A/S/D 이동, SPACE 돌진, TAB 패널에 보여줄 세포 변경, R 리셋, ESC 종료.

    python scripts/play.py                                 # configs/lineups.yaml 의 'play' 구성
    python scripts/play.py --lineup tournament             # 다른 구성을 상대로
    python scripts/play.py --opponents gold submissions/kim.py
    python scripts/play.py --seed 0 --fps 10 --max-steps 3000 --no-respawn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pygame  # noqa: E402

from cell_arena.play import BattleArena, lineup, load_agent  # noqa: E402
from cell_arena.play.human import HumanAgent  # noqa: E402
from cell_arena.play.render import ArenaViewer, HudInfo  # noqa: E402


def main() -> None:
    """게임 루프: 매 프레임 1 스텝."""
    parser = argparse.ArgumentParser(description="Cell Arena 사람 플레이")
    parser.add_argument("--lineup", default="play", help="상대 구성 (configs/lineups.yaml)")
    parser.add_argument("--opponents", nargs="*", default=None, help="구성 대신 상대 직접 지정 (봇 이름 | 파일.py)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20, help="초당 스텝 수")
    parser.add_argument("--max-steps", type=int, default=None, help="시간 제한 (기본 무한)")
    parser.add_argument("--no-respawn", action="store_true", help="탈락제 (기본: 사람·상대 모두 죽으면 리스폰)")
    args = parser.parse_args()

    human = HumanAgent()
    specs = args.opponents if args.opponents is not None else lineup(args.lineup)
    arena = BattleArena([human] + [load_agent(s) for s in specs], seed=args.seed, max_steps=args.max_steps,
                        respawn=not args.no_respawn)
    viewer = ArenaViewer(arena.cfg)
    clock = pygame.time.Clock()
    focus = 0
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
        viewer.draw(arena.state, HudInfo(arena.t, arena.max_steps, focus, arena.names, arena.status,
                                         human.last_action, arena.colors))
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
