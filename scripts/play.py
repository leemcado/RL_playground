"""사람 플레이 (인스트럭터 레포용 래퍼) — 실제 구현은 cell_arena.scripts.play (학생 레포에도 콘솔
스크립트 ``cell-arena-play`` 로 설치된다).

    python scripts/play.py                                 # 내장 기본 상대
    python scripts/play.py --lineup tournament             # configs/lineups.yaml 의 구성을 상대로
    python scripts/play.py --opponents gold submissions/kim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cell_arena.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    main()
