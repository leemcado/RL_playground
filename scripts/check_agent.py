"""학생 에이전트 점검 (인스트럭터 레포용 래퍼) — 실제 구현은 cell_arena.scripts.check_agent (학생 레포에도 콘솔
스크립트 ``cell-arena-check`` 로 설치된다). 이 파일은 이 레포 안에서 ``python scripts/check_agent.py`` 로도 쓸 수
있게 남겨 둔 얇은 진입점이다.

    python scripts/check_agent.py submissions/kim.py [submissions/lee.py ...]
    python scripts/check_agent.py agents/skeleton/agent.py --steps 300
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cell_arena.scripts.check_agent import main  # noqa: E402

if __name__ == "__main__":
    main()
