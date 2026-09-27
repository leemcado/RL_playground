"""설치·하드웨어 점검 (인스트럭터 레포용 래퍼) — 실제 구현은 cell_arena.scripts.doctor (학생 레포에도 콘솔
스크립트 ``cell-arena-doctor`` 로 설치된다).

    python scripts/doctor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cell_arena.scripts.doctor import main  # noqa: E402

if __name__ == "__main__":
    main()
