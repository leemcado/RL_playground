"""대결장: 실전 대결·관전·사람 플레이 (NumPy 엔진, 실시간).

렌더러(pygame)는 필요할 때만 ``cell_arena.play.render`` 에서 직접 가져온다.
"""

from cell_arena.play.arena import BattleArena
from cell_arena.play.loader import lineup, load_agent

__all__ = ["BattleArena", "lineup", "load_agent"]
