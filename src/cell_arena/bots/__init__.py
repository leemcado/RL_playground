"""연습용 상대 봇 — bronze / silver / gold / diamond (설명은 README.md).

``play.loader.load_agent``가 파일 경로로 동적 임포트하므로 여기서 다시 임포트하지 않는다.
"""

from __future__ import annotations

# check_agent / doctor 등 인스트럭터 레포 밖(학생 레포)에서도 practice 상대가 필요한 콘솔 스크립트가 쓰는 기본 구성.
# configs/lineups.yaml 의 'play' 와 같은 값 — 그건 인스트럭터의 대결·관전용, 이건 패키지에 내장된 기본값.
DEFAULT_OPPONENTS = ["bronze", "bronze", "bronze", "silver", "silver", "gold", "diamond"]

__all__ = ["DEFAULT_OPPONENTS"]
