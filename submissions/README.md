# 학생 제출물

학생 한 명 = **에이전트 파일 하나 + 가중치 파일** (`kim.py` + `kim.pt`). 이름·색·가중치 파일 이름은 `kim.py` 안의
`name` / `color` / `weights` 로 정해져 있다.

1. `kim.py`, `kim.pt` 를 이 폴더에 넣는다
2. 점검: `python scripts/check_agent.py submissions/kim.py`
3. `configs/lineups.yaml` 의 구성에 `submissions/kim.py` 한 줄 추가
4. 대결: `python scripts/battle.py --lineup tournament --render` (사람도 같이: `python scripts/play.py --lineup tournament`)
