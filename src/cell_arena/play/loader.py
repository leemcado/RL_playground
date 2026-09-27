"""참가자 지정 문자열 → Agent 인스턴스, 판 구성(configs/lineups.yaml) 읽기.

지정 형식:
- ``bronze``               봇 이름 → cell_arena 패키지에 내장된 봇 (cell_arena/bots/bronze.py)
- ``<경로>.py``            에이전트 파일 (현재 폴더 기준, 없으면 인스트럭터 레포 기준). 파일 안에 Agent 하위 클래스가 하나여야 한다
- ``<경로>.py@<가중치>``   학생 에이전트의 가중치 파일을 직접 지정 (기본은 클래스의 ``weights``, 에이전트 파일 폴더 기준)

학생 에이전트(StudentAgent)는 가중치 파일로 만든다 (``StudentAgent.load``). 봇은 인자 없이 만든다.
에이전트 파일은 같은 폴더의 다른 파일을 import 할 수 있고, 파일끼리 같은 이름의 보조 파일이 있어도 섞이지 않는다.

봇은 패키지 안(``BOTS_DIR``)에 있어 ``cell_arena`` 를 pip 로만 설치해도(학생 레포처럼 이 repo 소스가 없어도)
이름으로 바로 찾는다. ``REPO_ROOT``/``LINEUPS_FILE`` 은 인스트럭터 레포 전용 기능(``.py`` 상대 경로 해석, 토너먼트
구성 파일)이라 이 repo 밖에서 pip 설치됐을 때는 안 쓰인다 — 학생 쪽 경로(봇 이름, 자기 에이전트 파일)는 안 거친다.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import yaml

from cell_arena.agent import Agent, StudentAgent

REPO_ROOT = Path(__file__).resolve().parents[3]
BOTS_DIR = Path(__file__).resolve().parents[1] / "bots"
LINEUPS_FILE = REPO_ROOT / "configs" / "lineups.yaml"


def load_agent(spec: str) -> Agent:
    """지정 문자열로 참가자를 만든다.

    Raises:
        ValueError: 파일·가중치 없음, 또는 파일 안의 Agent 하위 클래스가 하나가 아님
    """
    spec, _, weights = spec.partition("@")
    path = _resolve(spec)
    cls = agent_class(path)
    if not issubclass(cls, StudentAgent):
        if weights:
            raise ValueError(f"{spec}: 봇은 가중치 파일을 받지 않는다")
        return cls()
    if weights:
        w = Path(weights)
        if not w.is_absolute() and not w.exists():
            w = path.parent / w
    elif cls.weights:
        w = path.parent / cls.weights
    else:
        raise ValueError(f"{path}: 학생 에이전트는 가중치 파일이 필요하다 (클래스의 weights 또는 '{spec}@가중치')")
    if not w.exists():
        raise ValueError(f"{cls.name}: 가중치 파일이 없다: {w}")
    return cls.load(w)


def lineup(name: str) -> list[str]:
    """configs/lineups.yaml 의 구성 하나 (참가자 지정 문자열 목록).

    Raises:
        KeyError: 없는 구성 이름
    """
    with LINEUPS_FILE.open(encoding="utf-8") as f:
        table = yaml.safe_load(f) or {}
    if name not in table:
        raise KeyError(f"{LINEUPS_FILE} 에 '{name}' 구성이 없다. 있는 구성: {list(table)}")
    return [str(s) for s in table[name]]


def agent_class(path: Path) -> type[Agent]:
    """에이전트 파일 안에 정의된 (추상이 아닌) Agent 하위 클래스 하나."""
    module = _load_file(path)
    found = [
        v for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Agent) and v.__module__ == module.__name__ and not inspect.isabstract(v)
    ]
    if len(found) != 1:
        names = [c.__name__ for c in found]
        raise ValueError(f"{path}: Agent 하위 클래스가 하나여야 한다 (찾은 것: {names}, 미완성 클래스는 제외)")
    return found[0]


def _resolve(spec: str) -> Path:
    if spec.endswith(".py"):
        path = Path(spec)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / spec
    else:
        path = BOTS_DIR / f"{spec}.py"
        if not path.exists():
            bots = sorted(p.stem for p in BOTS_DIR.glob("*.py"))
            raise ValueError(f"'{spec}': 봇 이름({', '.join(bots)}) 또는 에이전트 파일 경로(.py)여야 한다")
    if not path.exists():
        raise ValueError(f"에이전트 파일이 없다: {path}")
    return path.resolve()


def _load_file(path: Path) -> ModuleType:
    folder = path.parent
    name = f"_cell_arena_agent_{path.stem}_{abs(hash(path))}"
    if name in sys.modules:
        return sys.modules[name]
    before = set(sys.modules)
    sys.path.insert(0, str(folder))  # 같은 폴더의 보조 파일을 import 할 수 있게
    try:
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"모듈을 불러올 수 없다: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path.remove(str(folder))
        # 이 폴더에서 새로 불러온 보조 모듈은 캐시에서 빼서, 다른 파일의 같은 이름 보조 파일과 섞이지 않게 한다
        for mod_name in set(sys.modules) - before - {name}:
            mod_file = getattr(sys.modules[mod_name], "__file__", None)
            if mod_file and Path(mod_file).resolve().is_relative_to(folder):
                del sys.modules[mod_name]
    return module
