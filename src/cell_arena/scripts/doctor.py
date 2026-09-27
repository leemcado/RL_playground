"""설치·하드웨어 점검 — 실습 전에 한 번 돌린다. 문제가 있으면 해결 방법을 알려 준다.

    cell-arena-doctor

1. 파이썬·패키지 버전
2. JAX (시뮬레이션) 와 PyTorch (학습) 가 쓸 장치, torch 학습 한 스텝
3. wandb 로그인
4. 이 기기의 JAX 결과가 NumPy 참조 엔진과 같은지 (규칙 진행, state·image 관측)
5. 학습 env 처리량 (봇 7명 상대, 64 envs)
"""

from __future__ import annotations

import copy
import importlib
import netrc
import os
import platform
import shutil
import sys
import time
from functools import partial

import numpy as np

import cell_arena  # noqa: F401 — 환경 변수 설정 (jax 를 부르기 전에)
from cell_arena.core.api import IMAGE_CHANNELS
from cell_arena.hardware import jax_status, torch_device

INTEL_MAC = platform.system() == "Darwin" and platform.machine() == "x86_64"
failed: list[str] = []


def report(ok: bool | None, what: str, detail: str = "", fix: str = "") -> None:
    """[ OK ] / [WARN] (ok=None) / [FAIL] 한 줄, 실패·경고면 해결 방법."""
    tag = {True: "[ OK ]", None: "[WARN]", False: "[FAIL]"}[ok]
    print(f"{tag} {what:<22} {detail}")
    if fix and ok is not True:
        print(f"       → {fix}")
    if ok is False:
        failed.append(what)


def packages() -> None:
    py = sys.version_info
    report(py >= (3, 10), "python", f"{platform.python_version()} ({platform.system()} {platform.machine()})",
           "Python 3.10 이상이 필요하다")
    for mod, pip in (("numpy", "numpy"), ("yaml", "pyyaml"), ("torch", "torch"), ("wandb", "wandb"),
                     ("pygame", "pygame")):
        try:
            report(True, mod, importlib.import_module(mod).__version__)
        except Exception as err:  # noqa: BLE001
            report(False, mod, f"{type(err).__name__}: {err}", f"pip install cell-arena  (또는 pip install {pip})")


def devices() -> bool:
    ok, info = jax_status()
    fix = "pip install cell-arena" + ("  (Intel Mac 은 jax==0.4.38 이 마지막 판)" if INTEL_MAC else "")
    if ok and shutil.which("nvidia-smi") and "cpu" in info:
        report(None, "jax (시뮬레이션)", info, 'NVIDIA GPU 가 있다: pip install "cell-arena[cuda]" 로 JAX GPU 판 설치')
    else:
        report(ok if ok else None, "jax (시뮬레이션)", info if ok else f"없음 — NumPy env 로 대신 (느림): {info}", fix)
    try:
        import torch

        dev = torch_device()
        net = torch.nn.Sequential(torch.nn.Conv2d(len(IMAGE_CHANNELS), 16, 4, 2), torch.nn.ReLU(), torch.nn.Flatten(),
                                  torch.nn.LazyLinear(18)).to(dev)
        x = torch.rand(64, len(IMAGE_CHANNELS), 64, 64, device=dev)
        start = time.perf_counter()
        for _ in range(5):
            net(x).square().mean().backward()
        _ = float(next(net.parameters()).grad.sum())
        ms = (time.perf_counter() - start) / 5 * 1000
        hint = " (GPU 없음 — CPU 로 학습)" if dev.type == "cpu" else ""
        report(True, "torch (학습)", f"{dev} — CNN 배치 64 학습 한 스텝 {ms:.0f}ms{hint}")
    except Exception as err:  # noqa: BLE001
        report(False, "torch (학습)", f"{type(err).__name__}: {err}", "pip install cell-arena")
    return ok


def wandb_login() -> None:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            key = (netrc.netrc().authenticators("api.wandb.ai") or (None, None, None))[2]
        except (FileNotFoundError, netrc.NetrcParseError):
            key = None
    report(True if key else None, "wandb 로그인", "로그인됨" if key else "로그인 안 됨",
           "wandb login 으로 로그인하거나, 인터넷이 없으면 WANDB_MODE=offline 으로 학습 후 wandb sync")


def parity() -> None:
    """NumPy 참조 엔진과 이 기기의 JAX 결과 비교: 봇 8명 판의 여러 시점에서 규칙 한 스텝과 관측."""
    import jax

    from cell_arena.bots import DEFAULT_OPPONENTS
    from cell_arena.core.actions import decode_action
    from cell_arena.core.api import Observation
    from cell_arena.core.config import ArenaConfig
    from cell_arena.core.observation import build_observation, object_list, semantic_image
    from cell_arena.engine import jax_observation as jo
    from cell_arena.engine.jax_engine import from_numpy_state, world_step
    from cell_arena.engine.numpy_engine import NumpyArenaEngine
    from cell_arena.play import load_agent

    cfg = ArenaConfig()
    bots = [load_agent(s) for s in ["diamond", *DEFAULT_OPPONENTS]]
    n = len(bots)
    engine = NumpyArenaEngine(cfg, n, seed=0)
    for bot in bots:
        bot.reset(np.ones(1, dtype=bool))
    step = jax.jit(partial(world_step, cfg=cfg))
    image = jax.jit(jo.semantic_image, static_argnums=(1, 2, 3))
    objects = jax.jit(jo.object_list, static_argnums=(1, 2, 3))
    rng = np.random.default_rng(0)
    worst = {"rule": 0.0, "state": 0.0, "image": 0.0}
    for t in range(600):
        d, m, s = np.zeros((n, 2)), np.zeros(n, bool), np.zeros(n, bool)
        for i, bot in enumerate(bots):
            action = bot.act(Observation.stack([build_observation(engine.state, i, bot.obs_spec, cfg)]))
            (d[i], m[i], s[i]) = (a[0] for a in decode_action(action, bot.action_spec))
        if t % 60 == 59:
            twin = copy.deepcopy(engine)
            twin.state.cells.size += rng.uniform(0.0, 0.01, n)  # 같은 크기 동률(무작위 판정)을 피한다
            world = from_numpy_state(twin.state)
            for i in range(n):
                f_np, m_np = object_list(twin.state, i, cfg, 32)
                f_jx, m_jx = objects(world, i, cfg, 32)
                worst["state"] = max(worst["state"], float(np.abs(np.asarray(f_jx) - f_np).max()),
                                     float((np.asarray(m_jx) != m_np).any()))
                worst["image"] = max(worst["image"], float((np.asarray(image(world, i, cfg, 64))
                                                            != semantic_image(twin.state, i, cfg, 64)).mean()))
            ev_np = twin.step(d, m, s)
            new_world, ev_jx = step(world, jax.random.PRNGKey(t), d.astype(np.float32), m, s)
            cells = jax.device_get(new_world.cells)
            diffs = [np.abs(np.asarray(ev_jx[k], np.float64) - ev_np[k]).max() for k in ev_np]
            diffs += [np.abs(np.asarray(getattr(cells, k), np.float64) - getattr(twin.state.cells, k)).max()
                      for k in ("pos", "vel", "size", "alive")]
            worst["rule"] = max(worst["rule"], float(max(diffs)))
        for k in np.flatnonzero(engine.step(d, m, s)["died"]):
            engine.respawn(int(k))
    ok = worst["rule"] < 1e-2 and worst["state"] < 1e-3 and worst["image"] < 1e-3
    report(ok, "JAX = NumPy", f"규칙 최대 오차 {worst['rule']:.1e}, state {worst['state']:.1e}, "
                              f"image 픽셀 불일치 {worst['image'] * 100:.2f}% (장치 {jax.devices()[0].platform})",
           "이 기기의 JAX 결과가 참조 엔진과 다르다 — 강사에게 알린다")


def throughput(use_jax: bool) -> None:
    from cell_arena import ActionSpec, Config, ObsSpec, make_env
    from cell_arena.bots import DEFAULT_OPPONENTS

    class Probe:  # make_env 는 obs_spec / action_spec 만 읽는다
        action_spec = ActionSpec("discrete")

    cfg = Config({"opponents": DEFAULT_OPPONENTS, "num_envs": 64, "max_steps": 1000, "seed": 0})
    rng = np.random.default_rng(0)
    for mode in ("state", "image"):
        probe = Probe()
        probe.obs_spec = ObsSpec(mode=mode)
        env = make_env(cfg, probe, backend="jax" if use_jax else "numpy", verbose=False)
        env.reset()
        steps = 30 if use_jax else 3
        for i in range(steps + 3):
            if i == 3:
                start = time.perf_counter()
            env.step(rng.integers(0, 18, cfg.num_envs))
        sps = cfg.num_envs * steps / (time.perf_counter() - start)
        report(True, f"env 처리량 ({mode})", f"{sps:,.0f} 샘플/s ({'jax' if use_jax else 'numpy'}, 64 envs, 봇 7명)")


def main() -> None:
    print(f"Cell Arena doctor — {platform.platform()}\n")
    packages()
    use_jax = devices()
    wandb_login()
    if use_jax:
        parity()
    throughput(use_jax)
    print("\n" + ("모두 정상" if not failed else f"실패 {len(failed)}개: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
