"""하드웨어 대응 — 시뮬레이션(JAX)과 학습(PyTorch)이 어떤 기기에서도 코드 수정 없이 돌게 한다.

| 기기 | 시뮬레이션 | 학습 (torch) |
|---|---|---|
| Apple Silicon (M 칩) | JAX CPU | mps |
| Intel Mac | JAX CPU (jax 0.4.38 이 마지막 지원판) | cpu |
| Windows | JAX CPU (GPU 는 WSL2 에서만) | cuda 또는 cpu |
| Linux + NVIDIA | JAX CUDA (``pip install -e ".[cuda]"``) | cuda |

JAX 를 불러올 수 없으면 make_env 가 같은 입출력의 NumPy env 로 바꿔 쓴다 (느리지만 동작은 같다).
"""

from __future__ import annotations

import os
from typing import Any


def configure_env() -> None:
    """jax·torch 를 불러오기 전에 정할 환경 변수 (이미 정해져 있으면 그대로 둔다)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX 가 GPU 메모리를 선점하지 않게 (torch 와 공유)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 에 없는 연산은 CPU 로 대신


def jax_status() -> tuple[bool, str]:
    """JAX 를 쓸 수 있는지와 설명 (버전·장치, 또는 실패 이유)."""
    try:
        import jax

        devices = jax.devices()
        return True, f"jax {jax.__version__} ({devices[0].platform}, 장치 {len(devices)}개)"
    except Exception as err:  # noqa: BLE001 — 설치 안 됨, 드라이버 문제 등 어떤 이유든 NumPy 로 대체
        return False, f"{type(err).__name__}: {err}"


def torch_device(prefer: str | None = None) -> Any:
    """torch 장치. prefer 가 있으면 그대로, 없으면 cuda > mps > cpu."""
    import torch

    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
