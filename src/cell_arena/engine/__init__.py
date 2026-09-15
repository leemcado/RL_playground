"""규칙 엔진. 관측·보상·에피소드 관리는 하지 않고 규칙만 진행한다.

- ``numpy_engine``: 참조 구현 (대결장 백엔드, JAX 가 없을 때 학습 백엔드)
- ``jax_engine`` + ``jax_observation``: 학습용 (jit / vmap). jax 가 필요하므로 여기서 자동 import 하지 않는다
"""
