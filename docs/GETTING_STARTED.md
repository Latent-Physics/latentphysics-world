# Getting started

LPW's GPU engine runs on **Linux or WSL2 with an NVIDIA CUDA GPU** (Warp has
no Windows-native CUDA backend). Python 3.10–3.12.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[gpu,train,assets,dev]"

The engine core, `mujoco_warp`, is installed from our pinned fork — it is not
on PyPI. See [`third_party/README.md`](../third_party/README.md).

Recording gallery clips additionally needs the demo extra:

    pip install -e ".[demos]"

Offscreen rendering needs EGL: set `MUJOCO_GL=egl`.

## Assets (Franka demos and throughput tests)

    git clone https://github.com/google-deepmind/mujoco_menagerie ~/lpw/menagerie
    # or point LPW_MENAGERIE at an existing checkout
    python scripts/fetch_assets.py   # CC0 demo meshes -> ~/lpw/assets/library

## First run

    MUJOCO_GL=egl python examples/collision_tower.py
    MUJOCO_GL=egl PYTHONPATH=. python examples/parallel_worlds.py

Add `--record` to write the gallery clip to `docs/media/`.

## Tests

    python -m pytest -q

GPU tests skip automatically without CUDA + the engine; the scope guard and
CPU tests run anywhere. Menagerie-dependent tests skip when the asset
checkout is missing.
