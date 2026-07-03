"""Throughput floors (regression gates) — conservative fractions of measured.

Measured on RTX 5070 Ti (single-step CUDA-graph replay), 4096 worlds:
  - Franka manipulation scene:  8.6M steps/s @ 8192 worlds
  - 112-geom procedural room:   2.3M steps/s

The large-scene cliff is DYNAMIC-body count, not total geom count:
  - static furniture scales flat: 114/217/416/817 geoms -> ~2.3-2.4M steps/s
    (the S/F collision masks prune static-static pairs at model build)
  - dynamic clutter is the wall: 12/16/20/40 free bodies -> 0.93/0.75/0.52/
    0.12M steps/s (~1/n^2); >~40 exhausts contact budgets on 16 GB
So a static-AABB BVH would not move these numbers; the real lever is
dynamic-vs-* broadphase + constraint budgeting. Floors are ~2-3x below
measured to stay stable across driver/hw variance.
"""

import os
import time

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

MJCF = os.path.join(
    os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
    "franka_emika_panda", "mjx_single_cube.xml",
)


def _sps(scene, warmup=12, K=200):
    scene.step(warmup)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scene.step(K)
    torch.cuda.synchronize()
    return K * scene.n_worlds / (time.perf_counter() - t0)


@pytest.mark.skipif(not os.path.exists(MJCF), reason="menagerie not found")
def test_manipulation_scene_floor():
    scene = lpw.load_scene(MJCF, lpw.Config(n_worlds=4096))
    sps = _sps(scene)
    assert sps > 3_000_000, f"manipulation scene regressed: {sps:.0f} steps/s"


def test_indoor_room_floor(tmp_path):
    from latentphysics.assets.scene_gen import RoomSpec, generate_room
    p = generate_room(RoomSpec(seed=0, n_furniture=96, n_clutter=6),
                      str(tmp_path / "room.xml"))
    scene = lpw.load_scene(p, lpw.Config(n_worlds=4096))
    sps = _sps(scene)
    assert sps > 1_000_000, f"112-geom room regressed: {sps:.0f} steps/s"


def test_large_static_scene_scales_flat(tmp_path):
    """A heavily-furnished room (~800 geoms) must stay near the small-room
    rate: static-static pruning means static geom count is NOT a cliff.
    Measured ~2.4M steps/s; floor 1M."""
    from latentphysics.assets.scene_gen import RoomSpec, generate_room
    p = generate_room(RoomSpec(seed=0, n_furniture=760, n_clutter=2),
                      str(tmp_path / "big.xml"))
    scene = lpw.load_scene(p, lpw.Config(n_worlds=4096))
    sps = _sps(scene)
    assert sps > 1_000_000, f"static scaling regressed: {sps:.0f} steps/s"


def test_dynamic_clutter_floor(tmp_path):
    """The real large-scene cost: 20 free bodies per world. Measured
    ~0.52M steps/s @ 4096 worlds; floor 0.2M. Guards the dynamic-body
    broadphase/budget path against regression (and marks where a future
    dynamic broadphase would show up)."""
    from latentphysics.assets.scene_gen import RoomSpec, generate_room
    p = generate_room(RoomSpec(seed=0, n_furniture=40, n_clutter=20),
                      str(tmp_path / "clutter.xml"))
    scene = lpw.load_scene(p, lpw.Config(n_worlds=4096))
    sps = _sps(scene)
    assert sps > 200_000, f"dynamic-clutter path regressed: {sps:.0f} steps/s"
