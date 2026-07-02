"""Throughput floors (regression gates) — conservative fractions of measured.

Measured on RTX 5070 Ti (single-step CUDA-graph replay):
  - Franka manipulation scene: 8.6M physics steps/s @ 8192 worlds
  - 112-geom procedural room:  2.3M physics steps/s @ 4096 worlds
Floors are set ~2-3x below measured to stay stable across driver/hw variance.
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
