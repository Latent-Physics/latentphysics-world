"""GPU backend tests: budget autoscaling + world-state snapshot/branching.

Skipped automatically on hosts without the GPU engine (warp/mujoco_warp/CUDA).
Run on Linux/WSL2 + NVIDIA GPU:  pytest tests/test_backend_gpu.py -v
"""

import io
import contextlib

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # engine importable but no device
    pytest.skip("CUDA device required", allow_module_level=True)


@pytest.fixture(scope="module")
def torus_scene_path(tmp_path_factory):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from latentphysics.assets import mesh_to_mjcf

    out = tmp_path_factory.mktemp("assets")
    mesh = trimesh.creation.torus(major_radius=0.3, minor_radius=0.1)
    return mesh_to_mjcf(mesh, str(out), name="torus", pos=(0, 0, 0.6), threshold=0.05)


@pytest.fixture(scope="module")
def scene(torus_scene_path):
    return lpw.load_scene(torus_scene_path, lpw.Config(n_worlds=8))


def test_auto_budgets_no_overflow(scene):
    """11-hull concave body must not overflow auto-scaled contact/constraint buffers."""
    assert scene.data.njmax >= 256 and scene.data.naconmax >= 8 * 96
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scene.step(300)
    assert "overflow" not in buf.getvalue()
    z = scene.qpos()[:, 2]
    assert 0.0 < z[0].item() < 0.3  # settled on plane, no tunneling


def test_snapshot_restore_roundtrip(scene):
    scene.reset()
    scene.step(20)  # mid-fall: state evolves after this point
    snap = scene.snapshot()
    q0 = scene.qpos().clone()
    scene.step(100)
    assert not torch.allclose(q0, scene.qpos())
    scene.restore(snap)
    assert torch.allclose(q0, scene.qpos(), atol=1e-6)


def test_partial_restore_branches_worlds(scene):
    import numpy as np

    scene.reset()
    scene.step(20)
    snap = scene.snapshot()
    q0 = scene.qpos().clone()
    scene.step(100)
    mask = np.zeros(scene.n_worlds, dtype=bool)
    mask[:4] = True
    scene.restore(snap, worlds=mask)
    q = scene.qpos()
    assert (q[:4] - q0[:4]).abs().max().item() < 1e-6
    assert (q[4:] - q0[4:]).abs().max().item() > 1e-4


def test_replay_determinism_within_atomic_noise(scene):
    """Restore + identical steps must reproduce trajectories to float-atomic noise.

    Known engine limitation: contacts accumulate via GPU float atomics in
    nondeterministic order (~1e-9 noise / 50 steps). Bit-exact replay needs a
    fork-level deterministic contact ordering patch (roadmap R4).
    """
    scene.reset()
    scene.step(20)
    snap = scene.snapshot()
    scene.step(50)
    qa = scene.qpos().clone()
    scene.restore(snap)
    scene.step(50)
    qb = scene.qpos().clone()
    assert (qa - qb).abs().max().item() < 1e-7
