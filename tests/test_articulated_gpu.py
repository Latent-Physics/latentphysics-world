"""GPU tests for articulated furniture (R4): hinged doors + sliding drawers.

Verifies the joints are drivable on the batched engine, joint limits hold,
friction/damping park the parts when released, and the default room stays
articulation-free (reproducibility guard for existing seeds).
"""

import numpy as np
import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.assets.scene_gen import RoomSpec, generate_room  # noqa: E402

N = 8


def _articulated_joints(mjm):
    out = []
    for j in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if "_door" in name or "_drawer" in name:
            out.append((name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j]),
                        tuple(mjm.jnt_range[j])))
    return out


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("artic") / "room.xml")
    generate_room(RoomSpec(seed=3, n_articulated=2, n_furniture=40, n_clutter=4), path)
    return lpw.load_scene(path, lpw.Config(n_worlds=N))


def test_default_room_has_no_articulation(tmp_path):
    path = str(tmp_path / "room0.xml")
    generate_room(RoomSpec(seed=0), path)
    with open(path) as f:
        xml = f.read()
    assert "_door" not in xml and "_drawer" not in xml


def test_articulated_joints_exist(scene):
    joints = _articulated_joints(scene.mjm)
    assert any("_door" in n for n, *_ in joints)
    assert any("_drawer" in n for n, *_ in joints)


def test_joints_drivable_and_limited(scene):
    joints = _articulated_joints(scene.mjm)
    qpos, qvel = scene.qpos(), scene.qvel()
    scene.reset()

    # drive every articulated joint open (kinematic push, physics resolves)
    for _ in range(80):
        for name, _, dof, _ in joints:
            qvel[:, dof] = 1.5 if "_door" in name else 0.4
        scene.step()
    for name, qadr, _, (lo, hi) in joints:
        q = qpos[:, qadr]
        assert torch.isfinite(q).all(), f"{name} went non-finite"
        assert (q > lo + 0.05).all(), f"{name} did not open (q={q[0].item():.3f})"
        assert (q < hi + 1e-2).all(), f"{name} blew past its limit"

    # park test: once stopped, friction/damping must hold the parts in place
    # (no creep under gravity or solver noise) — coasting after a hard shove
    # is legitimate physics and NOT what this asserts
    for _, _, dof, _ in joints:
        qvel[:, dof] = 0.0
    held = {n: qpos[:, qadr].clone() for n, qadr, *_ in joints}
    scene.step(120)
    for name, qadr, _, (lo, hi) in joints:
        q = qpos[:, qadr]
        assert torch.isfinite(q).all()
        assert (q >= lo - 1e-3).all() and (q <= hi + 1e-2).all()
        drift = (q - held[name]).abs().max().item()
        assert drift < 0.05, f"{name} crept {drift:.3f} after being parked"


def test_room_state_stays_finite(scene):
    scene.reset()
    scene.step(200)
    assert torch.isfinite(scene.qpos()).all() and torch.isfinite(scene.qvel()).all()
