"""CPU test for the demo IK helper (examples/_ik.py) — converges to targets."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco  # noqa: E402

MJCF = os.path.join(os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
                    "franka_emika_panda", "mjx_single_cube.xml")
if not os.path.exists(MJCF):
    pytest.skip("mujoco_menagerie not found", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from _ik import gripper_down_quat, solve_ik  # noqa: E402


@pytest.fixture(scope="module")
def model():
    m = mujoco.MjModel.from_xml_path(MJCF)
    return m, mujoco.MjData(m), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")


@pytest.mark.parametrize("target", [
    (0.50, 0.00, 0.15), (0.45, 0.30, 0.10), (0.55, -0.25, 0.20), (0.40, 0.00, 0.30),
])
def test_ik_reaches_target(model, target):
    m, d, sid = model
    seed = m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")][:7].copy()
    downq = gripper_down_quat(m, sid)
    q = solve_ik(m, d, sid, np.array(target), downq, seed)
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    pos_err = np.linalg.norm(np.array(target) - d.site_xpos[sid])
    cq = np.zeros(4); neg = np.zeros(4); dq = np.zeros(4); rv = np.zeros(3)
    mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
    mujoco.mju_negQuat(neg, cq); mujoco.mju_mulQuat(dq, downq, neg); mujoco.mju_quat2Vel(rv, dq, 1.0)
    assert pos_err < 5e-3, f"position error {pos_err*1000:.1f} mm"
    assert np.linalg.norm(rv) < 0.05, f"orientation error {np.linalg.norm(rv):.3f} rad"
    assert np.all(q >= m.jnt_range[:7, 0] - 1e-6) and np.all(q <= m.jnt_range[:7, 1] + 1e-6)


def test_ik_respects_joint_limits_on_unreachable(model):
    # a far, unreachable target must still return within joint limits (no NaN)
    m, d, sid = model
    seed = m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")][:7].copy()
    q = solve_ik(m, d, sid, np.array([2.0, 2.0, 2.0]), gripper_down_quat(m, sid), seed)
    assert np.all(np.isfinite(q))
    assert np.all(q >= m.jnt_range[:7, 0] - 1e-6) and np.all(q <= m.jnt_range[:7, 1] + 1e-6)
