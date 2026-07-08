"""CPU semantic gate for the office-chair asset (examples/_chair.py).

Mechanical half of the CLAUDE.md fidelity bar for this asset (the rendered
half is `python examples/franka_push_chair.py --inspect`): the chair must
compile, pass the repo validity checks, settle grounded and upright at the
reference dimensions, and its articulation must behave (spring-loaded
recline returns; every declared joint exists).
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("imageio")  # procedural fabric textures need it once
import mujoco  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from _chair import BACK_Z1, BASE_R, HEAD_GAP, chair_scene_xml  # noqa: E402

from latentphysics.assets.validate import validate_model  # noqa: E402


@pytest.fixture(scope="module")
def settled():
    m = mujoco.MjModel.from_xml_string(chair_scene_xml())
    d = mujoco.MjData(m)
    for _ in range(600):
        mujoco.mj_step(m, d)
    return m, d


def _jadr(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert j >= 0, f"missing joint {name}"
    return m.jnt_qposadr[j]


def test_valid_and_articulated(settled):
    m, _ = settled
    rep = validate_model(m)
    assert rep.ok, str(rep)
    # 23 DoF: freejoint + 5x(swivel+roll) + lift + seat swivel + recline
    #         + 2x armrest slide + head slide + head pitch
    assert m.nv == 23
    for name in (["chair_lift", "chair_swivel", "chair_recline",
                  "chair_head_slide", "chair_head_pitch",
                  "chair_arm_l_slide", "chair_arm_r_slide"]
                 + [f"caster{i}_swivel" for i in range(5)]
                 + [f"wheel{i}_roll" for i in range(5)]):
        _jadr(m, name)


def test_settles_grounded_upright(settled):
    m, d = settled
    assert np.isfinite(d.qpos).all()
    assert float(np.abs(d.qvel).max()) < 0.05, "did not settle"
    quat = d.qpos[3:7]
    tilt = 2 * np.arccos(min(1.0, abs(quat[0])))
    assert tilt < np.radians(3), f"not upright: {np.degrees(tilt):.1f} deg"
    # grounded: every wheel pair's contact circle at the floor
    for i in range(5):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"wheel{i}")
        p = d.xpos[bid]
        assert abs(p[2] - 0.030) < 0.01, f"wheel{i} not on the floor: z={p[2]:.3f}"
        assert abs(np.hypot(p[0], p[1]) - BASE_R) < 0.05, "base span off"


def test_reference_dimensions(settled):
    m, d = settled
    # seat crown height (widest box in the seat body)
    seat_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chair_seat")
    tops = [d.geom_xpos[i][2] + m.geom_size[i][2] for i in range(m.ngeom)
            if m.geom_bodyid[i] == seat_body and m.geom_group[i] == 3
            and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            and m.geom_size[i][1] > 0.15]
    assert tops and 0.44 < max(tops) < 0.50, f"seat height {max(tops):.3f}"
    # headrest floats the spec'd gap above the backrest top
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chair_headrest")
    assert d.xpos[head][2] > BACK_Z1 + HEAD_GAP * 0.5
    # a real chair's mass, not a prop's
    assert 12.0 < float(m.body_mass[1:].sum()) < 22.0


def test_recline_spring_returns(settled):
    m, _ = settled
    d = mujoco.MjData(m)
    adr = _jadr(m, "chair_recline")
    d.qpos[adr] = 0.20
    mujoco.mj_forward(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[adr]) < 0.05, f"recline stuck at {d.qpos[adr]:.3f} rad"
