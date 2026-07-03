"""Damped least-squares inverse kinematics for scripting arm demos.

A deterministic kinematics solver (mujoco Jacobians), NOT a policy: given a
desired gripper world pose it returns the 7 arm joint angles, so demo scripts
can say "move the gripper to this point" instead of hand-tuning every joint.
Runs on a CPU MjData as an offline planning step; the resulting joint targets
are then servoed by the GPU sim.
"""

from __future__ import annotations

import numpy as np


def gripper_down_quat(mjm, site_id, keyframe="pickup"):
    """A known-good downward grasp orientation: the gripper quat at a keyframe."""
    import mujoco
    d = mujoco.MjData(mjm)
    kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    d.qpos[:] = mjm.key_qpos[kid]
    mujoco.mj_forward(mjm, d)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.site_xmat[site_id])
    return q


def solve_ik(mjm, mjd, site_id, target_pos, target_quat, q_seed,
             n_arm=7, iters=120, damping=0.12, rot_weight=0.5,
             pos_tol=1.5e-3, rot_tol=2e-2):
    """Return ``n_arm`` joint angles putting the site at (target_pos, target_quat).

    6-DoF damped least squares seeded at ``q_seed`` (a nearby valid arm pose
    keeps the branch/orientation sane). ``mjd`` is scratch — its qpos is
    overwritten. Joint limits are respected.
    """
    import mujoco

    q = np.array(q_seed[:n_arm], dtype=float)
    lo, hi = mjm.jnt_range[:n_arm, 0], mjm.jnt_range[:n_arm, 1]
    jacp = np.zeros((3, mjm.nv))
    jacr = np.zeros((3, mjm.nv))
    perr = np.zeros(3)
    qerr = np.zeros(3)
    cur_q = np.zeros(4)
    neg_cur = np.zeros(4)
    dq_quat = np.zeros(4)
    for _ in range(iters):
        mjd.qpos[:n_arm] = q
        mujoco.mj_forward(mjm, mjd)
        perr[:] = target_pos - mjd.site_xpos[site_id]
        # world-frame rotation error as an axis-angle vector: q_err = tgt ⊗ cur⁻¹,
        # then quat2Vel -> the angular velocity that pairs with the world jacr
        mujoco.mju_mat2Quat(cur_q, mjd.site_xmat[site_id])
        mujoco.mju_negQuat(neg_cur, cur_q)
        mujoco.mju_mulQuat(dq_quat, target_quat, neg_cur)
        mujoco.mju_quat2Vel(qerr, dq_quat, 1.0)
        if np.linalg.norm(perr) < pos_tol and np.linalg.norm(qerr) < rot_tol:
            break
        mujoco.mj_jacSite(mjm, mjd, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :n_arm], rot_weight * jacr[:, :n_arm]])   # 6 x n_arm
        err = np.concatenate([perr, rot_weight * qerr])
        # dq = J^T (J J^T + λ²I)^-1 err
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)
    return q
