"""Manipulation: franka opens a drawer — the arm actually does it (IK-scripted).

Ties the pieces together: the articulated drawer chest (R4 furniture), the
open-drawer benchmark scene, and the IK helper. The arm grasps the upper
drawer's handle and pulls it open — no kinematic joint drive, no policy; the
drawer moves because the gripper holds the handle and the arm pulls.

Run:  MUJOCO_GL=egl python examples/franka_open_drawer_arm.py --record
"""

import argparse

import numpy as np
import torch

import latentphysics as lpw
from latentphysics.envs.articulated_tasks import build_articulated_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _ik import gripper_down_quat, solve_ik

    scene = lpw.load_scene(build_articulated_scene("open_drawer"),
                           lpw.Config(n_worlds=4, njmax=2048))
    mjm = scene.mjm
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    # upper drawer: a top-down grasp has clearance above it (the lower drawer's
    # handle is blocked from above by the chest body)
    hgid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "f0w1h")
    jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "f0_drawer1")
    dr_adr = int(mjm.jnt_qposadr[jid])
    seed = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")][:7].copy()
    downq = gripper_down_quat(mjm, sid)
    ik_data = mujoco.MjData(mjm)

    # handle world pose at rest (read once from the CPU model)
    ik_data.qpos[:] = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")]
    mujoco.mj_forward(mjm, ik_data)
    handle = ik_data.geom_xpos[hgid].copy()

    cur = [seed.copy()]

    def arm_for(pos):
        cur[0] = solve_ik(mjm, ik_data, sid, np.asarray(pos, float), downq, cur[0])
        return cur[0]

    gz = float(handle[2]) + 0.015
    hx, hy = float(handle[0]), float(handle[1])
    waypoints = [
        ([hx, hy, gz + 0.12], True, 60),     # approach above the handle
        ([hx, hy, gz], True, 60),            # descend onto the handle bar
        ([hx, hy, gz], False, 60),           # grip the handle
        ([hx - 0.22, hy, gz], False, 160),   # pull the drawer open (toward the arm)
        ([hx - 0.22, hy, gz], False, 40),    # hold
    ]

    qpos, ctrl = scene.qpos(), scene.state("ctrl")
    qpos[:, :] = torch.as_tensor(mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")],
                                 dtype=torch.float32, device="cuda")
    scene.qvel().zero_()
    ctrl[:, :7] = torch.as_tensor(seed, dtype=torch.float32, device="cuda")
    ctrl[:, 7] = 0.04
    scene.forward()

    traj = []
    for pos, open_grip, n in waypoints:
        ctrl[:, :7] = torch.as_tensor(arm_for(pos), dtype=torch.float32, device="cuda")
        ctrl[:, 7] = 0.04 if open_grip else 0.0
        for _ in range(n):
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    opening = qpos[:, dr_adr]
    opened = int((opening > 0.15).sum().item())
    print(f"drawer opened by the arm in {opened}/{scene.n_worlds} worlds "
          f"(travel {[round(v, 3) for v in opening.tolist()]} m)")
    assert opened == scene.n_worlds, "arm failed to open the drawer"

    if args.record:
        from _record import record_webp
        record_webp(build_articulated_scene("open_drawer"), np.asarray(traj), "franka_open_drawer",
                    cam={"lookat": (0.33, 0.0, 0.5), "distance": 1.2,
                         "azimuth": 108, "elevation": -26}, every=4, quality=54)


if __name__ == "__main__":
    main()
