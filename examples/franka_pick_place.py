"""Rigid: franka pick & place — IK-scripted grasp, lift, carry, release.

The gripper poses are planned by the numerical IK helper (examples/_ik.py);
the arm is servoed to each waypoint on the LPW GPU engine. No policy, no
learning — a scripted trajectory demonstrating manipulation via the
simulator's kinematics.

Run:  MUJOCO_GL=egl python examples/franka_pick_place.py --record
"""

import argparse
import os

import numpy as np
import torch

import latentphysics as lpw

MJCF = os.path.join(os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
                    "franka_emika_panda", "mjx_single_cube.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _ik import gripper_down_quat, solve_ik

    scene = lpw.load_scene(MJCF, lpw.Config(n_worlds=4))
    mjm = scene.mjm
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "box")
    cube_adr = int(mjm.jnt_qposadr[mjm.body_jntadr[bid]])
    seed = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")][:7].copy()
    downq = gripper_down_quat(mjm, sid)
    ik_data = mujoco.MjData(mjm)

    def arm_for(pos):
        return solve_ik(mjm, ik_data, sid, np.asarray(pos, float), downq, seed)

    cube_start = np.array([0.5, 0.0, 0.03])   # comfortable, well inside the workspace
    grasp_z = float(cube_start[2]) + 0.015    # gripper site sits ~1.5 cm above box center
    goal = np.array([0.45, 0.32, 0.0])
    cx, cy = float(cube_start[0]), float(cube_start[1])
    # (gripper target xyz, gripper open? , settle steps)
    waypoints = [
        ([cx, cy, grasp_z + 0.14], True, 60),        # approach above cube
        ([cx, cy, grasp_z], True, 60),               # descend to grasp
        ([cx, cy, grasp_z], False, 60),              # close gripper
        ([cx, cy, grasp_z + 0.20], False, 60),       # lift
        ([goal[0], goal[1], grasp_z + 0.20], False, 90),   # carry to goal
        ([goal[0], goal[1], grasp_z + 0.03], False, 60),   # lower
        ([goal[0], goal[1], grasp_z + 0.03], True, 40),    # release
        ([goal[0], goal[1], grasp_z + 0.20], True, 50),    # retract
    ]

    qpos, ctrl = scene.qpos(), scene.state("ctrl")
    home_q = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    home_q[cube_adr:cube_adr + 3] = cube_start
    home_q[cube_adr + 3:cube_adr + 7] = (1, 0, 0, 0)
    qpos[:, :] = torch.as_tensor(home_q, dtype=torch.float32, device="cuda")
    scene.qvel().zero_()
    ctrl[:, :7] = torch.as_tensor(seed, dtype=torch.float32, device="cuda")
    ctrl[:, 7] = 0.04
    scene.forward()

    traj = []
    for pos, open_grip, n in waypoints:
        arm = torch.as_tensor(arm_for(pos), dtype=torch.float32, device="cuda")
        ctrl[:, :7] = arm
        ctrl[:, 7] = 0.04 if open_grip else 0.0
        for _ in range(n):
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    cube_final = qpos[:, cube_adr:cube_adr + 3]
    dxy = torch.linalg.norm(cube_final[:, :2] - torch.tensor(goal[:2], device="cuda"), dim=-1)
    placed = int((dxy < 0.08).sum().item())
    print(f"cube moved to goal in {placed}/{scene.n_worlds} worlds "
          f"(final xy err {[round(v, 3) for v in dxy.tolist()]})")
    assert placed == scene.n_worlds, "pick-and-place failed"

    if args.record:
        from _record import record_webp
        record_webp(MJCF, np.asarray(traj), "franka_pick_place",
                    cam={"lookat": (0.45, 0.15, 0.1), "distance": 1.5,
                         "azimuth": 140, "elevation": -20}, every=4, quality=52)


if __name__ == "__main__":
    main()
