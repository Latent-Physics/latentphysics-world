"""Rigid: franka push — non-prehensile manipulation, IK-scripted.

The closed gripper is placed behind a cube and swept forward, pushing the
cube to a goal without grasping it. Waypoints are planned by the IK helper;
the arm is servoed on the LPW GPU engine. No policy.

Run:  MUJOCO_GL=egl python examples/franka_push.py --record
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

    cube_start = np.array([0.5, 0.0, 0.03])
    goal_x = 0.66
    push_z = 0.05                         # fingertips at cube height
    behind = cube_start[0] - 0.09
    waypoints = [
        ([behind, 0.0, push_z + 0.15], 60),   # approach above, behind the cube
        ([behind, 0.0, push_z], 50),           # descend behind it
        ([goal_x, 0.0, push_z], 120),          # sweep forward, pushing the cube
        ([goal_x, 0.0, push_z + 0.15], 50),    # retract up
    ]

    qpos, ctrl = scene.qpos(), scene.state("ctrl")
    home_q = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    home_q[cube_adr:cube_adr + 3] = cube_start
    home_q[cube_adr + 3:cube_adr + 7] = (1, 0, 0, 0)
    qpos[:, :] = torch.as_tensor(home_q, dtype=torch.float32, device="cuda")
    scene.qvel().zero_()
    ctrl[:, :7] = torch.as_tensor(seed, dtype=torch.float32, device="cuda")
    ctrl[:, 7] = 0.0                      # gripper closed -> a solid pusher
    scene.forward()

    traj = []
    for pos, n in waypoints:
        ctrl[:, :7] = torch.as_tensor(arm_for(pos), dtype=torch.float32, device="cuda")
        for _ in range(n):
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    cube_x = qpos[:, cube_adr]
    pushed = int(((cube_x - cube_start[0]) > 0.10).sum().item())
    print(f"cube pushed forward in {pushed}/{scene.n_worlds} worlds "
          f"(final x {[round(v, 3) for v in cube_x.tolist()]}, start {cube_start[0]})")
    assert pushed == scene.n_worlds, "push failed"

    if args.record:
        from _record import record_webp
        record_webp(MJCF, np.asarray(traj), "franka_push",
                    cam={"lookat": (0.55, 0.0, 0.08), "distance": 1.4,
                         "azimuth": 150, "elevation": -18}, every=4, quality=52)


if __name__ == "__main__":
    main()
