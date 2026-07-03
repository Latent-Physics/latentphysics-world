"""Rigid: franka stack — grasp one cube and stack it on another (IK-scripted).

A precision task: pick the red cube and place it on top of the green cube.
Waypoints planned by the IK helper; arm servoed on the LPW GPU engine. No
policy.

Run:  MUJOCO_GL=egl python examples/franka_stack.py --record
"""

import argparse
import os

import numpy as np
import torch

import latentphysics as lpw

MEN = os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie"))
HALF = 0.025          # cube half-extent


def build_scene() -> str:
    """Franka + two free cubes on the table (own scene, written by the panda)."""
    # include mjx_scene (not mjx_panda): brings the arm PLUS the blue
    # checkerboard groundplane + skybox + lighting, matching pick & place
    xml = f"""<mujoco model="franka_stack">
  <include file="mjx_scene.xml"/>
  <worldbody>
    <body name="cubeA" pos="0.5 -0.12 {HALF}"><freejoint/>
      <geom type="box" size="{HALF} {HALF} {HALF}" rgba="0.2 0.7 0.3 1" mass="0.05"
            condim="4" friction="1 0.05 0.001" contype="1" conaffinity="1"/></body>
    <body name="cubeB" pos="0.5 0.12 {HALF}"><freejoint/>
      <geom type="box" size="{2*HALF} {2*HALF} {HALF}" rgba="0.8 0.25 0.2 1" mass="0.2"
            condim="4" friction="1 0.05 0.001" contype="1" conaffinity="1"/></body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04 0.5 -0.12 {HALF} 1 0 0 0 0.5 0.12 {HALF} 1 0 0 0"
         ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
"""
    path = os.path.join(MEN, "franka_emika_panda", "_lpw_stack.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    from _ik import gripper_down_quat, solve_ik

    scene = lpw.load_scene(build_scene(), lpw.Config(n_worlds=4))
    mjm = scene.mjm
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    aA = int(mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cubeA")]])
    aB = int(mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cubeB")]])
    seed = mjm.key_qpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")][:7].copy()
    downq = gripper_down_quat(mjm, sid)
    ik_data = mujoco.MjData(mjm)

    cur_seed = [seed.copy()]

    def arm_for(pos):
        # seed each solve from the previous solution so consecutive waypoints
        # give CONTINUOUS arm motion (seeding from home each time lets the IK
        # jump branches between waypoints and sweep the wrist sideways)
        cur_seed[0] = solve_ik(mjm, ik_data, sid, np.asarray(pos, float), downq, cur_seed[0])
        return cur_seed[0]

    A = np.array([0.5, -0.12])            # pick (green)
    B = np.array([0.5, 0.12])             # base (red)
    gz = HALF + 0.015                     # gripper-site z when grasping a cube on the table
    stack_z = 2 * HALF + gz               # site z to release A on top of B
    waypoints = [
        ([A[0], A[1], gz + 0.14], True, 60),
        ([A[0], A[1], gz], True, 60),
        ([A[0], A[1], gz], False, 60),          # grasp A
        ([A[0], A[1], gz + 0.20], False, 60),   # lift
        ([B[0], B[1], stack_z + 0.20], False, 90),   # carry over B
        ([B[0], B[1], stack_z + 0.004], False, 80),  # lower onto B (near resting)
        ([B[0], B[1], stack_z + 0.004], True, 50),   # open gripper, let A settle
        ([B[0], B[1], stack_z + 0.045], True, 40),   # gentle straight-up clearance
        ([B[0], B[1], stack_z + 0.24], True, 70),    # full retract (A stays on B)
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

    # success: A rests on top of B (A above B by ~one cube, xy aligned)
    za, zb = qpos[:, aA + 2], qpos[:, aB + 2]
    dxy = torch.linalg.norm(qpos[:, aA:aA + 2] - qpos[:, aB:aB + 2], dim=-1)
    stacked = int(((za - zb > 0.03) & (za - zb < 0.075) & (dxy < 0.04)).sum().item())
    print(f"stacked in {stacked}/{scene.n_worlds} worlds "
          f"(dz {[round(v,3) for v in (za-zb).tolist()]}, dxy {[round(v,3) for v in dxy.tolist()]})")
    assert stacked == scene.n_worlds, "stack failed"

    if args.record:
        from _record import record_webp
        record_webp(build_scene(), np.asarray(traj), "franka_stack",
                    cam={"lookat": (0.5, 0.0, 0.08), "distance": 1.4,
                         "azimuth": 145, "elevation": -18}, every=4, quality=52)


if __name__ == "__main__":
    main()
