"""Rigid: franka tool use — grasp a hooked stick, drag a far cube into reach.

The cube sits at x=0.68, beyond every direct-grasp target any demo uses
(max prior gripper x: 0.59). The arm grasps a stick whose far end carries a
downward blade, reaches PAST the cube with the shaft passing safely above
it, lowers the blade behind the cube, and drags it back inside the
workspace. IK-scripted waypoints — no policy, no learning.

Run:  MUJOCO_GL=egl python examples/franka_tool_use.py --record
"""

import argparse

import numpy as np

import latentphysics as lpw

STICK = (0.44, -0.16)   # shaft center; shaft along +x
CUBE = (0.68, 0.02)     # beyond direct reach; the tool brings it back
GRASP_X = 0.45          # grasp at the stick's balance point: near-zero pitch
                        # torque is what keeps the pinch from tumbling it
PHYS = 'condim="4" friction="1 0.05 0.001" contype="1" conaffinity="1"'
# the grasped shaft needs ROLLING friction (condim 6): with condim 4 the
# stick pivots freely about the finger-contact line and the end-weights'
# gravity torque tumbles it out of the pinch mid-carry
GRIP = 'condim="6" friction="1 0.05 0.02" contype="1" conaffinity="1"'


REST_Z = 0.058              # stick body z at rest (blade/foot bottoms on floor)


def bodies():
    stick = (
        f'<body name="stick" pos="{STICK[0]} {STICK[1]} {REST_Z}"><freejoint/>'
        # 32 cm shaft, ~2.2x2.4 cm cross-section (graspable on either finger
        # axis); carries almost all the mass so the pinch sits at the balance
        # point — an unbalanced tool pivots about the finger-contact line and
        # tumbles out of the grip mid-carry
        f'<geom type="box" size="0.16 0.011 0.012" rgba="0.62 0.42 0.22 1" mass="0.07" {GRIP}/>'
        # deep downward blade at the +x end: the shaft crosses ABOVE the cube
        # while the blade hangs behind it — a straight low stick would land
        # on the cube during the reach-over
        f'<geom type="box" pos="0.16 0 -0.023" size="0.012 0.06 0.035" '
        f'rgba="0.5 0.34 0.18 1" mass="0.008" {PHYS}/>'
        # matching foot at the -x end so the stick rests level for the grasp
        f'<geom type="box" pos="-0.15 0 -0.023" size="0.012 0.011 0.035" '
        f'rgba="0.5 0.34 0.18 1" mass="0.004" {PHYS}/>'
        '</body>')
    cube = (f'<body name="cube" pos="{CUBE[0]} {CUBE[1]} 0.025"><freejoint/>'
            f'<geom type="box" size="0.025 0.025 0.025" rgba="0.85 0.3 0.25 1" '
            f'mass="0.03" {PHYS}/></body>')
    return stick + cube


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    import mujoco
    import torch

    from _ik import ArmController, tilt_quat
    from _scene import franka_scene

    obj_qpos = (f"{STICK[0]} {STICK[1]} {REST_Z} 1 0 0 0  "
                f"{CUBE[0]} {CUBE[1]} 0.025 1 0 0 0")
    path = franka_scene("tool_use", bodies(), obj_qpos)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))
    arm = ArmController(scene)
    O, C = arm.OPEN, arm.CLOSED

    mjm = scene.mjm
    bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_adr = int(mjm.jnt_qposadr[mjm.body_jntadr[bid]])

    gx, gy = GRASP_X, STICK[1]
    gz = REST_Z + 0.008         # deep pinch: more finger-pad contact on the shaft
    # the default downq closes the fingers along world x — parallel to the
    # shaft, so they'd land ON it. Yaw the wrist 90° so they straddle it.
    yawq = tilt_quat(arm.downq, np.pi / 2, axis=(0, 0, 1))
    arm.move([gx, gy, gz + 0.14], O, 60, quat=yawq)    # approach above the shaft
    arm.move([gx, gy, gz], O, 60, quat=yawq)           # descend around it
    arm.hold(C, 60)                                    # grasp the shaft
    arm.move([gx, gy, 0.20], C, 70, quat=yawq)         # lift the tool
    # two-leg carry: straight +x on the -y side, then sidestep into the drag
    # lane from BEYOND the cube — a diagonal sweep grazes the cube's top edge
    # and knocks it out of the lane before the blade ever drops
    arm.move([0.585, -0.10, 0.20], C, 100, quat=yawq)  # leg 1: advance clear of the cube
    arm.move([0.585, CUBE[1], 0.20], C, 60, quat=yawq)  # leg 2: blade now past the cube
    arm.move([0.585, CUBE[1], 0.085], C, 60, quat=yawq)  # lower: shaft above cube, blade behind
    # drag as interpolated sub-waypoints: one far target makes the arm servo
    # at full joint speed and the blade jerks up over the cube; ~2 cm legs
    # keep the blade low, slow, and engaged all the way back
    for k in range(1, 10):
        arm.move([0.585 - 0.0228 * k, CUBE[1], 0.085], C, 20, quat=yawq)
    arm.move([0.38, CUBE[1], 0.20], C, 40, quat=yawq)  # lift away

    qpos = scene.qpos()
    assert torch.isfinite(qpos).all(), "tool-use sim went non-finite"
    cube_x = qpos[:, cube_adr]
    pulled = int((cube_x < 0.58).sum().item())
    print(f"cube dragged from x={CUBE[0]:.2f} into reach (<0.58) in "
          f"{pulled}/{scene.n_worlds} worlds "
          f"(final x {[round(v, 3) for v in cube_x.tolist()]})")
    assert pulled == scene.n_worlds, "tool-use drag failed"

    if args.record:
        from _record import record_webp
        record_webp(path, np.asarray(arm.traj), "franka_tool_use",
                    cam={"lookat": (0.5, -0.05, 0.1), "distance": 1.5,
                         "azimuth": 145, "elevation": -20}, every=5, quality=48)


if __name__ == "__main__":
    main()
