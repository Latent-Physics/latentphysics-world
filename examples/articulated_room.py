"""Worlds: articulated room — procedural interior with working doors & drawers.

Generates a seeded room whose wall furniture includes a hinged-door cabinet
and a two-drawer chest (R4), then scripts an open/close cycle on the GPU
engine and records world 0.

Run:  MUJOCO_GL=egl python examples/articulated_room.py --record
"""

import argparse
import math
import os

import numpy as np

import latentphysics as lpw
from latentphysics.assets.scene_gen import RoomSpec, generate_room


def articulated_joints(mjm):
    import mujoco
    out = []
    for j in range(mjm.njnt):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if "_door" in name or "_drawer" in name:
            out.append((name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/artic_room.xml")
    # table shifted off-center: frees the +x end of the room so both
    # articulated pieces can place near each other for one camera shot
    generate_room(RoomSpec(seed=args.seed, size=(6.2, 4.6), table_pos=(-1.4, 0.0),
                           n_articulated=2, n_furniture=30, n_clutter=5), path)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))
    joints = articulated_joints(scene.mjm)
    print("articulated joints:", [n for n, *_ in joints])

    qpos, qvel = scene.qpos(), scene.qvel()
    traj = []

    def run(n, door_v=0.0, drawer_v=0.0):
        for _ in range(n):
            for name, _, dof in joints:
                qvel[:, dof] = door_v if "_door" in name else drawer_v
            scene.step()
            traj.append(qpos[0].cpu().numpy().copy())

    run(40)                                   # settle
    run(160, door_v=1.0, drawer_v=0.28)       # open
    peak = {n: qpos[0, qa].item() for n, qa, _ in joints}
    run(60)                                   # hold (friction parks them)
    run(140, door_v=-1.0, drawer_v=-0.28)     # close
    run(40)

    print("peak openings:", {n: round(v, 3) for n, v in peak.items()})
    assert any(v > 0.7 for n, v in peak.items() if "_door" in n), "door failed to open"
    assert any(v > 0.15 for n, v in peak.items() if "_drawer" in n), "drawer failed to open"

    if args.record:
        from _record import record_webp
        # aim between the two articulated pieces
        centers = [scene.mjm.body(f"f{i}art").pos for i in range(2)]
        mid = np.mean(centers, axis=0)
        az = math.degrees(math.atan2(mid[1], mid[0]))   # look from room center
        record_webp(path, np.asarray(traj), "articulated_room",
                    cam={"lookat": (float(mid[0]) * 0.8, float(mid[1]) * 0.8, 0.5),
                         "distance": 3.4, "azimuth": az, "elevation": -22})


if __name__ == "__main__":
    main()
