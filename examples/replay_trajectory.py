"""Replay a recorded qpos trajectory and export an mp4 (offscreen render).

    MUJOCO_GL=egl python examples/replay_trajectory.py \
        --traj ~/lpw/runs/franka_reach_traj.npy --out ~/lpw/runs/franka_reach.mp4

Rendering uses the C-engine visualizer on the CPU model — replay only, no
physics re-simulation (the trajectory is the ground truth from the GPU run).
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    default_scene = os.path.join(
        os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
        "franka_emika_panda", "mjx_single_cube.xml")
    ap.add_argument("--mjcf", default=default_scene)
    ap.add_argument("--traj", default=os.path.expanduser("~/lpw/runs/franka_reach_traj.npy"))
    ap.add_argument("--target", default=os.path.expanduser("~/lpw/runs/franka_reach_target.npy"))
    ap.add_argument("--out", default=os.path.expanduser("~/lpw/runs/franka_reach.mp4"))
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--res", type=int, nargs=2, default=(640, 480))
    args = ap.parse_args()

    import imageio
    import mujoco

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    traj = np.load(args.traj)

    # visualize the reach target as a mocap body if the model has one
    target = None
    if os.path.exists(args.target):
        target = np.load(args.target)
        if m.nmocap > 0:
            d.mocap_pos[0] = target

    renderer = mujoco.Renderer(m, height=args.res[1], width=args.res[0])
    frames = []
    for qpos in traj:
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        renderer.update_scene(d)
        frames.append(renderer.render())
    renderer.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out} ({len(frames)} frames @ {args.fps} fps)")
    print("REPLAY_DONE")


if __name__ == "__main__":
    main()
