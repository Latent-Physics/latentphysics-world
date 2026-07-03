"""Perception: LiDAR point cloud — a batched multi-beam scan of a room.

Casts a (channels x h_res) LiDAR scan from a mount pose on the LPW GPU
engine and renders the returned point cloud as a rotating 3D view, colored
by height. All worlds scan at once; world 0 is shown.

Run:  MUJOCO_GL=egl python examples/perception_lidar.py --record
"""

import argparse
import os

import numpy as np

import latentphysics as lpw
from latentphysics.assets.scene_gen import RoomSpec, generate_room
from latentphysics.perception import Lidar, LidarConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/lidar_room.xml")
    generate_room(RoomSpec(seed=11, n_furniture=14, n_clutter=10), path)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=2))
    scene.step(100)

    lidar = Lidar(scene, origin=(0.0, -0.5, 1.0), cfg=LidarConfig(channels=24, h_res=240))
    out = lidar.scan()
    mask = out["mask"][0].cpu().numpy()
    pts = out["points"][0][out["mask"][0]].cpu().numpy()
    hit_rate = mask.mean()
    print(f"lidar: {lidar.n_rays} rays, hit rate {hit_rate:.0%}, {len(pts)} returns")
    assert hit_rate > 0.5, f"indoor scan hit rate too low: {hit_rate:.0%}"

    if args.record:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _record import save_frames

        z = pts[:, 2]
        frames = []
        for az in range(0, 360, 6):
            fig = plt.figure(figsize=(6.4, 4.0), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], z, s=0.6, c=z, cmap="turbo")
            ax.view_init(elev=26, azim=az)
            ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.7, 2.7); ax.set_zlim(0, 2.6)
            ax.set_axis_off(); fig.tight_layout(pad=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frames.append(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                          .reshape(h, w, 4)[..., :3].copy())
            plt.close(fig)
        save_frames(frames, "lidar_pointcloud", fps=12, quality=72)


if __name__ == "__main__":
    main()
