"""Perception: depth + segmentation — the engine's batched camera output.

Renders a procedural room from the corner camera on the LPW GPU batch
renderer and shows the two learning-relevant channels side by side: metric
depth (viridis) and instance segmentation (per-geom colors). All worlds
render at once; world 0 is shown.

Run:  MUJOCO_GL=egl python examples/perception_camera.py --record
"""

import argparse
import os

import numpy as np

import latentphysics as lpw
from latentphysics.assets.scene_gen import RoomSpec, generate_room
from latentphysics.perception import CameraConfig, DepthCamera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    path = os.path.expanduser("~/lpw/assets/demos/perception_room.xml")
    generate_room(RoomSpec(seed=11, n_furniture=14, n_clutter=10), path)
    scene = lpw.load_scene(path, lpw.Config(n_worlds=4))
    cam = DepthCamera(scene, CameraConfig(res=(320, 320), rgb=True, depth=True,
                                          segmentation=True, max_depth=8.0))

    rng = np.random.default_rng(0)
    palette = (rng.uniform(0.2, 1.0, (256, 3)) * 255).astype(np.uint8)
    frames, checked = [], False
    for _ in range(args.frames):
        scene.step(6)
        f = cam.render(camera=1)                       # corner camera
        dep = f["depth"][0].cpu().numpy()
        if not checked:
            finite = dep[(dep > 0) & (dep < 8.0)]
            assert finite.size > dep.size * 0.3, "depth camera saw almost nothing"
            checked = True
        dn = np.clip(dep / 6.0, 0, 1)
        from matplotlib import cm
        dimg = (cm.viridis(1.0 - dn)[..., :3] * 255).astype(np.uint8)
        seg = f["seg"][0, ..., 0].cpu().numpy().astype(np.int64) % 256
        simg = palette[seg]
        simg[seg == 255] = 20                          # background
        frames.append(np.concatenate([dimg, simg], axis=1))

    print(f"rendered {len(frames)} depth+segmentation frames on the GPU batch renderer")
    if args.record:
        from _record import save_frames
        save_frames(frames, "depth_segmentation", fps=10, quality=72)


if __name__ == "__main__":
    main()
