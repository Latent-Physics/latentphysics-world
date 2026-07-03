"""Assets: convex decomposition — a concave mesh becomes simulatable.

A torus (genuinely non-convex — it has a hole) is split by the CoACD
pipeline into convex collision parts. Each part is drawn a different color
and the assembled ring is dropped and tumbled on the LPW GPU engine: you
can see the decomposition itself, and the hole stays open as it rolls —
proof the parts captured the real shape, not a filled-in hull.

Run:  MUJOCO_GL=egl python examples/convex_decomposition.py --record
"""

import argparse
import os

import numpy as np
import trimesh

import latentphysics as lpw
from latentphysics.assets import convex_decompose
from latentphysics.assets.materials import material_assets

# distinct hull colors so the decomposition reads at a glance
_PALETTE = [(0.86, 0.32, 0.28), (0.30, 0.55, 0.80), (0.40, 0.70, 0.42),
            (0.88, 0.70, 0.28), (0.60, 0.42, 0.72), (0.32, 0.72, 0.72),
            (0.85, 0.52, 0.30), (0.70, 0.36, 0.52)]


def build_scene(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.creation.torus(major_radius=0.3, minor_radius=0.1)
    parts = convex_decompose(mesh, threshold=0.05)
    assets, geoms = [], []
    for i, p in enumerate(parts):
        fn = f"hull_{i}.obj"
        trimesh.Trimesh(vertices=p.vertices, faces=p.faces).export(os.path.join(out_dir, fn))
        col = _PALETTE[i % len(_PALETTE)]
        assets.append(f'<mesh name="h{i}" file="{fn}"/>')
        geoms.append(f'<geom type="mesh" mesh="h{i}" rgba="{col[0]} {col[1]} {col[2]} 1" '
                     f'contype="1" conaffinity="1"/>')
    xml = f"""<mujoco model="convex_decomposition">
  <compiler meshdir="." angle="radian"/>
  <option timestep="0.004"/>
  <asset>
    {material_assets()}
    {"".join(assets)}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9"
           diffuse="0.85 0.83 0.8" castshadow="true"/>
    <light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8"
           diffuse="0.3 0.3 0.33" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="mat_plaster"
          rgba=".8 .75 .68 1" contype="1" conaffinity="1"/>
    <body name="torus" pos="0 0 0.75" euler="0.5 0.2 0">
      <freejoint/>
      {"".join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""
    path = os.path.join(out_dir, "torus_hulls.xml")
    with open(path, "w") as f:
        f.write(xml)
    print(f"decomposed torus into {len(parts)} convex parts")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=360)
    args = ap.parse_args()

    import torch
    mjcf = build_scene(os.path.expanduser("~/lpw/assets/demos/torus"))
    scene = lpw.load_scene(mjcf, lpw.Config(n_worlds=4))
    qv = scene.qvel()
    qv[:, 1] = 0.6            # drift into frame
    qv[:, 3] = 6.0            # spin -> lands on edge and rolls
    scene.forward()

    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())
    assert torch.isfinite(scene.qpos()).all(), "torus sim went non-finite"
    z = scene.qpos()[:, 2]
    assert (z > -0.02).all(), "torus fell through the floor"
    print(f"simulated {args.steps} steps on GPU; resting z={z[0].item():.3f}")

    if args.record:
        from _record import record_webp
        record_webp(mjcf, np.asarray(traj), "convex_decomposition",
                    cam={"lookat": (0.0, 0.2, 0.15), "distance": 1.7,
                         "azimuth": 120, "elevation": -18, "azimuth_rate": 0.04},
                    every=4, quality=50)


if __name__ == "__main__":
    main()
