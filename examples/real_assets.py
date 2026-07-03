"""Assets: real CC0 meshes — scanned/modeled household objects, simulatable.

Loads a curated set of real CC0 objects (Poly Haven; fetch first with
`python scripts/fetch_assets.py`), runs them through the same importer +
convex decomposition + validity check as any asset, and drops them onto the
floor on the LPW GPU engine. Real mesh geometry — a vase, a bowl, an apple,
a rubber duck — not a box a sphere can fake.

Run:  python scripts/fetch_assets.py
      MUJOCO_GL=egl python examples/real_assets.py --record
"""

import argparse
import os

import numpy as np
import trimesh

import latentphysics as lpw
from latentphysics.assets.import_3d import (
    ImportSpec, SceneObject, _YUP_TO_ZUP, _rgba_trimesh, compose_mjcf,
)
from latentphysics.assets.validate import initial_penetration, settle, validate_model

LIB = os.path.expanduser("~/lpw/assets/library")
# (asset id, short node name, spawn xy, drop height)
LAYOUT = [
    ("ceramic_vase_01", "vase", (-0.28, 0.0), 0.55),
    ("wooden_bowl_01", "bowl", (0.26, 0.02), 0.35),
    ("food_apple_01", "apple", (0.02, 0.26), 0.30),
    ("rubber_duck_toy", "duck", (0.0, -0.28), 0.45),
]


def _load_object(asset_id, node, xy, drop):
    path = os.path.join(LIB, asset_id, f"{asset_id}_1k.gltf")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run: python scripts/fetch_assets.py")
    mesh = trimesh.load(path, force="scene").dump(concatenate=True)
    T = np.eye(4)
    T[:3, :3] = _YUP_TO_ZUP                         # glTF y-up -> MuJoCo z-up
    mesh.apply_transform(T)
    c = mesh.centroid
    lo = mesh.bounds[0]
    mesh.apply_translation([xy[0] - c[0], xy[1] - c[1], drop - lo[2]])
    return SceneObject(name=node, mesh=mesh, dynamic=True, rgba=_rgba_trimesh(mesh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=340)
    args = ap.parse_args()

    import mujoco

    objs = [_load_object(*item) for item in LAYOUT]
    out = os.path.expanduser("~/lpw/assets/demos/real_assets")
    mjcf = compose_mjcf(objs, out, "real_assets",
                        ImportSpec(threshold=0.05, max_hulls=24, density=300))

    m = mujoco.MjModel.from_xml_path(mjcf)
    print("structural:", validate_model(m))
    print("spawn contacts:", initial_penetration(m))
    assert validate_model(m).ok, "real-asset scene has structural defects"

    scene = lpw.load_scene(mjcf, lpw.Config(n_worlds=4))
    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())
    info = settle(scene, max_steps=200)
    print(f"settled: residual |v|={info['residual_vel']:.3f} converged={info['converged']}")

    import torch
    qpos = scene.qpos()
    assert torch.isfinite(qpos).all(), "real-asset sim went non-finite"
    for node in ("vase", "bowl", "apple", "duck"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, node)
        adr = int(m.jnt_qposadr[m.body_jntadr[bid]])
        z = qpos[:, adr + 2]
        assert (z > -0.02).all(), f"{node} fell through the floor"
    print("all four real objects imported, decomposed, and settled on the floor")

    if args.record:
        from _record import record_webp
        record_webp(mjcf, np.asarray(traj), "real_assets",
                    cam={"lookat": (0, 0, 0.12), "distance": 1.7,
                         "azimuth": 130, "elevation": -18, "azimuth_rate": 0.05},
                    every=4, quality=48)


if __name__ == "__main__":
    main()
