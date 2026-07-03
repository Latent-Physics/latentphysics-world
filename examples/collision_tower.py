"""Rigid: collision tower — a ring tower of boxes toppled by a heavy ball.

Physics runs on the LPW GPU engine (batched); world 0 is recorded for the
gallery. Run:  MUJOCO_GL=egl python examples/collision_tower.py --record
"""

import argparse
import math
import os

import numpy as np

import latentphysics as lpw

LAYERS = 6
PER_LAYER = 8
RADIUS = 0.24
BOX = (0.085, 0.032, 0.045)   # half-extents: length, thickness, height


ASSETS = ('<asset>'
          '<texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62" '
          'rgb2="0.12 0.14 0.18" width="256" height="256"/>'
          '<texture name="floortex" type="2d" builtin="checker" rgb1="0.78 0.74 0.68" '
          'rgb2="0.68 0.64 0.58" mark="edge" markrgb="0.55 0.52 0.48" width="300" height="300"/>'
          '<material name="floormat" texture="floortex" texrepeat="12 12" reflectance="0.12"/>'
          '</asset>')


def build_scene(path: str) -> str:
    body = ['<geom name="floor" type="plane" size="4 4 0.1" material="floormat"/>']
    body.append('<light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9" '
                'diffuse="0.8 0.78 0.75" castshadow="true"/>')
    body.append('<light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8" '
                'diffuse="0.25 0.25 0.28" castshadow="false"/>')
    woods = ((0.68, 0.52, 0.33), (0.6, 0.44, 0.28), (0.52, 0.38, 0.24))
    k = 0
    for lay in range(LAYERS):
        z = BOX[2] * (2 * lay + 1)
        for i in range(PER_LAYER):
            a = 2 * math.pi * (i + 0.5 * (lay % 2)) / PER_LAYER
            x, y = RADIUS * math.cos(a), RADIUS * math.sin(a)
            c = woods[k % 3]
            body.append(
                f'<body name="b{k}" pos="{x:.4f} {y:.4f} {z:.4f}" '
                f'euler="0 0 {math.degrees(a) + 90:.2f}"><freejoint/>'
                f'<geom type="box" size="{BOX[0]} {BOX[1]} {BOX[2]}" '
                f'rgba="{c[0]} {c[1]} {c[2]} 1" mass="0.08"/></body>')
            k += 1
    body.append('<body name="ball" pos="-1.6 0 0.35"><freejoint/>'
                '<geom type="sphere" size="0.09" rgba=".35 .35 .38 1" mass="2.5"/></body>')
    xml = ('<mujoco model="collision_tower">'
           '<option timestep="0.004" iterations="10" ls_iterations="10"/>'
           + ASSETS +
           '<worldbody>' + "".join(body) + '</worldbody></mujoco>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(xml)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--steps", type=int, default=560)
    args = ap.parse_args()

    scene_path = os.path.expanduser("~/lpw/assets/demos/tower.xml")
    build_scene(scene_path)
    # 49 free bodies in one pile need ~1152 constraint rows per world at
    # collapse. The auto budget scales njmax with dynamic-geom count (2048
    # here), and an undersized cap raises BudgetOverflow at step time instead
    # of silently corrupting qpos to NaN — no explicit njmax needed anymore.
    scene = lpw.load_scene(scene_path, lpw.Config(n_worlds=4))

    # fling the ball at the tower (free-joint linear dofs of the last body)
    qv = scene.qvel()
    qv[:, -6] = 5.5          # vx toward the tower
    qv[:, -4] = 1.2          # slight upward arc
    scene.forward()

    traj = []
    for _ in range(args.steps):
        scene.step()
        traj.append(scene.qpos()[0].cpu().numpy().copy())
    top_z = traj[-1][2 + 7 * (LAYERS * PER_LAYER - 1)]
    print(f"simulated {args.steps} steps on GPU; top-layer box final z = {top_z:.3f} "
          f"({'toppled' if top_z < 0.35 else 'still standing'})")
    import torch
    assert torch.isfinite(scene.qpos()).all(), "tower sim went non-finite"
    # the top ring rests at z=0.495 when standing; 0.35 is a real "toppled"
    assert top_z < 0.35, f"ball failed to topple the tower: top-layer z = {top_z:.3f}"

    if args.record:
        from _record import record_webp
        record_webp(scene_path, np.asarray(traj), "collision_tower",
                    cam={"lookat": (0, 0, 0.28), "distance": 2.3,
                         "azimuth": 150, "elevation": -16, "azimuth_rate": 0.05},
                    every=5, fps=12, quality=44)


if __name__ == "__main__":
    main()
