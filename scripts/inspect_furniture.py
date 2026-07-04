"""Render every articulated-furniture archetype, closed and open, and look.

The modeling-fidelity bar (CLAUDE.md) requires that a real object's model is
verified by rendering it — in every articulated state — and looking, not by
joint-travel asserts alone. This is that step, made a committed, repeatable
tool: one PNG per archetype, a 2-camera x 3-open-state grid, so drawer
cavities, handles, seams, and ground contact are all visible at a glance.

Run:  MUJOCO_GL=egl python scripts/inspect_furniture.py [out_dir]
"""

import os
import sys

import numpy as np

from latentphysics.assets.scene_gen import (
    _art_drawer_chest, _art_hinged_cabinet, _art_lid_chest,
    _art_sliding_door_cabinet,
)

_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'

# (label, archetype, (depth, lateral) half-sizes, height) — matches ART_SPECS
PIECES = [
    ("drawer_chest", _art_drawer_chest, (0.20, 0.35), 0.62),
    ("hinged_cabinet", _art_hinged_cabinet, (0.22, 0.40), 0.95),
    ("lid_chest", _art_lid_chest, (0.28, 0.36), 0.50),
    ("sliding_door", _art_sliding_door_cabinet, (0.26, 0.50), 0.85),
]

TILE = (360, 300)
ASSETS = ('<asset>'
          '<texture type="skybox" builtin="gradient" rgb1="0.5 0.57 0.65" '
          'rgb2="0.15 0.17 0.2" width="256" height="256"/>'
          '<texture name="ft" type="2d" builtin="checker" rgb1="0.8 0.77 0.72" '
          'rgb2="0.7 0.66 0.6" width="300" height="300"/>'
          '<material name="fm" texture="ft" texrepeat="8 8" reflectance="0.1"/>'
          '</asset>')


def _model(archetype, sz, h):
    import mujoco

    rng = np.random.default_rng(0)
    parts, _ = archetype(rng, 0, (0.0, 0.0), sz, h, _S, _D, room_half=(1.0, 1.0))
    xml = (f'<mujoco><option timestep="0.005"/>{ASSETS}'
           '<visual><global offwidth="1280" offheight="960"/></visual>'
           '<worldbody>'
           '<light directional="true" pos="0.6 -0.4 2.5" dir="-0.3 0.2 -0.9" diffuse="0.8 0.78 0.75"/>'
           '<light directional="true" pos="-1 1 1.5" dir="0.4 -0.4 -0.8" diffuse="0.3 0.3 0.33" castshadow="false"/>'
           f'<geom name="floor" type="plane" size="3 3 0.1" material="fm" {_S}/>'
           f'{"".join(parts)}</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def _cam(mujoco, name, h):
    c = mujoco.MjvCamera()
    c.lookat[:] = (0.0, 0.0, h * 0.55)
    if name == "front":     # look into the front (piece faces -x) from above
        c.distance, c.azimuth, c.elevation = h * 2.2, 0, -30
    else:                   # 3/4 view
        c.distance, c.azimuth, c.elevation = h * 2.4, 40, -22
    return c


def main():
    import mujoco

    out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/lpw/inspect")
    os.makedirs(out, exist_ok=True)
    tw, th = TILE
    for label, archetype, sz, h in PIECES:
        m = _model(archetype, sz, h)
        d = mujoco.MjData(m)
        # every non-free (articulated) joint, set to open fractions of its range
        jnt = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
               or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        r = mujoco.Renderer(m, height=th, width=tw)
        cams = ["front", "34"]
        canvas = np.zeros((th * len(cams), tw * 3, 3), dtype=np.uint8)
        for col, frac in enumerate((0.0, 0.5, 1.0)):
            mujoco.mj_resetData(m, d)
            for j in jnt:
                lo, hi = m.jnt_range[j]
                adr = m.jnt_qposadr[j]
                d.qpos[adr] = lo + frac * (hi - lo)
            mujoco.mj_forward(m, d)
            for row, cam in enumerate(cams):
                r.update_scene(d, camera=_cam(mujoco, cam, h))
                tile = r.render()
                canvas[row * th:(row + 1) * th, col * tw:(col + 1) * tw] = tile
        r.close()
        import imageio.v2 as imageio
        path = os.path.join(out, f"{label}.png")
        imageio.imwrite(path, canvas)
        print(f"wrote {path}  (cols: closed / half / open; rows: front / 3-4)")


if __name__ == "__main__":
    main()
