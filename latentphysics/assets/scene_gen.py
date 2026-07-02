"""Procedural indoor scene generation (R2) — endless-curriculum substrate.

Generates parameterized, reproducible room MJCFs: floor, walls, a work table,
furniture (box approximations at this stage), and tabletop clutter objects.
Every scene is seeded — the same (seed, params) always yields the same world,
which the RSI curriculum engine relies on.

Furniture uses primitive boxes for collision (fast, robust); mesh furniture
via the convex-decomposition pipeline plugs into the same composer later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = ["RoomSpec", "generate_room"]


@dataclass
class RoomSpec:
    seed: int = 0
    size: tuple = (6.0, 5.0)          # room extent (x, y) meters
    wall_height: float = 2.5
    n_furniture: int = 8              # box furniture pieces along walls
    n_clutter: int = 6                # free-moving objects on the table
    table_pos: tuple = (0.0, 0.0)     # table center (robot mounts at -x edge)
    table_size: tuple = (0.8, 1.2, 0.4)   # half-x, half-y, height
    clutter_size: tuple = (0.02, 0.045)   # min/max half-extent of clutter
    include_robot: bool = False       # attach Franka (needs menagerie path)
    menagerie: str | None = None


def _furniture_slots(rng, spec, k):
    """Place furniture along the walls, away from the table."""
    hx, hy = spec.size[0] / 2, spec.size[1] / 2
    slots = []
    for _ in range(k):
        wall = rng.integers(4)
        d = float(rng.uniform(0.35, 0.75))          # depth off the wall
        w = float(rng.uniform(0.3, 0.9))            # half width
        h = float(rng.uniform(0.3, 1.1))            # height
        t = float(rng.uniform(-0.75, 0.75))
        if wall == 0:
            pos = (hx - d, t * (hy - 1.0)); sz = (d * 0.9, w)
        elif wall == 1:
            pos = (-(hx - d), t * (hy - 1.0)); sz = (d * 0.9, w)
        elif wall == 2:
            pos = (t * (hx - 1.0), hy - d); sz = (w, d * 0.9)
        else:
            pos = (t * (hx - 1.0), -(hy - d)); sz = (w, d * 0.9)
        slots.append((pos, sz, h))
    return slots


def generate_room(spec: RoomSpec, out_path: str) -> str:
    """Write a room MJCF and return its path."""
    rng = np.random.default_rng(spec.seed)
    hx, hy = spec.size[0] / 2, spec.size[1] / 2
    wh, wt = spec.wall_height / 2, 0.05
    tx, ty = spec.table_pos
    thx, thy, tz = spec.table_size

    # Collision masks: static room geometry must never pair with itself —
    # with 100+ static geoms, static-static candidates dominate an O(n^2)
    # broadphase (measured 10x throughput loss). MuJoCo semantics: a pair
    # collides iff (contype_a & conaffinity_b) | (contype_b & conaffinity_a).
    #   static  contype=1 conaffinity=2  -> static x static  = 0 (pruned)
    #   dynamic contype=3 conaffinity=3  -> dyn x static, dyn x dyn != 0
    #   reachable static (table/legs/floor)  contype=1 conaffinity=2
    #   unreachable static (walls/furniture) contype=4 conaffinity=8
    #     -> never pairs with dynamic (3&8=0, 4&3=0): tabletop objects cannot
    #        reach wall furniture, so those 600+ convex pairs/world are pruned
    #        at model build instead of burning GJK every step (~10x pairs cut)
    S = 'contype="1" conaffinity="2"'
    F = 'contype="4" conaffinity="8"'
    D = 'contype="3" conaffinity="3"'

    body, asset = [], []
    # shell: floor + 4 walls (static)
    body.append(f'<geom name="floor" type="plane" size="{hx} {hy} 0.1" rgba=".7 .68 .65 1" {S}/>')
    for i, (p, s) in enumerate((
        ((hx + wt, 0), (wt, hy + wt)), ((-hx - wt, 0), (wt, hy + wt)),
        ((0, hy + wt), (hx + wt, wt)), ((0, -hy - wt), (hx + wt, wt)),
    )):
        body.append(f'<geom name="wall{i}" type="box" pos="{p[0]} {p[1]} {wh}" '
                    f'size="{s[0]} {s[1]} {wh}" rgba=".85 .84 .8 1" ' + F + '/>')
    # work table (static)
    body.append(f'<geom name="table" type="box" pos="{tx} {ty} {tz - 0.02}" '
                f'size="{thx} {thy} 0.02" rgba=".55 .4 .3 1" ' + S + '/>')
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        body.append(f'<geom name="tleg{i}" type="box" '
                    f'pos="{tx + sx * (thx - 0.05)} {ty + sy * (thy - 0.05)} {(tz - 0.04) / 2}" '
                    f'size="0.04 0.04 {(tz - 0.04) / 2}" rgba=".45 .33 .25 1" ' + S + '/>')
    # furniture along walls (static boxes)
    for i, (pos, sz, h) in enumerate(_furniture_slots(rng, spec, spec.n_furniture)):
        rgba = f"{rng.uniform(.4, .8):.2f} {rng.uniform(.4, .8):.2f} {rng.uniform(.4, .8):.2f} 1"
        body.append(f'<geom name="furn{i}" type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {h / 2:.3f}" '
                    f'size="{sz[0]:.3f} {sz[1]:.3f} {h / 2:.3f}" rgba="{rgba}" ' + F + '/>')
    # tabletop clutter (free bodies)
    for i in range(spec.n_clutter):
        cs = float(rng.uniform(*spec.clutter_size))
        cx = tx + float(rng.uniform(-thx + 0.1, thx - 0.1))
        cy = ty + float(rng.uniform(-thy + 0.1, thy - 0.1))
        rgba = f"{rng.uniform(.2, .95):.2f} {rng.uniform(.2, .95):.2f} {rng.uniform(.2, .95):.2f} 1"
        shape = rng.integers(3)
        if shape == 0:
            g = f'<geom type="box" size="{cs} {cs} {cs}" rgba="{rgba}" mass="0.1" ' + D + '/>'
        elif shape == 1:
            g = f'<geom type="sphere" size="{cs}" rgba="{rgba}" mass="0.1" ' + D + '/>'
        else:
            g = f'<geom type="cylinder" size="{cs} {cs}" rgba="{rgba}" mass="0.1" ' + D + '/>'
        body.append(f'<body name="clutter{i}" pos="{cx:.3f} {cy:.3f} {tz + cs + 0.005:.3f}">'
                    f'<freejoint/>{g}</body>')

    robot = ""
    if spec.include_robot:
        men = spec.menagerie or os.path.expanduser("~/lpw/menagerie")
        panda = os.path.join(men, "franka_emika_panda", "mjx_panda.xml")
        robot = f'<include file="{panda}"/>'
        # note: include-based composition places the arm at its own worldbody
        # pose; full attach/mount composition arrives with the scene composer.

    # cameras: overhead + oblique corner view (perception targets)
    body.append(f'<camera name="overhead" pos="{tx} {ty} {tz + 1.8}" '
                f'quat="1 0 0 0" fovy="60"/>')
    body.append(f'<camera name="corner" pos="{hx - 0.8:.2f} {-(hy - 0.8):.2f} 1.8" '
                f'mode="targetbody" target="clutter0" fovy="70"/>'
                if spec.n_clutter > 0 else
                f'<camera name="corner" pos="{hx - 0.8:.2f} {-(hy - 0.8):.2f} 1.8" fovy="70"/>')

    xml = f"""<mujoco model="lpw_room_seed{spec.seed}">
  <option timestep="0.005"/>
  {robot}
  <worldbody>
    {chr(10).join('    ' + b for b in body)}
  </worldbody>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
