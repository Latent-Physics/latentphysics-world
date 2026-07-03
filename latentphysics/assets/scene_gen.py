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
    # Newton solver budget. MuJoCo's defaults (100/50) are tuned for offline
    # accuracy; under CUDA-graph capture the solver while-loop unrolls into
    # iterations x ls_iterations conditional nodes, so defaults make graph
    # replay SLOWER than eager (measured 18.6 vs 12.9 ms/step). MJX-tuned
    # scenes ship 5/8; we default slightly higher for contact-rich clutter.
    solver_iterations: int = 8
    ls_iterations: int = 10


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


# curated interior palette (r, g, b) — furniture reads as wood/fabric, not crates
_WOODS = ((0.55, 0.38, 0.22), (0.68, 0.52, 0.33), (0.40, 0.28, 0.18), (0.62, 0.47, 0.32))
_FABRICS = ((0.52, 0.56, 0.48), (0.52, 0.53, 0.58), (0.40, 0.45, 0.55), (0.58, 0.50, 0.44))


def _jit(rng, c, s=0.04):
    return "%.2f %.2f %.2f 1" % tuple(min(1.0, max(0.0, v + rng.uniform(-s, s))) for v in c)


def _arch_shelf(rng, i, pos, sz, h, S):
    """Open bookshelf: two side panels + boards."""
    wood = _jit(rng, _WOODS[rng.integers(len(_WOODS))])
    x, y = pos
    sx, sy = sz
    t = 0.02
    g = []
    for k, yo in enumerate((-sy + t, sy - t)):
        g.append(f'<geom name="f{i}s{k}" type="box" pos="{x:.3f} {y + yo:.3f} {h/2:.3f}" '
                 f'size="{sx:.3f} {t} {h/2:.3f}" rgba="{wood}" {S}/>')
    nb = 3 if h > 0.7 else 2
    for k in range(nb):
        z = h * (k + 0.5) / nb
        g.append(f'<geom name="f{i}b{k}" type="box" pos="{x:.3f} {y:.3f} {z:.3f}" '
                 f'size="{sx:.3f} {sy - t:.3f} {t}" rgba="{wood}" {S}/>')
    return g


def _arch_cabinet(rng, i, pos, sz, h, S):
    """Cabinet/wardrobe: body + top overhang + darker kick base."""
    wood = _WOODS[rng.integers(len(_WOODS))]
    dark = _jit(rng, tuple(v * 0.6 for v in wood))
    x, y = pos
    sx, sy = sz
    return [
        f'<geom name="f{i}k" type="box" pos="{x:.3f} {y:.3f} 0.04" size="{sx*0.92:.3f} {sy*0.92:.3f} 0.04" rgba="{dark}" {S}/>',
        f'<geom name="f{i}b" type="box" pos="{x:.3f} {y:.3f} {h/2+0.04:.3f}" size="{sx:.3f} {sy:.3f} {h/2-0.06:.3f}" rgba="{_jit(rng, wood)}" {S}/>',
        f'<geom name="f{i}t" type="box" pos="{x:.3f} {y:.3f} {h:.3f}" size="{sx+0.015:.3f} {sy+0.015:.3f} 0.015" rgba="{dark}" {S}/>',
    ]


def _arch_sofa(rng, i, pos, sz, h, S):
    """Sofa: seat + backrest + two armrests (fabric tones)."""
    fab = _FABRICS[rng.integers(len(_FABRICS))]
    x, y = pos
    sx, sy = sz
    seat_h = 0.22
    back_h = max(min(h, 0.75), 2 * seat_h + 0.12)   # low slots: keep a positive backrest
    c1, c2 = _jit(rng, fab, 0.02), _jit(rng, tuple(v * 0.85 for v in fab), 0.02)
    ax = 0.9 * sx
    return [
        f'<geom name="f{i}seat" type="box" pos="{x:.3f} {y:.3f} {seat_h:.3f}" size="{sx:.3f} {sy:.3f} {seat_h:.3f}" rgba="{c1}" {S}/>',
        f'<geom name="f{i}back" type="box" pos="{x:.3f} {y:.3f} {seat_h*2 + (back_h-seat_h*2)/2:.3f}" size="{sx*0.35:.3f} {sy:.3f} {(back_h-seat_h*2)/2:.3f}" rgba="{c2}" {S}/>',
        f'<geom name="f{i}a0" type="box" pos="{x:.3f} {y - sy:.3f} {seat_h*1.5:.3f}" size="{ax:.3f} 0.07 {seat_h*1.5:.3f}" rgba="{c2}" {S}/>',
        f'<geom name="f{i}a1" type="box" pos="{x:.3f} {y + sy:.3f} {seat_h*1.5:.3f}" size="{ax:.3f} 0.07 {seat_h*1.5:.3f}" rgba="{c2}" {S}/>',
    ]


def _arch_sideboard(rng, i, pos, sz, h, S):
    """Low sideboard: top slab + two side panels."""
    wood = _WOODS[rng.integers(len(_WOODS))]
    hh = min(h, 0.5)
    x, y = pos
    sx, sy = sz
    c = _jit(rng, wood)
    return [
        f'<geom name="f{i}t" type="box" pos="{x:.3f} {y:.3f} {hh:.3f}" size="{sx:.3f} {sy:.3f} 0.02" rgba="{c}" {S}/>',
        f'<geom name="f{i}p0" type="box" pos="{x:.3f} {y - sy + 0.02:.3f} {hh/2:.3f}" size="{sx*0.9:.3f} 0.02 {hh/2:.3f}" rgba="{c}" {S}/>',
        f'<geom name="f{i}p1" type="box" pos="{x:.3f} {y + sy - 0.02:.3f} {hh/2:.3f}" size="{sx*0.9:.3f} 0.02 {hh/2:.3f}" rgba="{c}" {S}/>',
    ]


_ARCHETYPES = (_arch_cabinet, _arch_shelf, _arch_sofa, _arch_sideboard)


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
    # lighting: key light with shadows + soft fill (flat default headlight
    # alone reads like an untextured factory hall)
    body.append('<light name="key" directional="true" pos="0 0 3" dir="-0.35 0.25 -0.9" '
                'diffuse="0.75 0.73 0.70" specular="0.1 0.1 0.1" castshadow="true"/>')
    body.append('<light name="fill" directional="true" pos="0 0 3" dir="0.4 -0.3 -0.85" '
                'diffuse="0.34 0.35 0.38" specular="0 0 0" castshadow="false"/>')
    # shell: warm wood floor + TRANSLUCENT walls (alpha is visual only —
    # collision is unchanged; opaque walls occlude the interior from any
    # outside camera, which made demos unreadable)
    body.append(f'<geom name="floor" type="plane" size="{hx} {hy} 0.1" rgba=".72 .62 .48 1" {S}/>')
    for i, (p, s) in enumerate((
        ((hx + wt, 0), (wt, hy + wt)), ((-hx - wt, 0), (wt, hy + wt)),
        ((0, hy + wt), (hx + wt, wt)), ((0, -hy - wt), (hx + wt, wt)),
    )):
        body.append(f'<geom name="wall{i}" type="box" pos="{p[0]} {p[1]} {wh}" '
                    f'size="{s[0]} {s[1]} {wh}" rgba=".93 .91 .88 0.22" ' + F + '/>')
    # work table (static)
    body.append(f'<geom name="table" type="box" pos="{tx} {ty} {tz - 0.02}" '
                f'size="{thx} {thy} 0.02" rgba=".55 .4 .3 1" ' + S + '/>')
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        body.append(f'<geom name="tleg{i}" type="box" '
                    f'pos="{tx + sx * (thx - 0.05)} {ty + sy * (thy - 0.05)} {(tz - 0.04) / 2}" '
                    f'size="0.04 0.04 {(tz - 0.04) / 2}" rgba=".45 .33 .25 1" ' + S + '/>')
    # furniture along the walls — multi-box archetypes (cabinet / shelf /
    # sofa / sideboard) under a geom budget. n_furniture ~= number of static
    # furniture GEOMS (keeps large-scene benchmarks comparable); every piece
    # carries the unreachable-static mask, so visual richness costs nothing
    # in collision pairs.
    budget = spec.n_furniture
    fi = 0
    for pos, sz, h in _furniture_slots(rng, spec, max(4, spec.n_furniture)):
        if budget <= 0:
            break
        arch = _ARCHETYPES[int(rng.integers(len(_ARCHETYPES)))]
        geoms = arch(rng, fi, pos, sz, h, F)
        body.extend(geoms)
        budget -= len(geoms)
        fi += 1
    # tabletop clutter (free bodies) — warm accent palette, not random RGB
    accents = ((0.76, 0.33, 0.25), (0.86, 0.68, 0.30), (0.34, 0.55, 0.62),
               (0.55, 0.65, 0.45), (0.82, 0.79, 0.72), (0.47, 0.41, 0.60))
    for i in range(spec.n_clutter):
        cs = float(rng.uniform(*spec.clutter_size))
        cx = tx + float(rng.uniform(-thx + 0.1, thx - 0.1))
        cy = ty + float(rng.uniform(-thy + 0.1, thy - 0.1))
        rgba = _jit(rng, accents[int(rng.integers(len(accents)))], 0.05)
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
  <option timestep="0.005" iterations="{spec.solver_iterations}" ls_iterations="{spec.ls_iterations}"/>
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
