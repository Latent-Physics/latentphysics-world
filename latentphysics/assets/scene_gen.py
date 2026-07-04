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
    # Articulated furniture (R4): hinged cabinet doors + sliding drawers.
    # Default 0 keeps every existing (seed, spec) room byte-identical —
    # articulated pieces draw from their own rng stream for the same reason.
    n_articulated: int = 0
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


def _facing_quat(pos, room_half):
    """Rotation (as MJCF quat string) taking local +x to 'toward room center',
    plus the local-frame (depth, lateral) half-sizes given the world sz."""
    import math
    fx, fy = abs(pos[0]) / room_half[0], abs(pos[1]) / room_half[1]
    if fx >= fy:
        theta = 0.0 if pos[0] < 0 else math.pi          # face +x / -x
        along_x = True
    else:
        theta = math.pi / 2 if pos[1] < 0 else -math.pi / 2
        along_x = False
    q = f"{math.cos(theta/2):.6f} 0 0 {math.sin(theta/2):.6f}"
    return q, along_x


def _art_zone(pos, sz, room_half, reach):
    """AABB covering a wall piece's footprint PLUS the zone its door arc /
    drawer travel sweeps (``reach``, toward the room center). Articulated
    parts keep live collision with room contents — placement, not mask
    pruning, is what guarantees they can actually open."""
    fx, fy = abs(pos[0]) / room_half[0], abs(pos[1]) / room_half[1]
    # the arc/travel sweeps FORWARD into the room; laterally it only needs a
    # small margin (full-reach inflation on both sides pushed articulated
    # pieces onto opposite walls of any medium room)
    lat = 0.3 * reach + 0.1
    if fx >= fy:
        s = -1.0 if pos[0] > 0 else 1.0
        x0, x1 = sorted((pos[0] - s * sz[0], pos[0] + s * (sz[0] + reach)))
        y0, y1 = pos[1] - sz[1] - lat, pos[1] + sz[1] + lat
    else:
        s = -1.0 if pos[1] > 0 else 1.0
        y0, y1 = sorted((pos[1] - s * sz[1], pos[1] + s * (sz[1] + reach)))
        x0, x1 = pos[0] - sz[0] - lat, pos[0] + sz[0] + lat
    return x0, x1, y0, y1


def _overlaps(a, b):
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])


def _art_hinged_cabinet(rng, i, pos, sz, h, F, D, room_half):
    """Cabinet with a working hinged door (vertical axis, opens into the room).

    Carcass geoms are static (F mask: wall furniture is unreachable by table
    clutter). The door is a jointed body with the dynamic mask so it pairs
    with robots/objects; door-vs-carcass contact is pruned by the masks and
    constrained by the joint limit instead.
    """
    import math
    wood = _WOODS[rng.integers(len(_WOODS))]
    dark = _jit(rng, tuple(v * 0.6 for v in wood))
    c = _jit(rng, wood)
    q, along_x = _facing_quat(pos, room_half)
    dx = sz[0] if along_x else sz[1]                    # depth (local x)
    dy = sz[1] if along_x else sz[0]                    # lateral (local y)
    # cap the width: a single hinged door wider than ~1 m is neither
    # realistic nor placeable (swing arc grows with door width)
    dx, dy = max(dx, 0.22), min(max(dy, 0.35), 0.5)
    h = max(h, 0.9)
    t = 0.02
    zh = (h - 0.15) / 2                                 # door half-height
    zc = 0.12 + zh                                      # door center z
    g = [
        f'<geom name="f{i}k" type="box" pos="0 0 0.04" size="{dx*0.9:.3f} {dy*0.9:.3f} 0.04" rgba="{dark}" {F}/>',
        f'<geom name="f{i}bt" type="box" pos="0 0 0.10" size="{dx:.3f} {dy - 2*t:.3f} 0.02" rgba="{c}" {F}/>',
        f'<geom name="f{i}bk" type="box" pos="{-(dx - t):.3f} 0 {h/2:.3f}" size="{t} {dy - 2*t:.3f} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s0" type="box" pos="0 {-(dy - t):.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s1" type="box" pos="0 {dy - t:.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}t" type="box" pos="0 0 {h:.3f}" size="{dx + 0.015:.3f} {dy + 0.015:.3f} 0.015" rgba="{dark}" {F}/>',
    ]
    metal = "0.78 0.79 0.82 1"
    # a shelf inside at mid-height: an open cabinet must read as a cabinet,
    # not an empty shell
    g.append(f'<geom name="f{i}shelf" type="box" pos="0 0 {0.12 + (h - 0.12)*0.5:.3f}" '
             f'size="{dx*0.9:.3f} {dy - 2*t:.3f} 0.012" rgba="{c}" {F}/>')
    dw = dy - t                                          # door half-width
    hy = -2 * dw + 0.06                                  # handle near the free edge
    door = (
        f'<body name="f{i}door" pos="{dx + 0.02:.3f} {dw:.3f} {zc:.3f}">'
        f'<joint name="f{i}_door" type="hinge" axis="0 0 1" range="0 108" '
        f'damping="2.5" frictionloss="0.8"/>'
        f'<geom name="f{i}d" type="box" pos="0 {-dw:.3f} 0" size="0.015 {dw:.3f} {zh:.3f}" '
        f'rgba="{_jit(rng, wood, 0.02)}" density="500" {D}/>'
        # metal D-pull: two standoff posts + a vertical bar proud of the face
        f'<geom name="f{i}dpt" type="box" pos="0.028 {hy:.3f} 0.075" size="0.014 0.01 0.01" rgba="{metal}" density="700" {D}/>'
        f'<geom name="f{i}dpb" type="box" pos="0.028 {hy:.3f} -0.075" size="0.014 0.01 0.01" rgba="{metal}" density="700" {D}/>'
        f'<geom name="f{i}dh" type="box" pos="0.048 {hy:.3f} 0" size="0.01 0.012 0.078" '
        f'rgba="{metal}" density="700" {D}/>'
        f'</body>'
    )
    xml = (f'<body name="f{i}art" pos="{pos[0]:.3f} {pos[1]:.3f} 0" quat="{q}">'
           + "".join(g) + door + '</body>')
    return [xml], len(g) + 4


def _art_drawer_chest(rng, i, pos, sz, h, F, D, room_half):
    """Low face-frame chest with two working drawers on slide joints.

    Each drawer is a HOLLOW tray (floor + four low walls) behind a tall front
    face — pulled open it reads as a real drawer, not a solid block. The two
    faces fill the front with tight ~4 mm reveals around a visible face-frame
    rail (no floaty gaps), and each carries a proud metal U-pull that
    contrasts against the wood so the handle is actually visible.
    """
    wood = _WOODS[rng.integers(len(_WOODS))]
    dark = _jit(rng, tuple(v * 0.6 for v in wood))
    c = _jit(rng, wood)
    face = _jit(rng, wood, 0.02)
    metal = "0.78 0.79 0.82 1"                          # brushed pulls: contrast on any wood
    q, along_x = _facing_quat(pos, room_half)
    dx = sz[0] if along_x else sz[1]
    dy = sz[1] if along_x else sz[0]
    dx, dy = max(dx, 0.22), min(max(dy, 0.30), 0.6)
    h = min(max(h, 0.55), 0.85)
    t = 0.02
    # front opening spans from the bottom-panel top to under the top panel
    open_bot, open_top = 0.12, h - 0.015
    rail_h, rev = 0.02, 0.004                           # face-frame rail + reveal
    fhh = (open_top - open_bot - rail_h - 4 * rev) / 4  # front-face half-height
    f0z = open_bot + rev + fhh                          # lower face center z
    rlz = f0z + fhh + rev + rail_h / 2                  # mid rail center z
    f1z = rlz + rail_h / 2 + rev + fhh                  # upper face center z
    g = [
        # near-full-footprint base (toe kick inset only 1.5 cm — a 10% inset
        # read as a floating shadow gap)
        f'<geom name="f{i}k" type="box" pos="0 0 0.04" size="{dx - 0.015:.3f} {dy - 0.015:.3f} 0.04" rgba="{dark}" {F}/>',
        f'<geom name="f{i}bt" type="box" pos="0 0 0.10" size="{dx:.3f} {dy - 2*t:.3f} 0.02" rgba="{c}" {F}/>',
        f'<geom name="f{i}bk" type="box" pos="{-(dx - t):.3f} 0 {h/2:.3f}" size="{t} {dy - 2*t:.3f} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s0" type="box" pos="0 {-(dy - t):.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s1" type="box" pos="0 {dy - t:.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}t" type="box" pos="0 0 {h:.3f}" size="{dx + 0.015:.3f} {dy + 0.015:.3f} 0.015" rgba="{dark}" {F}/>',
        # face-frame rail between the drawers, set back so the faces slide clear
        f'<geom name="f{i}mid" type="box" pos="{dx - 0.006:.3f} 0 {rlz:.3f}" size="0.008 {dy - t:.3f} {rail_h/2:.3f}" rgba="{c}" {F}/>',
    ]
    travel = min(0.28, 1.2 * dx)
    tray_hd = min(0.16, dx * 0.72)                      # tray half-depth
    # ~3 cm clearance to the carcass walls: a slide joint gives no lateral
    # constraint, so a close-fit tray wobbles into the walls and DRAGS.
    tray_hw = dy - 0.08                                 # tray half-width
    wall_t, wall_hz = 0.006, min(0.05, fhh * 0.7)       # tray wall thickness / half-height
    fz = -fhh + 0.02                                    # tray floor center (drawer-local z)
    wz = fz + 0.006 + wall_hz                           # tray wall center
    drawers = []
    for k, zc in enumerate((f0z, f1z)):
        drawers.append(
            f'<body name="f{i}drw{k}" pos="{dx:.3f} 0 {zc:.3f}">'
            # friction/damping tuned for the now-hollow tray (the old values
            # were sized for a solid ~8 kg block and dragged a light drawer)
            f'<joint name="f{i}_drawer{k}" type="slide" axis="1 0 0" range="0 {travel:.3f}" '
            f'damping="3" frictionloss="0.8"/>'
            # tall front face
            f'<geom name="f{i}w{k}f" type="box" pos="0.008 0 0" size="0.008 {tray_hw + 0.012:.3f} {fhh:.3f}" rgba="{face}" density="500" {D}/>'
            # hollow tray: floor + back + two sides, all lower than the face
            f'<geom name="f{i}w{k}bot" type="box" pos="{-tray_hd:.3f} 0 {fz:.3f}" size="{tray_hd:.3f} {tray_hw:.3f} 0.006" rgba="{c}" density="400" {D}/>'
            f'<geom name="f{i}w{k}bk" type="box" pos="{-2*tray_hd + 0.006:.3f} 0 {wz:.3f}" size="0.006 {tray_hw:.3f} {wall_hz:.3f}" rgba="{c}" density="400" {D}/>'
            f'<geom name="f{i}w{k}sl" type="box" pos="{-tray_hd:.3f} {-(tray_hw - wall_t):.3f} {wz:.3f}" size="{tray_hd:.3f} {wall_t} {wall_hz:.3f}" rgba="{c}" density="400" {D}/>'
            f'<geom name="f{i}w{k}sr" type="box" pos="{-tray_hd:.3f} {tray_hw - wall_t:.3f} {wz:.3f}" size="{tray_hd:.3f} {wall_t} {wall_hz:.3f}" rgba="{c}" density="400" {D}/>'
            # metal U-pull: two standoff posts + a graspable bar proud of the face
            f'<geom name="f{i}w{k}pl" type="box" pos="0.032 -0.075 0" size="0.016 0.008 0.008" rgba="{metal}" density="700" {D}/>'
            f'<geom name="f{i}w{k}pr" type="box" pos="0.032 0.075 0" size="0.016 0.008 0.008" rgba="{metal}" density="700" {D}/>'
            f'<geom name="f{i}w{k}h" type="box" pos="0.048 0 0" size="0.008 0.083 0.011" rgba="{metal}" density="700" {D}/>'
            f'</body>'
        )
    xml = (f'<body name="f{i}art" pos="{pos[0]:.3f} {pos[1]:.3f} 0" quat="{q}">'
           + "".join(g) + "".join(drawers) + '</body>')
    return [xml], len(g) + 2 * 8


def _art_lid_chest(rng, i, pos, sz, h, F, D, room_half):
    """Storage chest with a top lid on a horizontal hinge (lifts open).

    Unlike the vertical-hinge door, the lid is gravity-loaded: released, it
    falls shut. Opening needs vertical headroom but almost no floor footprint,
    so it places easily (small reach zone).
    """
    wood = _WOODS[rng.integers(len(_WOODS))]
    dark = _jit(rng, tuple(v * 0.6 for v in wood))
    c = _jit(rng, wood)
    q, along_x = _facing_quat(pos, room_half)
    dx = sz[0] if along_x else sz[1]
    dy = sz[1] if along_x else sz[0]
    dx, dy = min(max(dx, 0.22), 0.5), min(max(dy, 0.30), 0.6)
    h = min(max(h, 0.4), 0.65)
    t = 0.02
    g = [
        f'<geom name="f{i}k" type="box" pos="0 0 0.04" size="{dx*0.95:.3f} {dy*0.95:.3f} 0.04" rgba="{dark}" {F}/>',
        f'<geom name="f{i}bot" type="box" pos="0 0 0.10" size="{dx:.3f} {dy:.3f} 0.02" rgba="{c}" {F}/>',
        f'<geom name="f{i}fx" type="box" pos="{dx - t:.3f} 0 {h/2:.3f}" size="{t} {dy:.3f} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}bx" type="box" pos="{-(dx - t):.3f} 0 {h/2:.3f}" size="{t} {dy:.3f} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}ly" type="box" pos="0 {-(dy - t):.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}ry" type="box" pos="0 {dy - t:.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
    ]
    # lid hinged at the back-top edge (x=-dx, z=h); axis 0 -1 0 so a positive
    # angle lifts the front of the slab up and back over the chest
    metal = "0.78 0.79 0.82 1"
    lid = (
        f'<body name="f{i}lid" pos="{-dx:.3f} 0 {h:.3f}">'
        f'<joint name="f{i}_lid" type="hinge" axis="0 -1 0" range="0 100" '
        f'damping="3" frictionloss="1.0"/>'
        f'<geom name="f{i}ld" type="box" pos="{dx:.3f} 0 0" size="{dx:.3f} {dy:.3f} 0.02" '
        f'rgba="{_jit(rng, wood, 0.02)}" density="400" {D}/>'
        # raised metal bar pull at the lid's front edge — fingers hook under it
        f'<geom name="f{i}lh" type="box" pos="{2*dx - 0.03:.3f} 0 0.032" size="0.012 0.07 0.018" '
        f'rgba="{metal}" density="700" {D}/>'
        f'</body>'
    )
    xml = (f'<body name="f{i}art" pos="{pos[0]:.3f} {pos[1]:.3f} 0" quat="{q}">'
           + "".join(g) + lid + '</body>')
    return [xml], len(g) + 2


def _art_sliding_door_cabinet(rng, i, pos, sz, h, F, D, room_half):
    """Cabinet with a sliding door (horizontal slide joint — no swing arc, so
    it fits flush against a wall where a hinged door could not open)."""
    wood = _WOODS[rng.integers(len(_WOODS))]
    dark = _jit(rng, tuple(v * 0.6 for v in wood))
    c = _jit(rng, wood)
    q, along_x = _facing_quat(pos, room_half)
    dx = sz[0] if along_x else sz[1]
    dy = sz[1] if along_x else sz[0]
    dx, dy = max(dx, 0.22), min(max(dy, 0.4), 0.7)
    h = max(h, 0.8)
    t = 0.02
    zh = (h - 0.15) / 2
    zc = 0.12 + zh
    g = [
        f'<geom name="f{i}k" type="box" pos="0 0 0.04" size="{dx*0.9:.3f} {dy*0.9:.3f} 0.04" rgba="{dark}" {F}/>',
        f'<geom name="f{i}bt" type="box" pos="0 0 0.10" size="{dx:.3f} {dy - 2*t:.3f} 0.02" rgba="{c}" {F}/>',
        f'<geom name="f{i}bk" type="box" pos="{-(dx - t):.3f} 0 {h/2:.3f}" size="{t} {dy - 2*t:.3f} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s0" type="box" pos="0 {-(dy - t):.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}s1" type="box" pos="0 {dy - t:.3f} {h/2:.3f}" size="{dx:.3f} {t} {h/2 - 0.02:.3f}" rgba="{c}" {F}/>',
        f'<geom name="f{i}t" type="box" pos="0 0 {h:.3f}" size="{dx + 0.015:.3f} {dy + 0.015:.3f} 0.015" rgba="{dark}" {F}/>',
    ]
    metal = "0.78 0.79 0.82 1"
    # a shelf inside at mid-height so the revealed half reads as a cabinet
    g.append(f'<geom name="f{i}shelf" type="box" pos="0 0 {0.12 + (h - 0.12)*0.5:.3f}" '
             f'size="{dx*0.9:.3f} {dy - 2*t:.3f} 0.012" rgba="{c}" {F}/>')
    # door covers the left half of the front (+x) face and slides +y to reveal it
    dw = dy * 0.5
    travel = dy
    hy = dw - 0.06                                       # handle near the door's leading edge
    door = (
        f'<body name="f{i}sd" pos="{dx + 0.02:.3f} {-dw:.3f} {zc:.3f}">'
        f'<joint name="f{i}_sdoor" type="slide" axis="0 1 0" range="0 {travel:.3f}" '
        f'damping="5" frictionloss="1.5"/>'
        f'<geom name="f{i}sdg" type="box" pos="0 0 0" size="0.015 {dw:.3f} {zh:.3f}" '
        f'rgba="{_jit(rng, wood, 0.02)}" density="400" {D}/>'
        # metal D-pull proud of the sliding panel
        f'<geom name="f{i}sdpt" type="box" pos="0.028 {hy:.3f} 0.06" size="0.014 0.01 0.01" rgba="{metal}" density="700" {D}/>'
        f'<geom name="f{i}sdpb" type="box" pos="0.028 {hy:.3f} -0.06" size="0.014 0.01 0.01" rgba="{metal}" density="700" {D}/>'
        f'<geom name="f{i}sdh" type="box" pos="0.048 {hy:.3f} 0" size="0.01 0.012 0.062" '
        f'rgba="{metal}" density="400" {D}/>'
        f'</body>'
    )
    xml = (f'<body name="f{i}art" pos="{pos[0]:.3f} {pos[1]:.3f} 0" quat="{q}">'
           + "".join(g) + door + '</body>')
    return [xml], len(g) + 4


_ART_ARCHETYPES = (_art_hinged_cabinet, _art_drawer_chest,
                   _art_lid_chest, _art_sliding_door_cabinet)

# clearance the piece's moving part sweeps into the room (see _art_zone):
# hinged door needs the full swing arc; a lid opens upward; a drawer needs its
# travel; a sliding door stays in-plane and needs almost nothing
_ART_REACH = {
    _art_hinged_cabinet: 1.0,
    _art_drawer_chest: 0.45,
    _art_lid_chest: 0.3,
    _art_sliding_door_cabinet: 0.2,
}


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
    # outside camera, which made demos unreadable). Materials are diffuse
    # grain textures (materials.py) that the geom rgba still tints; visual
    # only, physics untouched.
    Swood = S + ' material="mat_wood"'
    body.append(f'<geom name="floor" type="plane" size="{hx} {hy} 0.1" '
                f'rgba=".72 .62 .48 1" material="mat_plaster" {S}/>')
    for i, (p, s) in enumerate((
        ((hx + wt, 0), (wt, hy + wt)), ((-hx - wt, 0), (wt, hy + wt)),
        ((0, hy + wt), (hx + wt, wt)), ((0, -hy - wt), (hx + wt, wt)),
    )):
        body.append(f'<geom name="wall{i}" type="box" pos="{p[0]} {p[1]} {wh}" '
                    f'size="{s[0]} {s[1]} {wh}" rgba=".93 .91 .88 0.22" ' + F + '/>')
    # work table (static)
    body.append(f'<geom name="table" type="box" pos="{tx} {ty} {tz - 0.02}" '
                f'size="{thx} {thy} 0.02" rgba=".55 .4 .3 1" ' + Swood + '/>')
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        body.append(f'<geom name="tleg{i}" type="box" '
                    f'pos="{tx + sx * (thx - 0.05)} {ty + sy * (thy - 0.05)} {(tz - 0.04) / 2}" '
                    f'size="0.04 0.04 {(tz - 0.04) / 2}" rgba=".45 .33 .25 1" ' + Swood + '/>')
    # furniture along the walls — multi-box archetypes (cabinet / shelf /
    # sofa / sideboard) under a geom budget. n_furniture ~= number of static
    # furniture GEOMS (keeps large-scene benchmarks comparable); every piece
    # carries the unreachable-static mask, so visual richness costs nothing
    # in collision pairs.
    budget = spec.n_furniture
    fi = 0
    slots = _furniture_slots(rng, spec, max(4, spec.n_furniture))
    # articulated pieces (hinged doors / sliding drawers) claim slots whose
    # swing/travel zone clears the table, and draw from their OWN rng
    # stream: with n_articulated=0 the main stream is untouched and existing
    # seeded rooms stay byte-identical
    used = set()
    art_zones = []            # claimed swing/travel zones (articulated only)
    if spec.n_articulated > 0:
        art_rng = np.random.default_rng(spec.seed ^ 0x5EED)
        table_aabb = (tx - thx - 0.1, tx + thx + 0.1, ty - thy - 0.1, ty + thy + 0.1)
        # alternate door/drawer pieces, but let any still-needed archetype
        # claim a slot — if the door cabinet fits nowhere, drawers still place
        needed = [_ART_ARCHETYPES[k % len(_ART_ARCHETYPES)]
                  for k in range(spec.n_articulated)]
        for si, (pos, sz, h) in enumerate(slots):
            if not needed:
                break
            hit = None
            for arch in dict.fromkeys(needed):
                reach = _ART_REACH[arch]                 # per-archetype sweep
                zone = _art_zone(pos, sz, (hx, hy), reach)
                # the swept zone must clear the table AND every other
                # articulated piece (doors/drawers are mutually collidable,
                # unlike F statics)
                if _overlaps(zone, table_aabb) or any(_overlaps(zone, z) for z in art_zones):
                    continue
                hit = (arch, zone)
                break
            if hit is None:
                continue
            arch, zone = hit
            needed.remove(arch)
            # articulated furniture is all wood; append the material to both
            # the static-carcass and dynamic-part masks (archetypes stay
            # untouched — they just emit geoms with a material now)
            parts, ng = arch(art_rng, fi, pos, sz, h,
                             F + ' material="mat_wood"', D + ' material="mat_wood"', (hx, hy))
            body.extend(parts)
            budget -= ng
            fi += 1
            used.add(si)
            art_zones.append(zone)
    for si, (pos, sz, h) in enumerate(slots):
        if si in used:
            continue
        if budget <= 0:
            break
        # keep static furniture out of door/drawer swing zones — masks prune
        # the contacts, but a sofa inside a door arc still looks broken
        if art_zones:
            foot = (pos[0] - sz[0], pos[0] + sz[0], pos[1] - sz[1], pos[1] + sz[1])
            if any(_overlaps(foot, z) for z in art_zones):
                continue
        arch = _ARCHETYPES[int(rng.integers(len(_ARCHETYPES)))]
        # sofas read as upholstery (fabric weave); everything else as wood
        mat = ' material="mat_fabric"' if arch is _arch_sofa else ' material="mat_wood"'
        geoms = arch(rng, fi, pos, sz, h, F + mat)
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

    from .materials import material_assets
    xml = f"""<mujoco model="lpw_room_seed{spec.seed}">
  <option timestep="0.005" iterations="{spec.solver_iterations}" ls_iterations="{spec.ls_iterations}"/>
  {robot}
  <asset>
    {material_assets()}
  </asset>
  <worldbody>
    {chr(10).join('    ' + b for b in body)}
  </worldbody>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
