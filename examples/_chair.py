"""Articulated office-chair asset for the chair-push demo.

PROVENANCE: this asset's geometry was authored by an LLM from reference
product photos (owner-approved generative content, 2026-07-08 — see the
scope charter's sign-off rule). It is classical MJCF all the way down:
procedurally lofted meshes (see _chair_mesh.py) for the sculpted visual
surfaces, primitives for everything else, explicit masses, no learned
anything. Fidelity was verified the charter way — rendered in neutral and
articulated states and reviewed, plus the mechanical semantics test in
tests/test_chair_example.py.

Reference dimensions (annotated product photo, inches -> meters):
    headrest        13.39 x 7.48 in  -> 0.340 x 0.190 m
    headrest gap    3.94 in          -> 0.100 m   (stalk visible)
    backrest        21.26 x 17.72 in -> 0.540 h x 0.450 w m
    base diameter   21.26 in         -> 0.540 m   (star reach 0.27)

Articulation (23 DoF): freejoint root, 5x caster swivel, 5x wheel roll,
gas-lift height (sprung), seat swivel (yaw), spring-loaded backrest
recline, 2x armrest height (friction-locked), headrest height slide
(friction-locked), headrest pitch. Total mass ~19 kg. Visual geoms live
in group 2 (meshes carry no mass/contact), collision geoms in group 3
(repo convention); the caster wheels collide as sphere pairs.

Textures come from latentphysics.assets.materials (procedural fabric
grain), which needs imageio on first run — install the ``demos`` extra.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _chair_mesh import mesh_assets  # noqa: E402

from latentphysics.assets.materials import material_assets  # noqa: E402

# ---------------------------------------------------------------- dimensions
RW = 0.030          # caster wheel radius
BASE_R = 0.270      # star arm reach (base diameter 0.54)
SEAT_TOP = 0.482    # seat cushion crown height (at lift springref)
SEAT_W2 = 0.240     # seat half width  (0.48)
SEAT_D2 = 0.235     # seat half depth  (0.47)
BACK_X = -0.215     # backrest plane x (behind seat center)
BACK_Z0 = 0.487     # backrest bottom (sweeps close to the seat)
BACK_Z1 = 1.045     # backrest top    (0.54 tall)
HEAD_GAP = 0.100    # backrest top -> headrest bottom
HEAD_W2 = 0.170     # headrest half width (0.34)
HEAD_H2 = 0.093     # headrest half height (~0.19)

# chair qpos entries after the freejoint (keyframe plumbing):
# 10 casters + lift + swivel + recline + 2 armrests + head slide + head pitch
N_CHAIR_ZERO_QPOS = 17

FABRIC = "0.20 0.20 0.215 1"        # charcoal fabric (tints grain_fabric)
PLASTIC = "0.155 0.155 0.165 1"     # near-black frame plastic
PLASTIC_D = "0.125 0.125 0.135 1"   # darker plastic (wheels, underside)
METAL = "0.52 0.53 0.55 1"          # gas-lift piston


def _q(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_zy(yaw, pitch):
    """World-frame: pitch about y, then yaw about z."""
    qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    qy = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
    return _quat_mul(qz, qy)


def _geom(gtype, size, pos, *, quat=None, rgba=PLASTIC, material=None,
          mass=None, collide=False, friction=None, mesh=None):
    """Visual (group 2, no contact) or collision (group 3) geom.
    ``mesh`` swaps the primitive for a lofted mesh asset (visual only)."""
    if mesh:
        a = [f'type="mesh" mesh="{mesh}"', f'pos="{_q(*pos)}"']
    else:
        a = [f'type="{gtype}"', f'size="{_q(*size)}"', f'pos="{_q(*pos)}"']
    if quat is not None:
        a.append(f'quat="{_q(*quat)}"')
    if material:
        a.append(f'material="{material}"')
    a.append(f'rgba="{rgba}"')
    if collide:
        a.append(f'group="3" mass="{mass:.4f}"')
        if friction:
            a.append(f'friction="{friction}"')
    else:
        a.append('group="2" contype="0" conaffinity="0" mass="0"')
    return f'<geom {" ".join(a)}/>'


# ------------------------------------------------------------------- pieces
def _star_base():
    """5-arm star base + gas-lift column (root body geoms)."""
    g = []
    g.append(_geom("cylinder", (0.055, 0.030), (0, 0, 0.105), rgba=PLASTIC))
    g.append(_geom("cylinder", (0.058, 0.020), (0, 0, 0.132), mass=3.2,
                   rgba=PLASTIC, collide=True))
    for i in range(5):
        a = math.radians(18 + 72 * i)
        ca, sa = math.cos(a), math.sin(a)
        r0, r1, z0, z1 = 0.045, BASE_R - 0.012, 0.105, 0.058
        rm, zm = (r0 + r1) / 2, (z0 + z1) / 2
        pitch = math.atan2(z0 - z1, r1 - r0)
        quat = _quat_zy(a, pitch)
        g.append(_geom(None, None, (rm * ca, rm * sa, zm), quat=quat,
                       rgba=PLASTIC, mesh="star_arm"))
    g.append(_geom(None, None, (0, 0, 0), rgba=PLASTIC, mesh="lift_boot"))
    g.append(_geom("cylinder", (0.019, 0.062), (0, 0, 0.295), rgba=METAL,
                   material="mat_metal"))
    g.append(_geom("capsule", (0.030, 0.055), (0, 0, 0.20), mass=1.8,
                   rgba=PLASTIC, collide=True))
    return "\n      ".join(g)


def _caster(i):
    """Caster housing (swivel) + twin wheel (roll). Trail 20 mm.
    Collision is the sphere pair; the mesh discs are the visible wheels."""
    a = math.radians(18 + 72 * i)
    tip = (BASE_R * math.cos(a), BASE_R * math.sin(a), 0.058)
    return f"""
      <body name="caster{i}" pos="{_q(*tip)}">
        <joint name="caster{i}_swivel" type="hinge" axis="0 0 1"
               damping="0.02" armature="0.0005"/>
        <geom type="cylinder" size="0.020 0.013" pos="0 0 -0.004"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="box" size="0.011 0.019 0.022" pos="0.020 0 -0.022"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="box" size="0.018 0.028 0.007" pos="0.020 0 -0.004"
              rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
        <geom type="sphere" size="0.012" pos="0 0 -0.01" group="3" mass="0.08"/>
        <body name="wheel{i}" pos="0.020 0 -0.028">
          <joint name="wheel{i}_roll" type="hinge" axis="0 1 0"
                 damping="0.010" armature="0.0005"/>
          <geom type="sphere" size="{RW:.3f}" pos="0 0.0135 0" group="3"
                mass="0.05" friction="0.9 0.006 0.0002" rgba="{PLASTIC_D}"/>
          <geom type="sphere" size="{RW:.3f}" pos="0 -0.0135 0" group="3"
                mass="0.05" friction="0.9 0.006 0.0002" rgba="{PLASTIC_D}"/>
          <geom type="mesh" mesh="wheel_disc" pos="0 0.0135 0"
                rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
          <geom type="mesh" mesh="wheel_disc" pos="0 -0.0135 0"
                rgba="{PLASTIC_D}" group="2" contype="0" conaffinity="0" mass="0"/>
          <geom type="cylinder" size="0.011 0.023" pos="0 0 0"
                quat="0.7071 0.7071 0 0"
                rgba="0.22 0.22 0.24 1" group="2" contype="0" conaffinity="0" mass="0"/>
        </body>
      </body>"""


def _seat():
    """Seat cushion (lofted mesh) + tilt mechanism + paddles."""
    g = []
    g.append(_geom("box", (0.105, 0.090, 0.024), (0.01, 0, 0.388), rgba=PLASTIC_D))
    g.append(_geom("box", (0.10, 0.09, 0.02), (0.01, 0, 0.388), mass=2.4,
                   collide=True, rgba=PLASTIC_D))
    g.append(_geom("capsule", (0.009, 0.035), (0.06, -0.13, 0.385),
                   quat=_quat_zy(0, math.radians(90)), rgba=PLASTIC_D))
    g.append(_geom("capsule", (0.007, 0.028), (-0.04, -0.125, 0.380),
                   quat=_quat_zy(math.radians(20), math.radians(90)),
                   rgba=PLASTIC_D))
    g.append(_geom(None, None, (0, 0, 0), material="mat_fabric", rgba=FABRIC,
                   mesh="seat_cushion"))
    g.append(_geom("box", (SEAT_D2, SEAT_W2, 0.040), (0.012, 0, 0.442),
                   mass=4.2, collide=True, rgba=FABRIC))
    return "\n        ".join(g)


def _armrest(name, s):
    """Height-adjustable armrest: bolted mount -> angled post -> adjuster ->
    lofted pad, on a friction-locked slide joint."""
    post_tilt = 0.31
    pq = (math.cos(s * post_tilt / 2), -math.sin(s * post_tilt / 2), 0, 0)
    g = [
        _geom("box", (0.026, 0.022, 0.016), (0.02, s * 0.185, 0.386),
              rgba=PLASTIC_D),
        _geom("capsule", (0.016, 0.084), (0.02, s * 0.214, 0.482), quat=pq,
              rgba=PLASTIC),
        _geom("box", (0.028, 0.026, 0.026), (0.02, s * 0.245, 0.594),
              rgba=PLASTIC_D),
        _geom(None, None, (0.035, s * 0.245, 0.640), rgba=PLASTIC_D,
              mesh="arm_pad"),
        _geom("box", (0.125, 0.042, 0.015), (0.035, s * 0.245, 0.634),
              mass=0.7, collide=True, rgba=PLASTIC_D),
        _geom("capsule", (0.016, 0.084), (0.02, s * 0.214, 0.482), quat=pq,
              mass=0.3, collide=True, rgba=PLASTIC),
    ]
    geoms = "\n          ".join(g)
    return f"""
        <body name="{name}" pos="0 0 0">
          <joint name="{name}_slide" type="slide" axis="0 0 1"
                 range="0 0.08" frictionloss="80" damping="5" armature="0.02"/>
          {geoms}
        </body>"""


def _backrest():
    """Sculpted backrest: lofted cushion + rear shell meshes, spine frame,
    L-bracket to the tilt mechanism, headrest stalk. Collision stays 3
    coarse slabs following the curve."""
    g = []
    g.append(_geom(None, None, (0, 0, 0), material="mat_fabric", rgba=FABRIC,
                   mesh="back_cushion"))
    g.append(_geom(None, None, (0, 0, 0), rgba=PLASTIC, mesh="back_shell"))
    # collision: 3 coarse slabs following the curve
    g.append(_geom("box", (0.036, 0.20, 0.095), (BACK_X + 0.045, 0, 0.60),
                   quat=_quat_zy(0, -0.04), mass=1.0, collide=True, rgba=FABRIC))
    g.append(_geom("box", (0.036, 0.225, 0.10), (BACK_X + 0.032, 0, 0.79),
                   quat=_quat_zy(0, 0.08), mass=1.0, collide=True, rgba=FABRIC))
    g.append(_geom("box", (0.036, 0.205, 0.10), (BACK_X - 0.03, 0, 0.975),
                   quat=_quat_zy(0, 0.21), mass=1.0, collide=True, rgba=FABRIC))
    # L-bracket to the tilt mechanism: visible side load path seat<->back
    g.append(_geom("box", (0.070, 0.042, 0.017), (BACK_X + 0.062, 0, 0.442),
                   quat=_quat_zy(0, -0.52), rgba=PLASTIC))
    g.append(_geom("box", (0.048, 0.042, 0.015), (BACK_X + 0.095, 0, 0.402),
                   rgba=PLASTIC))
    # spine frame
    g.append(_geom("box", (0.016, 0.052, 0.26), (BACK_X - 0.055, 0, 0.78),
                   quat=_quat_zy(0, 0.10), rgba=PLASTIC))
    g.append(_geom("box", (0.016, 0.052, 0.26), (BACK_X - 0.055, 0, 0.78),
                   quat=_quat_zy(0, 0.10), mass=1.2, collide=True, rgba=PLASTIC))
    return "\n          ".join(g)


def _headrest_stalk():
    """Stalk from the backrest top up to the headrest slide/pitch joints."""
    top_x = BACK_X - 0.062
    return _geom("box", (0.010, 0.032, 0.078),
                 (top_x - 0.020, 0, BACK_Z1 + 0.052),
                 quat=_quat_zy(0, 0.18), rgba=PLASTIC)


def _headrest():
    """Lofted pillow on a friction-locked height slide + pitch hinge. The
    bracket bridges the hinge and overlaps the stalk top so the pivot reads
    as one continuous column in every pose."""
    g = []
    g.append(_geom("box", (0.010, 0.027, 0.046), (-0.018, 0, -HEAD_H2 + 0.016),
                   quat=_quat_zy(0, 0.18), rgba=PLASTIC))
    core_q = _quat_zy(0, 0.14)
    g.append(_geom(None, None, (0, 0, 0), quat=core_q, material="mat_fabric",
                   rgba=FABRIC, mesh="headrest_pillow"))
    g.append(_geom("box", (0.040, HEAD_W2 - 0.01, HEAD_H2 - 0.01), (0, 0, 0),
                   quat=core_q, mass=0.55, collide=True, rgba=FABRIC))
    return "\n            ".join(g)


def chair_body(name="chair", pos=(0, 0, 0.004), yaw_deg=0.0):
    """The chair as a worldbody fragment (freejoint root)."""
    yaw = math.radians(yaw_deg)
    quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    casters = "".join(_caster(i) for i in range(5))
    hz = BACK_Z1 + HEAD_GAP + HEAD_H2 - 0.01
    return f"""
    <body name="{name}" pos="{_q(*pos)}" quat="{_q(*quat)}">
      <freejoint name="{name}_free"/>
      {_star_base()}
      {casters}
      <body name="{name}_seat" pos="0 0 0">
        <joint name="{name}_lift" type="slide" axis="0 0 1"
               range="-0.03 0.05" stiffness="22000" damping="600"
               armature="0.05"/>
        <joint name="{name}_swivel" type="hinge" axis="0 0 1" pos="0 0 0.35"
               damping="0.6" armature="0.01" frictionloss="0.4"/>
        {_seat()}
        {_armrest(f"{name}_arm_l", 1)}
        {_armrest(f"{name}_arm_r", -1)}
        <body name="{name}_back" pos="0 0 0">
          <joint name="{name}_recline" type="hinge" axis="0 1 0"
                 pos="-0.12 0 0.40" range="-0.03 0.22"
                 stiffness="360" damping="28" armature="0.01"/>
          {_backrest()}
          {_headrest_stalk()}
          <body name="{name}_headrest" pos="{_q(BACK_X - 0.092, 0, hz)}">
            <joint name="{name}_head_slide" type="slide" axis="0 0 1"
                   range="-0.03 0.04" frictionloss="50" damping="3"
                   armature="0.02"/>
            <joint name="{name}_head_pitch" type="hinge" axis="0 1 0"
                   pos="0.0 0 {-HEAD_H2 - 0.005:.3f}" range="-0.35 0.35"
                   stiffness="60" damping="4" armature="0.005"/>
            {_headrest()}
          </body>
        </body>
      </body>
    </body>"""


def chair_assets():
    extra = ('<material name="mat_metal" specular="0.9" shininess="0.8" '
             'reflectance="0.35"/>')
    return material_assets() + mesh_assets() + extra


def chair_scene_xml():
    """Studio scene: the chair alone on a light floor (for inspection)."""
    return f"""<mujoco model="lpw_chair_studio">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <statistic center="0 0 0.65" extent="2.2"/>
  <visual>
    <global offwidth="2560" offheight="1920"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.36 0.36 0.38" diffuse="0.62 0.62 0.62" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    {chair_assets()}
  </asset>
  <worldbody>
    <light pos="1.8 -1.2 2.6" dir="-0.55 0.35 -0.75" diffuse="0.75 0.74 0.72" castshadow="true"/>
    <light pos="-1.6 1.8 2.2" dir="0.5 -0.55 -0.68" diffuse="0.34 0.35 0.38" castshadow="false"/>
    <geom name="floor" type="plane" size="3.2 3.2 0.1" material="mat_plaster"
          quat="0.981 0 0 0.195" rgba="0.87 0.87 0.88 1"
          friction="0.9 0.005 0.0001"/>
    {chair_body()}
  </worldbody>
</mujoco>"""
