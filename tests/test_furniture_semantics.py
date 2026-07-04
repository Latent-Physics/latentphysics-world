"""Semantic gate for articulated furniture — the mechanical half of the
modeling-fidelity bar (CLAUDE.md).

A joint that travels far enough says nothing about whether the piece LOOKS
like the real object. These CPU checks assert the geometry a human would
look for: the piece is grounded, its handle is proud of the face and
contrasts with the wood, and a drawer is a hollow tray behind a filled,
tight-seamed front — not a solid block with floaty gaps. Rendered proof
lives in scripts/inspect_furniture.py; this is the regression guard.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from latentphysics.assets.scene_gen import (  # noqa: E402
    _art_drawer_chest, _art_hinged_cabinet, _art_lid_chest,
    _art_sliding_door_cabinet,
)

_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'

# (archetype, sz, h). pos=(-1,0) → facing quat is identity, so local axes map
# straight to world: depth is +x, and geom AABBs are axis-aligned.
PIECES = {
    "drawer": (_art_drawer_chest, (0.20, 0.35), 0.62),
    "hinged": (_art_hinged_cabinet, (0.22, 0.40), 0.95),
    "lid": (_art_lid_chest, (0.28, 0.36), 0.50),
    "sliding": (_art_sliding_door_cabinet, (0.26, 0.50), 0.85),
}


def _compile(archetype, sz, h):
    rng = np.random.default_rng(0)
    parts, _ = archetype(rng, 0, (-1.0, 0.0), sz, h, _S, _D, room_half=(1.0, 1.0))
    xml = (f'<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/>'
           f'{"".join(parts)}</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                     # qpos=0 → everything closed
    return m, d


def _geoms(m, d):
    """name -> (center xyz, half-size xyz, rgba). Body rotation is identity."""
    out = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name:
            out[name] = (d.geom_xpos[g].copy(), m.geom_size[g].copy(), m.geom_rgba[g].copy())
    return out


def _mean_rgb(rgba):
    return float(np.mean(rgba[:3]))


@pytest.mark.parametrize("key", list(PIECES))
def test_piece_is_grounded(key):
    """No floating furniture: the lowest geom must reach the floor."""
    m, d = _compile(*PIECES[key])
    g = _geoms(m, d)
    lowest = min(c[2] - s[2] for n, (c, s, _) in g.items() if n != "floor")
    assert lowest <= 0.005, f"{key}: lowest geom bottom at z={lowest:.3f}, not grounded"


# (handle geom, face geom) whose front (+x) faces are compared
_HANDLE_FACE = {
    "drawer": ("f0w1h", "f0w1f"),
    "hinged": ("f0dh", "f0d"),
    "lid": ("f0lh", "f0ld"),
    "sliding": ("f0sdh", "f0sdg"),
}


@pytest.mark.parametrize("key", list(PIECES))
def test_handle_is_visible_and_grippable(key):
    m, d = _compile(*PIECES[key])
    g = _geoms(m, d)
    hname, fname = _HANDLE_FACE[key]
    (hc, hs, hrgba), (fc, fs, _) = g[hname], g[fname]
    # the lid pull stands up (+z); every other handle stands out (+x)
    ax = 2 if key == "lid" else 0
    protrusion = (hc[ax] + hs[ax]) - (fc[ax] + fs[ax])
    assert protrusion >= 0.03, f"{key}: handle only {protrusion*100:.1f} cm proud of the face"
    # contrast: the handle must not vanish into the wood
    face_rgba = next(v[2] for n, v in g.items() if n == fname)
    diff = abs(_mean_rgb(hrgba) - _mean_rgb(face_rgba))
    assert diff >= 0.15, f"{key}: handle/face brightness diff only {diff:.2f} — handle blends in"


def test_drawer_is_a_hollow_tray():
    """The old drawer was a solid block; a real drawer is an open tray."""
    m, d = _compile(*PIECES["drawer"])
    g = _geoms(m, d)
    for k in (0, 1):
        floor_top = g[f"f0w{k}bot"][0][2] + g[f"f0w{k}bot"][1][2]
        wall_top = g[f"f0w{k}sl"][0][2] + g[f"f0w{k}sl"][1][2]
        cavity = wall_top - floor_top
        assert cavity >= 0.05, f"drawer {k}: cavity only {cavity*100:.1f} cm — not a tray"
        # interior span (between side walls) is most of the face width
        face_w = 2 * g[f"f0w{k}f"][1][1]
        inner_w = 2 * (g[f"f0w{k}sl"][0][1].__abs__())
        assert inner_w >= 0.6 * face_w, f"drawer {k}: interior {inner_w:.2f} < 60% of face {face_w:.2f}"


def test_drawer_front_has_tight_seams():
    """Closed, the two faces fill the front around the rail with ~mm reveals,
    not the 3-6 cm gaps that read as a floating stack of boards."""
    m, d = _compile(*PIECES["drawer"])
    g = _geoms(m, d)
    f0_top = g["f0w0f"][0][2] + g["f0w0f"][1][2]
    f1_bot = g["f0w1f"][0][2] - g["f0w1f"][1][2]
    rail_bot = g["f0mid"][0][2] - g["f0mid"][1][2]
    rail_top = g["f0mid"][0][2] + g["f0mid"][1][2]
    assert rail_bot - f0_top <= 0.008, "gap below the rail too wide"
    assert f1_bot - rail_top <= 0.008, "gap above the rail too wide"
