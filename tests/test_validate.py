"""CPU tests for asset validity checks + mesh sanitization (no GPU needed).

The GPU settle() path is covered in test_validate_gpu.py.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco  # noqa: E402

from latentphysics.assets.validate import (  # noqa: E402
    initial_penetration, validate_model,
)


def _model(xml):
    return mujoco.MjModel.from_xml_string(xml)


def test_validate_flags_no_collision_body():
    # a free body whose only geom is visual-only (contype/conaffinity=0)
    xml = """<mujoco><worldbody>
      <body name="ghost" pos="0 0 1"><freejoint/>
        <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>"""
    rep = validate_model(_model(xml))
    assert not rep.ok
    assert any(i.kind == "no_collision" and i.body == "ghost" for i in rep.issues)


def test_validate_flags_zero_mass_body():
    # explicit tiny mass on a movable body
    xml = """<mujoco><worldbody>
      <body name="feather" pos="0 0 1"><freejoint/>
        <geom type="box" size="0.1 0.1 0.1" mass="1e-9"/>
      </body></worldbody></mujoco>"""
    rep = validate_model(_model(xml))
    assert any(i.kind in ("zero_mass", "singular_inertia") and i.body == "feather"
               for i in rep.issues)


def test_validate_passes_healthy_body():
    xml = """<mujoco><worldbody>
      <body name="cube" pos="0 0 1"><freejoint/>
        <geom type="box" size="0.1 0.1 0.1" density="500"/>
      </body></worldbody></mujoco>"""
    rep = validate_model(_model(xml))
    assert rep.ok, str(rep)


def test_validate_ignores_static_bodies():
    # a static (welded) massless-looking geom must NOT be flagged
    xml = """<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <geom name="wall" type="box" pos="0 0 1" size="1 0.05 1"/>
    </worldbody></mujoco>"""
    assert validate_model(_model(xml)).ok


def test_initial_penetration_detects_overlap():
    # two unit boxes overlapping by ~0.1 m
    xml = """<mujoco><worldbody>
      <geom name="a" type="box" pos="0 0 0.5" size="0.2 0.2 0.2"/>
      <body name="b" pos="0.3 0 0.5"><freejoint/>
        <geom name="bg" type="box" size="0.2 0.2 0.2"/></body>
    </worldbody></mujoco>"""
    rep = initial_penetration(_model(xml), tol=0.005)
    assert not rep.ok and any(i.kind == "penetration" for i in rep.issues)


def test_initial_penetration_clean_scene():
    xml = """<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <body name="b" pos="0 0 1.0"><freejoint/>
        <geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>"""
    assert initial_penetration(_model(xml)).ok


def test_import_strict_raises_on_defect(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from latentphysics.assets.import_3d import ImportSpec, import_glb

    # near-zero density -> sub-threshold mass on the dynamic body: the wired
    # validate_model check must surface it, and strict must turn it fatal
    s = trimesh.Scene()
    s.add_geometry(trimesh.creation.box(extents=(0.2, 0.2, 0.2)), node_name="obj")
    glb = str(tmp_path / "light.glb")
    s.export(glb)
    spec = ImportSpec(up="z", dynamic=("obj",), density=1e-4, strict=True)
    with pytest.raises(ValueError):
        import_glb(glb, str(tmp_path / "out"), name="light", spec=spec)


def test_convex_decompose_rejects_empty_mesh():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from latentphysics.assets import convex_decompose
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        convex_decompose(empty)


def test_convex_decompose_cleans_nan_vertices():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from latentphysics.assets import convex_decompose
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    v = np.asarray(box.vertices, dtype=float).copy()
    v = np.vstack([v, [np.nan, np.nan, np.nan]])          # stray non-finite vertex
    dirty = trimesh.Trimesh(vertices=v, faces=box.faces, process=False)
    parts = convex_decompose(dirty)
    assert len(parts) >= 1
    for p in parts:
        assert np.isfinite(p.vertices).all()
