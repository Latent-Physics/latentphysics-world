"""CPU tests: GLB diffuse UV textures pass through to MuJoCo materials.

Visual-only invariant: with textures on vs off, collision masks, geom
counts, and body masses must be byte-identical — the texture path may not
touch physics.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")

from latentphysics.assets.import_3d import ImportSpec, import_glb  # noqa: E402


@pytest.fixture(scope="module")
def textured_glb(tmp_path_factory):
    """A box with per-vertex UVs and a red/blue checker baseColorTexture."""
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    v = box.vertices
    uv = (v[:, :2] - v[:, :2].min(axis=0)) / np.ptp(v[:, :2], axis=0)
    checker = np.zeros((8, 8, 3), dtype=np.uint8)
    checker[::2, ::2] = (255, 0, 0)
    checker[1::2, 1::2] = (0, 0, 255)
    img = Image.fromarray(checker)
    box.visual = TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=img))
    path = tmp_path_factory.mktemp("glb") / "texbox.glb"
    box.export(str(path))
    return str(path)


def _compile(glb, out_dir, **spec_kw):
    xml = import_glb(glb, str(out_dir), "texbox",
                     ImportSpec(threshold=0.1, max_hulls=4, **spec_kw))
    return mujoco.MjModel.from_xml_path(xml)


def test_texture_reaches_visual_geom(textured_glb, tmp_path):
    m = _compile(textured_glb, tmp_path / "on")
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == 2]
    assert vis, "no visual geom emitted"
    for g in vis:
        assert m.geom_matid[g] >= 0, "visual geom lost its material"
    mesh_id = int(m.geom_dataid[vis[0]])
    assert m.mesh_texcoordnum[mesh_id] > 0, "UVs did not survive the OBJ export"
    # floor checker + the passed-through diffuse
    assert m.ntex >= 2 and m.nmat >= 2


def test_textures_are_visual_only(textured_glb, tmp_path):
    m_on = _compile(textured_glb, tmp_path / "on")
    m_off = _compile(textured_glb, tmp_path / "off", textures=False)
    assert m_on.ngeom == m_off.ngeom
    assert (m_on.geom_contype == m_off.geom_contype).all()
    assert (m_on.geom_conaffinity == m_off.geom_conaffinity).all()
    np.testing.assert_array_equal(m_on.body_mass, m_off.body_mass)
    # the kill-switch actually kills: no per-asset texture without it
    assert m_off.ntex < m_on.ntex
