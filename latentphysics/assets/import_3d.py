"""GLB/glTF + USD indoor scene import (R4) — real 3D assets to engine-ready MJCF.

Both importers walk the source scene graph, bake every node transform into
world-space vertices (sidesteps MuJoCo's inability to express non-uniform
node scale), and feed the same composer: CoACD convex hulls for collision,
the original mesh as a collision-free visual geom, and the same
collision-mask scheme as the procedural generator (static x static pruned
at model build).

Up-axis: glTF is +y-up by convention (``ImportSpec.up``); USD stages declare
``upAxis`` and ``metersPerUnit`` metadata, which are honored automatically.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from . import convex_decompose

__all__ = ["ImportSpec", "SceneObject", "import_glb", "import_usd"]

# masks match scene_gen: static never pairs with itself, dynamic pairs with all
_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'

_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


@dataclass
class ImportSpec:
    threshold: float = 0.06        # CoACD concavity (lower = tighter hulls)
    max_hulls: int = 32            # hull cap per object
    dynamic: tuple = ()            # node-name substrings imported as free bodies
    up: str = "y"                  # GLB source up-axis: "y" (glTF standard) or "z"
    scale: float = 1.0             # uniform rescale on top of source units
    add_floor: bool = True         # add a ground plane (off if the scan has one)
    density: float = 400.0         # dynamic-body geom density (kg/m^3)
    solver_iterations: int = 8
    ls_iterations: int = 10
    validate: bool = True          # run structural validity check on the output
    strict: bool = False           # raise on validity issues instead of warning


@dataclass
class SceneObject:
    """One imported object: a world-baked trimesh ready for the composer."""
    name: str
    mesh: "object"                 # trimesh.Trimesh, vertices in world frame
    dynamic: bool = False
    rgba: str = "0.7 0.7 0.7 1"


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "node"


def _rgba_trimesh(mesh) -> str:
    """Mean vertex color (how GLB color data survives a trimesh round-trip),
    falling back to the PBR base color, then neutral gray."""
    for get in (lambda: np.asarray(mesh.visual.vertex_colors, dtype=float).mean(axis=0),
                lambda: np.asarray(mesh.visual.material.main_color, dtype=float)):
        try:
            c = get()[:4] / 255.0
            return "%.3f %.3f %.3f %.3f" % (c[0], c[1], c[2], max(c[3], 0.05))
        except Exception:
            continue
    return "0.7 0.7 0.7 1"


def compose_mjcf(objects, out_dir: str, name: str, spec: ImportSpec) -> str:
    """Convex-decompose ``objects`` (list of SceneObject) and write one MJCF."""
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    assets, bodies, seen = [], [], {}
    for obj in objects:
        if len(getattr(obj.mesh, "faces", ())) == 0:
            continue
        raw = _sanitize(obj.name)
        n_prev = seen.get(raw, 0)
        seen[raw] = n_prev + 1
        base = raw if n_prev == 0 else f"{raw}_{n_prev}"

        world = obj.mesh
        # dynamic bodies get a local frame at their centroid so the free
        # joint / inertia are well-conditioned; statics stay in world frame
        origin = world.vertices.mean(axis=0) if obj.dynamic else np.zeros(3)
        world.apply_translation(-origin)

        vis = f"{base}_visual.obj"
        world.export(os.path.join(out_dir, vis))
        assets.append(f'<mesh name="{base}_v" file="{vis}"/>')

        hulls = convex_decompose(world, threshold=spec.threshold,
                                 max_hulls=spec.max_hulls)
        # visual geom carries NO mass: without this it contributes inertia at
        # MuJoCo's default density (1000) and silently overrides spec.density,
        # so every dynamic body's mass came from the render mesh, not physics
        geoms = [f'<geom type="mesh" mesh="{base}_v" rgba="{obj.rgba}" '
                 f'contype="0" conaffinity="0" group="2" mass="0"/>']
        for i, part in enumerate(hulls):
            fn = f"{base}_c{i}.obj"
            trimesh.Trimesh(vertices=part.vertices, faces=part.faces).export(
                os.path.join(out_dir, fn))
            assets.append(f'<mesh name="{base}_c{i}" file="{fn}"/>')
            mask = _D if obj.dynamic else _S
            dens = f' density="{spec.density}"' if obj.dynamic else ""
            geoms.append(f'<geom type="mesh" mesh="{base}_c{i}" group="3" '
                         f'rgba="{obj.rgba}"{dens} {mask}/>')

        joint = "<freejoint/>" if obj.dynamic else ""
        pos = "%.5f %.5f %.5f" % tuple(origin)
        bodies.append(f'<body name="{base}" pos="{pos}">{joint}'
                      + "".join(geoms) + "</body>")

    floor = (f'<geom name="floor" type="plane" size="10 10 0.1" '
             f'material="floormat" {_S}/>' if spec.add_floor else "")
    assets.append('<texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62" '
                  'rgb2="0.12 0.14 0.18" width="256" height="256"/>')
    if spec.add_floor:
        assets.append('<texture name="floortex" type="2d" builtin="checker" '
                      'rgb1="0.78 0.74 0.68" rgb2="0.68 0.64 0.58" mark="edge" '
                      'markrgb="0.55 0.52 0.48" width="300" height="300"/>')
        assets.append('<material name="floormat" texture="floortex" '
                      'texrepeat="16 16" reflectance="0.12"/>')
    xml = f"""<mujoco model="{name}">
  <compiler meshdir="." angle="radian"/>
  <option timestep="0.005" iterations="{spec.solver_iterations}" ls_iterations="{spec.ls_iterations}"/>
  <asset>
    {chr(10).join('    ' + a for a in assets).lstrip()}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9"
           diffuse="0.8 0.78 0.75" castshadow="true"/>
    <light name="fill" directional="true" pos="2 -2 2" dir="0.4 0.5 -0.8"
           diffuse="0.3 0.3 0.33" castshadow="false"/>
    {floor}
    {chr(10).join('    ' + b for b in bodies).lstrip()}
  </worldbody>
</mujoco>
"""
    path = os.path.join(out_dir, f"{name}.xml")
    with open(path, "w") as f:
        f.write(xml)

    # structural body-check on the compiled result: a ghost body or a
    # near-singular inertia is far cheaper to catch here than as a NaN mid-sim
    if spec.validate:
        import mujoco

        from .validate import validate_model
        report = validate_model(mujoco.MjModel.from_xml_path(path))
        if not report.ok:
            if spec.strict:
                raise ValueError(f"imported scene failed validation:\n{report}")
            import warnings
            warnings.warn(f"imported scene has validity issues:\n{report}", stacklevel=2)
    return path


# --- GLB / glTF ---------------------------------------------------------------

def import_glb(glb_path: str, out_dir: str, name: str = "scene",
               spec: ImportSpec | None = None) -> str:
    """Import a GLB/glTF scene into ``<out_dir>/<name>.xml``; returns the path."""
    import trimesh

    spec = spec or ImportSpec()
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded, node_name="object")
    else:
        scene = loaded

    conv = np.eye(4)
    if spec.up == "y":
        conv[:3, :3] = _YUP_TO_ZUP.copy()
    conv[:3, :3] *= spec.scale

    objects = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph.get(node)
        mesh = scene.geometry[gname]
        if not hasattr(mesh, "vertices") or len(mesh.faces) == 0:
            continue
        world = mesh.copy()
        world.apply_transform(conv @ T)        # bake node transform + up-axis
        objects.append(SceneObject(name=node, mesh=world,
                                   dynamic=any(s in node for s in spec.dynamic),
                                   rgba=_rgba_trimesh(mesh)))
    return compose_mjcf(objects, out_dir, name, spec)


# --- USD ------------------------------------------------------------------------

def _fan_triangulate(counts, idx):
    """glTF is all-triangles; USD faces are ngons — fan-triangulate them."""
    faces, k = [], 0
    for c in counts:
        for i in range(1, int(c) - 1):
            faces.append((idx[k], idx[k + i], idx[k + i + 1]))
        k += int(c)
    return np.asarray(faces, dtype=np.int64)


def _rgba_usd(gmesh) -> str:
    try:
        cols = gmesh.GetDisplayColorAttr().Get()
        c = np.asarray(cols, dtype=float).reshape(-1, 3).mean(axis=0)
        op = gmesh.GetDisplayOpacityAttr().Get()
        a = float(np.asarray(op, dtype=float).mean()) if op else 1.0
        return "%.3f %.3f %.3f %.3f" % (c[0], c[1], c[2], max(a, 0.05))
    except Exception:
        return "0.7 0.7 0.7 1"


def import_usd(usd_path: str, out_dir: str, name: str = "scene",
               spec: ImportSpec | None = None) -> str:
    """Import a USD stage into ``<out_dir>/<name>.xml``; returns the path.

    Honors the stage's ``upAxis`` and ``metersPerUnit`` metadata (times
    ``spec.scale``); ``spec.up`` is GLB-only and ignored here.
    """
    import trimesh
    from pxr import Usd, UsdGeom

    spec = spec or ImportSpec()
    stage = Usd.Stage.Open(usd_path)
    conv = np.eye(4)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        conv[:3, :3] = _YUP_TO_ZUP.copy()
    conv[:3, :3] *= UsdGeom.GetStageMetersPerUnit(stage) * spec.scale

    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    objects = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        gmesh = UsdGeom.Mesh(prim)
        pts = np.asarray(gmesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(gmesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        idx = np.asarray(gmesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if pts.size == 0 or counts.size == 0:
            continue
        # Gf matrices use the row-vector convention (v' = v @ M) — transpose
        # to the column convention trimesh expects
        M = np.asarray(xcache.GetLocalToWorldTransform(prim), dtype=float).T
        world = trimesh.Trimesh(vertices=pts, faces=_fan_triangulate(counts, idx),
                                process=False)
        world.apply_transform(conv @ M)
        pname = prim.GetName()
        objects.append(SceneObject(name=pname, mesh=world,
                                   dynamic=any(s in pname for s in spec.dynamic),
                                   rgba=_rgba_usd(gmesh)))
    return compose_mjcf(objects, out_dir, name, spec)
