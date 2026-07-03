"""Large-scene broadphase (our IP) — CHANGE/BUILD, readiness report §4①.

The engine's stock broadphase is SAP/NxN (O(n^2)); fully furnished rooms
(100-200+ geoms) hit a performance cliff. This adds a BVH broadphase with static/dynamic
separation: static furniture builds an AABB tree once (cached); only dynamic
bodies (arm / mobile base) do incremental queries.

This is a patch to the engine's collision driver, exposed as a selectable
`BroadphaseType.BVH` — kept here so our modification is isolated and auditable.
"""

from __future__ import annotations

__all__ = ["BVHBroadphase"]


class BVHBroadphase:
    """BVH broadphase with static/dynamic layering.

    Planned API:
        bp = BVHBroadphase(static_geoms=..., dynamic_geoms=...)
        pairs = bp.query(scene)      # candidate colliding pairs for narrowphase
    """

    def __init__(self, **kw) -> None:
        raise NotImplementedError(
            "TODO(P2): BVH over static AABBs (build once) + incremental dynamic query; "
            "integrate as BroadphaseType.BVH in the forked engine collision driver"
        )
