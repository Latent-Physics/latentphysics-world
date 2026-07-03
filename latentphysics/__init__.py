"""Latent Physics World — indoor robot simulation platform.

Proprietary. Physics core built on mujoco_warp (Apache-2.0); see NOTICE.

Public API (stable surface we own; the engine is wrapped, never exposed raw):

    import latentphysics as lpw
    cfg   = lpw.Config(n_worlds=256, device="cuda")
    scene = lpw.load_scene("scenes/kitchen.xml", cfg)   # -> engine-backed Scene
    for _ in range(1000):
        scene.step()

Layers (all under `latentphysics/`, all our IP — roles in CLAUDE.md):
    backend/      thin adapter over the mujoco_warp engine
    assets/       procedural worlds, GLB/USD import, convex decomposition
    perception/   LiDAR / point cloud / depth / segmentation
    envs/         batched task containers + benchmark verifiers
    domain_rand/  domain randomization + sim-to-real calibration
    broadphase/   large-scene BVH broadphase
"""

from .version import __version__
from .config import Config, resolve_device

__all__ = ["__version__", "Config", "resolve_device", "load_scene"]


def load_scene(mjcf_path, config=None):
    """Load an MJCF scene onto the GPU engine and return a Scene facade.

    Thin entry point; the heavy lifting lives in `latentphysics.backend`.
    Kept here so user code depends only on the top-level package.
    """
    from .backend import WarpEngine  # lazy: avoids importing warp/CUDA at import time
    from .config import Config

    cfg = config or Config()
    engine = WarpEngine(cfg)
    return engine.load_mjcf(mjcf_path)
