"""Batched depth / RGB / segmentation cameras via the engine's BVH renderer.

Wraps the native batch render context: one ``render()`` rasterizes every
model camera across every world on the GPU, then per-camera tensors are read
out zero-copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CameraConfig:
    res: tuple = (128, 128)        # (width, height) for all cameras
    rgb: bool = True
    depth: bool = True
    segmentation: bool = False
    max_depth: float = 20.0        # meters; engine normalizes depth by this
                                   # (values at max_depth = miss/far clip)


class Camera3Out:
    """Output holder: torch views onto preallocated warp buffers."""


class DepthCamera:
    def __init__(self, scene, cfg: CameraConfig | None = None):
        import warp as wp
        import mujoco_warp as mjw

        self.scene = scene
        self.cfg = cfg or CameraConfig()
        self.n = scene.n_worlds
        ncam = scene.mjm.ncam
        if ncam == 0:
            raise ValueError(
                "model has no <camera> — add one to the MJCF (scene_gen adds "
                "an overhead camera to generated rooms)")
        self.ncam = ncam
        w, h = self.cfg.res
        self._rc = mjw.create_render_context(
            scene.mjm, nworld=self.n, cam_res=(w, h),
            render_rgb=self.cfg.rgb, render_depth=self.cfg.depth,
            render_seg=self.cfg.segmentation,
        )
        self._wp, self._mjw = wp, mjw
        self._rgb = wp.zeros((self.n, h, w), dtype=wp.vec3, device="cuda") if self.cfg.rgb else None
        self._depth = wp.zeros((self.n, h, w), dtype=wp.float32, device="cuda") if self.cfg.depth else None
        self._seg = wp.zeros((self.n, h, w), dtype=wp.vec2i, device="cuda") if self.cfg.segmentation else None

    @torch.no_grad()
    def render(self, camera: int = 0) -> dict:
        """Render all worlds; return dict of GPU tensors for one camera:
        rgb (n,h,w,3) float, depth (n,h,w) meters, seg (n,h,w,2) int."""
        mjw, wp = self._mjw, self._wp
        mjw.refit_bvh(self.scene.model, self.scene.data, self._rc)
        mjw.render(self.scene.model, self.scene.data, self._rc)
        out = {}
        if self._rgb is not None:
            mjw.get_rgb(self._rc, camera, self._rgb)
            out["rgb"] = wp.to_torch(self._rgb)
        if self._depth is not None:
            # engine returns depth normalized by the scale (1.0 == miss/far);
            # multiply back so callers get METERS
            mjw.get_depth(self._rc, camera, self.cfg.max_depth, self._depth)
            out["depth"] = wp.to_torch(self._depth) * self.cfg.max_depth
        if self._seg is not None:
            mjw.get_segmentation(self._rc, camera, self._seg)
            out["seg"] = wp.to_torch(self._seg)
        wp.synchronize()
        return out
