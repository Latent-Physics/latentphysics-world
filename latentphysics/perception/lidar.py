"""Scanning LiDAR on the engine's batched multi-ray kernel.

A LiDAR is (channels x horizontal_resolution) rays cast from a mount pose —
either a named site (moves with the robot) or a fixed world pose. One
``scan()`` casts n_worlds * n_rays rays in a single GPU launch.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import torch


@dataclass
class LidarConfig:
    channels: int = 16                 # vertical beams
    h_res: int = 180                   # horizontal samples per revolution
    v_fov: tuple = (-15.0, 15.0)       # degrees
    h_fov: tuple = (-180.0, 180.0)     # degrees (full revolution)
    max_range: float = 20.0
    dropout: float = 0.0               # random beam dropout probability
    noise_std: float = 0.0             # gaussian range noise (m)
    seed: int = 0


class Lidar:
    def __init__(self, scene, mount_site: str | None = None,
                 origin=(0.0, 0.0, 0.5), cfg: LidarConfig | None = None):
        import warp as wp
        from mujoco_warp._src import types as mjwt

        self.scene = scene
        self.cfg = cfg or LidarConfig()
        self.n = scene.n_worlds
        self.n_rays = self.cfg.channels * self.cfg.h_res
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(self.cfg.seed)

        # local ray directions (n_rays, 3)
        c = self.cfg
        v = np.deg2rad(np.linspace(c.v_fov[0], c.v_fov[1], c.channels))
        h = np.deg2rad(np.linspace(c.h_fov[0], c.h_fov[1], c.h_res, endpoint=False))
        vv, hh = np.meshgrid(v, h, indexing="ij")
        dirs = np.stack([np.cos(vv) * np.cos(hh), np.cos(vv) * np.sin(hh), np.sin(vv)], -1)
        self._local_dirs = torch.as_tensor(
            dirs.reshape(-1, 3), dtype=torch.float32, device=self.device)

        # mount
        self._site = None
        if mount_site is not None:
            import mujoco
            sid = mujoco.mj_name2id(scene.mjm, mujoco.mjtObj.mjOBJ_SITE, mount_site)
            if sid < 0:
                raise ValueError(f"site {mount_site!r} not found")
            self._site = sid
            self._site_xpos = scene.state("site_xpos")
            self._site_xmat = scene.state("site_xmat")
        self._origin = torch.tensor(origin, dtype=torch.float32, device=self.device)

        # engine buffers (allocated once, torch views for zero-copy writes)
        self._wp, self._mjwt = wp, mjwt
        self._pnt = wp.zeros((self.n, self.n_rays), dtype=wp.vec3, device="cuda")
        self._vec = wp.zeros((self.n, self.n_rays), dtype=wp.vec3, device="cuda")
        self._dist = wp.zeros((self.n, self.n_rays), dtype=wp.float32, device="cuda")
        self._geomid = wp.zeros((self.n, self.n_rays), dtype=wp.int32, device="cuda")
        self._normal = wp.zeros((self.n, self.n_rays), dtype=wp.vec3, device="cuda")
        self._exclude = wp.array(-np.ones(self.n, dtype=np.int32), dtype=wp.int32, device="cuda")
        self._pnt_t = wp.to_torch(self._pnt)
        self._vec_t = wp.to_torch(self._vec)

    @torch.no_grad()
    def scan(self) -> dict:
        """Cast all beams for all worlds; returns GPU tensors:
        dist (n, n_rays) [inf -> max_range], points (n, n_rays, 3) world-frame
        hit positions, geom_id (n, n_rays), mask (valid hits)."""
        import mujoco_warp as mjw

        if self._site is not None:
            org = self._site_xpos[:, self._site]                       # (n,3)
            rot = self._site_xmat[:, self._site].reshape(self.n, 3, 3)  # (n,3,3)
            dirs = torch.einsum("nij,rj->nri", rot, self._local_dirs)
        else:
            org = self._origin.expand(self.n, 3)
            dirs = self._local_dirs.unsqueeze(0).expand(self.n, -1, -1)

        self._pnt_t.copy_(org.unsqueeze(1).expand(-1, self.n_rays, -1))
        self._vec_t.copy_(dirs)

        geomgroup = self._mjwt.vec6f(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        mjw.rays(self.scene.model, self.scene.data, self._pnt, self._vec,
                 geomgroup, True, self._exclude,
                 self._dist, self._geomid, self._normal)
        self._wp.synchronize()

        dist = self.scene.engine._to_torch(self._dist).clone()
        gid = self.scene.engine._to_torch(self._geomid).clone()
        mask = (gid >= 0) & (dist < self.cfg.max_range)
        dist = torch.where(mask, dist, torch.full_like(dist, self.cfg.max_range))
        if self.cfg.noise_std > 0:
            dist = dist + torch.randn(dist.shape, device=self.device,
                                      generator=self.gen) * self.cfg.noise_std
        if self.cfg.dropout > 0:
            drop = torch.rand(dist.shape, device=self.device, generator=self.gen) < self.cfg.dropout
            mask = mask & ~drop
        points = self._pnt_t + self._vec_t * dist.unsqueeze(-1)
        return {"dist": dist, "points": points, "geom_id": gid, "mask": mask}
