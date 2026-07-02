"""Batched point clouds (LiDAR output container + export)."""

from __future__ import annotations

import torch


class PointCloud:
    def __init__(self, points: torch.Tensor, mask: torch.Tensor | None = None):
        """points: (n_worlds, n_pts, 3) GPU tensor; mask: (n_worlds, n_pts) valid."""
        self.points = points
        self.mask = mask if mask is not None else torch.ones(
            points.shape[:2], dtype=torch.bool, device=points.device)

    def world(self, i: int) -> torch.Tensor:
        """Valid points of one world, (k, 3)."""
        return self.points[i][self.mask[i]]

    def save_ply(self, i: int, path: str) -> str:
        pts = self.world(i).cpu().numpy()
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in pts:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        return path
