"""Perception (our IP) — GPU-batched LiDAR, depth, segmentation (R2).

Built directly on the engine's batched ray casting (``rays``) and native BVH
batch renderer (``create_render_context``/``render``). Everything returns
torch tensors shaped ``(n_worlds, ...)`` on the GPU.
"""

from .lidar import Lidar, LidarConfig
from .camera import DepthCamera, CameraConfig
from .pointcloud import PointCloud

__all__ = ["Lidar", "LidarConfig", "DepthCamera", "CameraConfig", "PointCloud"]
