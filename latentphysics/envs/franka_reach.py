"""FrankaReach — first R1 task: drive the gripper to a random 3D target.

Contact-light on purpose: it validates the full env/training loop (obs, reward,
auto-reset, throughput) before contact-rich tasks (push/pick) stack physics
difficulty on top.
"""

from __future__ import annotations

import torch

from .base import TaskConfig, VecTask

# workspace box for target sampling (in front of the arm, above the floor)
_WS_LO = (0.30, -0.35, 0.15)
_WS_HI = (0.75, 0.35, 0.60)
_SUCCESS_RADIUS = 0.05


class FrankaReach(VecTask):
    def __init__(self, scene, cfg: TaskConfig | None = None):
        super().__init__(scene, cfg)
        self._gripper = self._site_id("gripper")
        self.targets = torch.zeros(self.n, 3, device=self.device)
        self._lo = torch.tensor(_WS_LO, device=self.device)
        self._hi = torch.tensor(_WS_HI, device=self.device)
        # obs = qpos + qvel + gripper_pos + target
        self.obs_dim = scene.mjm.nq + scene.mjm.nv + 3 + 3

    def _task_reset(self, mask: torch.Tensor) -> None:
        k = int(mask.sum().item())
        if k == 0:
            return
        u = torch.rand(k, 3, device=self.device, generator=self.gen)
        self.targets[mask] = self._lo + u * (self._hi - self._lo)

    def _compute(self):
        grip = self.site_xpos[:, self._gripper]           # (n,3)
        dist = torch.linalg.norm(grip - self.targets, dim=-1)
        success = dist < _SUCCESS_RADIUS
        reward = -dist + success.float()
        obs = torch.cat([self.qpos, self.qvel, grip, self.targets], dim=-1)
        return obs, reward, success
