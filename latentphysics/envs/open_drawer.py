"""Open-drawer manipulation task (R4) — Franka + articulated drawer chest.

First task to exercise R4 articulated furniture in the RL stack: the arm
must pull the UPPER drawer of a chest open past 15 cm (of ~26 cm travel).
Physically checkable predicate (a joint coordinate), same auto-verified
contract as the R2 suite.

Not registered in ``envs.suite.SUITE``: suite tasks share the tabletop
cube scene, while this one needs its own scene — build it with
:func:`build_scene` and pass the loaded scene in.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .base import TaskConfig, VecTask
from ..assets.scene_gen import _art_drawer_chest

__all__ = ["build_scene", "OpenDrawer"]

_S = 'contype="1" conaffinity="2"'
_D = 'contype="3" conaffinity="3"'


def build_scene(out_path: str | None = None, menagerie: str | None = None) -> str:
    """Franka at the origin facing a drawer chest 0.72 m away.

    The MJCF is written INTO the menagerie panda directory (as
    ``_lpw_open_drawer.xml``) so the panda's relative mesh paths resolve —
    MuJoCo resolves ``meshdir`` against the main file, not the include.
    ``out_path`` is accepted for API symmetry but only its basename is used.
    """
    men = menagerie or os.path.expanduser("~/lpw/menagerie")
    panda = "mjx_panda.xml"
    base = os.path.basename(out_path) if out_path else "_lpw_open_drawer.xml"
    out_path = os.path.join(men, "franka_emika_panda", base)
    rng = np.random.default_rng(0)
    # carcass gets the REACHABLE-static mask (S): unlike wall furniture, the
    # robot must collide with the chest body, not just its drawers
    parts, _ = _art_drawer_chest(rng, 0, (0.72, 0.0), (0.20, 0.35), 0.62,
                                 _S, _D, room_half=(1.0, 1.0))
    xml = f"""<mujoco model="open_drawer">
  <include file="{panda}"/>
  <option timestep="0.005" iterations="8" ls_iterations="10"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.53 0.62"
             rgb2="0.12 0.14 0.18" width="256" height="256"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.78 0.74 0.68"
             rgb2="0.68 0.64 0.58" mark="edge" markrgb="0.55 0.52 0.48"
             width="300" height="300"/>
    <material name="floormat" texture="floortex" texrepeat="10 10" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 0 3" dir="-0.3 0.2 -0.9"
           diffuse="0.8 0.78 0.75" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floormat" {_S}/>
    {"".join(parts)}
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04 0 0"
         ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
  </keyframe>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


class OpenDrawer(VecTask):
    """Pull the upper drawer open. Verify: slide joint > 15 cm."""

    OPEN_THRESH = 0.15

    def __init__(self, scene, cfg: TaskConfig | None = None):
        super().__init__(scene, cfg)
        import mujoco

        mjm = scene.mjm
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "f0_drawer1")
        self._drawer_adr = int(mjm.jnt_qposadr[jid])
        self._gripper = self._site_id("gripper")
        self._handle = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "f0w1h")
        self.geom_xpos = scene.state("geom_xpos")
        self.obs_dim = mjm.nq + mjm.nv + 3 + 3 + 1

    def drawer_q(self) -> torch.Tensor:
        return self.qpos[:, self._drawer_adr]

    def _task_reset(self, mask: torch.Tensor) -> None:
        pass                       # fixed chest; DR arrives with R3 calibration

    def _compute(self):
        grip = self.site_xpos[:, self._gripper]
        handle = self.geom_xpos[:, self._handle]
        q = self.drawer_q()
        success = q > self.OPEN_THRESH
        reward = (-torch.linalg.norm(grip - handle, dim=-1)
                  + 4.0 * q + 2.0 * success.float())
        obs = torch.cat([self.qpos, self.qvel, grip, handle, q.unsqueeze(-1)], dim=-1)
        return obs, reward, success
