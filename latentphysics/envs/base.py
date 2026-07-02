"""Vectorized task base — gym-style batched envs on the GPU engine.

Everything stays GPU-resident: actions are written into the engine's ctrl
buffer through a zero-copy torch view, observations/rewards/dones are computed
in torch from zero-copy state views, and per-world auto-reset uses the
engine-level snapshot/restore primitive (the same one that powers RSI
branching). No host round-trips inside the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..backend import Scene


@dataclass
class TaskConfig:
    episode_len: int = 200        # control steps per episode
    substeps: int = 4             # physics steps per control step (dt*4 = control period)
    seed: int = 0


class VecTask:
    """Base class: subclasses implement observation/reward/task-reset hooks.

    Contract for subclasses:
      - ``obs_dim`` / ``act_dim`` attributes (act_dim defaults to nu)
      - ``_task_reset(mask)``: (re)randomize task variables for masked worlds
      - ``_compute()``: return (obs, reward, success) — all torch, batched
    """

    def __init__(self, scene: Scene, cfg: TaskConfig | None = None):
        self.scene = scene
        self.cfg = cfg or TaskConfig()
        self.n = scene.n_worlds
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(self.cfg.seed)

        # zero-copy engine views (torch tensors backed by warp memory)
        self.ctrl = scene.state("ctrl")            # (n, nu)
        self.qpos = scene.state("qpos")            # (n, nq)
        self.qvel = scene.state("qvel")            # (n, nv)
        self.site_xpos = scene.state("site_xpos")  # (n, nsite, 3)

        mjm = scene.mjm
        self.nu = mjm.nu
        self.act_dim = self.nu
        cr = torch.as_tensor(mjm.actuator_ctrlrange, dtype=torch.float32, device=self.device)
        self._ctrl_lo, self._ctrl_hi = cr[:, 0], cr[:, 1]

        self.progress = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._init_snap = None

    # --- keyframe / initial state ------------------------------------------------
    def _write_keyframe(self, name: str) -> None:
        """Set all worlds to a model keyframe and refresh kinematics."""
        import mujoco

        mjm = self.scene.mjm
        kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise ValueError(f"keyframe {name!r} not found")
        qpos0 = torch.as_tensor(mjm.key_qpos[kid], dtype=torch.float32, device=self.device)
        ctrl0 = torch.as_tensor(mjm.key_ctrl[kid], dtype=torch.float32, device=self.device)
        self.qpos.copy_(qpos0.expand_as(self.qpos))
        self.qvel.zero_()
        self.ctrl.copy_(ctrl0.expand_as(self.ctrl))
        self.scene.forward()

    # --- gym-style API -------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset ALL worlds to the initial state; snapshot it for fast auto-reset."""
        self._write_keyframe("home") if self._has_key("home") else self.scene.reset()
        self.progress.zero_()
        self._task_reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        self._init_snap = self.scene.snapshot(self._init_snap)
        obs, _, _ = self._compute()
        return obs

    def step(self, action: torch.Tensor):
        """action in [-1,1]^(n,act_dim) -> (obs, reward, done, info)."""
        a = torch.clamp(action, -1.0, 1.0)
        target = self._ctrl_lo + (a + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)
        self.ctrl.copy_(target)
        self.scene.step(self.cfg.substeps)
        self.progress += 1

        obs, reward, success = self._compute()
        timeout = self.progress >= self.cfg.episode_len
        done = success | timeout

        if done.any():
            self._auto_reset(done)
            # recompute obs for the reset worlds so the policy sees fresh state
            obs_new, _, _ = self._compute()
            obs = torch.where(done.unsqueeze(-1), obs_new, obs)

        info = {"success": success, "timeout": timeout}
        return obs, reward, done, info

    def _auto_reset(self, mask: torch.Tensor) -> None:
        self.scene.restore(self._init_snap, worlds=mask.cpu().numpy())
        self.progress[mask] = 0
        self._task_reset(mask)
        self.scene.forward()

    # --- helpers -------------------------------------------------------------------
    def _has_key(self, name: str) -> bool:
        import mujoco

        return mujoco.mj_name2id(self.scene.mjm, mujoco.mjtObj.mjOBJ_KEY, name) >= 0

    def _site_id(self, name: str) -> int:
        import mujoco

        sid = mujoco.mj_name2id(self.scene.mjm, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site {name!r} not found")
        return sid

    def _body_qpos_addr(self, body_name: str) -> int:
        """qpos address of a body's free joint (for object poses in obs)."""
        import mujoco

        mjm = self.scene.mjm
        bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"body {body_name!r} not found")
        jadr = mjm.body_jntadr[bid]
        return int(mjm.jnt_qposadr[jadr])

    # --- subclass hooks --------------------------------------------------------------
    def _task_reset(self, mask: torch.Tensor) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _compute(self):  # pragma: no cover - interface
        raise NotImplementedError
