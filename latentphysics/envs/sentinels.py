"""Physics sentinels v0 — the first anti-exploit guardrail (RSI principle #2).

RL at scale *will* find and exploit simulator artifacts (deep penetration,
energy injection, NaN states). Sentinels watch for physically impossible
signatures during training and report per-world violations, so exploited
rollouts can be flagged, logged, or terminated.

v0 checks (cheap, model-agnostic, GPU-resident):
  - penetration: contact depth beyond tolerance (contacts should be shallow
    under soft-constraint physics; deep interpenetration = broken states)
  - velocity explosion: |qvel| above a hard bound
  - non-finite state: NaN/Inf in qpos/qvel

Usage (periodic, not per-step — keep the hot loop clean):

    sentinel = PhysicsSentinel(scene)
    report = sentinel.check()          # dict of per-world tensors
    if report["any"].any(): ...        # flag/reset offending worlds
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SentinelConfig:
    penetration_tol: float = 0.01     # m; max tolerated contact depth
    qvel_max: float = 100.0           # rad/s or m/s; explosion bound


class PhysicsSentinel:
    def __init__(self, scene, cfg: SentinelConfig | None = None):
        self.scene = scene
        self.cfg = cfg or SentinelConfig()
        self.n = scene.n_worlds
        eng = scene.engine
        self._qpos = eng._to_torch(scene.data.qpos)
        self._qvel = eng._to_torch(scene.data.qvel)
        self._dist = eng._to_torch(scene.data.contact.dist)        # (naconmax,)
        self._worldid = eng._to_torch(scene.data.contact.worldid)  # (naconmax,)
        self._nacon = eng._to_torch(scene.data.nacon)
        self.violation_counts = torch.zeros(self.n, dtype=torch.long, device="cuda")

    @torch.no_grad()
    def check(self) -> dict:
        n_active = int(self._nacon.sum().item())
        pen = torch.zeros(self.n, device="cuda")
        if n_active > 0:
            d = self._dist[:n_active]
            w = self._worldid[:n_active].long()
            depth = torch.clamp(-d, min=0.0)   # dist<0 == penetration
            pen.scatter_reduce_(0, w, depth, reduce="amax")
        vel = self._qvel.abs().amax(dim=-1)
        finite = torch.isfinite(self._qpos).all(-1) & torch.isfinite(self._qvel).all(-1)

        v_pen = pen > self.cfg.penetration_tol
        v_vel = vel > self.cfg.qvel_max
        v_nan = ~finite
        any_v = v_pen | v_vel | v_nan
        self.violation_counts += any_v.long()
        return {
            "penetration_depth": pen,
            "qvel_max": vel,
            "nonfinite": v_nan,
            "penetration": v_pen,
            "velocity": v_vel,
            "any": any_v,
        }

    @torch.no_grad()
    def check_energy(self) -> torch.Tensor:
        """Total (potential+kinetic) energy per world — eager engine call,
        use sparingly (e.g. once per training iteration)."""
        eng = self.scene.engine
        eng._mjw.energy_pos(self.scene.model, self.scene.data)
        eng._mjw.energy_vel(self.scene.model, self.scene.data)
        eng._wp.synchronize()
        return eng._to_torch(self.scene.data.energy).sum(dim=-1)
