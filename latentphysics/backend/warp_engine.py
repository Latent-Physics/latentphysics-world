"""Adapter over the mujoco_warp GPU physics engine.

Wraps mujoco_warp's ``put_model`` / ``make_data`` / ``step`` behind a small,
stable ``Scene`` facade. The rest of Latent Physics World depends only on
``Scene`` — not on mujoco_warp — so upstream can be patched/swapped freely.

State is batched over ``n_worlds`` (mujoco_warp's ``nworld``): ``qpos`` is
``(n_worlds, nq)`` etc. Engine arrays are exposed as **zero-copy torch tensors**
on the GPU via ``warp.to_torch``.

Platform: needs an NVIDIA CUDA GPU (Linux / WSL2). Importing this module is
cheap and safe everywhere; the engine deps import lazily in ``_require_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Config, resolve_device


class EngineUnavailable(RuntimeError):
    """Raised when the GPU physics engine (warp / mujoco_warp) can't be used."""


def _require_engine():
    """Lazily import the engine, with an actionable error if unavailable."""
    try:
        import mujoco
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as e:
        raise EngineUnavailable(
            "The GPU physics engine is not available.\n"
            "Latent Physics World runs physics on mujoco_warp, which needs an "
            "NVIDIA CUDA GPU (Linux / WSL2).\n"
            "Install with:  pip install -e '.[gpu]'  (see docs/PLAN.md)\n"
            f"Underlying import error: {e}"
        ) from e
    return mujoco, wp, mjw


def _auto_budgets(mjm, n_worlds: int) -> dict:
    """Estimate contact/constraint buffer budgets from scene geometry.

    The engine's defaults are sized for small scenes and *silently drop*
    contacts/constraints on overflow (readiness report §4②) — fatal for
    contact-rich indoor scenes and for RSI anti-exploit guarantees. We scale
    with geom count; explicit Config values always win.
    """
    import numpy as np

    # Contacts require at least one DYNAMIC geom: static room geometry
    # (walls, furniture on the world body) never collides with itself, so
    # budgets must scale with movable collision geoms, not total geoms.
    # Getting this wrong is a direct OOM/throughput cliff: a 106-geom room
    # budgeted by total count allocated 2.5 GB of EPA buffers alone.
    col = (mjm.geom_contype | mjm.geom_conaffinity) != 0
    dyn = mjm.body_dofnum[mjm.geom_bodyid] > 0
    n_dyn = max(int(np.count_nonzero(col & dyn)), 1)
    per_world = min(max(16 * n_dyn + 16, 64), 512)   # contacts per world
    naconmax = min(per_world * max(n_worlds, 1), 1 << 19)  # pooled across worlds
    return {
        "naconmax": naconmax,
        # njmax is per world; ~4 constraint rows per pyramidal condim-3 contact
        "njmax": min(max(4 * per_world, 128), 2048),
        # convex-CCD (GJK/EPA) slots cost ~3 KB each. Demand is ~2-4 slots per
        # dynamic geom per world (measured ~64/world on a 32-object room), so
        # scale per world with margin but hard-cap total: the cap trades
        # dropped convex-contact candidates for not OOMing — the R2 BVH
        # broadphase is the real fix for huge cluttered batches.
        "naccdmax": min(max(2 * n_dyn, 16) * max(n_worlds, 1), 1 << 17),
    }


@dataclass
class Scene:
    """User-facing handle to a loaded, batched simulation on the GPU."""

    engine: "WarpEngine"
    mjm: Any            # mujoco.MjModel (CPU; kept for compilation/introspection)
    model: Any          # mujoco_warp Model
    data: Any           # mujoco_warp Data
    n_worlds: int = 1
    _graph: Any = None      # captured CUDA graph of a step sequence
    _graph_n: int = 0       # substep count the graph was captured for
    _fwd_graph: Any = None  # captured CUDA graph of forward() (used by resets)
    _no_graph: bool = False  # sleep-enabled models can't use naive capture

    # --- stepping -------------------------------------------------------------
    def step(self, n: int = 1) -> "Scene":
        """Advance the simulation ``n`` steps across all worlds."""
        self.engine._step(self, n)
        return self

    def reset(self) -> "Scene":
        """Reset all worlds to the model's initial state."""
        self.engine._reset(self)
        return self

    def forward(self) -> "Scene":
        """Recompute derived quantities (kinematics, sensors) without
        advancing time — call after writing qpos/qvel directly."""
        self.engine._forward(self)
        return self

    # --- state (zero-copy torch views, shaped (n_worlds, ...)) ----------------
    def state(self, name: str):
        """Zero-copy torch view of any engine Data field (e.g. 'qpos','xpos')."""
        return self.engine._to_torch(getattr(self.data, name))

    def qpos(self):
        """Generalized positions, torch (n_worlds, nq) on GPU."""
        return self.state("qpos")

    def qvel(self):
        """Generalized velocities, torch (n_worlds, nv) on GPU."""
        return self.state("qvel")

    @property
    def time(self) -> float:
        return float(self.engine._to_torch(self.data.time)[0].item())

    # --- world-state snapshot / restore (RSI branching primitive) --------------
    def snapshot(self, buf=None):
        """Capture full physics state of ALL worlds into a (n_worlds, size)
        float32 warp array. Reuse ``buf`` (a previous snapshot) to avoid
        reallocation. This is the engine-level primitive behind branching /
        rollback / deterministic replay in the RSI loop."""
        return self.engine._get_state(self, buf)

    def restore(self, snap, worlds=None) -> "Scene":
        """Restore state captured by :meth:`snapshot`. ``worlds`` optionally
        selects a boolean mask (n_worlds,) of worlds to restore."""
        self.engine._set_state(self, snap, worlds)
        return self

    # --- contacts (P1; overflow-aware, readiness report §4②) ------------------
    def num_contacts(self) -> int:
        """Current active contact count; warns if the buffer overflowed."""
        d = self.data
        nacon = getattr(d, "nacon", None)
        cap = int(getattr(d, "naconmax", 0) or 0)
        try:
            n = int(self.engine._to_torch(nacon).sum().item()) if nacon is not None else -1
        except Exception:
            n = -1
        if cap and n >= cap:
            import warnings
            warnings.warn(
                f"contact buffer full ({n}>={cap}); contacts were dropped. "
                f"Increase Config.naconmax for cluttered indoor scenes.",
                stacklevel=2,
            )
        return n


class WarpEngine:
    """Owns engine model/data creation and stepping for a given Config."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.device = resolve_device(self.config.device)
        self._mujoco, self._wp, self._mjw = _require_engine()

    def load_mjcf(self, mjcf_path: str) -> Scene:
        """Compile an MJCF and place it on the GPU engine, batched to n_worlds."""
        mujoco, wp, mjw = self._mujoco, self._wp, self._mjw
        mjm = mujoco.MjModel.from_xml_path(mjcf_path)
        if self.config.timestep is not None:
            mjm.opt.timestep = self.config.timestep
        model = mjw.put_model(mjm)
        budgets = _auto_budgets(mjm, self.config.n_worlds)
        make_kw = {
            "nworld": self.config.n_worlds,
            "naconmax": self.config.naconmax or budgets["naconmax"],
            "njmax": self.config.njmax or budgets["njmax"],
            "naccdmax": budgets["naccdmax"],
        }
        data = mjw.make_data(mjm, **make_kw)
        no_graph = bool(mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP)
        return Scene(engine=self, mjm=mjm, model=model, data=data,
                     n_worlds=self.config.n_worlds, _no_graph=no_graph)

    # --- internals ------------------------------------------------------------
    def _step(self, scene: Scene, n: int) -> None:
        """Advance n physics steps.

        Hot path uses a captured CUDA graph of the whole n-step sequence:
        mujoco_warp's step launches hundreds of kernels, and Python-driven
        launches dominate wall-clock otherwise (~40x slower measured). Capture
        once per (scene, n); replay every call. Falls back to eager launches
        if capture is unsupported.
        """
        mjw, wp = self._mjw, self._wp
        if scene._no_graph:
            # sleep-enabled models change kernel launch shapes dynamically —
            # incompatible with naive whole-step graph capture (sleep-aware
            # capture is fork-level engine work, tracked for R2)
            for _ in range(n):
                mjw.step(scene.model, scene.data)
            wp.synchronize()
            return
        if scene._graph is not None and scene._graph_n == n:
            wp.capture_launch(scene._graph)
        else:
            try:
                # warm up once eagerly so all modules are JIT-compiled,
                # then capture the sequence
                for _ in range(n):
                    mjw.step(scene.model, scene.data)
                with wp.ScopedCapture() as cap:
                    for _ in range(n):
                        mjw.step(scene.model, scene.data)
                scene._graph, scene._graph_n = cap.graph, n
            except Exception:
                scene._graph, scene._graph_n = None, 0
                for _ in range(n):
                    mjw.step(scene.model, scene.data)
        wp.synchronize()

    def _reset(self, scene: Scene) -> None:
        self._mjw.reset_data(scene.model, scene.data)

    def _forward(self, scene: Scene) -> None:
        """Recompute derived state. Graph-captured like _step — resets call this
        every episode boundary, and an eager forward (~hundreds of launches)
        would halve short-episode training throughput (measured)."""
        wp, mjw = self._wp, self._mjw
        if scene._fwd_graph is not None:
            wp.capture_launch(scene._fwd_graph)
        else:
            try:
                mjw.forward(scene.model, scene.data)  # warmup/JIT
                with wp.ScopedCapture() as cap:
                    mjw.forward(scene.model, scene.data)
                scene._fwd_graph = cap.graph
            except Exception:
                scene._fwd_graph = None
                mjw.forward(scene.model, scene.data)
        wp.synchronize()

    def _to_torch(self, warp_array):
        """Zero-copy view of a warp array as a torch tensor (GPU)."""
        return self._wp.to_torch(warp_array)

    # --- state snapshot internals ----------------------------------------------
    def _state_spec(self, mjm) -> tuple[int, int]:
        """(sig, per-world size) for full-physics snapshots.

        mujoco_warp's State bitflags mirror mujoco.mjtState, which lets us size
        the buffer with mujoco.mj_stateSize. Verified at runtime — if upstream
        ever diverges, fail loudly rather than corrupt state.
        """
        mujoco, mjw = self._mujoco, self._mjw
        for name in ("TIME", "QPOS", "QVEL", "ACT"):
            ours = int(getattr(mjw.State, name))
            theirs = int(getattr(mujoco.mjtState, f"mjSTATE_{name}"))
            if ours != theirs:
                raise RuntimeError(
                    f"mujoco_warp State.{name}={ours} != mujoco mjSTATE_{name}={theirs}; "
                    "snapshot sizing assumption broken by upstream — update _state_spec."
                )
        # INTEGRATION = full physics + warmstart + ctrl + applied forces + mocap.
        # Warmstart inclusion is what makes restore→replay bit-reproducible
        # (FULLPHYSICS omits it, leaving the solver a different starting point).
        sig = int(self._mjw.State.INTEGRATION)
        size = mujoco.mj_stateSize(mjm, sig)
        return sig, size

    def _get_state(self, scene: Scene, buf=None):
        wp, mjw = self._wp, self._mjw
        sig, size = self._state_spec(scene.mjm)
        if buf is None:
            buf = wp.zeros((scene.n_worlds, size), dtype=wp.float32, device="cuda")
        mjw.get_state(scene.model, scene.data, buf, sig)
        wp.synchronize()
        return buf

    def _set_state(self, scene: Scene, snap, worlds=None) -> None:
        wp, mjw = self._wp, self._mjw
        sig, _size = self._state_spec(scene.mjm)
        active = None
        if worlds is not None:
            import numpy as np
            mask = np.asarray(worlds)
            if mask.dtype != np.bool_ or mask.shape != (scene.n_worlds,):
                raise ValueError(f"worlds must be bool mask of shape ({scene.n_worlds},)")
            active = wp.array(mask, dtype=wp.bool, device="cuda")
        mjw.set_state(scene.model, scene.data, snap, sig, active)
        wp.synchronize()
