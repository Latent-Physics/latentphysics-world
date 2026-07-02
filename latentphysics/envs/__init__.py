"""Environment layer (our IP) — vectorized, GPU-resident, gym-style tasks.

    import latentphysics as lpw
    from latentphysics.envs import FrankaReach, TaskConfig

    scene = lpw.load_scene(FRANKA_MJCF, lpw.Config(n_worlds=2048))
    env = FrankaReach(scene, TaskConfig(episode_len=200))
    obs = env.reset()
    obs, reward, done, info = env.step(action)   # torch tensors on cuda

Design: zero-copy views into the engine, per-world auto-reset via the
snapshot/restore branching primitive, no host round-trips in the loop.
Manager-based composition (obs/reward/termination managers, mjlab-style)
lands in R4; R1 keeps a lean subclass API (VecTask hooks).
"""

from .base import TaskConfig, VecTask
from .franka_reach import FrankaReach

__all__ = ["TaskConfig", "VecTask", "FrankaReach"]
