"""Minimal RL baselines for validating the training loop end-to-end.

Not a training framework — external stacks plug in through the vectorized env
API. This exists so the platform ships with a self-contained proof that
policies actually train on it.
"""

from .ppo import PPO, PPOConfig, ActorCritic

__all__ = ["PPO", "PPOConfig", "ActorCritic"]
