"""Single-file PPO on GPU-resident vectorized envs (R1 baseline).

Standard clipped-surrogate PPO with GAE; every tensor stays on the GPU end to
end (rollout, advantage, update) — the point is to demonstrate the platform's
zero-copy training loop, not to be a feature-complete RL library.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def dist(self, obs):
        mean = self.pi(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.rsample()
        return a, d.log_prob(a).sum(-1), self.v(obs).squeeze(-1)


@dataclass
class PPOConfig:
    horizon: int = 32
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0


class PPO:
    def __init__(self, env, cfg: PPOConfig | None = None, device: str = "cuda"):
        self.env = env
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.net = ActorCritic(env.obs_dim, env.act_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self._obs = env.reset()
        # rolling success statistics
        self._succ, self._eps = 0, 0

    @torch.no_grad()
    def _rollout(self):
        c, n = self.cfg, self.env.n
        T = c.horizon
        obs = torch.empty(T, n, self.env.obs_dim, device=self.device)
        act = torch.empty(T, n, self.env.act_dim, device=self.device)
        logp = torch.empty(T, n, device=self.device)
        rew = torch.empty(T, n, device=self.device)
        done = torch.empty(T, n, device=self.device)
        val = torch.empty(T + 1, n, device=self.device)

        for t in range(T):
            obs[t] = self._obs
            a, lp, v = self.net.act(self._obs)
            act[t], logp[t], val[t] = a, lp, v
            self._obs, r, d, info = self.env.step(a)
            rew[t], done[t] = r, d.float()
            self._succ += int(info["success"].sum().item())
            self._eps += int(d.sum().item())
        val[T] = self.net.act(self._obs)[2]

        adv = torch.zeros(T, n, device=self.device)
        gae = torch.zeros(n, device=self.device)
        for t in reversed(range(T)):
            nonterm = 1.0 - done[t]
            delta = rew[t] + c.gamma * val[t + 1] * nonterm - val[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val[:T]
        return (x.reshape(-1, *x.shape[2:]) for x in (obs, act, logp, adv, ret))

    def train_iter(self):
        c = self.cfg
        obs, act, logp_old, adv, ret = self._rollout()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = obs.shape[0]
        idx = torch.randperm(B, device=self.device)
        mb = B // c.minibatches
        for _ in range(c.epochs):
            for k in range(c.minibatches):
                i = idx[k * mb:(k + 1) * mb]
                d = self.net.dist(obs[i])
                logp = d.log_prob(act[i]).sum(-1)
                ratio = (logp - logp_old[i]).exp()
                pg = -torch.min(
                    ratio * adv[i],
                    ratio.clamp(1 - c.clip, 1 + c.clip) * adv[i],
                ).mean()
                v = self.net.v(obs[i]).squeeze(-1)
                vloss = 0.5 * (v - ret[i]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + c.vf_coef * vloss - c.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
        return {"pg": pg.item(), "v": vloss.item(), "ent": ent.item()}

    def success_rate(self, reset: bool = True) -> float:
        r = self._succ / max(self._eps, 1)
        if reset:
            self._succ, self._eps = 0, 0
        return r

    @torch.no_grad()
    def evaluate(self, episodes_per_world: int = 1) -> float:
        """Deterministic-policy success rate over fresh episodes."""
        obs = self.env.reset()
        succ, eps = 0, 0
        max_steps = self.env.cfg.episode_len * episodes_per_world + 1
        for _ in range(max_steps):
            a, _, _ = self.net.act(obs, deterministic=True)
            obs, _, d, info = self.env.step(a)
            succ += int(info["success"].sum().item())
            eps += int(d.sum().item())
            if eps >= self.env.n * episodes_per_world:
                break
        return succ / max(eps, 1)
