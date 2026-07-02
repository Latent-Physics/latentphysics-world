"""Train PPO on FrankaReach — the R1 end-to-end training proof.

    python examples/train_franka_reach.py --n-worlds 2048 --iters 300

Saves the policy to ~/lpw/runs/franka_reach.pt and a qpos trajectory of the
trained policy (single world) for viewer replay.
"""

import argparse
import os
import time

import torch

import latentphysics as lpw
from latentphysics.envs import FrankaReach, PhysicsSentinel, TaskConfig
from latentphysics.rl import PPO, PPOConfig

DEFAULT_MJCF = os.path.join(
    os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
    "franka_emika_panda", "mjx_single_cube.xml",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--n-worlds", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--episode-len", type=int, default=150)
    ap.add_argument("--out", default=os.path.expanduser("~/lpw/runs"))
    args = ap.parse_args()

    scene = lpw.load_scene(args.mjcf, lpw.Config(n_worlds=args.n_worlds))
    env = FrankaReach(scene, TaskConfig(
        episode_len=args.episode_len, substeps=4, seed=0,
        obs_noise=0.005, act_noise=0.01,     # DR v0
    ))
    agent = PPO(env, PPOConfig())
    sentinel = PhysicsSentinel(scene)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        stats = agent.train_iter()
        if it % 10 == 0:
            sr = agent.success_rate()
            rep = sentinel.check()
            sps = it * agent.cfg.horizon * args.n_worlds / (time.perf_counter() - t0)
            print(f"iter {it:4d} | success {sr:6.1%} | env-sps {sps:9.0f} "
                  f"| pg {stats['pg']:+.3f} v {stats['v']:.3f} "
                  f"| sentinel viol {int(rep['any'].sum())}", flush=True)

    final = agent.evaluate()
    print(f"FINAL deterministic success rate: {final:.1%}", flush=True)
    torch.save(agent.net.state_dict(), os.path.join(args.out, "franka_reach.pt"))

    # record a single-world trajectory of the trained policy for replay
    obs = env.reset()
    traj = []
    for _ in range(env.cfg.episode_len):
        a, _, _ = agent.net.act(obs, deterministic=True)
        obs, _, _, _ = env.step(a)
        traj.append(env.qpos[0].detach().cpu().numpy().copy())
    import numpy as np
    np.save(os.path.join(args.out, "franka_reach_traj.npy"), np.asarray(traj))
    tgt = env.targets[0].detach().cpu().numpy()
    np.save(os.path.join(args.out, "franka_reach_target.npy"), tgt)
    print("saved policy + trajectory to", args.out, flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
