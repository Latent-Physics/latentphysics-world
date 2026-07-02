"""GPU env-layer tests: FrankaReach mechanics + throughput floor.

Needs the GPU engine and a mujoco_menagerie checkout (set LPW_MENAGERIE or
clone to ~/lpw/menagerie). Skips cleanly otherwise.
"""

import os
import time

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

MJCF = os.path.join(
    os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie")),
    "franka_emika_panda", "mjx_single_cube.xml",
)
if not os.path.exists(MJCF):
    pytest.skip("mujoco_menagerie Franka scene not found", allow_module_level=True)

from latentphysics.envs import FrankaReach, TaskConfig  # noqa: E402


@pytest.fixture(scope="module")
def env():
    scene = lpw.load_scene(MJCF, lpw.Config(n_worlds=512))
    return FrankaReach(scene, TaskConfig(episode_len=50, substeps=4, seed=0))


def test_reset_and_shapes(env):
    obs = env.reset()
    assert obs.shape == (512, env.obs_dim) and obs.is_cuda
    assert env.act_dim == 8


def test_step_autoreset_and_reward_range(env):
    env.reset()
    n_done = 0
    for _ in range(60):  # > episode_len -> timeouts must fire
        a = torch.rand(512, env.act_dim, device="cuda") * 2 - 1
        obs, rew, done, info = env.step(a)
        n_done += int(done.sum().item())
    assert obs.shape == (512, env.obs_dim) and rew.is_cuda and done.is_cuda
    assert n_done >= 512, "every world should have finished at least one episode"
    assert env.progress.max().item() <= 50
    assert rew.min().item() > -3.0 and rew.max().item() <= 1.0


def test_throughput_floor(env):
    """R1 KPI: >=500k physics steps/s on the Franka scene (CUDA-graph path)."""
    env.reset()
    a = torch.zeros(512, env.act_dim, device="cuda")
    for _ in range(12):
        env.step(a)  # warmup + graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        env.step(a)
    torch.cuda.synchronize()
    phys_sps = 50 * 512 * env.cfg.substeps / (time.perf_counter() - t0)
    assert phys_sps > 500_000, f"physics steps/s too low: {phys_sps:.0f}"
