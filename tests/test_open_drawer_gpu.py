"""GPU tests for the open-drawer benchmark task (R4 articulated furniture)."""

import os

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

MEN = os.environ.get("LPW_MENAGERIE", os.path.expanduser("~/lpw/menagerie"))
if not os.path.exists(os.path.join(MEN, "franka_emika_panda", "mjx_panda.xml")):
    pytest.skip("mujoco_menagerie not found", allow_module_level=True)

from latentphysics.envs import TaskConfig  # noqa: E402
from latentphysics.envs.open_drawer import OpenDrawer, build_scene  # noqa: E402

N = 16


def _retract_arm(env):
    """Lift the arm clear of the chest for a pure joint-drive check. At home
    the gripper is staged in the drawer mouth, and a servo-held arm resting
    there drags the (realistically light hollow) drawer; the arm actually
    opening it is covered by examples/franka_open_drawer_arm.py."""
    qp, ct = env.scene.qpos(), env.scene.state("ctrl")
    qp[:, 1] = -1.3
    ct[:, 1] = -1.3
    env.scene.qvel().zero_()
    for _ in range(30):
        env.scene.step()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    path = build_scene(str(tmp_path_factory.mktemp("dw") / "open_drawer.xml"))
    scene = lpw.load_scene(path, lpw.Config(n_worlds=N))
    return OpenDrawer(scene, TaskConfig(episode_len=60, substeps=4, seed=1))


def test_reset_and_step_shapes(env):
    obs = env.reset()
    assert obs.shape == (N, env.obs_dim) and obs.is_cuda
    act = torch.zeros(N, env.act_dim, device="cuda")
    obs, reward, done, info = env.step(act)
    assert obs.shape == (N, env.obs_dim)
    assert reward.shape == done.shape == (N,)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    # arm holding home pose does not accidentally solve the task
    assert not info["success"].any()


def test_opening_drawer_is_success(env):
    env.reset()
    _retract_arm(env)
    _, r0, _ = env._compute()
    jid = mujoco.mj_name2id(env.scene.mjm, mujoco.mjtObj.mjOBJ_JOINT, "f0_drawer1")
    dof = int(env.scene.mjm.jnt_dofadr[jid])
    qvel = env.scene.qvel()
    for _ in range(200):
        qvel[:, dof] = 0.45          # kinematic pull on the upper drawer
        env.scene.step()
    obs, r1, success = env._compute()
    assert success.all(), f"drawer q={env.drawer_q()[0].item():.3f}"
    assert (r1 > r0).all(), "reward did not increase as the drawer opened"
    assert torch.isfinite(obs).all()


def test_auto_reset_restores_closed_drawer(env):
    env.reset()
    _retract_arm(env)
    jid = mujoco.mj_name2id(env.scene.mjm, mujoco.mjtObj.mjOBJ_JOINT, "f0_drawer1")
    dof = int(env.scene.mjm.jnt_dofadr[jid])
    qvel = env.scene.qvel()
    for _ in range(200):
        qvel[:, dof] = 0.45
        env.scene.step()
    # a step observing success must auto-reset those worlds to closed
    obs, _, done, info = env.step(torch.zeros(N, env.act_dim, device="cuda"))
    assert info["success"].any()
    assert (env.drawer_q() < 0.02).all(), "auto-reset left drawers open"
