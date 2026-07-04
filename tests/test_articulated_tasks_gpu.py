"""GPU tests for the articulated-furniture task family (open door/lid/drawer/slide)."""

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
from latentphysics.envs.articulated_tasks import ART_SPECS, build_articulated_scene, make  # noqa: E402

N = 8
TASKS = sorted(ART_SPECS)


def _env(task, tmp):
    path = build_articulated_scene(task, os.path.join(tmp, f"{task}.xml"))
    scene = lpw.load_scene(path, lpw.Config(n_worlds=N, njmax=2048))
    return make(task, scene, TaskConfig(episode_len=80, substeps=4, seed=1))


def test_registry_covers_four_archetypes():
    assert set(ART_SPECS) == {"open_drawer", "open_door", "open_lid", "slide_door"}


@pytest.mark.parametrize("task", TASKS)
def test_reset_step_and_no_false_success(tmp_path, task):
    env = _env(task, str(tmp_path))
    obs = env.reset()
    assert obs.shape == (N, env.obs_dim) and obs.is_cuda
    obs, reward, done, info = env.step(torch.zeros(N, env.act_dim, device="cuda"))
    assert reward.shape == done.shape == (N,)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    # holding home pose must not already count as opened
    assert not info["success"].any(), f"{task}: home pose falsely marked success"


@pytest.mark.parametrize("task", TASKS)
def test_opening_flips_success_and_raises_reward(tmp_path, task):
    env = _env(task, str(tmp_path))
    env.reset()
    # This is a pure joint-drive check: driving the target joint must flip
    # success and raise reward. Retract the arm clear of the piece first — at
    # home the gripper sits in the mouth of the piece (it is staged to grasp
    # the handle), and a servo-held arm resting there drags the drawer, which
    # is now a realistically light hollow tray rather than the old solid block.
    # The arm ACTUALLY opening a piece is covered end-to-end by
    # examples/franka_open_drawer_arm.py.
    qp, ct = env.scene.qpos(), env.scene.state("ctrl")
    qp[:, 1] = -1.3      # lift the shoulder so the gripper clears the piece
    ct[:, 1] = -1.3
    env.scene.qvel().zero_()
    for _ in range(30):
        env.scene.step()
    _, r0, _ = env._compute()
    jid = mujoco.mj_name2id(env.scene.mjm, mujoco.mjtObj.mjOBJ_JOINT, ART_SPECS[task].joint)
    dof = int(env.scene.mjm.jnt_dofadr[jid])
    qvel = env.scene.qvel()
    # kinematic drive on the target joint (hinges faster than slides)
    v = 1.2 if "door" in task or "lid" in task else 0.4
    for _ in range(240):
        qvel[:, dof] = v
        env.scene.step()
    obs, r1, success = env._compute()
    assert torch.isfinite(obs).all()
    assert success.all(), f"{task}: q={env.joint_q()[0].item():.3f} < {ART_SPECS[task].thresh}"
    assert (r1 > r0).all(), f"{task}: reward did not rise as it opened"
