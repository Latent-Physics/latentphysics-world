"""GPU tests: procedural rooms + LiDAR + depth/seg cameras (R2)."""

import os

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from latentphysics.assets.scene_gen import RoomSpec, generate_room  # noqa: E402
from latentphysics.perception import (  # noqa: E402
    CameraConfig, DepthCamera, Lidar, LidarConfig, PointCloud)


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    out = tmp_path_factory.mktemp("rooms")
    return generate_room(RoomSpec(seed=3, n_furniture=12, n_clutter=6),
                         str(out / "room.xml"))


@pytest.fixture(scope="module")
def scene(room):
    s = lpw.load_scene(room, lpw.Config(n_worlds=64))
    s.step(30)
    return s


def test_room_generation_reproducible(tmp_path):
    a = generate_room(RoomSpec(seed=7), str(tmp_path / "a.xml"))
    b = generate_room(RoomSpec(seed=7), str(tmp_path / "b.xml"))
    assert open(a).read() == open(b).read()
    c = generate_room(RoomSpec(seed=8), str(tmp_path / "c.xml"))
    assert open(a).read() != open(c).read()


def test_room_masks_prune_static_pairs(room):
    import mujoco
    m = mujoco.MjModel.from_xml_path(room)
    # walls/furniture (contype=4) must not pair with each other or clutter
    assert (m.geom_contype == 4).sum() >= 12
    assert (m.geom_contype == 3).sum() >= 6


def test_lidar_scan(scene):
    lidar = Lidar(scene, origin=(0.0, 0.0, 0.8),
                  cfg=LidarConfig(channels=8, h_res=90))
    out = lidar.scan()
    assert out["dist"].shape == (64, 8 * 90) and out["dist"].is_cuda
    assert out["points"].shape == (64, 8 * 90, 3)
    hit = out["mask"].float().mean().item()
    assert hit > 0.4, f"indoor scan should hit walls/floor (hit-rate {hit:.0%})"
    pc = PointCloud(out["points"], out["mask"])
    assert pc.world(0).shape[0] > 100


def test_depth_camera_meters(scene, tmp_path):
    cam = DepthCamera(scene, CameraConfig(res=(64, 64), rgb=True, depth=True,
                                          segmentation=True))
    f = cam.render(camera=0)  # overhead camera at ~2.2 m above the floor
    d = f["depth"]
    assert d.shape == (64, 64, 64)
    assert 1.0 < d.min().item() < 2.0, "table should be ~1.4-1.8 m below cam"
    assert 2.0 < d.max().item() < 3.0, "floor should be ~2.2 m below cam"
    assert f["rgb"].mean().item() > 0.05, "rgb should have content"
    assert len(f["seg"][..., 0].unique()) > 3, "seg should distinguish geoms"
