"""Smoke tests that must pass on ANY host (incl. Windows dev box, no CUDA).

They verify the package imports, config works, and the engine boundary degrades
gracefully when GPU deps are absent — WITHOUT importing warp/mujoco_warp.
"""

import importlib

import pytest


def test_package_imports():
    lpw = importlib.import_module("latentphysics")
    assert lpw.__version__
    for sub in ["backend", "assets", "perception", "envs", "domain_rand", "broadphase"]:
        importlib.import_module(f"latentphysics.{sub}")


def test_config_and_device():
    import latentphysics as lpw

    cfg = lpw.Config(n_worlds=8)
    assert cfg.n_worlds == 8
    assert lpw.resolve_device("cpu") == "cpu"
    assert lpw.resolve_device("auto") in ("cpu", "cuda")


def test_engine_unavailable_is_graceful():
    """Off-GPU, loading a scene must raise EngineUnavailable (not ImportError)."""
    import latentphysics as lpw
    from latentphysics.backend import EngineUnavailable

    try:
        import mujoco_warp  # noqa: F401

        pytest.skip("engine present; unavailability path not exercised here")
    except ImportError:
        pass

    with pytest.raises(EngineUnavailable):
        lpw.load_scene("nonexistent.xml", lpw.Config())
