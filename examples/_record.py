"""Shared demo recorder: replay a GPU-simulated qpos trajectory offscreen
and write an animated webp for the gallery.

Physics always runs on the LPW GPU engine; this module only re-renders the
recorded trajectory with the CPU reference renderer (visuals, not physics).
"""

from __future__ import annotations

import os
import subprocess
import tempfile


def record_webp(mjcf_path, qpos_traj, out_name, cam=None, every=3, fps=15,
                size=(640, 400), media_dir=None, quality=70, ssaa=2):
    """Render qpos_traj (list/array of qpos) into docs/media/<out_name>.webp.

    ``size`` is the OUTPUT resolution; ``ssaa`` supersamples (renders at
    ssaa*size, then box-downscales) since MuJoCo's Renderer has no MSAA —
    supersampling is the only anti-aliasing available."""
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco
    import numpy as np

    media_dir = media_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "media")
    os.makedirs(media_dir, exist_ok=True)

    # MuJoCo's default offscreen framebuffer is 640x480; enlarge it at compile
    # time so the supersampled render fits. MjSpec preserves includes/meshdir
    # (needed for menagerie scenes); the fallback keeps ssaa within 640x480.
    try:
        spec = mujoco.MjSpec.from_file(mjcf_path)
        spec.visual.global_.offwidth = max(size[0] * ssaa, 640)
        spec.visual.global_.offheight = max(size[1] * ssaa, 480)
        m = spec.compile()
    except Exception:
        m = mujoco.MjModel.from_xml_path(mjcf_path)
        ssaa = max(1, min(640 // size[0], 480 // size[1]))
    rw, rh = size[0] * ssaa, size[1] * ssaa
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=rh, width=rw)
    c = mujoco.MjvCamera()
    cam = cam or {}
    c.lookat[:] = cam.get("lookat", (0, 0, 0.3))
    c.distance = cam.get("distance", 2.0)
    c.azimuth = cam.get("azimuth", 120)
    c.elevation = cam.get("elevation", -20)

    def _downscale(img):
        if ssaa == 1:
            return img
        h, w = size[1], size[0]
        return (img.reshape(h, ssaa, w, ssaa, 3).mean(axis=(1, 3))).astype(np.uint8)

    frames = []
    for k, q in enumerate(qpos_traj):
        if k % every:
            continue
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        if "azimuth_rate" in cam:
            c.azimuth = cam.get("azimuth", 120) + k * cam["azimuth_rate"]
        r.update_scene(d, camera=c)
        frames.append(_downscale(r.render()).copy())
    r.close()
    return save_frames(frames, out_name, fps=fps, quality=quality, media_dir=media_dir)


def save_frames(frames, out_name, fps=15, quality=70, media_dir=None):
    """Encode a list of HxWx3 uint8 frames to docs/media/<out_name>.webp
    (gif fallback). Used by clips that produce image frames directly —
    perception overlays, matplotlib point clouds — not a qpos replay."""
    import imageio.v2 as imageio
    import imageio_ffmpeg

    media_dir = media_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "media")
    os.makedirs(media_dir, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), out_name + ".mp4")
    imageio.mimsave(tmp, frames, fps=fps, quality=8)
    out = os.path.join(media_dir, out_name + ".webp")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    res = subprocess.run([ffmpeg, "-y", "-i", tmp, "-vcodec", "libwebp", "-lossless", "0",
                          "-q:v", str(quality), "-loop", "0", "-an", out], capture_output=True)
    if res.returncode != 0:
        out = os.path.join(media_dir, out_name + ".gif")
        imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"wrote {out} ({len(frames)} frames)")
    return out
