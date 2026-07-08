"""Procedural lofted meshes for the office-chair asset (visual layer only).

Pure numpy: every part is a superellipse cross-section swept along an axis
(a loft), written as OBJ with UVs. Deterministic — same code, same bytes.
Meshes are generated on demand into ~/lpw/assets/chair_meshes_v1 (never
committed, same policy as fetched assets). Collision stays primitive —
these meshes carry no mass and no contype.
"""

from __future__ import annotations

import os

import numpy as np

MESH_DIR = os.path.expanduser("~/lpw/assets/chair_meshes_v1")


# ------------------------------------------------------------ mesh plumbing
def _superellipse(K, a, b, p):
    """(K,2) rounded-rectangle ring, CCW, p>=2 (2=ellipse, high=boxy)."""
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / p),
                     b * np.sign(s) * np.abs(s) ** (2 / p)], axis=1)


def _loft(rings):
    """Watertight loft through same-K rings -> (V, F, VT). Rings must be
    ordered along the sweep; caps are centroid fans."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    K = rings[0].shape[0]
    n = len(rings)
    V = np.concatenate(rings, axis=0)
    VT = np.stack([np.tile(np.linspace(0, 1, K, endpoint=False), n),
                   np.repeat(np.linspace(0, 1, n), K)], axis=1)
    F = []
    for i in range(n - 1):
        for k in range(K):
            k1 = (k + 1) % K
            a, b = i * K + k, i * K + k1
            c, d = (i + 1) * K + k1, (i + 1) * K + k
            F.append((a, b, c))
            F.append((a, c, d))
    # caps
    c0 = len(V)
    V = np.concatenate([V, rings[0].mean(0, keepdims=True),
                        rings[-1].mean(0, keepdims=True)], axis=0)
    VT = np.concatenate([VT, [[0.5, 0.0], [0.5, 1.0]]], axis=0)
    for k in range(K):
        k1 = (k + 1) % K
        F.append((c0, k1, k))                                  # bottom cap
        F.append((c0 + 1, (n - 1) * K + k, (n - 1) * K + k1))  # top cap
    return V, np.asarray(F, dtype=int), VT


def _write_obj(path, V, F, VT):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in VT:
            f.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
        for a, b, c in F:
            f.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")


def _shrink_caps(build_ring, ts, shrink=(1.0, 0.82, 0.5, 0.18), pad=0.014):
    """Wrap a ring builder with rounded end closures: extra rings at each
    end, scaled toward the centroid and pushed outward along the sweep."""
    t0, t1 = ts[0], ts[-1]
    rings = []
    for s, dz in zip(shrink[:0:-1], np.linspace(pad, 0, len(shrink) - 1, endpoint=False)):
        rings.append(build_ring(t0, scale=s, axis_pad=-dz))
    rings += [build_ring(t) for t in ts]
    for s, dz in zip(shrink[1:], np.linspace(0, pad, len(shrink) - 1, endpoint=False) + pad / (len(shrink) - 1)):
        rings.append(build_ring(t1, scale=s, axis_pad=dz))
    return rings


# ------------------------------------------------------------ chair pieces
# dimensions duplicated from _chair.py would be circular; the builders take
# the numbers they need as arguments instead.

def back_cushion(back_x, z0, z1, K=40):
    """Sculpted front cushion: lumbar bulge, curled wings, rounded top."""
    ts = np.linspace(0, 1, 44)

    def ring(t, scale=1.0, axis_pad=0.0):
        z = z0 + 0.010 + t * (z1 - z0 - 0.020) + axis_pad
        w2 = (0.186 + 0.040 * np.sin(np.pi * (t * 0.92 + 0.04))) * scale
        x_off = (0.050 * np.sin(np.pi * min(t * 1.8, 1.0))
                 - 0.090 * max(t - 0.45, 0.0) ** 1.5)
        r = _superellipse(K, 0.036 * scale, w2, 4.5)
        # wings curl forward near the edges
        curl = 0.030 * (np.abs(r[:, 1]) / max(w2, 1e-6)) ** 3.5
        x = back_x + x_off + r[:, 0] + curl
        return np.stack([x, r[:, 1], np.full(K, z)], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.012))


def back_shell(back_x, z0, z1, K=36):
    """Clean plastic rear shell, offset behind the cushion."""
    ts = np.linspace(0, 1, 30)

    def ring(t, scale=1.0, axis_pad=0.0):
        z = z0 + 0.030 + t * (z1 - z0 - 0.055) + axis_pad
        w2 = (0.170 + 0.036 * np.sin(np.pi * (t * 0.92 + 0.04))) * scale
        x_off = (0.050 * np.sin(np.pi * min(t * 1.8, 1.0))
                 - 0.090 * max(t - 0.45, 0.0) ** 1.5)
        r = _superellipse(K, 0.016 * scale, w2, 3.5)
        return np.stack([back_x - 0.038 + x_off + r[:, 0], r[:, 1],
                         np.full(K, z)], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.010))


def seat_cushion(K=40):
    """Plump seat: crowned top, side bolsters, waterfall front edge."""
    ts = np.linspace(0, 1, 40)
    x0, x1 = -0.223, 0.232

    def ring(t, scale=1.0, axis_pad=0.0):
        x = x0 + t * (x1 - x0) + axis_pad + 0.012
        w2 = 0.240 * (1 - 0.10 * (1 - t) ** 2 - 0.06 * t ** 4) * scale
        # cross-section rounds toward the front (waterfall)
        p = 4.6 - 2.2 * max(t - 0.72, 0.0) / 0.28
        h2 = (0.042 - 0.004 * max(t - 0.8, 0.0) / 0.2) * scale
        r = _superellipse(K, w2, h2, max(p, 2.2))
        y, z = r[:, 0], r[:, 1]
        # crown + side bolsters on the top half only
        top = z > 0
        z = z + top * (0.004 * np.cos(y / 0.24 * np.pi / 2)
                       + 0.008 * np.exp(-((np.abs(y) - 0.205) / 0.035) ** 2)
                       * np.sin(np.pi * min(max(t, 0.08), 0.92)))
        return np.stack([np.full(K, x), y, 0.442 + z], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.010))


def headrest_pillow(K=36):
    """Wide flat pillow, rounded everywhere."""
    ts = np.linspace(0, 1, 30)
    W = 0.168

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * W + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 3.2) ** (1 / 3.2)
        r = _superellipse(K, 0.042 * env * scale + 1e-4, 0.090 * env * scale + 1e-4, 2.6)
        return np.stack([r[:, 0], np.full(K, y), r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004))


def arm_pad(K=28):
    """Soft rounded armrest pad, lofted along its length."""
    ts = np.linspace(0, 1, 26)
    L = 0.150

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * L + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 4.0) ** (1 / 4.0)
        r = _superellipse(K, 0.046 * env * scale + 1e-4, 0.017 * env * scale + 1e-4, 3.0)
        return np.stack([np.full(K, x), r[:, 0], r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004))


def star_arm(reach, K=24):
    """One tapered base arm along +x (positioned/rotated by the geom)."""
    ts = np.linspace(0, 1, 16)
    L2 = reach / 2

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * L2 + axis_pad
        w = (0.034 - 0.010 * t) * scale
        h = (0.021 - 0.007 * t) * scale
        arch = 0.004 * np.sin(np.pi * t)
        r = _superellipse(K, w, h, 3.2)
        return np.stack([np.full(K, x), r[:, 0], r[:, 1] + arch], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.75, 0.35), pad=0.008))


def lift_boot(K=28):
    """Accordion bellows around the gas lift."""
    zs = np.linspace(0.125, 0.255, 40)
    rings = []
    for i, z in enumerate(zs):
        t = i / (len(zs) - 1)
        r = 0.0295 + 0.0045 * (0.5 + 0.5 * np.cos(2 * np.pi * 6.0 * t))
        ring = _superellipse(K, r, r, 2.0)
        rings.append(np.stack([ring[:, 0], ring[:, 1], np.full(K, z)], axis=1))
    return _loft(rings)


def wheel_disc(R=0.031, W=0.0080, K=28):
    """Caster wheel: flat disc with rounded rim (lofted along the axle)."""
    ts = np.linspace(0, 1, 14)

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * W + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 6.0) ** (1 / 6.0)
        r = R * env * scale + 2e-4
        ring = _superellipse(K, r, r, 2.0)
        return np.stack([ring[:, 0], np.full(K, y), ring[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.6), pad=0.001))


_BUILDERS = {
    "back_cushion": lambda: back_cushion(-0.215, 0.487, 1.045),
    "back_shell": lambda: back_shell(-0.215, 0.487, 1.045),
    "seat_cushion": seat_cushion,
    "headrest_pillow": headrest_pillow,
    "arm_pad": arm_pad,
    "star_arm": lambda: star_arm(0.215),
    "lift_boot": lift_boot,
    "wheel_disc": wheel_disc,
}


def ensure_meshes(cache_dir: str = MESH_DIR) -> dict:
    """Generate any missing OBJ into ``cache_dir``; return name -> path."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name, build in _BUILDERS.items():
        p = os.path.join(cache_dir, f"{name}.obj")
        if not os.path.exists(p):
            V, F, VT = build()
            _write_obj(p, V, F, VT)
        paths[name] = p
    return paths


def mesh_assets(cache_dir: str = MESH_DIR) -> str:
    """MJCF <asset> inner block declaring every chair mesh."""
    paths = ensure_meshes(cache_dir)
    return "".join(
        f'<mesh name="{name}" file="{p}" smoothnormal="true"/>'
        for name, p in paths.items())
