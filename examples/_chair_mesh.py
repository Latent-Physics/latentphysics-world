"""Procedural lofted meshes + textures for the office-chair asset (visuals).

Pure numpy: parts are superellipse cross-sections swept along an axis (a
loft) or tubes swept along a 3D path (piping, brackets), written as OBJ
with UVs. Deterministic — same code, same bytes. Assets are generated on
demand into ~/lpw/assets/chair_meshes_v3 (never committed, same policy as
fetched assets). Collision stays primitive — these meshes carry no mass
and no contype.

Modeled from three reference photos (side / back / front). The back view
drives the rear architecture: a central sculpted spine column sweeping
from the mechanism pod up past the backrest to carry the headrest T-bar,
two butterfly brackets tying the spine to a horizontally ribbed rear
shell, and an hourglass (waisted) backrest silhouette.
"""

from __future__ import annotations

import os

import numpy as np

MESH_DIR = os.path.expanduser("~/lpw/assets/chair_meshes_v3")

BACK_X, BACK_Z0, BACK_Z1 = -0.215, 0.487, 1.045


# ------------------------------------------------------------ mesh plumbing
def _superellipse(K, a, b, p):
    """(K,2) rounded-rectangle ring, CCW, p>=2 (2=ellipse, high=boxy)."""
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / p),
                     b * np.sign(s) * np.abs(s) ** (2 / p)], axis=1)


def _loft(rings, uv=(1.0, 1.0), closed=False):
    """Loft through same-K rings -> (V, F, VT). Open lofts get centroid-fan
    caps; ``closed=True`` wraps the last ring to the first. ``uv`` is the
    texture repeat around/along the sweep (MuJoCo ignores material
    texrepeat for meshes, so tiling lives in the texcoords)."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    K = rings[0].shape[0]
    n = len(rings)
    V = np.concatenate(rings, axis=0)
    VT = np.stack([np.tile(np.linspace(0, uv[0], K, endpoint=False), n),
                   np.repeat(np.linspace(0, uv[1], n), K)], axis=1)
    F = []
    strips = n if closed else n - 1
    for i in range(strips):
        i1 = (i + 1) % n
        for k in range(K):
            k1 = (k + 1) % K
            a, b = i * K + k, i * K + k1
            c, d = i1 * K + k1, i1 * K + k
            F.append((a, b, c))
            F.append((a, c, d))
    if not closed:
        c0 = len(V)
        V = np.concatenate([V, rings[0].mean(0, keepdims=True),
                            rings[-1].mean(0, keepdims=True)], axis=0)
        VT = np.concatenate([VT, [[0.5, 0.0], [0.5, 1.0]]], axis=0)
        for k in range(K):
            k1 = (k + 1) % K
            F.append((c0, k1, k))                                  # bottom cap
            F.append((c0 + 1, (n - 1) * K + k, (n - 1) * K + k1))  # top cap
    return V, np.asarray(F, dtype=int), VT


def _merge(*parts):
    """Concatenate (V, F, VT) parts into one mesh."""
    Vs, Fs, Ts = [], [], []
    off = 0
    for V, F, VT in parts:
        Vs.append(V)
        Fs.append(np.asarray(F) + off)
        Ts.append(VT)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ts)


def _sweep_tube(path, r, K=12, closed=True, uv=(1.0, 1.0)):
    """Circular tube swept along a 3D path. Ring normals point from the
    path centroid outward (stable frames on convex-ish loops — no twist)."""
    P = np.asarray(path, dtype=float)
    n = len(P)
    nxt, prv = np.roll(P, -1, 0), np.roll(P, 1, 0)
    tan = nxt - prv
    if not closed:
        tan[0], tan[-1] = P[1] - P[0], P[-1] - P[-2]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    centroid = P.mean(0)
    th = np.linspace(0, 2 * np.pi, K, endpoint=False)
    rings = []
    for i in range(n):
        radial = P[i] - centroid
        radial -= tan[i] * radial.dot(tan[i])
        nrm = np.linalg.norm(radial)
        if nrm < 1e-9:
            radial = np.array([1.0, 0, 0]) - tan[i] * tan[i][0]
            nrm = np.linalg.norm(radial)
        nvec = radial / nrm
        bvec = np.cross(tan[i], nvec)
        rings.append(P[i] + r * (np.outer(np.cos(th), nvec)
                                 + np.outer(np.sin(th), bvec)))
    return _loft(rings, uv=uv, closed=closed)


def _smooth_loop(P, passes=2):
    """Circular moving average — rounds polyline corners."""
    P = np.asarray(P, dtype=float)
    for _ in range(passes):
        P = (np.roll(P, 1, 0) + P + np.roll(P, -1, 0)) / 3.0
    return P


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


# ---------------------------------------------------- backrest silhouette
def _back_w2(t):
    """Hourglass half-width (front photo): wide shoulders, narrow waist,
    slight flare at the bottom where the back meets the seat."""
    return (0.150
            + 0.072 * np.exp(-((t - 0.84) / 0.16) ** 2)
            + 0.040 * np.exp(-(t / 0.12) ** 2))


def _back_x_off(t):
    return (0.050 * np.sin(np.pi * min(t * 1.8, 1.0))
            - 0.090 * max(t - 0.45, 0.0) ** 1.5)


def _back_t(z):
    return float(np.clip((z - BACK_Z0 - 0.010) / (BACK_Z1 - BACK_Z0 - 0.020), 0, 1))


def _spine_x(z):
    """Spine column centerline: sweeps back from the mechanism pod, hugs
    the ribbed shell with a standoff, carries on up to the headrest T-bar."""
    x_track = BACK_X + _back_x_off(_back_t(min(z, BACK_Z1))) - 0.068
    if z < 0.50:
        s = np.clip((z - 0.375) / 0.125, 0, 1)
        return -0.115 * (1 - s) + x_track * s
    return x_track


# ------------------------------------------------------------ chair pieces
def back_cushion(K=40):
    """Upholstered hourglass backrest cushion with lumbar bulge."""
    ts = np.linspace(0, 1, 44)

    def ring(t, scale=1.0, axis_pad=0.0):
        z = BACK_Z0 + 0.010 + t * (BACK_Z1 - BACK_Z0 - 0.020) + axis_pad
        w2 = _back_w2(t) * scale
        r = _superellipse(K, 0.034 * scale, w2, 4.2)
        curl = 0.022 * (np.abs(r[:, 1]) / max(w2, 1e-6)) ** 3.5
        x = BACK_X + _back_x_off(t) + r[:, 0] + curl
        return np.stack([x, r[:, 1], np.full(K, z)], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.012), uv=(7.0, 8.0))


def back_piping(K=10):
    """Piped seam around the backrest face, just inside the edge."""
    pts = []
    ts = np.linspace(0, 1, 46)
    zline = BACK_Z0 + 0.016 + ts * (BACK_Z1 - BACK_Z0 - 0.032)
    for t, z in zip(ts, zline):
        pts.append((BACK_X + _back_x_off(float(t)) + 0.030,
                    _back_w2(float(t)) - 0.008, z))
    for phi in np.linspace(0, np.pi, 14)[1:-1]:
        w = (_back_w2(1.0) - 0.008) * np.cos(phi)
        pts.append((BACK_X + _back_x_off(1.0) + 0.030, w,
                    zline[-1] + 0.010 * np.sin(phi)))
    for t, z in zip(ts[::-1], zline[::-1]):
        pts.append((BACK_X + _back_x_off(float(t)) + 0.030,
                    -(_back_w2(float(t)) - 0.008), z))
    for phi in np.linspace(np.pi, 2 * np.pi, 14)[1:-1]:
        w = (_back_w2(0.0) - 0.008) * np.cos(phi)
        pts.append((BACK_X + _back_x_off(0.0) + 0.030, w,
                    zline[0] - 0.008 * np.abs(np.sin(phi))))
    return _sweep_tube(_smooth_loop(pts, passes=3), r=0.004, K=K, closed=True)


def back_ribs(n_ribs=7, K=18):
    """Horizontally ribbed rear shell (back photo): rounded slats following
    the hourglass width and the shell curvature, with visible grooves."""
    parts = []
    zs = np.linspace(BACK_Z0 + 0.040, BACK_Z1 - 0.045, n_ribs)
    for zc in zs:
        t = _back_t(float(zc))
        w = _back_w2(t) - 0.010
        x_c = BACK_X + _back_x_off(t) - 0.050
        ys = np.linspace(-w, w, 22)
        rings = []
        for y in ys:
            wrap = 0.042 * (y / w) ** 2          # shell curls forward at edges
            r = _superellipse(K, 0.0135, 0.0295, 2.6)
            rings.append(np.stack([x_c + wrap + r[:, 0],
                                   np.full(K, y),
                                   zc + r[:, 1]], axis=1))
        parts.append(_loft(rings))
    return _merge(*parts)


def spine():
    """Central sculpted spine column: mechanism pod -> up behind the ribbed
    shell -> past the backrest top to the headrest T-bar."""
    zs = np.linspace(0.385, 1.155, 40)
    K = 22
    rings = []
    for z in zs:
        u = (z - zs[0]) / (zs[-1] - zs[0])
        ay = 0.054 - 0.020 * np.sin(np.pi * u) + 0.008 * u ** 3
        ax = 0.024 - 0.006 * u
        r = _superellipse(K, ax, ay, 2.4)
        rings.append(np.stack([_spine_x(float(z)) + r[:, 0],
                               r[:, 1], np.full(K, z)], axis=1))
    col = _loft(rings)
    # headrest T-bar across the spine top
    ys = np.linspace(-0.088, 0.088, 16)
    x_t = _spine_x(1.15) + 0.004
    tbar = _loft([np.stack([x_t + _superellipse(12, 0.013, 0.020, 2.5)[:, 0],
                            np.full(12, y),
                            1.148 + _superellipse(12, 0.013, 0.020, 2.5)[:, 1]],
                           axis=1) for y in ys])
    return _merge(col, tbar)


def _butterfly(z_c, spread_y, spread_z):
    """X-shaped bracket tying the spine to the ribbed shell: four arm tubes
    splaying from a hub pad on the spine to the shell corners."""
    x_hub = _spine_x(z_c) + 0.016
    parts = []
    hub = _loft([np.stack([x_hub - 0.006 + _superellipse(14, 0.010, 0.040, 2.5)[:, 0],
                           np.full(14, y),
                           z_c + _superellipse(14, 0.010, 0.040, 2.5)[:, 1]], axis=1)
                 for y in np.linspace(-0.034, 0.034, 6)])
    parts.append(hub)
    for sy in (1, -1):
        for sz in (1, -1):
            za = z_c + sz * spread_z
            ta = _back_t(za)
            anchor = (BACK_X + _back_x_off(ta) - 0.030,
                      sy * spread_y, za)
            path = [(x_hub, sy * 0.020, z_c + sz * 0.016)]
            mid = (0.5 * (x_hub + anchor[0]) - 0.008,
                   0.5 * (0.020 * sy + anchor[1]),
                   0.5 * (z_c + sz * 0.016 + anchor[2]))
            path += [mid, anchor]
            # densify + smooth the 3-point elbow into a curve
            P = np.asarray(path)
            dense = []
            for i in range(len(P) - 1):
                for s in np.linspace(0, 1, 8, endpoint=False):
                    dense.append(P[i] * (1 - s) + P[i + 1] * s)
            dense.append(P[-1])
            dense = np.asarray(dense)
            for _ in range(2):
                dense[1:-1] = (dense[:-2] + dense[1:-1] + dense[2:]) / 3.0
            parts.append(_sweep_tube(dense, r=0.0130, K=10, closed=False))
    return _merge(*parts)


def butterfly_up():
    return _butterfly(0.865, 0.132, 0.088)


def butterfly_lo():
    return _butterfly(0.640, 0.115, 0.078)


def mech_housing(K=30):
    """Rounded under-seat mechanism pod (replaces the flat box visual)."""
    ts = np.linspace(0, 1, 22)

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * 0.125 + 0.012 + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 2.6) ** (1 / 2.6)
        r = _superellipse(K, 0.098 * env * scale + 1e-4,
                          0.046 * env * scale + 1e-4, 2.6)
        return np.stack([np.full(K, x), r[:, 0], 0.377 + r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004))


def paddle():
    """Adjustment lever: arm tube out of the pod + flat paddle tip."""
    ys = np.linspace(0, -0.150, 18)
    path = [(0.055 + 0.018 * (y / -0.150) ** 2, y,
             0.372 - 0.020 * (y / -0.150)) for y in ys]
    arm = _sweep_tube(path, r=0.0085, K=10, closed=False)
    tip_c = np.array(path[-1])
    ts = np.linspace(0, 1, 10)

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * 0.030 + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 2.4) ** (1 / 2.4)
        r = _superellipse(12, 0.036 * env * scale + 1e-4,
                          0.0075 * env * scale + 1e-4, 2.2)
        return tip_c + np.stack([r[:, 0], np.full(12, y), r[:, 1]], axis=1)

    tip = _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.6), pad=0.002))
    return _merge(arm, tip)


def seat_cushion(K=40):
    """Plump seat: crowned top, rolled-down side edges, waterfall front."""
    ts = np.linspace(0, 1, 40)
    x0, x1 = -0.223, 0.232

    def ring(t, scale=1.0, axis_pad=0.0):
        x = x0 + t * (x1 - x0) + axis_pad + 0.012
        w2 = 0.240 * (1 - 0.10 * (1 - t) ** 2 - 0.06 * t ** 4) * scale
        p = 4.6 - 2.2 * max(t - 0.72, 0.0) / 0.28
        h2 = (0.042 - 0.004 * max(t - 0.8, 0.0) / 0.2) * scale
        r = _superellipse(K, w2, h2, max(p, 2.2))
        y, z = r[:, 0], r[:, 1]
        top = z > 0
        z = z + top * (0.005 * np.cos(y / 0.24 * np.pi / 2)
                       - 0.012 * np.clip((np.abs(y) - 0.205) / 0.035, 0, 1))
        return np.stack([np.full(K, x), y, 0.442 + z], axis=1)

    return _loft(_shrink_caps(ring, ts, pad=0.010), uv=(9.0, 7.0))


def seat_piping(K=10):
    """Piped seam cord around the seat cushion's top shoulder."""
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    p = 3.6
    x = 0.012 + 0.212 * np.sign(c) * np.abs(c) ** (2 / p)
    y = 0.222 * np.sign(s) * np.abs(s) ** (2 / p)
    z = (0.442 + 0.0315
         - 0.010 * np.clip((np.abs(y) - 0.200) / 0.030, 0, 1)     # side rolloff
         - 0.010 * np.clip((x - 0.14) / 0.09, 0, 1))              # waterfall
    path = _smooth_loop(np.stack([x, y, z], axis=1), passes=2)
    return _sweep_tube(path, r=0.0045, K=K, closed=True)


def headrest_pillow(K=36):
    """Wide flat pillow whose ends curl forward to cradle the head."""
    ts = np.linspace(0, 1, 30)
    W = 0.168

    def ring(t, scale=1.0, axis_pad=0.0):
        y = (t * 2 - 1) * W + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 3.2) ** (1 / 3.2)
        curl = 0.020 * (y / W) ** 2
        r = _superellipse(K, 0.042 * env * scale + 1e-4, 0.090 * env * scale + 1e-4, 2.6)
        return np.stack([r[:, 0] + curl, np.full(K, y), r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004), uv=(4.0, 5.0))


def headrest_piping(K=10):
    """Piped seam around the pillow's silhouette (pillow local frame)."""
    W = 0.168
    pts = []
    for phi in np.linspace(0, np.pi, 40)[1:-1]:
        y = W * 0.985 * np.cos(phi)
        env = (1 - min(abs(y / W), 1.0) ** 3.2) ** (1 / 3.2)
        pts.append((0.020 * (y / W) ** 2, y, 0.090 * env))
    for phi in np.linspace(np.pi, 2 * np.pi, 40)[1:-1]:
        y = W * 0.985 * np.cos(phi)
        env = (1 - min(abs(y / W), 1.0) ** 3.2) ** (1 / 3.2)
        pts.append((0.020 * (y / W) ** 2, y, -0.090 * env))
    return _sweep_tube(_smooth_loop(pts, passes=3), r=0.0035, K=K, closed=True)


def arm_pad(K=28):
    """Soft rounded armrest pad, lofted along its length."""
    ts = np.linspace(0, 1, 26)
    L = 0.160

    def ring(t, scale=1.0, axis_pad=0.0):
        x = (t * 2 - 1) * L + axis_pad
        env = (1 - min(abs(t * 2 - 1), 1.0) ** 4.0) ** (1 / 4.0)
        r = _superellipse(K, 0.048 * env * scale + 1e-4, 0.018 * env * scale + 1e-4, 3.0)
        return np.stack([np.full(K, x), r[:, 0], r[:, 1]], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.3), pad=0.004))


def arm_post(K=18):
    """Sculpted armrest column (replaces the bare capsule): elliptical
    taper with a slight forward bow, built along z (the geom quat tilts it)."""
    ts = np.linspace(0, 1, 18)

    def ring(t, scale=1.0, axis_pad=0.0):
        z = (t * 2 - 1) * 0.098 + axis_pad
        ay = (0.032 - 0.010 * t) * scale
        ax = (0.021 - 0.005 * t) * scale
        bow = 0.006 * np.sin(np.pi * t)
        r = _superellipse(K, ax, ay, 2.6)
        return np.stack([r[:, 0] + bow, r[:, 1], np.full(K, z)], axis=1)

    return _loft(_shrink_caps(ring, ts, shrink=(1.0, 0.7, 0.35), pad=0.006))


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
    "back_cushion": back_cushion,
    "back_piping": back_piping,
    "back_ribs": back_ribs,
    "spine": spine,
    "butterfly_up": butterfly_up,
    "butterfly_lo": butterfly_lo,
    "mech_housing": mech_housing,
    "paddle": paddle,
    "seat_cushion": seat_cushion,
    "seat_piping": seat_piping,
    "headrest_pillow": headrest_pillow,
    "headrest_piping": headrest_piping,
    "arm_pad": arm_pad,
    "arm_post": arm_post,
    "star_arm": lambda: star_arm(0.215),
    "lift_boot": lift_boot,
    "wheel_disc": wheel_disc,
}


# ------------------------------------------------------- chair textures
def _to_png(arr):
    a = np.clip(arr, 0.0, 1.0)
    return (np.repeat(a[:, :, None], 3, axis=2) * 255).astype(np.uint8)


def _stitch_fabric(res=256):
    """Woven grain + one dashed stitch channel per tile: quilted seams."""
    rng = np.random.default_rng(11)
    u = np.linspace(0, 2 * np.pi, res, endpoint=False)
    warp = 0.5 + 0.5 * np.sin(24 * u)[:, None]
    weft = 0.5 + 0.5 * np.sin(24 * u)[None, :]
    base = 0.82 + 0.14 * np.maximum(warp, weft)
    base = base + 0.025 * rng.standard_normal((res, res))
    col = res // 2
    base[:, col - 2:col + 3] *= 0.90                        # seam groove
    dash = ((np.arange(res) // 7) % 2 == 0).astype(float)   # stitch dashes
    base[:, col - 1:col + 2] *= (1.0 - 0.22 * dash)[:, None]
    return _to_png(base)


_TEXTURES = {"chair_stitch": _stitch_fabric}


def ensure_meshes(cache_dir: str = MESH_DIR) -> dict:
    """Generate any missing OBJ/PNG into ``cache_dir``; return name -> path."""
    import imageio.v2 as imageio

    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name, build in _BUILDERS.items():
        p = os.path.join(cache_dir, f"{name}.obj")
        if not os.path.exists(p):
            V, F, VT = build()
            _write_obj(p, V, F, VT)
        paths[name] = p
    for name, gen in _TEXTURES.items():
        p = os.path.join(cache_dir, f"{name}.png")
        if not os.path.exists(p):
            imageio.imwrite(p, gen())
        paths[name] = p
    return paths


def mesh_assets(cache_dir: str = MESH_DIR) -> str:
    """MJCF <asset> inner block: chair meshes + chair-specific materials."""
    paths = ensure_meshes(cache_dir)
    parts = [f'<mesh name="{n}" file="{paths[n]}" smoothnormal="true"/>'
             for n in _BUILDERS]
    parts.append(f'<texture name="chair_stitch" type="2d" file="{paths["chair_stitch"]}"/>')
    parts.append('<material name="mat_fabric_stitch" texture="chair_stitch" '
                 'reflectance="0.05"/>')
    return "".join(parts)
