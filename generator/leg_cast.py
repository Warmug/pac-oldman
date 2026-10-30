"""Plaster long-leg cast, upper thigh to just above the ankle, fitted to the model's leg.

Each piece rides its own bone, so it moves with the animations without the trousers poking through:
- thigh plaster on the thigh bone, a stockinette cuff rolled over its top edge, which sits higher
  on the outside of the thigh than toward the groin,
- the knee: a rounded drum around the knee's hinge axis on the calf bone. The animations bend the
  knee up to 115 degrees; turning about its own axis the drum stays put, so no gap opens in front.
  Behind the knee the thigh and shin plaster overlap instead (no lump in the hollow of the knee),
- shin plaster on the calf bone, ending just above the ankle with the stockinette rolled over the
  edge, so the foot and shoe move freely below it.

Every piece is lofted from convex hulls of the leg's surface in thin slabs along the piece's axis
(points taken from many animation poses, so it fits however the joints bend), smoothed and pushed
out by the cast's thickness.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import ConvexHull

import meshkit as mk

CUFF_V = (0.0, 0.117)        # texture rows of the stockinette cuff
PLASTER_V = (0.125, 1.0)     # texture rows of the plaster wrap


# --- cross-section fields ------------------------------------------------------------------

def _ray_polygon(center, poly, angles):
    """Distance from `center` (inside the convex polygon) to its boundary along each angle."""
    d = np.stack([np.cos(angles), np.sin(angles)], 1)
    p, q = poly, np.roll(poly, -1, axis=0)
    e = q - p
    out = np.full(len(angles), np.inf)
    for k in range(len(p)):
        # center + t d = p + s e
        den = d[:, 0] * (-e[k, 1]) - d[:, 1] * (-e[k, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = p[k] - center
            t = (rel[0] * (-e[k, 1]) - rel[1] * (-e[k, 0])) / den
            s = (d[:, 0] * rel[1] - d[:, 1] * rel[0]) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
        out = np.where(ok, np.minimum(out, t), out)
    return out


def section_field(pts, axis, e1, e2, s_values, n_ang, slab, min_pts=6):
    """For slabs centred on s_values (half-width `slab`) along `axis`: hull centre (in e1/e2) and
    the hull's radius at n_ang angles around it. Slabs without enough points are interpolated."""
    s = pts @ axis
    a, b = pts @ e1, pts @ e2
    angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    C = np.full((len(s_values), 2), np.nan)
    R = np.full((len(s_values), n_ang), np.nan)
    for i, s0 in enumerate(s_values):
        sel = np.abs(s - s0) <= slab
        if sel.sum() < min_pts:
            continue
        p2 = np.stack([a[sel], b[sel]], 1)
        try:
            hull = ConvexHull(p2)
        except Exception:
            continue
        poly = p2[hull.vertices]
        # area centroid of the hull polygon
        x, y = poly[:, 0], poly[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2
        if area < 0.05:                    # a sliver at the very end of a piece: treat as missing
            continue
        c = np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * area)
        r = _ray_polygon(c, poly, angles)
        if not np.isfinite(r).all():
            continue
        C[i] = c
        R[i] = r
    good = ~np.isnan(C[:, 0])
    idx = np.arange(len(s_values))
    for k in range(2):
        C[:, k] = np.interp(idx, idx[good], C[good, k])
    for j in range(n_ang):
        R[:, j] = np.interp(idx, idx[good], R[good, j])
    return C, R, angles


def smooth_field(C, R, s_sigma=1.2, ang_sigma=1.5):
    C = gaussian_filter1d(C, s_sigma, axis=0, mode="nearest")
    R = gaussian_filter1d(R, s_sigma, axis=0, mode="nearest")
    R = gaussian_filter1d(R, ang_sigma, axis=1, mode="wrap")
    return C, R


def _interp_rows(s_values, field, s):
    """Linear interpolation of per-slab rows (field[n_s, ...]) at positions s (any shape)."""
    s = np.clip(s, s_values[0], s_values[-1])
    i = np.clip(np.searchsorted(s_values, s) - 1, 0, len(s_values) - 2)
    t = (s - s_values[i]) / (s_values[i + 1] - s_values[i])
    t = t.reshape(t.shape + (1,) * (field.ndim - 1))
    return field[i] * (1 - t) + field[i + 1] * t


def _ang_interp(R_rows, angles, u):
    """R_rows[..., n_ang] sampled at fractional angle u (0..1 around)."""
    n = len(angles)
    x = (u % 1.0) * n
    j = np.floor(x).astype(int) % n
    t = x - np.floor(x)
    take = lambda jj: np.take_along_axis(R_rows, jj[..., None], axis=-1)[..., 0]
    return take(j) * (1 - t) + take((j + 1) % n) * t


def _grid(P, UV, centre_of, smooth_angle=65):
    """Mesh from a (nu+1, nv+1, 3) point grid with its uvs, faces turned to point away from
    centre_of(points) (the axis point each face wraps around)."""
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    v, uv = P.reshape(-1, 3), UV.reshape(-1, 2)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    keep = np.linalg.norm(fn, axis=1) > 1e-10
    f, fn = f[keep], fn[keep]
    cen = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3
    if np.sum(np.einsum("ij,ij->i", fn, cen - centre_of(cen))) < 0:
        f = f[:, ::-1]
    return mk.Mesh(v, f, uv).smooth(smooth_angle)


# --- pieces ------------------------------------------------------------------------------------

def edge(back, side_pos, front, side_neg):
    """Smooth edge height around a limb from its values at the back (+y), the +z side, the front and
    the -z side (u = 0, 0.25, 0.5, 0.75), as a two-harmonic curve through those four points."""
    a0 = (back + side_pos + front + side_neg) / 4
    a1, b1 = (back - front) / 2, (side_pos - side_neg) / 2
    a2 = (back + front - side_pos - side_neg) / 4

    def f(u):
        th = 2 * np.pi * np.asarray(u, float)
        return a0 + a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2 * th)
    return f


def plaster_tube(pts, top, bottom, offset=0.75, thickness=0.45, n_ang=48, n_len=60, cuff_top=True,
                 cuff_bottom=False):
    """Plaster around a limb in its bone frame: x runs along the limb, y to the back, z to one side.
    top(u)/bottom(u) give the ends (u: 0..1 around, 0 = +y). An end with a cuff has the stockinette
    rolled over it (inner wall, rounded rim, knitted band); an end without one is plain and open
    (it sits inside the knee piece)."""
    axis, e1, e2 = np.eye(3)
    u_s = np.linspace(0, 1, 97)
    lo, hi = float(np.min(top(u_s))), float(np.max(bottom(u_s)))
    s_values = np.linspace(lo - 0.6, hi + 0.6, max(24, int((hi - lo) / 0.4)))
    C, R, angles = section_field(pts, axis, e1, e2, s_values, n_ang, slab=0.45)
    C, R = smooth_field(C, R)
    R = R + offset

    def point(u, s, dr=0.0):
        c = _interp_rows(s_values, C, s)
        r = _ang_interp(_interp_rows(s_values, R, s), angles, u) + dr
        th = 2 * np.pi * u
        return np.stack([s, c[..., 0] + r * np.cos(th), c[..., 1] + r * np.sin(th)], -1)

    seg = 0.05                                   # profile share of each cuff sub-segment
    T0 = 3 * seg if cuff_top else 0.0
    T1 = 1 - 3 * seg if cuff_bottom else 1.0
    band, depth = 0.85, 0.9

    def fn(u, t):
        st, sb = top(u), bottom(u)
        out = np.zeros(u.shape + (3,))
        if cuff_top:                                             # inner wall -> rim -> band
            k = t < seg
            out[k] = point(u[k], st[k] + depth * (1 - t[k] / seg), -thickness)
            k = (t >= seg) & (t < 2 * seg)
            a = np.pi * (t[k] - seg) / seg
            out[k] = point(u[k], st[k] - np.sin(a) * thickness * 0.55, -thickness / 2 - np.cos(a) * thickness / 2)
            k = (t >= 2 * seg) & (t < T0)
            tt = (t[k] - 2 * seg) / seg
            out[k] = point(u[k], st[k] + band * tt, 0.12 * np.sin(np.pi * np.minimum(tt * 1.4, 1)))
        s0 = st + (band if cuff_top else 0.0)
        s1 = sb - (band if cuff_bottom else 0.0)
        k = (t >= T0) & (t <= T1)
        tt = (t[k] - T0) / (T1 - T0)
        out[k] = point(u[k], s0[k] + (s1[k] - s0[k]) * tt)
        if cuff_bottom:                                          # band -> rim -> inner wall
            k = (t > T1) & (t < T1 + seg)
            tt = (t[k] - T1) / seg
            out[k] = point(u[k], s1[k] + band * tt, 0.12 * np.sin(np.pi * np.minimum((1 - tt) * 1.4, 1)))
            k = (t >= T1 + seg) & (t < T1 + 2 * seg)
            a = np.pi * (t[k] - T1 - seg) / seg
            out[k] = point(u[k], sb[k] + np.sin(a) * thickness * 0.55, -thickness / 2 + np.cos(a) * thickness / 2)
            k = t >= T1 + 2 * seg
            out[k] = point(u[k], sb[k] - depth * (t[k] - T1 - 2 * seg) / seg, -thickness)
        return out

    uu, tt = np.meshgrid(np.linspace(0, 1, n_ang + 1), np.linspace(0, 1, n_len + 1), indexing="ij")
    P = fn(uu, tt)
    cuff_v = lambda x: CUFF_V[0] + (CUFF_V[1] - CUFF_V[0]) * x
    v = PLASTER_V[0] + (PLASTER_V[1] - PLASTER_V[0]) * np.clip((tt - T0) / (T1 - T0), 0, 1)
    if cuff_top:
        v = np.where(tt < T0, cuff_v(tt / T0), v)
    if cuff_bottom:
        v = np.where(tt > T1, cuff_v((1 - tt) / (1 - T1)), v)
    UV = np.stack([uu, v], -1)

    def centre_of(q):
        c = _interp_rows(s_values, C, q[:, 0])
        return np.stack([q[:, 0], c[:, 0], c[:, 1]], 1)

    # a cuffed end is one sheet folding from the outside over the rim to the inner wall, so once the
    # outside faces out, the inner wall faces the leg, as it should
    m = _grid(P, UV, centre_of)
    m.field = dict(kind="tube", s_values=s_values, C=C, R=R, angles=angles,
                   top=(lambda u: top(u) + (band if cuff_top else 0.0)),
                   bottom=(lambda u: bottom(u) - (band if cuff_bottom else 0.0)))
    return m


def knee_drum(z, r, arc, taper=(0.0, 0.12), segments=40, power=2.6, n_prof=33):
    """Rounded barrel around the knee's hinge (the calf frame's z axis through the knee joint),
    at least r[k] in radius at z[k]. Bending the knee turns the shin about this axis, so the barrel
    turns onto itself and never opens a gap. Its sides are domes (a superellipse profile) rather
    than flat discs, so it blends into the thigh and shin plaster instead of reading as a hinge cap.
    Only the front `arc` (fractions of a turn from +x, the shin direction) is built: the gap opens
    in front; behind the knee the thigh and shin plaster overlap. The radius eases off by
    taper=(at the start, at the end) of the arc so the cut edges sink into the plaster."""
    z = np.asarray(z, float)
    r = np.asarray(r, float)
    zc = (z[0] + z[-1]) / 2
    # half-width a little past the required span; then the smallest R0 whose superellipse
    # R0 * (1 - |dz/W|^p)^(1/p) covers every r[k]
    W = (z[-1] - z[0]) / 2 + 0.35
    shape = np.clip(1 - np.abs((z - zc) / W) ** power, 1e-6, 1) ** (1 / power)
    R0 = float(np.max(r / shape))
    zz = zc + W * np.sin(np.linspace(-np.pi / 2, np.pi / 2, n_prof))     # denser near the poles
    rr = R0 * np.clip(1 - np.abs((zz - zc) / W) ** power, 0, 1) ** (1 / power)
    prof = np.stack([rr, zz], 1)
    a0, a1 = arc
    phis = np.linspace(a0, a1, segments + 1) * 2 * np.pi
    frac = phis / (2 * np.pi)
    w0 = np.clip((frac - a0) / 0.05, 0, 1)
    w1 = np.clip((a1 - frac) / 0.05, 0, 1)
    shrink = 1 - taper[0] * (1 - np.sin(w0 * np.pi / 2)) - taper[1] * (1 - np.sin(w1 * np.pi / 2))
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    vt = seg / seg[-1]
    P = np.zeros((segments + 1, len(prof), 3))
    P[:, :, 0] = (prof[None, :, 0] * shrink[:, None]) * np.cos(phis)[:, None]
    P[:, :, 1] = (prof[None, :, 0] * shrink[:, None]) * np.sin(phis)[:, None]
    P[:, :, 2] = prof[None, :, 1]
    UV = np.stack(np.meshgrid(np.linspace(0, 1, segments + 1), PLASTER_V[0] + (PLASTER_V[1] - PLASTER_V[0]) * vt,
                              indexing="ij"), -1)

    def centre_of(q):
        return np.stack([np.zeros(len(q)), np.zeros(len(q)), q[:, 2]], 1)

    m = _grid(P, UV, centre_of)
    m.field = dict(kind="drum", z=prof[:, 1], r=prof[:, 0], arc=arc)
    return m


def margin(m, pts):
    """How far inside each point (in the piece's bone frame) is: positive inside, negative out."""
    fd = m.field
    if fd.get("kind") == "drum":
        rad = np.hypot(pts[:, 0], pts[:, 1])
        phi = (np.arctan2(pts[:, 1], pts[:, 0]) / (2 * np.pi)) % 1
        z, r = fd["z"], fd["r"]
        rz = np.interp(pts[:, 2], z, r, left=-np.inf, right=-np.inf)
        a0, a1 = fd["arc"]
        in_arc = ((phi - a0) % 1) <= ((a1 - a0) % 1)
        return np.where(in_arc, rz - rad, -np.inf)
    sv, C, R, angles = fd["s_values"], fd["C"], fd["R"], fd["angles"]
    s, a, b = pts[:, 0], pts[:, 1], pts[:, 2]
    c = _interp_rows(sv, C, s)
    da, db = a - c[:, 0], b - c[:, 1]
    u = (np.arctan2(db, da) / (2 * np.pi)) % 1
    r = _ang_interp(_interp_rows(sv, R, s), angles, u)
    out = r - np.hypot(da, db)
    return np.where((s >= fd["top"](u)) & (s <= fd["bottom"](u)), out, -np.inf)


# --- texture -------------------------------------------------------------------------------

def paint(path_jpg, path_normal, size=512, seed=11):
    """Off-white plaster wrap (gauze weave, bandage layer edges spiralling down the leg) with a
    knitted stockinette band at the top rows, and a matching normal map."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    h = np.zeros((size, size))
    # gauze weave: two thread directions
    h += 0.35 * np.sin(2 * np.pi * (xx + yy) / 8.0) ** 2
    h += 0.35 * np.sin(2 * np.pi * (xx - yy) / 8.0) ** 2
    # bandage layer edges spiralling around (slanted ridges every ~64 px, tileable in x)
    phase = (yy + xx * 64 / size * 2) / 64.0
    ridge = np.exp(-((phase - np.round(phase)) ** 2) / (2 * 0.035 ** 2))
    h += 0.45 * ridge
    h += gaussian_filter(rng.normal(0, 1, (size, size)), 1.2) * 0.25
    plaster = np.array([236.0, 233.0, 224.0])
    shade = 1 - 0.035 * (h - h.mean()) / h.std()
    blotch = gaussian_filter(rng.normal(0, 1, (size, size)), 18, mode="wrap")
    shade *= 1 - 0.035 * blotch / blotch.std()
    rgb = plaster[None, None, :] * shade[..., None]
    # stockinette cuff: soft white knit (vertical ribs), top rows
    cuff_rows = int(CUFF_V[1] * size) + 1
    rib = 0.5 + 0.5 * np.sin(2 * np.pi * xx[:cuff_rows] / 6.0)
    cuff = np.array([246.0, 246.0, 242.0])[None, None, :] * (0.92 + 0.08 * rib[..., None])
    rgb[:cuff_rows] = cuff
    h[:cuff_rows] = 0.6 * rib
    # gap row between the bands
    gap0, gap1 = cuff_rows, int(PLASTER_V[0] * size)
    rgb[gap0:gap1] = rgb[gap1:gap1 + (gap1 - gap0)]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.SMOOTH).save(path_jpg, quality=92)
    # normal map from the height (tangent space, y up in the image)
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / 2
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / 2
    k = 0.9
    n = np.stack([-gx * k, gy * k, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(path_normal)

