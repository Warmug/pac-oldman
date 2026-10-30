"""Plaster short-leg cast with a post-op cast shoe, fitted to the model's lower leg and shoe.

Three pieces, each on its own bone, so it moves with the walk cycle without the shoe or trouser
leg poking through and without the foot sinking into the floor:
- plaster tube on the calf bone, from below the knee down over the ankle, a stockinette cuff
  rolled over its top edge, cut lower behind the knee so the knee can still bend,
- cast shoe on the foot bone (dark fabric upper, two Velcro straps, rubber sole) over the foot
  and the bottom of the plaster,
- the shoe's toe on the toe bone, tucked into it at the ball of the foot.
The ankle and the ball of the foot bend up to 30 degrees in the walk/run animations; a shoe
flexing there reads naturally, a rigid plaster foot would not.

Every piece is lofted from convex hulls of the leg/shoe surface in thin slabs along the piece's
axis (points taken from many animation poses, so it fits however the joints bend), smoothed and
pushed out by the cast's thickness.
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
        if area < 0.05:                    # a sliver (the very tip of a shoe): treat as missing
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


# --- pieces ------------------------------------------------------------------------------------

def shin_tube(pts, s_front, s_back, s_bottom, offset=0.75, thickness=0.45, n_ang=48, n_len=40):
    """Calf-bone frame: x runs down the shin, y to the back, z to his outer side.
    The top edge runs from s_front (in front) to s_back (behind the knee)."""
    axis, e1, e2 = np.eye(3)
    s_values = np.linspace(min(s_front, s_back) - 0.5, s_bottom + 0.5, 46)
    C, R, angles = section_field(pts, axis, e1, e2, s_values, n_ang, slab=0.45)
    C, R = smooth_field(C, R)
    R = R + offset

    def top(u):          # angle 0 points to the back (+y)
        return s_front + (s_back - s_front) * (1 + np.cos(2 * np.pi * u)) / 2

    def point(u, s, dr=0.0):
        c = _interp_rows(s_values, C, s)
        r = _ang_interp(_interp_rows(s_values, R, s), angles, u) + dr
        th = 2 * np.pi * u
        return np.stack([s, c[..., 0] + r * np.cos(th), c[..., 1] + r * np.sin(th)], -1)

    # profile per angle: inner wall (0..0.08) -> rolled rim (0.08..0.16) -> cuff band (0.16..0.24)
    # -> plaster down to the bottom (0.24..1)
    def fn(u, t):
        st = top(u)
        out = np.zeros(u.shape + (3,))
        k = t < 0.08                                  # inner wall, 0.9 deep
        tt = t[k] / 0.08
        out[k] = point(u[k], st[k] + 0.9 * (1 - tt), -thickness)
        k = (t >= 0.08) & (t < 0.16)                  # half-round over the rim
        tt = (t[k] - 0.08) / 0.08
        a = np.pi * tt
        out[k] = point(u[k], st[k] - np.sin(a) * thickness * 0.55, -thickness / 2 - np.cos(a) * thickness / 2)
        k = (t >= 0.16) & (t < 0.24)                  # stockinette band, slightly proud
        tt = (t[k] - 0.16) / 0.08
        out[k] = point(u[k], st[k] + 0.85 * tt, 0.12 * np.sin(np.pi * np.minimum(tt * 1.4, 1)))
        k = t >= 0.24
        tt = (t[k] - 0.24) / 0.76
        out[k] = point(u[k], st[k] + 0.85 + (s_bottom - st[k] - 0.85) * tt)
        return out

    uu, tt = np.meshgrid(np.linspace(0, 1, n_ang + 1), np.linspace(0, 1, n_len + 1), indexing="ij")
    P = fn(uu, tt)
    v = np.where(tt < 0.24, CUFF_V[0] + (CUFF_V[1] - CUFF_V[0]) * tt / 0.24,
                 PLASTER_V[0] + (PLASTER_V[1] - PLASTER_V[0]) * (tt - 0.24) / 0.76)
    UV = np.stack([uu, v], -1)

    def centre_of(q):
        c = _interp_rows(s_values, C, q[:, 0])
        return np.stack([q[:, 0], c[:, 0], c[:, 1]], 1)

    # one sheet folds from the inner wall over the rim to the outside, so once the outside faces
    # out, the inner wall faces the leg, as it should
    m = _grid(P, UV, centre_of)
    # the outer plaster surface starts 0.85 below the (slanted) top edge; test below the lowest point
    m.field = dict(s_values=s_values, C=C, R=R, angles=angles, s_top=max(s_front, s_back) + 0.85,
                   s_bottom=s_bottom)
    return m


def closed_loft(pts, axis, e1, e2, s0, s1, offset, round_back, round_front, n_ang=48, n_len=36,
                bottom_offset=None, bottom_dir=None, uv_along=None):
    """Closed blob lofted along `axis` from s0 to s1 with rounded ends; e1/e2 span the sections
    (angle 0 = e1). bottom_offset: different push-out on the side facing bottom_dir (the sole).
    uv_along=(u0, u1): texture u runs along the axis over that range and v around the section
    (shoe layout); default: u around, v along the plaster rows."""
    s_values = np.linspace(s0, s1, 40)
    C, R, angles = section_field(pts, axis, e1, e2, s_values, n_ang, slab=0.45)
    C, R = smooth_field(C, R)
    off = np.full(n_ang, offset)
    if bottom_offset is not None:
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        down = np.array([bottom_dir @ e1, bottom_dir @ e2])
        down /= np.linalg.norm(down)
        w = np.clip(dirs @ down, 0, 1) ** 2
        off = offset * (1 - w) + bottom_offset * w
    R = R + off[None, :]
    span = s1 - s0

    def fn(u, t):
        s = s0 + span * t
        c = _interp_rows(s_values, C, s)
        r = _ang_interp(_interp_rows(s_values, R, s), angles, u)
        # round the ends: radius falls to 0 over round_back / round_front
        fb = np.clip((s - s0) / round_back, 0, 1)
        ff = np.clip((s1 - s) / round_front, 0, 1)
        r = r * np.sqrt(1 - (1 - fb) ** 2) * np.sqrt(1 - (1 - ff) ** 2)
        th = 2 * np.pi * u
        a = c[..., 0] + r * np.cos(th)
        b = c[..., 1] + r * np.sin(th)
        return s[..., None] * axis + a[..., None] * e1 + b[..., None] * e2

    uu, tt = np.meshgrid(np.linspace(0, 1, n_ang + 1), np.linspace(0, 1, n_len + 1), indexing="ij")
    P = fn(uu, tt)
    if uv_along is None:
        UV = np.stack([uu, PLASTER_V[0] + (PLASTER_V[1] - PLASTER_V[0]) * tt], -1)
    else:
        UV = np.stack([uv_along[0] + (uv_along[1] - uv_along[0]) * tt, uu], -1)

    def centre_of(q):
        sq = q @ axis
        c = _interp_rows(s_values, C, sq)
        return sq[:, None] * axis + c[:, :1] * e1 + c[:, 1:] * e2

    m = _grid(P, UV, centre_of)
    m.field = dict(s_values=s_values, C=C, R=R, angles=angles, axis=axis, e1=e1, e2=e2,
                   s0=s0, s1=s1, round_back=round_back, round_front=round_front)
    return m


def margin(m, pts):
    """How far inside each point (in the piece's bone frame) is: positive inside, negative out.
    Measured radially in the piece's sections; the tube is open, so points beyond its ends are out."""
    fd = m.field
    if "axis" not in fd:                                   # shin tube: x is the axis, y/z the section
        sv, C, R, angles = fd["s_values"], fd["C"], fd["R"], fd["angles"]
        s, a, b = pts[:, 0], pts[:, 1], pts[:, 2]
        lo, hi = fd["s_top"], fd["s_bottom"]
    else:
        sv, C, R, angles = fd["s_values"], fd["C"], fd["R"], fd["angles"]
        s, a, b = pts @ fd["axis"], pts @ fd["e1"], pts @ fd["e2"]
        lo, hi = fd["s0"], fd["s1"]
    c = _interp_rows(sv, C, s)
    da, db = a - c[:, 0], b - c[:, 1]
    u = (np.arctan2(db, da) / (2 * np.pi)) % 1
    r = _ang_interp(_interp_rows(sv, R, s), angles, u)
    if "axis" in fd:
        fb = np.clip((s - lo) / fd["round_back"], 0, 1)
        ff = np.clip((hi - s) / fd["round_front"], 0, 1)
        r = r * np.sqrt(1 - (1 - fb) ** 2) * np.sqrt(1 - (1 - ff) ** 2)
    out = r - np.hypot(da, db)
    return np.where((s >= lo) & (s <= hi), out, -np.inf)


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


# shoe texture layout: u runs heel -> toe, v runs around the foot starting on top (angle 0 = up),
# so the sole is the band around v = 0.5 and straps are bands of u
SHOE_FOOT_U = (0.0, 0.70)     # the shoe body
SHOE_TOE_U = (0.70, 1.0)      # its toe piece
STRAPS_U = ((0.30, 0.40), (0.50, 0.60))
SOLE_V = (0.36, 0.64)


def paint_shoe(path, size=512, seed=5):
    """Post-op cast shoe: navy mesh fabric, two grey Velcro straps over the instep, black rubber sole."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    u, v = xx / size, yy / size
    fabric = np.array([38.0, 46.0, 66.0])
    weave = 0.5 + 0.5 * np.sin(2 * np.pi * xx / 5.0) * np.sin(2 * np.pi * yy / 5.0)
    rgb = fabric[None, None, :] * (0.88 + 0.16 * weave[..., None])
    for a, b in STRAPS_U:
        k = (u >= a) & (u < b)
        hook = 0.9 + 0.1 * (rng.random((size, size)) > 0.5)
        rgb[k] = (np.array([118.0, 122.0, 128.0])[None, :] * hook[k][:, None])
        edge = k & ((u - a < 0.008) | (b - u < 0.008))
        rgb[edge] = [70, 72, 78]
    k = (v >= SOLE_V[0]) & (v < SOLE_V[1])
    rgb[k] = np.array([24.0, 24.0, 26.0]) * (0.9 + 0.2 * rng.random((int(k.sum()), 1)))
    for vv in SOLE_V:                          # welt line where the sole meets the fabric
        k = np.abs(v - vv) < 0.006
        rgb[k] = [60, 60, 64]
    rgb += gaussian_filter(rng.normal(0, 1, (size, size, 1)), (1, 1, 0)) * 3
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, quality=92)
