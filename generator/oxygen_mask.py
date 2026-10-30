"""Clear oxygen mask with green elastic strap, nose clip and elbow outlet, built on a face.

Everything is built in the head bone's frame from a `Face` (landmarks + a ray caster
against the head mesh), so it sits on that playermodel's face with no offsets in pac.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import meshkit as mk

# o2_strap.jpg atlas (256x256): elastic band on the left, aluminium nose clip and lugs on the right
STRAP_ELASTIC = (0.02, 0.02, 0.48, 0.98)
STRAP_ALU = (0.54, 0.02, 0.98, 0.48)
STRAP_LUG = (0.54, 0.54, 0.98, 0.98)

# o2_mask.png (512x512): u runs around the rim, v from the rim (top) to the dome apex (bottom).
# The outlet and its swivel use the bottom strip.
MASK_SHELL = (0.0, 0.0, 1.0, 0.86)
MASK_PORT = (0.0, 0.88, 1.0, 0.99)
HOLE_U = (0.25, 0.75)       # left and right side of the dome
HOLE_V = 0.52


def paint_tube(path):
    """64x8 strip for the beam: u spans the tube's width, v runs along it."""
    w, h = 64, 8
    x = (np.arange(w) + 0.5) / w * 2 - 1
    edge = np.clip(np.abs(x), 0, 1)
    alpha = 0.35 + 0.6 * edge ** 6 + 0.25 * np.exp(-((x + 0.35) / 0.12) ** 2)
    shade = 205 + 45 * np.exp(-((x + 0.35) / 0.12) ** 2) - 40 * edge ** 4
    rgba = np.zeros((h, w, 4))
    rgba[..., 0] = shade - 6
    rgba[..., 1] = shade
    rgba[..., 2] = shade + 4
    rgba[..., 3] = np.clip(alpha, 0, 1) * 255
    Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA").save(path)


def paint_mask(path):
    size = 512
    rng = np.random.default_rng(3)
    rgb = np.ones((size, size, 3)) * np.array([226.0, 236.0, 240.0])
    rgb += rng.normal(0, 3, (size, size, 1))
    alpha = np.ones((size, size)) * 0.9
    # frosted band near the rim, clearer toward the front of the dome
    v = (np.arange(size) + 0.5) / size
    alpha *= (0.75 + 0.25 * np.clip(1 - v / 0.25, 0, 1))[:, None]
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    # exhalation holes: a ring of small round holes on each side (fully see-through)
    for hu in HOLE_U:
        cx, cy = hu * size, HOLE_V * MASK_SHELL[3] * size
        d.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=(205, 215, 222, 240))
        for k in range(7):
            a = 2 * np.pi * k / 7
            r = 0 if k == 0 else 16
            hx, hy = cx + r * np.cos(a), cy + r * np.sin(a)
            d.ellipse((hx - 5, hy - 5, hx + 5, hy + 5), fill=(0, 0, 0, 0))
    # outlet: slightly frosted plastic
    y0 = int(MASK_PORT[1] * size)
    d.rectangle((0, y0, size, size), fill=(215, 225, 230, 215))
    img.save(path)


def paint_strap(path):
    size = 256
    rng = np.random.default_rng(5)
    img = np.zeros((size, size, 3))
    green = np.array([40.0, 168.0, 120.0])
    x0, y0, x1, y1 = (int(c * size) for c in STRAP_ELASTIC)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weave = 10 * np.sin(yy * 1.9) * np.sin(xx * 1.9) + 8 * np.sin(xx * 0.9)
    img[y0:y1, x0:x1] = green + weave[..., None] + rng.normal(0, 4, (y1 - y0, x1 - x0, 1))
    x0, y0, x1, y1 = (int(c * size) for c in STRAP_ALU)
    img[y0:y1, x0:x1] = np.array([188.0, 192.0, 198.0]) + 25 * np.sin(np.linspace(0, 3, y1 - y0))[:, None, None]
    x0, y0, x1, y1 = (int(c * size) for c in STRAP_LUG)
    img[y0:y1, x0:x1] = np.array([210.0, 222.0, 226.0]) + rng.normal(0, 3, (y1 - y0, x1 - x0, 1))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.SMOOTH).save(path, quality=92)


# --- geometry ---------------------------------------------------------------------

def catmull_rom_closed(points, per_segment=8):
    p = np.asarray(points, float)
    n = len(p)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        for t in np.linspace(0, 1, per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


class Face:
    """Landmarks in the head bone frame plus a surface sampler.

    frame: F forward (out of the face), U up, L to the character's left; origin anywhere.
    surface(u, v) -> point on the skin seen from the front at lateral u, height v (face coords)."""

    def __init__(self, origin, F, U, L, surface, normal, landmarks):
        self.o, self.F, self.U, self.L = map(np.asarray, (origin, F, U, L))
        self.surface = surface
        self.normal = normal
        self.lm = landmarks

    def to_head(self, u, v, d):
        return self.o + self.L * u + self.U * v + self.F * d

    def coords(self, p):
        q = np.asarray(p) - self.o
        return q @ self.L, q @ self.U, q @ self.F


def rim_contour(face, m):
    """Closed rim loop (face coords u, v) around nose and mouth, as control points."""
    vt, vm, vb = m["v_top"], m["v_mouth"], m["v_bottom"]
    half = [(0.0, vt), (0.62, vt - 0.3), (1.25, vt - 1.15), (m["w_cheek"], vm + 0.45),
            (m["w_cheek"] + 0.05, vm - 0.45), (1.55, vb + 0.45), (0.75, vb + 0.06)]
    right = [(-u, v) for u, v in reversed(half[1:])]
    pts = half + [(0.0, vb)] + right
    return catmull_rom_closed(pts, 10)


def build_mask(face, m):
    """Returns (shell mesh, strap+clip mesh, outlet position, outlet direction) in head frame."""
    loop2d = rim_contour(face, m)
    rim = np.array([face.surface(u, v) + face.normal(u, v) * m["rim_lift"] for u, v in loop2d])
    rim_d = (rim - face.o) @ face.F
    apex = face.to_head(0.0, m["v_apex"], m["d_apex"])

    def shell_fn(uu, ss, offset=0.0):
        idx = uu * len(rim)
        i0 = np.floor(idx).astype(int) % len(rim)
        i1 = (i0 + 1) % len(rim)
        t = (idx - np.floor(idx))[..., None]
        r = rim[i0] * (1 - t) + rim[i1] * t
        rd = rim_d[i0] * (1 - t[..., 0]) + rim_d[i1] * t[..., 0]
        c = r + face.F * (m["d_apex"] - rd)[..., None] * m["dome_round"]
        s = ss[..., None]
        p = (1 - s) ** 2 * r + 2 * (1 - s) * s * c + s * s * apex
        return p

    nu, nv = len(rim), 22
    outer = mk.surface(lambda u, s: shell_fn(u, s), nu, nv, MASK_SHELL, flip=False, smooth_angle=70)
    # orientation: make sure outer faces point away from the face
    centre = face.to_head(0.0, m["v_apex"], m["d_apex"] - 1.0)
    if np.mean(np.einsum("ij,ij->i", outer.n, outer.v - centre)) < 0:
        outer = mk.surface(lambda u, s: shell_fn(u, s), nu, nv, MASK_SHELL, flip=True, smooth_angle=70)
    inner = outer.copy()
    inner.v = inner.v - inner.n * m["thickness"]
    inner = inner.flipped()
    # rim edge strip joining outer and inner along s = 0
    edge_out = np.array([shell_fn(np.array(u), np.array(0.0)) for u in np.linspace(0, 1, nu + 1)])
    nrm0 = [outer.n[np.argmin(np.linalg.norm(outer.v - p, axis=1))] for p in edge_out]
    edge_in = edge_out - np.array(nrm0) * m["thickness"]
    edge = _strip(edge_out, edge_in, MASK_SHELL)
    if np.mean(np.einsum("ij,ij->i", edge.n, edge.v - centre)) > 0:
        edge = edge.flipped()

    # elbow outlet at the bottom front of the dome, pointing down and slightly forward
    port_base = shell_fn(np.array(m["port_u"]), np.array(m["port_s"]))
    d = face.F * m["port_forward"] - face.U * (1 - m["port_forward"] ** 2) ** 0.5
    d = d / np.linalg.norm(d)
    p1 = port_base - d * 0.3
    p2 = port_base + d * 0.75
    p3 = p2 + d * 0.65
    port = mk.concat([
        mk.tube([p1, p2], 0.36, 16, MASK_PORT, up_hint=face.F),
        mk.tube([p2 - d * 0.05, p2 + d * 0.2], 0.42, 16, MASK_PORT, up_hint=face.F),
        mk.tube([p2 + d * 0.2, p3], 0.2, 12, MASK_PORT, up_hint=face.F),
    ])
    shell = mk.concat([outer, inner, edge, port])

    # nose clip: aluminium strip just inside the rim across the bridge of the nose
    clip_u = np.linspace(-m["clip_span"], m["clip_span"], 15)
    clip_pts = np.array([shell_fn(np.array(u % 1.0), np.array(0.09)) for u in clip_u])
    clip = _band_on_surface(clip_pts, lambda p: outer.n[np.argmin(np.linalg.norm(outer.v - p, axis=1))],
                            0.24, 0.035, STRAP_ALU, lift=0.03)

    # elastic strap: from one side of the dome around the back of the head to the other
    lugs, ends = [], []
    for side in (1, -1):
        u_side = 0.25 if side > 0 else 0.75
        p = shell_fn(np.array(u_side), np.array(0.06))
        n = outer.n[np.argmin(np.linalg.norm(outer.v - p, axis=1))]
        ends.append(p + n * 0.05)
        lugs.append(mk.tube([p - n * 0.02, p + n * 0.16], 0.16, 10, STRAP_LUG, up_hint=face.U))
    path = strap_path(face, m, ends[0], ends[1])
    strap = _band_on_surface(path, lambda p: m["head_normal"](p), m["strap_width"], 0.055, STRAP_ELASTIC, lift=0.0)
    strap_mesh = mk.concat([strap, clip] + lugs)
    return shell, strap_mesh, p3, d


def strap_path(face, m, start, end):
    """Around the back of the head below the ears, hugging the skin."""
    centre = face.to_head(0.0, m["strap_axis_v"], m["strap_axis_d"])
    a0 = np.arctan2((start - centre) @ face.L, (start - centre) @ face.F)
    a1 = np.arctan2((end - centre) @ face.L, (end - centre) @ face.F)
    if a1 < a0:
        a1 += 2 * np.pi
    pts = [start]
    for a in np.linspace(a0, a1, 64)[1:-1]:
        back = (1 - np.cos(a)) / 2          # 0 at the front, 1 at the back of the head
        v = m["strap_v_front"] * (1 - back) + m["strap_v_back"] * back
        if abs(np.sin(a)) > 0.6:
            v = min(v, m["strap_v_ear"])
        direction = face.F * np.cos(a) + face.L * np.sin(a)
        origin = centre + face.U * (v - m["strap_axis_v"])
        hit = m["cast_out"](origin, direction)
        pts.append(hit + m["head_normal"](hit) * m["strap_lift"])
    pts.append(end)
    pts = np.array(pts)
    # light smoothing so the band does not follow every skin bump
    sm = pts.copy()
    for _ in range(3):
        sm[1:-1] = 0.25 * sm[:-2] + 0.5 * sm[1:-1] + 0.25 * sm[2:]
    return sm


def _strip(a, b, rect):
    n = len(a)
    v = np.vstack([a, b])
    s = np.linspace(0, 1, n)
    uv = np.vstack([mk.map_uv(s, np.zeros(n), rect), mk.map_uv(s, np.ones(n) * 0.05, rect)])
    f = []
    for i in range(n - 1):
        f += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    return mk.Mesh(v, np.array(f), uv).smooth(60)


def _band_on_surface(path, normal_at, width, thick, rect, lift=0.0):
    """Flat rectangular band lying on a surface along `path`."""
    p = np.asarray(path, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.array([normal_at(q) for q in p])
    n = n - t * np.einsum("ij,ij->i", n, t)[:, None]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    s = np.cross(t, n)
    base = p + n * (lift + thick / 2)
    corners = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]   # (side, normal) going around the section
    rings = [base + s * (cs * width / 2) + n * (cn * thick / 2) for cs, cn in corners]
    k = len(p)
    v = np.vstack(rings + [rings[0]])
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    su = seg / max(seg[-1], 1e-9)
    uv = np.vstack([mk.map_uv(su, np.full(k, j / 4), rect) for j in range(5)])
    f = []
    for j in range(4):
        a0, b0 = j * k, (j + 1) * k
        for i in range(k - 1):
            f += [[a0 + i, b0 + i, b0 + i + 1], [a0 + i, b0 + i + 1, a0 + i + 1]]
    band = mk.Mesh(v, np.array(f), uv).smooth(40)
    # outward check: the top face should point along n
    if np.mean(np.einsum("ij,ij->i", band.n[:k], n)) < 0:
        band = band.flipped()
    for end, sign in ((0, -1), (k - 1, 1)):
        cap = np.array([r[end] for r in rings])
        c = mk.Mesh(cap, np.array([[0, 1, 2], [0, 2, 3]]), mk.map_uv(np.array([0, 1, 1, 0.0]), np.array([0, 0, 1, 1.0]), rect))
        c.smooth(10)
        if np.mean(c.n @ (t[end] * sign)) < 0:
            c = c.flipped()
        band = mk.concat([band, c])
    return band
