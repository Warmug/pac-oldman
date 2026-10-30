"""Procedural meshes for PAC3 OBJ props.

Conventions
- Coordinates are whatever frame the part is placed in (Source: x forward, y left, z up).
- Faces are counter-clockwise seen from the front, like any OBJ; pac reverses the
  winding on load, which is what Source expects.
- UV rects are given in image pixels-space fractions with y going DOWN the image
  (x0, y0, x1, y1). They are written as OBJ v = 1 - y because pac flips v back.
- pac wraps every uv with `% 1`, so no coordinate may land exactly on 0 or 1.
"""
import numpy as np

UV_EPS = 0.0008
MAX_TRIS = 21845  # pac3 OBJ limit per mesh (65535 vertices / 3)


class Mesh:
    def __init__(self, v=None, f=None, uv=None, n=None):
        self.v = np.zeros((0, 3)) if v is None else np.asarray(v, float)
        self.f = np.zeros((0, 3), int) if f is None else np.asarray(f, int)
        self.uv = np.zeros((len(self.v), 2)) if uv is None else np.asarray(uv, float)
        self.n = None if n is None else np.asarray(n, float)

    @property
    def tris(self):
        return len(self.f)

    def copy(self):
        return Mesh(self.v.copy(), self.f.copy(), self.uv.copy(), None if self.n is None else self.n.copy())

    def transformed(self, m):
        """Apply a 4x4 matrix (rotation part must be orthonormal or uniformly scaled)."""
        m = np.asarray(m, float)
        out = self.copy()
        out.v = self.v @ m[:3, :3].T + m[:3, 3]
        if self.n is not None:
            nm = np.linalg.inv(m[:3, :3]).T
            n = self.n @ nm.T
            out.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if np.linalg.det(m[:3, :3]) < 0:
            out.f = out.f[:, ::-1].copy()
        return out

    def flipped(self):
        out = self.copy()
        out.f = out.f[:, ::-1].copy()
        if out.n is not None:
            out.n = -out.n
        return out

    def smooth(self, angle=45.0):
        """Per-corner normals: average faces that share a position and lie within `angle` degrees."""
        v, f = self.v, self.f
        if len(f) == 0:
            self.n = np.zeros_like(v)
            return self
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        area = np.linalg.norm(fn, axis=1)
        fnu = fn / np.maximum(area[:, None], 1e-12)

        keys = np.round(v / 1e-4).astype(np.int64)
        _, pos_id = np.unique(keys, axis=0, return_inverse=True)
        pos_id = pos_id.ravel()
        faces_at = {}
        for fi, tri in enumerate(f):
            for vi in tri:
                faces_at.setdefault(pos_id[vi], []).append(fi)

        cos_t = np.cos(np.radians(angle))
        new_v, new_uv, new_n, index = [], [], [], {}
        new_f = np.zeros_like(f)
        for fi, tri in enumerate(f):
            for k, vi in enumerate(tri):
                nb = faces_at[pos_id[vi]]
                dots = fnu[nb] @ fnu[fi]
                sel = np.asarray(nb)[dots > cos_t]
                n = fn[sel].sum(axis=0)
                ln = np.linalg.norm(n)
                n = n / ln if ln > 1e-12 else fnu[fi]
                key = (vi, tuple(np.round(n, 4)))
                idx = index.get(key)
                if idx is None:
                    idx = index[key] = len(new_v)
                    new_v.append(v[vi])
                    new_uv.append(self.uv[vi])
                    new_n.append(n)
                new_f[fi, k] = idx
        self.v, self.uv, self.n, self.f = np.array(new_v), np.array(new_uv), np.array(new_n), new_f
        return self

    def signed_volume(self):
        a, b, c = self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def bounds(self):
        return self.v.min(axis=0), self.v.max(axis=0)

    def write_obj(self, path, comment=""):
        if self.n is None:
            self.smooth()
        if self.tris > MAX_TRIS:
            raise ValueError(f"{path}: {self.tris} triangles > pac limit {MAX_TRIS}")
        uv = np.clip(self.uv, UV_EPS, 1 - UV_EPS)
        lines = [f"# {line}" for line in comment.splitlines()] if comment else []
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.v]
        lines += [f"vt {u:.5f} {1 - t:.5f}" for u, t in uv]
        lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.n]
        lines += ["f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}".format(*(t + 1)) for t in self.f]
        with open(path, "w", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")


def concat(meshes):
    meshes = [m for m in meshes if m is not None and len(m.v)]
    out = Mesh()
    vs, fs, uvs, ns, off = [], [], [], [], 0
    have_n = all(m.n is not None for m in meshes)
    for m in meshes:
        vs.append(m.v)
        fs.append(m.f + off)
        uvs.append(m.uv)
        if have_n:
            ns.append(m.n)
        off += len(m.v)
    out.v, out.f, out.uv = np.vstack(vs), np.vstack(fs), np.vstack(uvs)
    out.n = np.vstack(ns) if have_n else None
    return out


def map_uv(s, t, rect):
    """s,t in 0..1 -> atlas rect (x0, y0, x1, y1) with y down the image."""
    x0, y0, x1, y1 = rect
    return np.stack([x0 + s * (x1 - x0), y0 + t * (y1 - y0)], axis=-1)


def surface(fn, nu, nv, rect=(0, 0, 1, 1), flip=False, smooth_angle=45.0):
    """Parametric grid. fn(u, v) takes broadcast arrays in 0..1 and returns (..., 3) points.
    Faces are CCW when looking along -(dP/du x dP/dv); pass flip=True to turn them around."""
    u = np.linspace(0, 1, nu + 1)
    t = np.linspace(0, 1, nv + 1)
    uu, tt = np.meshgrid(u, t, indexing="ij")
    pts = np.asarray(fn(uu, tt), float).reshape(nu + 1, nv + 1, 3)
    v = pts.reshape(-1, 3)
    uv = map_uv(uu.ravel(), tt.ravel(), rect)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    if flip:
        f = f[:, ::-1]
    # drop degenerate triangles (poles of lathes, collapsed caps)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    keep = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) > 1e-10
    m = Mesh(v, f[keep], uv)
    return m.smooth(smooth_angle)


def lathe(profile, segments, rect=(0, 0, 1, 1), smooth_angle=40.0, arc=(0.0, 1.0)):
    """Revolve a (radius, z) polyline around +z. Profile should run bottom to top for
    outward-facing normals. v follows the profile arclength, u goes around."""
    prof = np.asarray(profile, float)
    seg = np.linalg.norm(np.diff(prof, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s / s[-1] if s[-1] > 0 else s
    # rings sit exactly on the profile points so its corners stay sharp after smoothing
    return _lathe_exact(prof, s, segments, rect, arc[0], arc[1]).smooth(smooth_angle)


def _lathe_exact(prof, s, segments, rect, a0, a1):
    nu, nv = segments, len(prof) - 1
    u = np.linspace(0, 1, nu + 1)
    ang = 2 * np.pi * (a0 + (a1 - a0) * u)
    r, z = prof[:, 0], prof[:, 1]
    x = np.cos(ang)[:, None] * r[None, :]
    y = np.sin(ang)[:, None] * r[None, :]
    zz = np.broadcast_to(z[None, :], x.shape)
    v = np.stack([x, y, zz], -1).reshape(-1, 3)
    uu, tt = np.meshgrid(u, s, indexing="ij")
    uv = map_uv(uu.ravel(), (1 - tt).ravel(), rect)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    keep = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) > 1e-10
    return Mesh(v, f[keep], uv)


def frames_along(path, up_hint=(0, 0, 1)):
    """Rotation-minimizing frames (tangent, normal, binormal) along a polyline."""
    p = np.asarray(path, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.zeros_like(t)
    up = np.asarray(up_hint, float)
    n0 = np.cross(t[0], up)
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(t[0], [1, 0, 0])
    n[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(p)):
        # double reflection method (Wang et al. 2008)
        v1 = p[i] - p[i - 1]
        c1 = v1 @ v1
        if c1 < 1e-14:
            n[i] = n[i - 1]
            continue
        nl = n[i - 1] - (2 / c1) * (v1 @ n[i - 1]) * v1
        tl = t[i - 1] - (2 / c1) * (v1 @ t[i - 1]) * v1
        v2 = t[i] - tl
        c2 = v2 @ v2
        n[i] = nl - (2 / c2) * (v2 @ nl) * v2 if c2 > 1e-14 else nl
        n[i] /= np.linalg.norm(n[i])
    b = np.cross(t, n)
    return t, n, b


def sweep(path, shape, rect=(0, 0, 1, 1), up_hint=(0, 0, 1), closed_shape=True,
          smooth_angle=50.0, scale=None):
    """Sweep a 2D cross-section (list of (a, b) in the normal/binormal plane, CCW) along a path.
    `scale` optionally scales the section per path point."""
    p = np.asarray(path, float)
    sh = np.asarray(shape, float)
    if closed_shape:
        sh = np.vstack([sh, sh[:1]])
    t, n, b = frames_along(p, up_hint)
    k = np.ones(len(p)) if scale is None else np.asarray(scale, float)
    pts = p[:, None, :] + k[:, None, None] * (sh[None, :, 0, None] * n[:, None, :] + sh[None, :, 1, None] * b[:, None, :])
    nu, nv = len(p) - 1, len(sh) - 1
    v = pts.reshape(-1, 3)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s = s / max(s[-1], 1e-9)
    w = np.linspace(0, 1, nv + 1)
    uu, tt = np.meshgrid(s, w, indexing="ij")
    uv = map_uv(uu.ravel(), tt.ravel(), rect)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, bb = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    # t x (section tangent) points inward for a CCW section, so wind the other way
    f = np.concatenate([np.stack([a, c, bb], 1), np.stack([a, d, c], 1)])
    return Mesh(v, f, uv).smooth(smooth_angle)


def circle(radius, segments, start=0.0):
    a = start + np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], 1)


def rounded_rect(w, h, r, per_corner=4):
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for a in np.linspace(a0, a0 + 90, per_corner + 1):
            pts.append((cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))))
    return np.array(pts)


def tube(path, radius, segments=10, rect=(0, 0, 1, 1), up_hint=(0, 0, 1), caps=True, cap_rect=None):
    m = sweep(path, circle(radius, segments), rect, up_hint)
    if not caps:
        return m
    p = np.asarray(path, float)
    t, n, b = frames_along(p, up_hint)
    parts = [m]
    for end, sign in ((0, -1), (len(p) - 1, 1)):
        parts.append(disc_at(p[end], t[end] * sign, n[end], radius, segments, cap_rect or rect))
    return concat(parts)


def disc_at(center, normal, ref, radius, segments, rect=(0, 0, 1, 1)):
    """Flat disc facing `normal`. Seen from the front, `ref` is the image's right and
    normal x ref its up, so a texture in `rect` reads the right way round."""
    nrm = np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    e1 = np.asarray(ref, float) - nrm * (np.asarray(ref, float) @ nrm)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = center + radius * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    v = np.vstack([center, ring])
    uv = map_uv(np.concatenate([[0.5], 0.5 + 0.5 * np.cos(a)]), np.concatenate([[0.5], 0.5 - 0.5 * np.sin(a)]), rect)
    f = np.array([[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)])
    m = Mesh(v, f, uv)
    m.n = np.tile(nrm, (len(v), 1))
    return m


def box(size, center=(0, 0, 0), rect=(0, 0, 1, 1)):
    sx, sy, sz = (np.asarray(size, float) / 2)
    c = np.asarray(center, float)
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            nrm = np.zeros(3)
            nrm[axis] = sign
            u_ax, v_ax = [(1, 2), (2, 0), (0, 1)][axis]
            e1 = np.zeros(3)
            e1[u_ax] = 1
            e2 = np.cross(nrm, e1)
            half = np.array([sx, sy, sz])
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append(c + nrm * half[axis] + e1 * a * half[u_ax] + e2 * b * half[np.argmax(np.abs(e2))])
            v = np.array(corners)
            uv = map_uv(np.array([0, 1, 1, 0.0]), np.array([1, 1, 0, 0.0]), rect)
            m = Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), uv)
            m.n = np.tile(nrm, (4, 1))
            faces.append(m)
    return concat(faces)


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    r = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                  [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                  [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def trans(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def source_angle_matrix(p, y, r):
    """Rotation matrix of a Source Angle(p, y, r); columns are forward, left, up."""
    p, y, r = np.radians([p, y, r])
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    forward = np.array([cp * cy, cp * sy, -sp])
    right = np.array([-sr * sp * cy + cr * sy, -sr * sp * sy - cr * cy, -sr * cp])
    up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
    return np.stack([forward, -right, up], axis=1)


def matrix_to_source_angle(m):
    """Inverse of source_angle_matrix (columns forward, left, up)."""
    fwd, left, up = m[:, 0], m[:, 1], m[:, 2]
    xy = np.hypot(fwd[0], fwd[1])
    if xy > 1e-6:
        yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
        pitch = np.degrees(np.arctan2(-fwd[2], xy))
        roll = np.degrees(np.arctan2(left[2], up[2]))
    else:
        yaw = np.degrees(np.arctan2(-left[0], left[1]))
        pitch = np.degrees(np.arctan2(-fwd[2], xy))
        roll = 0.0
    return pitch, yaw, roll
