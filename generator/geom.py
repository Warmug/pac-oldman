"""Ray casting and nearest-point queries against a triangle mesh (numpy, brute force)."""
import numpy as np


class RayCaster:
    def __init__(self, v, f, n=None):
        self.v = np.asarray(v, float)
        self.f = np.asarray(f, int)
        self.a = self.v[self.f[:, 0]]
        self.e1 = self.v[self.f[:, 1]] - self.a
        self.e2 = self.v[self.f[:, 2]] - self.a
        fn = np.cross(self.e1, self.e2)
        self.fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        self.vn = None if n is None else np.asarray(n, float)

    def hits(self, origin, direction):
        """All (t, triangle) hits along the ray, sorted by t (Moller-Trumbore, both sides)."""
        o = np.asarray(origin, float)
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        p = np.cross(d, self.e2)
        det = np.einsum("ij,ij->i", self.e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        s = o - self.a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, self.e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", self.e2, q) * inv
        m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        idx = np.nonzero(m)[0]
        order = np.argsort(t[idx])
        return t[idx][order], idx[order]

    def first(self, origin, direction):
        t, idx = self.hits(origin, direction)
        if len(t) == 0:
            return None, None
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        return np.asarray(origin, float) + d * t[0], idx[0]

    def nearest_vertex_normal(self, p):
        i = np.argmin(np.linalg.norm(self.v - p, axis=1))
        if self.vn is not None:
            return self.vn[i]
        tri = np.nonzero((self.f == i).any(axis=1))[0]
        n = self.fn[tri].sum(axis=0)
        return n / np.linalg.norm(n)

    def closest_distance(self, points):
        """Approximate distance from each point to the mesh (nearest vertex)."""
        pts = np.asarray(points, float)
        out = np.empty(len(pts))
        for i, p in enumerate(pts):
            out[i] = np.min(np.linalg.norm(self.v - p, axis=1))
        return out

    def inside(self, p, direction=(0.577, 0.577, 0.577)):
        """Parity test: odd number of crossings means inside a closed mesh."""
        t, _ = self.hits(p, direction)
        return len(t) % 2 == 1
