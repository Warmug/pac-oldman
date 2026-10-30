"""Round wire spectacles fitted to a face (head bone frame), plus their lenses."""
import numpy as np

import meshkit as mk

WIRE = (0.05, 0.05, 0.95, 0.95)   # oldman_glasses.jpg is a flat gold wire colour


def build_glasses(face, g):
    """g: eye_l, eye_r (eyeball centres), rim_radius, lens_ahead (in front of the eyeball
    centre), bridge_lift, wire, temple_end_l/r (points just above each ear root),
    hook_drop, clear(p) -> point pushed out of the skin. Returns (frames, lenses)."""
    F, U, L = face.F, face.U, face.L
    r = g["rim_radius"]
    parts, lenses = [], []
    centres = {}
    for side, eye in (("l", g["eye_l"]), ("r", g["eye_r"])):
        c = eye + F * g["lens_ahead"] + U * g.get("rim_drop", 0.0)
        centres[side] = c
        rim = [c + r * (np.cos(a) * L + np.sin(a) * U) for a in np.linspace(0, 2 * np.pi, 49)]
        parts.append(mk.sweep(rim, mk.circle(g["wire"], 6), WIRE, up_hint=F))
        # lens: a shallow dome so it catches highlights
        def lens_fn(uu, ss, c=c):
            ang = uu * 2 * np.pi
            rad = ss * (r - g["wire"] * 0.5)
            bulge = 0.08 * (1 - ss ** 2)
            return (c[None, None, :] + rad[..., None] * (np.cos(ang)[..., None] * L + np.sin(ang)[..., None] * U)
                    + bulge[..., None] * F)
        lens = mk.surface(lens_fn, 24, 4, (0.05, 0.05, 0.95, 0.95), smooth_angle=80)
        if np.mean(lens.n @ F) < 0:
            lens = lens.flipped()
        lenses += [lens, lens.flipped()]

    # bridge: arch over the nose between the inner edges of the rims
    inner_l = centres["l"] - L * r
    inner_r = centres["r"] + L * r
    mid = (inner_l + inner_r) / 2 + U * g["bridge_lift"]
    mid = g["clear"](mid, 0.12)
    bridge = [(1 - t) ** 2 * inner_l + 2 * (1 - t) * t * mid + t * t * inner_r for t in np.linspace(0, 1, 13)]
    parts.append(mk.sweep(bridge, mk.circle(g["wire"], 6), WIRE, up_hint=F))

    # temples: hinge at the outer edge, back along the head just above the ear, then hook down
    for side, sign in (("l", 1), ("r", -1)):
        c = centres[side]
        hinge = c + L * sign * (r + 0.02) + U * 0.12
        end = g["temple_end_" + side]
        pts = []
        for t in np.linspace(0, 1, 24):
            p = hinge * (1 - t) + end * t
            pts.append(g["clear"](p, 0.1))
        back = -F
        for k, t in enumerate(np.linspace(0, 1, 9)[1:]):
            a = t * np.pi * 0.55
            pts.append(end + back * 0.55 * np.sin(a) - U * g["hook_drop"] * (1 - np.cos(a)) / (1 - np.cos(np.pi * 0.55)))
        pts = np.array(pts)
        for _ in range(2):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        parts.append(mk.tube(pts, g["wire"] * 1.15, 6, WIRE, up_hint=U))
    return mk.concat(parts), mk.concat(lenses)
