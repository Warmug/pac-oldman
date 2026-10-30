"""Measures male_08 and solves the model-specific values of the outfit.

Coordinates: the dump puts the model at the world origin facing +x (Source: x forward,
y left, z up). Head-space values are in the ValveBiped.Bip01_Head1 frame, which is
exactly the frame pac uses for Position/Angles of parts on the "head" bone.
"""
import os

import numpy as np

import glasses as gl
import meshkit as mk
import oxygen_cart as oc
import oxygen_mask as om
from geom import RayCaster

IDLE = "aim_normal_0"
HEAD = "ValveBiped.Bip01_Head1"


def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v), np.array(f)


def to_local(world_mat, p):
    inv = np.linalg.inv(world_mat)
    return inv[:3, :3] @ np.asarray(p) + inv[:3, 3]


def dir_local(world_mat, d):
    return world_mat[:3, :3].T @ np.asarray(d)


def angles_from_axes(fwd, up_hint):
    fwd = np.asarray(fwd, float) / np.linalg.norm(fwd)
    left = np.cross(up_hint, fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    return mk.matrix_to_source_angle(np.column_stack([fwd, left, up]))


class HeadData:
    """Head-space mesh, face frame and landmarks."""

    def __init__(self, model, pose=IDLE):
        mats = model.pose(pose)
        self.head_world = mats[model.bone(HEAD)]
        inv = np.linalg.inv(self.head_world)
        verts, faces, norms, eyes, off = [], [], [], {}, 0
        for m in model.skinned(mats):
            v = m.v @ inv[:3, :3].T + inv[:3, 3]
            n = m.n @ inv[:3, :3].T
            name = m.material.lower()
            if "eyeball" in name:
                eyes["l" if name.endswith("_l") or "eyeball_l" in name else "r"] = (v.mean(axis=0), np.linalg.norm(v - v.mean(axis=0), axis=1).max())
            keep = np.linalg.norm(v[m.f].mean(axis=1) - [4, -1, 0], axis=1) < 11
            verts.append(v)
            norms.append(n)
            faces.append(m.f[keep] + off)
            off += len(v)
        self.v = np.vstack(verts)
        self.n = np.vstack(norms)
        self.f = np.vstack(faces)
        self.rc = RayCaster(self.v, self.f, self.n)
        (self.eye_l, self.eye_r_rad), (self.eye_r, _) = eyes["l"], eyes["r"]
        self.eye_radius = self.eye_r_rad

        # forward = where the model's "eyes" attachment looks, up = world up in this stance
        label = pose if pose in model.attachments else next(iter(model.attachments))
        _, _, eye_ang = model.attachments[label]["eyes"]
        R = self.head_world[:3, :3]
        F = R.T @ mk.source_angle_matrix(*eye_ang)[:, 0]
        U = R.T @ np.array([0.0, 0.0, 1.0])
        U = U - (U @ F) * F
        U /= np.linalg.norm(U)
        F /= np.linalg.norm(F)
        L = np.cross(U, F)
        if (self.eye_l - self.eye_r) @ L < 0:
            self.eye_l, self.eye_r = self.eye_r, self.eye_l
        self.origin = (self.eye_l + self.eye_r) / 2
        self.F, self.U, self.L = F, U, L
        self.face = om.Face(self.origin, F, U, L, self.surface, self.normal, {})
        self._landmarks()

    def surface(self, u, v):
        o = self.origin + self.L * u + self.U * v + self.F * 15
        p, _ = self.rc.first(o, -self.F)
        return p if p is not None else self.origin + self.L * u + self.U * v

    def depth(self, u, v):
        return (self.surface(u, v) - self.origin) @ self.F

    def normal(self, u, v):
        return self.rc.nearest_vertex_normal(self.surface(u, v))

    def _landmarks(self):
        vs = np.arange(-0.3, -4.6, -0.05)
        mid = np.array([self.depth(0.0, v) for v in vs])
        self.profile = (vs, mid)
        i_nose = int(np.argmax(np.where(vs > -2.4, mid, -1e9)))
        self.nose_v, self.nose_d = vs[i_nose], mid[i_nose]
        # below the nose: subnasale (min), upper lip (max), mouth line (min), lower lip (max), chin
        below = vs < self.nose_v - 0.35
        vv, dd = vs[below], mid[below]
        peaks = [i for i in range(1, len(dd) - 1) if dd[i] >= dd[i - 1] and dd[i] >= dd[i + 1]]
        self.upper_lip_v = vv[peaks[0]] if peaks else self.nose_v - 0.9
        lips = [i for i in peaks if vv[i] < self.upper_lip_v - 0.25]
        self.lower_lip_v = vv[lips[0]] if lips else self.upper_lip_v - 0.5
        between = (vv < self.upper_lip_v) & (vv > self.lower_lip_v)
        self.mouth_v = vv[between][np.argmin(dd[between])] if between.any() else (self.upper_lip_v + self.lower_lip_v) / 2
        chin_zone = (vv < self.lower_lip_v - 0.35) & (vv > self.lower_lip_v - 2.2)
        self.chin_v = vv[chin_zone][np.argmax(dd[chin_zone])]
        self.chin_d = dd[chin_zone].max()
        # where the jaw turns under: depth falls well behind the chin
        under = vv[(vv < self.chin_v) & (dd < self.chin_d - 1.2)]
        self.chin_bottom_v = under[0] if len(under) else self.chin_v - 0.8
        hv = self.v[np.linalg.norm(self.v - self.origin, axis=1) < 9]
        q = hv - self.origin
        self.centre_d = float(np.median(q @ self.F)) - 0.6
        self.centre_v = float(np.median(q @ self.U))

    def coords(self, p):
        q = np.asarray(p) - self.origin
        return np.array([q @ self.L, q @ self.U, q @ self.F])

    def point(self, u, v, d):
        return self.origin + self.L * u + self.U * v + self.F * d

    def cast_out(self, origin, direction):
        t, idx = self.rc.hits(origin, direction)
        if len(t) == 0:
            return origin + direction * 4
        return origin + np.asarray(direction) / np.linalg.norm(direction) * t[-1]

    def clear(self, p, margin):
        """Push p outward (from the head's centre axis) until it is `margin` above the skin."""
        c = self.point(0.0, self.coords(p)[1], self.centre_d)
        d = p - c
        d /= np.linalg.norm(d)
        hit = self.cast_out(c, d)
        need = np.linalg.norm(hit - c) + margin
        return p if np.linalg.norm(p - c) >= need else c + d * need


def fit_mask(fit, hd):
    m = dict(
        v_top=-0.42, v_mouth=hd.mouth_v, v_bottom=hd.chin_v - 0.15, w_cheek=1.95, rim_lift=0.11,
        v_apex=(hd.nose_v + hd.mouth_v) / 2, d_apex=hd.nose_d + 1.0, dome_round=0.85, thickness=0.07,
        port_u=0.5, port_s=0.5, port_forward=0.5, clip_span=0.075,
        strap_axis_v=hd.nose_v, strap_axis_d=hd.centre_d,
        strap_v_front=hd.mouth_v + 0.45, strap_v_back=hd.nose_v - 0.55, strap_v_ear=hd.nose_v - 0.75,
        strap_lift=0.05, strap_width=0.42,
        cast_out=hd.cast_out, head_normal=hd.rc.nearest_vertex_normal,
    )
    shell, strap, outlet, odir = om.build_mask(hd.face, m)
    fit.mask_params = m
    fit.mask_mesh, fit.strap_mesh = shell, strap
    fit.mask_outlet = (tuple(outlet), angles_from_axes(odir, hd.F))
    return shell, strap


def fit_glasses(fit, hd):
    ipd = np.linalg.norm(hd.eye_l - hd.eye_r)
    g = dict(eye_l=hd.eye_l, eye_r=hd.eye_r, rim_radius=min(0.8, ipd / 2 - 0.33),
             lens_ahead=hd.eye_radius + 0.5, rim_drop=-0.05, bridge_lift=0.18, wire=0.045,
             hook_drop=0.75, clear=hd.clear)
    for side, sign in (("l", 1), ("r", -1)):
        c = hd.point(0.0, 0.1, hd.centre_d)
        direction = hd.L * sign * np.cos(np.radians(12)) - hd.F * np.sin(np.radians(12))
        hit = hd.cast_out(c, direction)
        g["temple_end_" + side] = hit + direction * 0.12
    frames, lenses = gl.build_glasses(hd.face, g)
    fit.glasses_params = g
    fit.glasses_mesh, fit.lenses_mesh = frames, lenses


def fit_face_props(fit, hd):
    """Cigarette between the lips (left corner) and where the exhaled smoke comes out."""
    corner = hd.surface(0.62, hd.mouth_v)
    filt = corner - hd.F * 0.1
    d = hd.F * 0.82 - hd.U * 0.42 + hd.L * 0.38
    d /= np.linalg.norm(d)
    origin = filt + d * 0.45
    fit.cig_mouth = (tuple(origin), angles_from_axes(d, hd.U))
    fit.cig_mouth_tip = tuple(origin + d * 2.85)
    lips = hd.surface(0.0, hd.mouth_v) + hd.F * 0.35
    fit.mouth = tuple(lips)
    fit.exhale_angles = angles_from_axes(hd.F - hd.U * 0.25, hd.U)


def fit_cart_and_tube(fit, model, hd):
    mats = model.pose(IDLE, posture_bone3(fit))
    pelvis = mats[model.bone("ValveBiped.Bip01_Pelvis")]
    target = np.array([-7.0, 23.0, pelvis[2, 3]])
    fit.cart_pos = tuple(to_local(pelvis, target))
    fit.cart_ang = mk.matrix_to_source_angle(pelvis[:3, :3].T)
    fit.barb_pos = tuple(oc.barb_tip())

    # clip on the shirt: front-left of the belly, a little off the skin
    body = [m for m in model.skinned(mats)]
    v = np.vstack([m.v for m in body])
    off, faces = 0, []
    for m in body:
        faces.append(m.f + off)
        off += len(m.v)
    rc = RayCaster(v, np.vstack(faces))
    sp = model.bone("ValveBiped.Bip01_Spine1")
    spine = mats[sp]
    start = spine[:3, 3] + np.array([0, 0, 1.5])
    direction = np.array([np.cos(np.radians(35)), np.sin(np.radians(35)), 0.0])
    hit = rc.hits(start, direction)
    clip_world = start + direction * (hit[0][-1] + 0.35) if len(hit[0]) else start + direction * 7
    fit.tube_clip_bone = "spine 1"
    fit.tube_clip = (tuple(to_local(spine, clip_world)), angles_from_axes(dir_local(spine, [0, 0, -1]), dir_local(spine, [1, 0, 0])))
    fit.clip_world = clip_world


def posture_bone3(fit, arms=True, cane=False, smoke=False):
    """The always-on posture as bone3 manipulations {bone: (pos, ang)} (unarmed)."""
    b = {"pelvis": (fit.pelvis_sink, (0, 0, 0)), "spine": ((0, 0, 0), fit.hunch[0]),
         "spine 1": ((0, 0, 0), fit.hunch[1]), "spine 2": ((0, 0, 0), fit.hunch[2]),
         "spine 4": ((0, 0, 0), fit.hunch[3]), "neck": ((0, 0, 0), fit.neck)}
    for side in ("left", "right"):
        b[f"{side} thigh"] = ((0, 0, 0), fit.thigh)
        b[f"{side} calf"] = ((0, 0, 0), fit.calf)
        b[f"{side} foot"] = ((0, 0, 0), fit.foot)
    if arms:
        b["left clavicle"] = ((0, 0, 0), fit.clavicle_l)
        b["right clavicle"] = ((0, 0, 0), fit.clavicle_r)
        if not smoke:
            b["left upperarm"] = ((0, 0, 0), fit.upperarm_hang)
            b["left forearm"] = ((0, 0, 0), fit.forearm_bend)
        if not cane:
            b["right upperarm"] = ((0, 0, 0), fit.upperarm_hang)
            b["right forearm"] = ((0, 0, 0), fit.forearm_bend)
    return b


def apply(fit, model):
    fit.model = model
    fit.head = hd = HeadData(model)
    fit_mask(fit, hd)
    fit_glasses(fit, hd)
    fit_face_props(fit, hd)
    fit_cart_and_tube(fit, model, hd)
    fit.glasses_obj = "props/glasses_m08.obj"
    fit.lenses_obj = "props/lenses_m08.obj"
    import solve_poses
    solve_poses.solve(fit, model)


def write_head_assets(fit, props):
    fit.mask_mesh.write_obj(os.path.join(props, "o2_mask.obj"),
                            "clear oxygen mask shell + elbow outlet - procedural, CC0\nHead1 frame of male_08_shirt_tie, inches")
    fit.strap_mesh.write_obj(os.path.join(props, "o2_strap.obj"),
                             "oxygen mask elastic strap, nose clip, lugs - procedural, CC0\nHead1 frame of male_08_shirt_tie, inches")
    fit.glasses_mesh.write_obj(os.path.join(props, "glasses_m08.obj"),
                               "round wire spectacles - procedural, CC0\nHead1 frame of male_08_shirt_tie, inches")
    fit.lenses_mesh.write_obj(os.path.join(props, "lenses_m08.obj"),
                              "spectacle lenses - procedural, CC0\nHead1 frame of male_08_shirt_tie, inches")
    om.paint_mask(os.path.join(props, "o2_mask.png"))
    om.paint_strap(os.path.join(props, "o2_strap.jpg"))
    om.paint_tube(os.path.join(props, "o2_tube.png"))
