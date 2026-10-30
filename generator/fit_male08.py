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
from pacgen import ORIGIN_BONE

IDLE = "aim_normal_0"
REF = "seq0"      # reference (ragdoll) pose from the first export; head-space fitting uses it
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

    def __init__(self, model, pose=REF):
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

        # forward = where the model's "eyes" attachment looks (attachments were recorded in the
        # reference pose, so use that pose's head rotation), up = world up in the same pose
        label = REF if REF in model.attachments else next(iter(model.attachments))
        _, _, eye_ang = model.attachments[label]["eyes"]
        R = model.pose(REF)[model.bone(HEAD)][:3, :3]
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
        att = model.attachments[label].get("mouth")
        inv = np.linalg.inv(self.head_world)
        self.mouth_attachment = None if att is None else inv[:3, :3] @ att[1] + inv[:3, 3]
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
        if self.mouth_attachment is not None:
            # the model's own "mouth" attachment marks the lip line; average with the profile dip
            self.mouth_v = float((self.mouth_v + self.coords(self.mouth_attachment)[1]) / 2)
        chin_zone = (vv < self.lower_lip_v - 0.35) & (vv > self.lower_lip_v - 2.2)
        self.chin_v = vv[chin_zone][np.argmax(dd[chin_zone])]
        self.chin_d = dd[chin_zone].max()
        # where the jaw turns under: depth falls well behind the chin
        under = vv[(vv < self.chin_v) & (dd < self.chin_d - 1.2)]
        self.chin_bottom_v = under[0] if len(under) else self.chin_v - 0.8
        # front/back of the skull at nose height give the head's centre depth
        front = (self.surface(0.0, -1.0) - self.origin) @ self.F
        back = (self.cast_out(self.point(0.0, -1.0, 0.0), -self.F) - self.origin) @ self.F
        self.centre_d = float((front + back) / 2)
        self.back_d = float(back)

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
        v_top=-0.45, v_mouth=hd.mouth_v, v_bottom=hd.chin_v - 0.05, w_cheek=1.95, rim_lift=0.11,
        v_apex=(hd.nose_v + hd.mouth_v) / 2, d_apex=hd.nose_d + 1.0, dome_round=0.85, thickness=0.07,
        port_u=0.5, port_s=0.5, port_forward=0.5, clip_span=0.075,
        strap_axis_v=hd.nose_v, strap_axis_d=hd.centre_d,
        strap_v_front=hd.mouth_v + 0.5, strap_v_back=hd.mouth_v + 0.35, strap_v_ear=hd.mouth_v - 0.15,
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
    d = hd.F * 0.92 - hd.U * 0.34 + hd.L * 0.26
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
    # Hung off pac's no-bone frame (his origin turned to his eye yaw, never pitched or rolled), not
    # the pelvis: rifle and physgun stances twist the pelvis 36-72 degrees and crouching tilts it,
    # which swung the cart around him and, with ConstrainPitch, could freeze it tilted. The dump's
    # entity sits at the origin facing +x, so that frame is the identity here. The drop trace starts
    # at pelvis height, 24 units out: outside the player's 32x32 hull so it can't hit him.
    fit.cart_bone = ORIGIN_BONE
    fit.cart_pos = (-7.0, 23.0, round(float(pelvis[2, 3]), 2))
    fit.cart_ang = (0.0, 0.0, 0.0)
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


def fit_cane_belt(fit, model):
    """Cane hooked on the belt at the right hip, slung diagonally back like a sword so it clears
    the floor in low stances. Defined in the pelvis's own frame (taken from the idle pose, where
    the pelvis faces forward), so it stays on the right hip whatever the weapon stance does."""
    mats = model.pose(IDLE, {"pelvis": (fit.pelvis_sink, (0, 0, 0))})
    pelvis = mats[model.bone("ValveBiped.Bip01_Pelvis")]
    hook = pelvis[:3, 3] + np.array([-2.6, -7.0, 2.6])        # right hip, a little back, belt height
    shaft = np.array([-0.5, -0.18, -0.85])                     # hangs down and back, slightly out
    up = -shaft / np.linalg.norm(shaft)
    inward = np.array([0.25, 1.0, 0.0])                         # the handle hooks over the belt
    x = inward - (inward @ up) * up
    x /= np.linalg.norm(x)
    world = np.eye(4)
    world[:3, :3] = np.column_stack([x, np.cross(up, x), up])
    world[:3, 3] = hook
    local = np.linalg.inv(pelvis) @ world
    fit.cane_belt = (tuple(np.round(local[:3, 3], 3)), tuple(np.round(mk.matrix_to_source_angle(local[:3, :3]), 2)))


CAST_POSES = ("aim_normal_0", "seq0", "walk_0", "walk_0.25", "walk_0.5", "walk_0.75", "run_0", "run_0.5",
              "crouch_idle", "cwalk_0", "cwalk_0.5", "aim_pistol_0", "aim_ar2_0", "aim_melee_0")


_BARY = np.array([(i, j, 6 - i - j) for i in range(7) for j in range(7 - i)], float) / 6   # 28 per triangle


def leg_points(model, bone_name, poses, post, min_weight=0.25, extra=None):
    """Skin surface weighted to `bone_name`, densely sampled over its triangles (the mesh is coarse;
    sparse points let the per-slab hulls cut corners inside the real surface), expressed in that
    bone's frame, from every pose in `poses`. extra(mats) can add world-space points per pose."""
    b = model.bone(bone_name)
    weights = [np.array([sum(x for bb, x in ws if bb == b) for ws in src["w"]]) for src in model.meshes]
    out = []
    for label in poses:
        mats = model.pose(label, post)
        inv = np.linalg.inv(mats[b])
        pts = [extra(mats)] if extra is not None else []
        for mesh, w in zip(model.skinned(mats), weights):
            sel = w >= min_weight
            if not sel.any():
                continue
            v = mesh.v
            f = mesh.f[sel[mesh.f].any(axis=1)]          # every triangle touching the bone's skin
            tri = v[f]                                   # (n, 3, 3)
            pts += [v[sel], np.einsum("bk,nkd->nbd", _BARY, tri).reshape(-1, 3)]
        p = np.vstack(pts)
        out.append(p @ inv[:3, :3].T + inv[:3, 3])
    return np.vstack(out)


def fit_leg_cast(fit, model, side="L", poses=CAST_POSES, max_ankle=21.0):
    """Short-leg plaster cast with a cast shoe: plaster tube (calf bone), shoe (foot bone), shoe toe
    (toe bone). The shoe's collar is sized for ankle bends up to max_ankle degrees from standing
    (standing, walking, weapon stances); deeper bends (push-off, sprint, crouch) are brief."""
    import leg_cast as lc
    post = posture_bone3(fit)
    calf_n, foot_n, toe_n = (f"ValveBiped.Bip01_{side}_{b}" for b in ("Calf", "Foot", "Toe0"))
    calf, foot, toe = model.bone(calf_n), model.bone(foot_n), model.bone(toe_n)
    idle = model.pose(IDLE, post)

    # plaster: from just below the knee in front (3.0 down the calf bone) and lower behind it
    # (5.0, the back of the knee) down over the ankle joint
    shin_len = np.linalg.norm(idle[foot][:3, 3] - idle[calf][:3, 3])
    tube = lc.shin_tube(leg_points(model, calf_n, poses, post), s_front=3.0, s_back=5.0, s_bottom=shin_len + 0.4)

    # shoe: the leather shoe, plus the bottom 1.4 units of the plaster so its collar sits over it
    rel0 = idle[calf][:3, :3].T @ idle[foot][:3, :3]

    def ankle_bend(mats):
        d = rel0.T @ (mats[calf][:3, :3].T @ mats[foot][:3, :3])
        return np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))

    moderate = [p for p in poses if ankle_bend(model.pose(p, post)) <= max_ankle]
    tube_end = tube.v[tube.v[:, 0] > shin_len - 1.0]

    def tube_end_world(mats):
        return tube_end @ mats[calf][:3, :3].T + mats[calf][:3, 3]

    foot_pts = np.vstack([leg_points(model, foot_n, poses, post),
                          leg_points(model, foot_n, moderate, post, min_weight=2.0,   # extra points only
                                     extra=tube_end_world)])
    rf, rt = idle[foot][:3, :3], idle[toe][:3, :3]
    fwd = rf.T @ rt[:, 0]                                    # sole-forward (toe bone x) in the foot frame
    up = rf.T @ np.array([0, 0, 1.0])
    up = up - (up @ fwd) * fwd
    up /= np.linalg.norm(up)
    side_ax = np.cross(fwd, up)
    ball = (idle[toe][:3, 3] - idle[foot][:3, 3]) @ rf          # toe bone origin in the foot frame
    s_ball = ball @ fwd
    s_heel = (foot_pts @ fwd).min()
    shoe = lc.closed_loft(foot_pts[(foot_pts @ fwd) < s_ball + 1.4], fwd, up, side_ax, s_heel - 0.2, s_ball + 1.4,
                          offset=0.45, round_back=0.9, round_front=1.2, bottom_offset=0.55, bottom_dir=-up,
                          uv_along=lc.SHOE_FOOT_U)

    # the shoe's toe: front of the leather shoe in the toe frame, tucked 1.5 back into the shoe
    toe_pts = leg_points(model, toe_n, poses, post)
    t_fwd = np.array([1.0, 0, 0])
    t_up = rt.T @ np.array([0, 0, 1.0])
    t_up = t_up - (t_up @ t_fwd) * t_fwd
    t_up /= np.linalg.norm(t_up)
    t_side = np.cross(t_fwd, t_up)
    toe_cap = lc.closed_loft(toe_pts[toe_pts[:, 0] > -1.5], t_fwd, t_up, t_side, -1.5, toe_pts[:, 0].max() + 0.6,
                             offset=0.45, round_back=0.6, round_front=1.9, bottom_offset=0.55, bottom_dir=-t_up,
                             n_len=20, uv_along=lc.SHOE_TOE_U)
    fit.cast_meshes = {"plaster": tube, "shoe": shoe, "toe": toe_cap}
    fit.cast_bones = {"plaster": pac_name(calf_n), "shoe": pac_name(foot_n), "toe": pac_name(toe_n)}
    fit.cast_moderate_poses = moderate


def pac_name(real):
    from male08 import pac_bone_name
    return pac_bone_name(real)


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
        hang = getattr(fit, "arm_hang", None)
        for side, skip in (("left", smoke), ("right", cane)):
            if skip:
                continue
            ua = hang[side]["upperarm"] if hang else fit.upperarm_hang
            fa = hang[side]["forearm"] if hang else fit.forearm_bend
            b[f"{side} upperarm"] = ((0, 0, 0), ua)
            b[f"{side} forearm"] = ((0, 0, 0), fa)
    return b


def apply(fit, model):
    fit.model = model
    fit.head = hd = HeadData(model, REF)
    fit_mask(fit, hd)
    fit_glasses(fit, hd)
    fit_face_props(fit, hd)
    fit_cart_and_tube(fit, model, hd)
    # the cane is shortened 7% along its shaft: hunched and on bent knees his wrist hangs lower than
    # the cane's full height, which forced a ~59 degree elbow bend; this gives the ~30 degrees of a
    # cane fitted to the person
    fit.cane_scale = 0.93
    fit_cane_belt(fit, model)
    fit.glasses_obj = "props/glasses_m08.obj"
    fit.lenses_obj = "props/lenses_m08.obj"
    fit_leg_cast(fit, model)
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
    import leg_cast as lc
    frames = {"plaster": "left calf", "shoe": "left foot", "toe": "left toe"}
    names = {"plaster": ("cast_plaster.obj", "plaster short-leg cast with stockinette cuff"),
             "shoe": ("cast_shoe.obj", "post-op cast shoe"), "toe": ("cast_shoe_toe.obj", "post-op cast shoe, toe")}
    for k, mesh in fit.cast_meshes.items():
        fname, what = names[k]
        mesh.write_obj(os.path.join(props, fname),
                       f"{what} - procedural, CC0\n{frames[k]} bone frame of male_08_shirt_tie, inches")
    lc.paint(os.path.join(props, "cast_plaster.jpg"), os.path.join(props, "cast_plaster_n.png"))
    lc.paint_shoe(os.path.join(props, "cast_shoe.jpg"))
