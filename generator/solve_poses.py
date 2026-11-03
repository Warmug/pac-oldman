"""Re-solves the posed props on the real male_08 skeleton with pac's bone3 math.

- cane: right-arm angles so the cane tip rests on the floor beside the right foot
- cigarette: drag pose so the filter reaches the lips at the peak of the drag
- aim: how the hunch interacts with the aim layers, giving the neck/upper-back
  compensation used while aiming up/down and while a weapon is out
"""
import os

import numpy as np
from scipy.spatial import cKDTree

import meshkit as mk
from fit_male08 import IDLE, HEAD, load_obj, posture_bone3, to_local

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lm(residual, x0, iters=60, step=1e-3, lam=1e-2):
    """Levenberg-Marquardt with a finite-difference Jacobian."""
    x = np.asarray(x0, float).copy()
    r = residual(x)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = step
            J[:, i] = (residual(x + dx) - r) / step
        A = J.T @ J
        g = J.T @ r
        while True:
            dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn = x + dx
            rn = residual(xn)
            cn = rn @ rn
            if cn < cost:
                x, r, cost, lam = xn, rn, cn, lam * 0.3
                break
            lam *= 4
            if lam > 1e8:
                return x, cost
        if np.linalg.norm(dx) < 1e-5:
            break
    return x, cost


def part_matrix(pos, ang):
    m = np.eye(4)
    m[:3, :3] = mk.source_angle_matrix(*ang)
    m[:3, 3] = pos
    return m


def cane_geometry(scale=1.0):
    """Tip and top of the shaft in the cane part's frame, with the part's Scale z applied."""
    v, _ = load_obj(os.path.join(REPO, "props/cane.obj"))
    shaft = v[v[:, 2] < -6]
    axis = shaft[:, :2].mean(axis=0)
    tip = np.array([axis[0], axis[1], v[:, 2].min() * scale])
    top = np.array([axis[0], axis[1], -6.0 * scale])
    return tip, top


def leg_tree(model, mats):
    """KD-tree of the leg and hip skin (vertices mostly weighted to pelvis/thigh/calf/foot)."""
    legs = {model.bone(f"ValveBiped.Bip01_{s}_{b}") for s in "LR" for b in ("Thigh", "Calf", "Foot", "Toe0")}
    legs.add(model.bone("ValveBiped.Bip01_Pelvis"))
    pts = [mesh.v[k] for mesh, src in zip(model.skinned(mats), model.meshes)
           for k, ws in enumerate(src["w"]) if max(ws, key=lambda t: t[1])[0] in legs]
    return cKDTree(np.array(pts))


def solve_cane(fit, model, ahead=6.0, outside=9.0, clearance=3.0):
    """Right-arm angles so the cane tip rests on the floor `ahead` units in front of and
    `outside` units to the right of the right foot, out where it can take his weight, with
    the hand beside his thigh and the shaft kept `clearance` off his legs."""
    tip_l, top_l = cane_geometry(fit.cane_scale)
    names = ["right upperarm", "right forearm", "right hand"]
    x0 = np.array([*fit.cane_arm["right upperarm"], fit.cane_arm["right forearm"][1], *fit.cane_arm["right hand"]])
    foot = model.bone("ValveBiped.Bip01_R_Foot")
    hand = model.bone("ValveBiped.Bip01_R_Hand")
    cane_m = part_matrix(*fit.cane_hand)
    base = posture_bone3(fit, cane=True)
    legs = leg_tree(model, model.pose(IDLE, base))            # the arm angles don't move the legs
    shaft_l = np.array([top_l + (tip_l - top_l) * t for t in np.linspace(0, 0.97, 14)])

    def pose_with(x):
        b = dict(base)
        b["right upperarm"] = ((0, 0, 0), tuple(x[0:3]))
        b["right forearm"] = ((0, 0, 0), (0, x[3], 0))
        b["right hand"] = ((0, 0, 0), tuple(x[4:7]))
        return model.pose(IDLE, b)

    def residual(x):
        mats = pose_with(x)
        w = mats[hand] @ cane_m
        tip = w[:3, :3] @ tip_l + w[:3, 3]
        top = w[:3, :3] @ top_l + w[:3, 3]
        f = mats[foot][:3, 3]
        shaft = (top - tip) / np.linalg.norm(top - tip)
        lean = np.degrees(np.arccos(np.clip(shaft[2], -1, 1)))
        gap, _ = legs.query(shaft_l @ w[:3, :3].T + w[:3, 3])
        return np.array([
            (tip[2] - 0.2) * 3.0,                 # resting on the floor
            (tip[0] - (f[0] + ahead)) * 0.6,      # a step ahead of the right foot
            (tip[1] - (f[1] - outside)) * 0.6,    # a little outside it
            max(0.0, lean - 10.0) * 0.5,          # close to vertical
            *(np.maximum(0.0, clearance - gap) * 3.0),   # not touching his legs
            *((x - x0) * 0.02),                   # stay near the old look
        ])

    x, cost = lm(residual, x0)
    fit.cane_arm = {"right upperarm": tuple(np.round(x[0:3], 2)), "right forearm": (0, round(float(x[3]), 2), 0),
                    "right hand": tuple(np.round(x[4:7], 2))}
    mats = pose_with(x)
    w = mats[hand] @ cane_m
    fit.cane_report = {"tip": w[:3, :3] @ tip_l + w[:3, 3], "foot": mats[foot][:3, 3], "cost": cost}


SIT = "sit_act_head0"            # the chair sit (ACT_HL2MP_SIT); sit_rollercoaster is the same pose 4 lower


def torso_tree(model, mats, extra=(), dense=False):
    """KD-tree of the torso and neck skin (vertices mostly weighted to pelvis/spine/neck/clavicles),
    plus any `extra` bones (the thighs, when he sits). dense: skin sampled over the triangles, for
    thin things (a cigarette) that could slip between the vertices."""
    bones = {model.bone(f"ValveBiped.Bip01_{n}") for n in
             ("Pelvis", "Spine", "Spine1", "Spine2", "Spine4", "Neck1", "L_Clavicle", "R_Clavicle") + tuple(extra)}
    if dense:
        from fit_male08 import skin_samples
        return cKDTree(skin_samples(model, mats, bones, 0.4))
    pts = [mesh.v[k] for mesh, src in zip(model.skinned(mats), model.meshes)
           for k, ws in enumerate(src["w"]) if max(ws, key=lambda t: t[1])[0] in bones]
    return cKDTree(np.array(pts))


CIG_NAMES = ("left upperarm", "left forearm", "left hand")


def cig_split(v):
    """8 numbers -> {bone: angles}: upper arm p/y/r, forearm bend (yaw) and twist (roll), hand p/y/r."""
    return {"left upperarm": tuple(v[0:3]), "left forearm": (0, v[3], v[4]), "left hand": tuple(v[5:8])}


def cig_pose_params(x):
    """16 numbers -> {bone: angles} for the hold pose (x[:8]) and the drag offset (x[8:])."""
    return cig_split(x[:8]), cig_split(x[8:])


def solve_cig(fit, model, upper_r=3.0, fore_r=2.6, path=(0.0, 0.25, 0.5, 0.75, 1.0), label=IDLE,
              attr="cig_arm", torso_extra=(), rest_target=None, extra_clear=None, start_from=None):
    """Left arm hold pose and the drag that brings the filter to his lips, on top of the animation
    `label` (standing idle, or the seated pose); the result goes to fit.<attr>.

    pac moves between the two by blending the bone3 angles (hold + drag * offset), so the arm is
    checked along that whole path: the upper arm and forearm axes stay upper_r/fore_r away from his
    torso skin (no clipping into the chest, even through a jacket), and at the drag the elbow sits
    out to his left and in front of the chest, the forearm rising to the mouth.

    Seated extras: rest_target (where the resting hand goes, instead of just above the animation's),
    extra_clear(mats) -> residuals added at every path sample (e.g. keeping the hand out of the leg
    cast), start_from: an earlier {bone: (hold, drag, ...)} solution to also start from."""
    hand = model.bone("ValveBiped.Bip01_L_Hand")
    head = model.bone(HEAD)
    sh, el = model.bone("ValveBiped.Bip01_L_UpperArm"), model.bone("ValveBiped.Bip01_L_Forearm")
    chest_b = model.bone("ValveBiped.Bip01_Spine4")
    cig_m = part_matrix(*fit.cig_hand)
    filt_l = np.array([-0.45, 0, 0])
    tip_l = np.array([2.85, 0, 0])
    base = posture_bone3(fit, smoke=True)
    torso = torso_tree(model, model.pose(label, base), torso_extra)   # the left arm angles don't move the torso
    hd = fit.head
    lips_head = hd.surface(0.22, hd.mouth_v) + hd.F * 0.05

    def pose_at(x, t):
        hold, drag = cig_pose_params(x)
        b = dict(base)
        for n in CIG_NAMES:
            b[n] = ((0, 0, 0), tuple(np.array(hold[n]) + t * np.array(drag[n])))
        return model.pose(label, b)

    def arm_axis(mats):
        S, E, W = mats[sh][:3, 3], mats[el][:3, 3], mats[hand][:3, 3]
        return (np.array([S + (E - S) * f for f in np.linspace(0.45, 1, 6)]),
                np.array([E + (W - E) * f for f in np.linspace(0, 1, 8)]))

    def residual(x):
        r = []
        for t in path:
            mats = pose_at(x, t)
            up, fo = arm_axis(mats)
            du, _ = torso.query(up)
            dfo, _ = torso.query(fo)
            r += list(np.maximum(0, upper_r - du) * 3.0) + list(np.maximum(0, fore_r - dfo) * 3.0)
            if extra_clear is not None:
                r += list(extra_clear(mats))
        # drag: filter at the lips, cigarette pointing away from the face, elbow out and forward
        mats = pose_at(x, 1.0)
        w = mats[hand] @ cig_m
        filt = w[:3, :3] @ filt_l + w[:3, 3]
        tip = w[:3, :3] @ tip_l + w[:3, 3]
        hw = mats[head]
        lips = hw[:3, :3] @ lips_head + hw[:3, 3]
        fwd = hw[:3, :3] @ hd.F
        out = (tip - filt) / np.linalg.norm(tip - filt)
        r += list((filt - lips) * 3.0)
        r += [max(0.0, 0.55 - out @ fwd) * 4.0]
        r += list((mats[el][:3, 3] - elbow_goal) * 0.5)    # elbow forward and out, not across the chest
        # resting: upper arm hangs at his side like the other arm, elbow bent about a right angle
        # so the hand holds the cigarette out in front of his belly, pointing forward
        mats = pose_at(x, 0.0)
        w = mats[hand] @ cig_m
        hold_dir = w[:3, :3] @ np.array([1.0, 0, 0])
        r += [max(0.0, 0.3 - hold_dir[0]) * 2.0]
        if label == IDLE:
            r += list((mats[el][:3, 3] - elbow_rest) * 0.8)
            r += [max(0.0, mats[el][2, 3] + 2.0 - mats[hand][2, 3]) * 1.0]
        else:
            # sitting: the hand rests just above his left thigh, where the sit animation has it
            r += list((mats[hand][:3, 3] - hand_rest) * 0.8)
        # elbow is a hinge: keep the bend between straight and fully folded
        hold, drag = cig_pose_params(x)
        for fa in (hold["left forearm"][1], hold["left forearm"][1] + drag["left forearm"][1]):  # hinge
            r += [max(0.0, fa) * 0.5, max(0.0, -150.0 - fa) * 0.5]
        # pac blends the angles linearly, so a big wrist/forearm-twist offset spins the hand on the way
        # up; twisting also wrings the skin around the wrist and elbow, so keep both moderate
        for v in (np.array(drag["left hand"]), np.array(hold["left hand"]) + np.array(drag["left hand"]) - hold0[5:8]):
            r += list(v * 0.04) + list(np.maximum(0, np.abs(v) - 70) * 0.5)
        for tw in (hold["left forearm"][2], hold["left forearm"][2] + drag["left forearm"][2]):
            r += [tw * 0.03, max(0.0, abs(tw) - 70) * 0.5]
        # sitting: a big upper-arm offset makes the blended path swing and twist the whole arm on the
        # way up (standing keeps its own, already settled, solution)
        if label != IDLE:
            r += list(np.maximum(0, np.abs(np.array(drag["left upperarm"])) - 75) * 0.5)
        r += list((x - x0) * 0.01)
        return np.array(r)

    old = [fit.cig_arm[n] for n in CIG_NAMES]
    hold0 = np.array([*old[0][0], old[1][0][1], old[1][0][2], *old[2][0]], float)
    peak0 = hold0 + np.array([*old[0][1], old[1][1][1], old[1][1][2], *old[2][1]], float)

    # Stage 1, the drag pose alone, pulled toward an elbow placed by triangle geometry: with the
    # filter at his lips the elbow lies on a circle around the shoulder->lips line; take the point
    # 30 degrees from straight down toward forward-left (forward and out, below the shoulder).
    m0 = model.pose(label, base)
    S = m0[sh][:3, 3]
    lips0 = m0[head][:3, :3] @ lips_head + m0[head][:3, 3]
    a = np.linalg.norm(m0[el][:3, 3] - S)
    b = np.linalg.norm(m0[hand][:3, 3] - m0[el][:3, 3]) + np.linalg.norm(cig_m[:3, :3] @ filt_l + cig_m[:3, 3])
    c = np.linalg.norm(lips0 - S)
    u = (lips0 - S) / c
    cos_t = np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1)
    v1 = np.array([0, 0, -1.0]) - (np.array([0, 0, -1.0]) @ u) * u
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    if v2[1] < 0:
        v2 = -v2                                             # toward his left
    phi = np.radians(30)
    elbow_goal = S + a * (cos_t * u + np.sqrt(1 - cos_t ** 2) * (np.cos(phi) * v1 + np.sin(phi) * v2))
    hang = np.array([0.1, 0.18, -1.0])                       # same hang as the other arm (solve_arm_hang)
    elbow_rest = S + a * hang / np.linalg.norm(hang)
    hand_rest = m0[hand][:3, 3] + np.array([1.0, 0, 2.5])     # sitting: just above where it lies on the thigh
    if rest_target is not None:
        hand_rest = np.asarray(rest_target, float)

    def drag_pose(y):
        b3 = dict(base)
        b3.update({n: ((0, 0, 0), a) for n, a in cig_split(y).items()})
        return model.pose(label, b3)

    def r_peak(y):
        mats = drag_pose(y)
        w = mats[hand] @ cig_m
        filt = w[:3, :3] @ filt_l + w[:3, 3]
        tip = w[:3, :3] @ tip_l + w[:3, 3]
        hw = mats[head]
        out = (tip - filt) / np.linalg.norm(tip - filt)
        up, fo = arm_axis(mats)
        return np.array([*((filt - (hw[:3, :3] @ lips_head + hw[:3, 3])) * 3.0),
                         *((mats[el][:3, 3] - elbow_goal) * 0.8),
                         *((y[5:8] - hold0[5:8]) * 0.04),           # wrist turns little from the hold
                         *(np.maximum(0, np.abs(y[5:8] - hold0[5:8]) - 70) * 0.5),
                         y[4] * 0.03, max(0.0, abs(y[4]) - 70) * 0.5,  # moderate forearm twist
                         max(0.0, 0.55 - out @ (hw[:3, :3] @ hd.F)) * 4.0,
                         *(np.maximum(0, upper_r - torso.query(up)[0]) * 3.0),
                         *(np.maximum(0, fore_r - torso.query(fo)[0]) * 3.0),
                         *((np.maximum(0, np.abs(y[0:3] - hold0[0:3]) - 75) * 0.5) if label != IDLE else ())])

    best = None
    starts = [peak0]
    for k, d in ((0, 30), (0, -30), (1, 40), (2, 40), (2, -40), (4, 45), (4, -45)):
        st = peak0.copy()
        st[k] += d
        starts.append(st)
    for start in starts:
        y, cost = lm(r_peak, start, iters=80)
        if best is None or cost < best[1]:
            best = (y, cost)
    peak = best[0]

    # Stage 2: hold pose and drag offset together, clearance checked along the blend
    x0 = np.concatenate([hold0, peak - hold0])
    x, cost = lm(residual, x0, iters=80)
    if label != IDLE:
        # sitting: also start from the seated animation's own arm (hand on the thigh) as the hold
        alt0 = np.concatenate([np.zeros(8), peak])
        x0_saved = x0
        x0 = alt0
        xa, ca = lm(residual, alt0, iters=80)
        if ca < cost:
            x, cost = xa, ca
        else:
            x0 = x0_saved
        if start_from:
            st = [np.array([*start_from["left upperarm"][k], *start_from["left forearm"][k][1:],
                            *start_from["left hand"][k]], float) for k in (0, 1)]
            xs, cs = lm(residual, np.concatenate(st), iters=80)
            if cs < cost:
                x, cost = xs, cs
    hold, drag = cig_pose_params(np.round(x, 2))
    setattr(fit, attr, {n: (tuple(float(a) for a in hold[n]), tuple(float(a) for a in drag[n])) for n in CIG_NAMES})
    mats = pose_at(x, 1.0)
    w = mats[hand] @ cig_m
    hw = mats[head]
    report = {"filter": w[:3, :3] @ filt_l + w[:3, 3], "lips": hw[:3, :3] @ lips_head + hw[:3, 3], "cost": cost,
              "elbow_vs_chest": mats[el][:3, 3] - mats[chest_b][:3, 3], "clearance": []}
    for t in path:
        m_t = pose_at(x, t)
        up, fo = arm_axis(m_t)
        report["clearance"].append((t, float(torso.query(up)[0].min()), float(torso.query(fo)[0].min())))
    setattr(fit, attr.replace("arm", "report"), report)
    # tip of the cigarette in the hand frame, for the ember and smoke
    fit.cig_hand_tip = tuple(np.round(cig_m[:3, :3] @ tip_l + cig_m[:3, 3], 4))


def solve_cig_bow(fit, model, obstacles, label=SIT, attr="cig_arm_sit", torso_extra=("L_Thigh", "R_Thigh"),
                  path=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), gap=0.8, skin_gap=0.5, upper_r=3.0,
                  fore_r=2.4, extra_clear=None):
    """A bow in the drag's path, keeping its two ends (the hold and the drag) as solved.

    pac blends the arm's bone3 angles straight from the hold to the drag; seated, that line takes
    his left hand through his right fist and the cane's handle. A third set of angles, `bow`, is added
    as bow * 4s(1 - s) (s = the drag blend, so it is zero at both ends and full halfway), solved so
    the hand, fingers and cigarette stay `gap` off the obstacles (cane, right hand and forearm) and
    `skin_gap` off his belly and thighs, and the arm stays off his torso. fit.<attr> entries become
    (hold, drag, bow)."""
    from make_outfit import CIG_FINGERS
    from fit_male08 import skin_samples
    hand = model.bone("ValveBiped.Bip01_L_Hand")
    sh, el = model.bone("ValveBiped.Bip01_L_UpperArm"), model.bone("ValveBiped.Bip01_L_Forearm")
    cig_m = part_matrix(*fit.cig_hand)
    cig_pts = np.array([[-0.45 + 3.3 * f_, 0, 0] for f_ in np.linspace(0, 1, 8)])   # filter -> tip axis
    base = posture_bone3(fit, smoke=True)
    base.update({b: ((0, 0, 0), a) for b, a in CIG_FINGERS.items()})
    m0 = model.pose(label, base)
    torso = torso_tree(model, m0, torso_extra)
    skin = torso_tree(model, m0, torso_extra, dense=True)
    obst = cKDTree(obstacles)
    # the left hand and its fingers as they hold the cigarette, in the hand's frame
    hb = {k for k, n in enumerate(model.names) if "L_Finger" in n or n.endswith("L_Hand")}
    inv = np.linalg.inv(m0[hand])
    hp = skin_samples(model, m0, hb, 0.3) @ inv[:3, :3].T + inv[:3, 3]
    _, keep = np.unique(np.round(hp / 0.5), axis=0, return_index=True)
    hp = hp[keep]
    near = lambda tree, pts, r: np.minimum(tree.query(pts, distance_upper_bound=r)[0], r)   # only what's close
    hold, drag = ([np.array([*getattr(fit, attr)[n][k]], float) for n in CIG_NAMES] for k in (0, 1))
    hold_v = np.array([*hold[0], hold[1][1], hold[1][2], *hold[2]])
    drag_v = np.array([*drag[0], drag[1][1], drag[1][2], *drag[2]])

    def pose_at(bow, s_):
        b = dict(base)
        b.update({n: ((0, 0, 0), a) for n, a in cig_split(hold_v + s_ * drag_v + 4 * s_ * (1 - s_) * bow).items()})
        return model.pose(label, b)

    def gaps(mats):
        hm, cm = mats[hand], mats[hand] @ cig_m
        h = hp @ hm[:3, :3].T + hm[:3, 3]
        c = cig_pts @ cm[:3, :3].T + cm[:3, 3]
        S, E, W = mats[sh][:3, 3], mats[el][:3, 3], mats[hand][:3, 3]
        up = np.array([S + (E - S) * f_ for f_ in np.linspace(0.45, 1, 6)])
        fo = np.array([E + (W - E) * f_ for f_ in np.linspace(0, 1, 8)])
        return (near(obst, h, 2 * gap), near(obst, c, 2 * gap) - 0.15, near(skin, h, 2 * skin_gap),
                near(skin, c, 2 * skin_gap) - 0.15, near(torso, up, 2 * upper_r), near(torso, fo, 2 * fore_r),
                near(obst, fo, 2 * fore_r))

    def residual(bow):
        r = []
        for s_ in path:
            mats = pose_at(bow, s_)
            dh, dc, sh_, sc, du, df, dfo = gaps(mats)
            if extra_clear is not None:
                r += list(extra_clear(mats))
            # one number per check and sample: the root of the summed squared intrusions
            for d, lim, wgt in ((dh, gap, 3.0), (dc, gap, 3.0), (sh_, skin_gap, 3.0), (sc, skin_gap, 3.0),
                                (du, upper_r, 2.0), (df, fore_r, 2.0), (dfo, fore_r, 2.0)):
                r.append(np.sqrt(np.sum(np.maximum(0, lim - d) ** 2)) * wgt)
        r += list(bow * 0.03) + list(np.maximum(0, np.abs(bow) - 60) * 0.5)
        return np.array(r)

    best = None
    for start in (np.zeros(8), np.array([0, -25, 0, 0, 0, 0, 0, 0.0]), np.array([0, 25, 0, 0, 0, 0, 0, 0.0]),
                  np.array([-20, 0, 0, 0, 0, 0, 0, 0.0])):
        bow, cost = lm(residual, start, iters=50)
        if best is None or cost < best[1]:
            best = (bow, cost)
    bow, cost = best
    bow = np.round(bow, 2)
    split = cig_split(bow)
    setattr(fit, attr, {n: (*getattr(fit, attr)[n][:2], tuple(float(a) for a in split[n])) for n in CIG_NAMES})
    rows = []
    for s_ in (0.0, *path, 1.0):
        g = gaps(pose_at(bow, s_))
        rows.append((s_, *(float(v.min()) for v in g)))
    getattr(fit, attr.replace("arm", "report"))["bow"] = dict(cost=cost, bow=bow, rows=rows)


def _local_skin(model, mats, bone_idx, owners, min_weight=0.3, cell=0.35):
    """Skin of the `owners` bones, densely sampled, in bone `bone_idx`'s frame (thinned to one
    point per `cell`)."""
    from fit_male08 import skin_samples
    inv = np.linalg.inv(mats[bone_idx])
    p = skin_samples(model, mats, owners, min_weight) @ inv[:3, :3].T + inv[:3, 3]
    _, keep = np.unique(np.round(p / cell), axis=0, return_index=True)
    return p[keep]


def solve_cast_rest(fit, model, label=SIT, obstacles=None, touch=0.15, clear=0.15):
    """Sitting with the leg cast on. The sit animation rests his left hand on his left thigh, which
    the plaster now covers, so the hand sank into it. Small extra arm angles set it on top instead.

    pac applies several bone parts on one bone in the order they were last shown, and rotations
    don't commute, so the lift never adds a second part to a bone that already has one:

    - fit.cast_lift_sit (no cigarette; the arm-hang parts are on): more elbow bend - a yaw, the same
      axis as the hang's elbow part, so the two commute - and a turn of the hand (no other part moves
      the left hand while he sits). Hand resting `touch` above the plaster, none of it or the
      forearm inside.
    - fit.cig_arm_sit_cast (holding the cigarette): its own seated solve_cig + solve_cig_bow (a copy
      of the seated cigarette shown only with the cast on), resting where the bare hand rests on the
      plaster, with the hand, forearm and cigarette kept out of it at rest and all the way up."""
    import leg_cast as lc
    from make_outfit import CIG_FINGERS
    tube = fit.cast_parts["thigh"]
    tb = model.bone("ValveBiped.Bip01_L_Thigh")
    hand, fore = model.bone("ValveBiped.Bip01_L_Hand"), model.bone("ValveBiped.Bip01_L_Forearm")
    hand_bones = {k for k, n in enumerate(model.names) if "L_Finger" in n or n.endswith("L_Hand")}
    fore_bones = {k for k, n in enumerate(model.names) if any(t in n for t in ("L_Forearm", "L_Wrist", "L_Ulna"))}

    def margins(mats, pts_by_bone):
        inv = np.linalg.inv(mats[tb])
        out = []
        for b, pts in pts_by_bone:
            w = pts @ mats[b][:3, :3].T + mats[b][:3, 3]
            out.append(lc.margin(tube, w @ inv[:3, :3].T + inv[:3, 3]))
        mg = np.concatenate(out)
        return mg[np.isfinite(mg)]

    def rest_terms(mats, pts_by_bone, touch_w=1.0):
        mg = margins(mats, pts_by_bone)
        near = min(-float(mg.max()), 2.0) if len(mg) else 2.0          # gap from the hand to the plaster
        return [np.sqrt(np.sum(np.maximum(0, mg + clear) ** 2)) * 4.0, (near - touch) * touch_w], near

    # --- no cigarette ------------------------------------------------------------------------
    base = posture_bone3(fit)
    elbow0 = base["left forearm"][1][1]

    def plain_pose(x):
        b = dict(base)
        b["left forearm"] = ((0, 0, 0), (0, elbow0 + x[0], 0))
        b["left hand"] = ((0, 0, 0), tuple(x[1:4]))
        return model.pose(label, b)

    m0 = plain_pose(np.zeros(4))
    pts = [(hand, _local_skin(model, m0, hand, hand_bones)), (fore, _local_skin(model, m0, fore, fore_bones))]

    def residual(x):
        r, _ = rest_terms(plain_pose(x), pts)
        return np.array(r + list(x * 0.03) + [max(0.0, elbow0 + x[0]) * 0.5])

    x, cost = lm(residual, np.zeros(4), iters=60)
    x = np.round(x, 2)
    fit.cast_lift_sit = {"left forearm": (0, float(x[0]), 0), "left hand": tuple(float(a) for a in x[1:4])}
    sunk = margins(m0, pts)
    fit.cast_rest_report = dict(cost=cost, sunk_before=float(sunk.max()), gap_after=rest_terms(plain_pose(x), pts)[1])

    # --- holding the cigarette: its own seated solve, the hand resting on the plaster ------------
    if not getattr(fit, "cig_arm_sit", None):
        return
    cig_m = part_matrix(*fit.cig_hand)
    cig_pts = np.array([[-0.45 + 3.3 * f_, 0, 0] for f_ in np.linspace(0, 1, 8)]) @ cig_m[:3, :3].T + cig_m[:3, 3]
    cbase = posture_bone3(fit, smoke=True)
    cbase.update({b: ((0, 0, 0), a) for b, a in CIG_FINGERS.items()})
    m1 = model.pose(label, cbase)
    cpts = [(hand, _local_skin(model, m1, hand, hand_bones)), (fore, _local_skin(model, m1, fore, fore_bones))]

    def cast_clear(mats):
        mg, mc = margins(mats, cpts), margins(mats, [(hand, cig_pts)])
        return [np.sqrt(np.sum(np.maximum(0, mg + clear) ** 2)) * 4.0,
                np.sqrt(np.sum(np.maximum(0, mc + 0.25) ** 2)) * 4.0]          # the cigarette's axis, 0.1 off

    rest = plain_pose(x)[hand][:3, 3] + np.array([0.0, 0.0, 0.6])     # where the bare hand rests on it
    solve_cig(fit, model, label=label, attr="cig_arm_sit_cast", torso_extra=("L_Thigh", "R_Thigh"),
              path=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0), rest_target=rest,
              extra_clear=cast_clear, start_from=fit.cig_arm_sit)
    solve_cig_bow(fit, model, obstacles, label=label, attr="cig_arm_sit_cast", extra_clear=cast_clear)
    arm = fit.cig_arm_sit_cast
    vec = lambda k: np.array([*arm["left upperarm"][k], *arm["left forearm"][k][1:], *arm["left hand"][k]], float)
    deepest = []
    for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = dict(cbase)
        b.update({n: ((0, 0, 0), a) for n, a in
                  cig_split(vec(0) + s_ * vec(1) + 4 * s_ * (1 - s_) * vec(2)).items()})
        mats = model.pose(label, b)
        mg, mc = margins(mats, cpts), margins(mats, [(hand, cig_pts)])
        deepest.append((s_, float(mg.max()) if len(mg) else None, float(mc.max()) if len(mc) else None))
    fit.cast_rest_report.update(cig_rest_target=rest, cig_in_plaster=deepest)


WATCH_LOOK_DOWN = 20.0          # degrees his head tips down to read the watch


def solve_watch_check(fit, model, upper_r=3.0, fore_r=2.6, path=(0.33, 0.66, 1.0)):
    """Left arm offsets (on top of the hanging arm) that bring the watch from his pocket to in front
    of his chest, face toward his eyes, while his head tips down WATCH_LOOK_DOWN degrees. pac blends
    the offsets in from zero, so the arm is kept off his torso along the whole motion, and the chain
    from the pocket to the watch stays a believable length."""
    import pocket_watch as pw
    hand = model.bone("ValveBiped.Bip01_L_Hand")
    head = model.bone(HEAD)
    pel = model.bone("ValveBiped.Bip01_Pelvis")
    sh, el = model.bone("ValveBiped.Bip01_L_UpperArm"), model.bone("ValveBiped.Bip01_L_Forearm")
    base = posture_bone3(fit)                                  # cigarette off: the left arm hangs
    torso = torso_tree(model, model.pose(IDLE, base))
    hd = fit.head
    watch_m = part_matrix(*fit.watch_hand)
    bow_l = np.array(pw.bow_point())
    pocket_l = np.array(fit.watch_pocket[0])

    def items(x, t):
        d = cig_split(x * t)
        out = [(b, p_, a) for b, (p_, a) in base.items()]
        out += [(n, (0, 0, 0), d[n]) for n in CIG_NAMES]
        out.append(("head", (0, 0, 0), (0, -WATCH_LOOK_DOWN * t, 0)))
        return out

    def arm_axis(mats):
        S, E, W = mats[sh][:3, 3], mats[el][:3, 3], mats[hand][:3, 3]
        return (np.array([S + (E - S) * f for f in np.linspace(0.45, 1, 6)]),
                np.array([E + (W - E) * f for f in np.linspace(0, 1, 8)]))

    def residual(x):
        r = []
        for t in path:
            mats = model.pose(IDLE, items(x, t))
            up, fo = arm_axis(mats)
            r += list(np.maximum(0, upper_r - torso.query(up)[0]) * 3.0)
            r += list(np.maximum(0, fore_r - torso.query(fo)[0]) * 3.0)
        mats = model.pose(IDLE, items(x, 1.0))
        w = mats[hand] @ watch_m
        hw = mats[head]
        eyes = hw[:3, :3] @ hd.origin + hw[:3, 3]
        fwd, upv, left = hw[:3, :3] @ hd.F, hw[:3, :3] @ hd.U, hw[:3, :3] @ hd.L
        gaze = np.cos(np.radians(58)) * fwd - np.sin(np.radians(58)) * upv
        target = eyes + gaze * 10.0 + left * 1.5
        to_eyes = (eyes - w[:3, 3]) / np.linalg.norm(eyes - w[:3, 3])
        r += list((w[:3, 3] - target) * 1.5)
        r += [max(0.0, 0.85 - w[:3, 0] @ to_eyes) * 6.0]                      # face toward his eyes
        r += [max(0.0, -(w[:3, 2] @ fwd)) * 1.0]                              # 12 o'clock away from him
        # a reading arm, not a reaching one: elbow well below the shoulder, by his side, not thrown out front
        S, E = mats[sh][:3, 3], mats[el][:3, 3]
        r += [max(0.0, E[2] - (S[2] - 7.0)) * 2.0, max(0.0, E[0] - (S[0] + 5.0)) * 2.0,
              max(0.0, (S[1] - 1.0) - E[1]) * 2.0]
        bow = w[:3, :3] @ bow_l + w[:3, 3]
        pocket = mats[pel][:3, :3] @ pocket_l + mats[pel][:3, 3]
        r += [max(0.0, np.linalg.norm(bow - pocket) - 18.0) * 1.0]         # chain length
        d = cig_split(x)
        tw = d["left forearm"][2]
        r += [tw * 0.03, max(0.0, abs(tw) - 70) * 0.5]
        hnd = np.array(d["left hand"])
        r += list(hnd * 0.02) + list(np.maximum(0, np.abs(hnd) - 70) * 0.5)
        fa = d["left forearm"][1]
        r += [max(0.0, fa) * 0.5, max(0.0, -140.0 - fa) * 0.5]
        r += list(x * 0.003)
        return np.array(r)

    best = None
    for start in ([0, 0, 0, -90, 0, 0, 0, 0], [20, 0, 0, -100, 0, 0, 0, 0], [-20, 0, 0, -100, 0, 0, 0, 0],
                  [0, -40, 0, -100, 30, 0, 0, 0], [0, 40, 0, -100, -30, 0, 0, 0], [0, 0, 40, -110, 0, 0, 0, 0],
                  [0, 0, -40, -110, 0, 0, 0, 0]):
        x, cost = lm(residual, np.array(start, float), iters=80)
        if best is None or cost < best[1]:
            best = (x, cost)
    x, cost = best
    fit.watch_delta = {n: tuple(round(float(a), 2) for a in ang) for n, ang in cig_split(x).items()}
    mats = model.pose(IDLE, items(x, 1.0))
    w = mats[hand] @ watch_m
    hw = mats[head]
    eyes = hw[:3, :3] @ hd.origin + hw[:3, 3]
    # turn the watch in his palm (about its own face) so 12 o'clock points away from him as he reads
    # it; its centre and facing, which the solve placed, stay the same
    face = w[:3, 0]
    away = hw[:3, :3] @ hd.F                   # his forward: on the dial that is its far edge
    twelve = away - (away @ face) * face
    twelve /= np.linalg.norm(twelve)
    r_world = np.column_stack([face, np.cross(twelve, face), twelve])
    r_hand = mats[hand][:3, :3].T @ r_world
    fit.watch_hand = (fit.watch_hand[0], tuple(round(float(a), 3) for a in mk.matrix_to_source_angle(r_hand)))
    watch_m = part_matrix(*fit.watch_hand)
    w = mats[hand] @ watch_m
    bow = w[:3, :3] @ bow_l + w[:3, 3]
    pocket = mats[pel][:3, :3] @ pocket_l + mats[pel][:3, 3]
    clear = []
    for t in path:
        m_t = model.pose(IDLE, items(x, t))
        up, fo = arm_axis(m_t)
        clear.append((t, float(torso.query(up)[0].min()), float(torso.query(fo)[0].min())))
    # second hand on the hand bone: at the subdial, the watch's own angles (the proxy turns its roll,
    # which is a turn about the dial's axis); bow marker pointing down at the check so the chain sags
    sub = watch_m @ part_matrix(pw.SUBDIAL, (0, 0, 0))
    fit.watch_second_hand = (tuple(np.round(sub[:3, 3], 3)), tuple(round(float(a), 3) for a in fit.watch_hand[1]))
    down_in_hand = mats[hand][:3, :3].T @ np.array([0, 0, -1.0])
    bow_in_hand = watch_m[:3, :3] @ bow_l + watch_m[:3, 3]
    side = np.cross(down_in_hand, [0, 0, 1.0])
    side = side if np.linalg.norm(side) > 1e-6 else np.array([0, 1.0, 0])
    from fit_male08 import angles_from_axes
    fit.watch_bow_marker = (tuple(np.round(bow_in_hand, 3)), angles_from_axes(down_in_hand, np.cross(side, down_in_hand)))
    fit.watch_look_down = WATCH_LOOK_DOWN
    fit.watch_report = dict(cost=cost, eye_dist=float(np.linalg.norm(eyes - w[:3, 3])),
                            facing=float(w[:3, 0] @ ((eyes - w[:3, 3]) / np.linalg.norm(eyes - w[:3, 3]))),
                            chain=float(np.linalg.norm(bow - pocket)), clearance=clear)


def _rot_angle(ang):
    """Size, in degrees, of the rotation a bone3 Angles applies."""
    r = mk.source_angle_matrix(*ang)
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def sit_grip(fit):
    """The standing grip turned 90 degrees about the handle (the hand's z axis runs along it in the
    fist): the fist stays round the handle but the shaft drops out of the palm side instead of along
    the fingers, so a palm-down hand resting on top of the handle stands the cane up. Holding it the
    standing way would need the wrist bent ~90 degrees with the forearm level."""
    m = part_matrix(*fit.cane_hand)
    rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # -90 about z
    return (tuple(round(float(a), 3) for a in m[:3, 3]),
            tuple(round(float(a), 3) for a in mk.matrix_to_source_angle(rz @ m[:3, :3])))


def seated_cane_obstacles(fit, model, label=SIT, radius=0.42):
    """Surface points of the cane standing between his knees and of the right forearm and hand
    resting on its handle (the seated cane pose): the seated cigarette arm keeps clear of them."""
    from fit_male08 import skin_samples
    b = posture_bone3(fit, cane=True)
    b.update({n: ((0, 0, 0), a) for n, a in fit.cane_arm_sit.items()})
    mats = model.pose(label, b)
    w = mats[model.bone("ValveBiped.Bip01_R_Hand")] @ part_matrix(*fit.cane_hand_sit)
    tip_l, _ = cane_geometry(fit.cane_scale)
    a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    ex, ey, ez = np.eye(3)
    ring = lambda c, u, v: c + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    pts = [ring(np.array([tip_l[0], tip_l[1], z]), ex, ey) for z in np.arange(tip_l[2], -0.8 * fit.cane_scale, 0.3)]
    pts += [ring(np.array([x, 0.0, 0.08]), ey, ez) for x in np.arange(-2.28, tip_l[0], 0.3)]     # the handle
    pts = np.vstack(pts) @ w[:3, :3].T + w[:3, 3]
    arm = {k for k, n in enumerate(model.names) if any(t in n for t in ("R_Forearm", "R_Hand", "R_Finger", "R_Wrist", "R_Ulna"))}
    return np.vstack([pts, skin_samples(model, mats, arm, 0.4)])


def solve_cane_seated(fit, model, label=SIT, clearance=1.2, max_lean=12.0, back=3.0, wrist_max=30.0,
                      upper_r=2.6, fore_r=2.2):
    """Sitting: his right hand resting on top of the cane's handle, the cane standing between his
    knees with its tip on the floor between his feet, the top leaning `back` units toward him
    (absolute bone3 angles for the right arm, like the standing grip; the seated grip is sit_grip)."""
    tip_l, top_l = cane_geometry(fit.cane_scale)
    shaft_l = np.array([top_l + (tip_l - top_l) * t for t in np.linspace(0, 0.97, 14)])
    hand = model.bone("ValveBiped.Bip01_R_Hand")
    sh, el = model.bone("ValveBiped.Bip01_R_UpperArm"), model.bone("ValveBiped.Bip01_R_Forearm")
    fit.cane_hand_sit = sit_grip(fit)
    cane_m = part_matrix(*fit.cane_hand_sit)
    base = posture_bone3(fit, cane=True)
    mats0 = model.pose(label, base)
    from fit_male08 import skin_samples
    leg_bones = {model.bone(f"ValveBiped.Bip01_{s_}_{b_}") for s_ in "LR" for b_ in ("Thigh", "Calf", "Foot", "Toe0")}
    legs = cKDTree(skin_samples(model, mats0, leg_bones, 0.5))     # dense: the knees are sparse in vertices
    body = torso_tree(model, mats0, ("L_Thigh", "R_Thigh"))
    feet_w = [np.array([sum(x for bb, x in ws if model.names[bb].endswith(("Foot", "Toe0"))) for ws in src["w"]])
              for src in model.meshes]
    floor = skin_samples(model, mats0, None, 0.5, feet_w)[:, 2].min()
    B = lambda n: mats0[model.bone(f"ValveBiped.Bip01_{n}")][:3, 3]
    mid_y = (B("L_Calf")[1] + B("R_Calf")[1]) / 2
    tip_goal = np.array([(B("L_Foot")[0] + B("R_Foot")[0]) / 2 - 1.0, mid_y, floor + 0.2])

    def pose_with(x):
        b = dict(base)
        b["right upperarm"] = ((0, 0, 0), tuple(x[0:3]))
        b["right forearm"] = ((0, 0, 0), (0, x[3], 0))
        b["right hand"] = ((0, 0, 0), tuple(x[4:7]))
        return model.pose(label, b)

    def residual(x):
        mats = pose_with(x)
        w = mats[hand] @ cane_m
        tip = w[:3, :3] @ tip_l + w[:3, 3]
        top = w[:3, :3] @ top_l + w[:3, 3]
        d = (top - tip) / np.linalg.norm(top - tip)
        lean = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
        gap, _ = legs.query(shaft_l @ w[:3, :3].T + w[:3, 3])
        S, E, W = mats[sh][:3, 3], mats[el][:3, 3], mats[hand][:3, 3]
        up = np.array([S + (E - S) * f for f in np.linspace(0.45, 1, 6)])
        fo = np.array([E + (W - E) * f for f in np.linspace(0, 0.8, 6)])
        return np.array([*((tip - tip_goal) * np.array([1.5, 1.5, 3.0])),
                         (top[1] - mid_y) * 1.5,                      # handle centred between the knees too
                         (top[0] - (tip[0] - back)) * 0.5,            # leaning a little back toward him
                         # a resting reach: elbow low and to his right, not cocked up or behind him
                         max(0.0, E[2] - (S[2] - 6.0)) * 2.0, max(0.0, E[1] - (S[1] + 2.0)) * 2.0,
                         max(0.0, (S[0] - 1.0) - E[0]) * 2.0,
                         max(0.0, lean - max_lean) * 0.5,
                         *(np.maximum(0.0, clearance - gap) * 3.0),
                         *(np.maximum(0.0, upper_r - body.query(up)[0]) * 2.0),
                         *(np.maximum(0.0, fore_r - body.query(fo)[0]) * 2.0),
                         max(0.0, x[3]) * 0.5,                       # the elbow only bends one way
                         max(0.0, _rot_angle(x[4:7]) - wrist_max) * 0.5,   # wrist near the sit's own
                         *((x - x0) * 0.008)])

    x0 = np.array([*fit.cane_arm["right upperarm"], fit.cane_arm["right forearm"][1], 0, 0, 0], float)
    best = None
    for dx in ((0, 0, 0, 0, 0, 0, 0), (0, -30, 0, 0, 0, 0, 0), (0, 0, 30, -40, 0, 0, 0), (0, -60, 0, -10, 0, 0, 0),
               (20, -50, 20, -15, 0, 0, 0), (-20, -50, -20, -15, 0, 0, 0)):
        x, cost = lm(residual, x0 + np.array(dx, float), iters=80)
        if best is None or cost < best[1]:
            best = (x, cost)
    x, cost = best
    fit.cane_arm_sit = {"right upperarm": tuple(np.round(x[0:3], 2)), "right forearm": (0, round(float(x[3]), 2), 0),
                        "right hand": tuple(np.round(x[4:7], 2))}
    mats = pose_with(x)
    w = mats[hand] @ cane_m
    tip = w[:3, :3] @ tip_l + w[:3, 3]
    top = w[:3, :3] @ top_l + w[:3, 3]
    d = (top - tip) / np.linalg.norm(top - tip)
    gap, _ = legs.query(shaft_l @ w[:3, :3].T + w[:3, 3])
    fit.cane_report_sit = dict(cost=cost, tip=tip, top=top, goal=tip_goal, mid_y=mid_y, floor=floor,
                               lean=float(np.degrees(np.arccos(d[2]))), leg_gap=float(gap.min()),
                               elbow=mats[el][:3, 3] - mats[sh][:3, 3], wrist=_rot_angle(x[4:7]))


def head_pitch(mats, model, hd):
    """World pitch of the face direction (positive = looking down) and the neck bend."""
    hw = mats[model.bone(HEAD)]
    f = hw[:3, :3] @ hd.F
    return np.degrees(np.arctan2(-f[2], np.hypot(f[0], f[1])))


def aim_report(fit, model, holdtypes=("normal", "pistol", "ar2", "smg", "melee", "passive", "fist"),
               pitches=(-75, -45, -20, 0, 20, 45, 75), bone3_for=None):
    """Head pitch per aim pitch: animation alone vs with the posture."""
    hd = fit.head
    rows = []
    for h in holdtypes:
        for p in pitches:
            label = f"aim_{h}_{p}"
            if label not in model.poses:
                continue
            anim = head_pitch(model.pose(label), model, hd)
            b3 = bone3_for(h, p) if bone3_for else posture_bone3(fit, arms=False)
            post = head_pitch(model.pose(label, b3), model, hd)
            rows.append((h, p, anim, post))
    return rows


def solve_arm_hang(fit, model, elbow=-12.0):
    """Upper arm straight down beside the body (slightly forward and out) under the hunch.
    Only the swing (pitch/yaw) is solved; no roll so the sleeve doesn't twist, fixed gentle elbow."""
    base = posture_bone3(fit, smoke=True, cane=True)      # hunch + clavicles, no hang parts
    fit.arm_hang = {}
    for side in ("left", "right"):
        S = "L" if side == "left" else "R"
        ua, hand = model.bone(f"ValveBiped.Bip01_{S}_UpperArm"), model.bone(f"ValveBiped.Bip01_{S}_Hand")
        sgn = 1 if side == "left" else -1
        target = np.array([0.0, 0.18 * sgn, -1.0])
        target /= np.linalg.norm(target)

        def pose_with(x):
            b = dict(base)
            b[f"{side} upperarm"] = ((0, 0, 0), (x[0], x[1], 0.0))
            b[f"{side} forearm"] = ((0, 0, 0), (0, elbow, 0))
            return model.pose(IDLE, b)

        def residual(x):
            return (pose_with(x)[ua][:3, 0] - target) * 10.0

        x, cost = lm(residual, np.array([0.0, -20.0]))
        mats = pose_with(x)
        fit.arm_hang[side] = {"upperarm": (round(float(x[0]), 2), round(float(x[1]), 2), 0.0), "forearm": (0, elbow, 0),
                              "dir": np.round(mats[ua][:3, 0], 3), "hand": np.round(mats[hand][:3, 3], 2), "cost": cost}


def solve_cane_upright(fit, model, reach=400.0):
    """Stops the cane swinging with the walk cycle (it poked forward and across him mid-stride).
    The cane part aims (pac AimPart) at a marker far ahead on the pelvis, so it keeps its standing
    lean whatever the arm does and only its grip follows the hand. pac adds the aim direction's
    angles to the part's Angles; the marker sits straight ahead of the standing grip, where those
    aim angles are zero, so the cane's Angles are just its standing world angles."""
    b = posture_bone3(fit, cane=True)
    b.update({name: ((0, 0, 0), ang) for name, ang in fit.cane_arm.items()})
    mats = model.pose(IDLE, b)
    w = mats[model.bone("ValveBiped.Bip01_R_Hand")] @ part_matrix(*fit.cane_hand)
    pelvis = mats[model.bone("ValveBiped.Bip01_Pelvis")]
    fit.cane_aim = ("pelvis", tuple(np.round(to_local(pelvis, w[:3, 3] + np.array([reach, 0.0, 0.0])), 3)))
    fit.cane_upright_ang = tuple(round(float(a), 3) for a in mk.matrix_to_source_angle(w[:3, :3]))


def solve(fit, model):
    solve_arm_hang(fit, model)
    solve_cane(fit, model)
    solve_cane_upright(fit, model)
    solve_cig(fit, model)
    solve_watch_check(fit, model)
    if SIT in model.poses:
        solve_cane_seated(fit, model)
        solve_cig(fit, model, label=SIT, attr="cig_arm_sit", torso_extra=("L_Thigh", "R_Thigh"),
                  path=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
        obstacles = seated_cane_obstacles(fit, model)
        solve_cig_bow(fit, model, obstacles)
        if getattr(fit, "cast_parts", None):
            solve_cast_rest(fit, model, obstacles=obstacles)
