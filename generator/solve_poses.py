"""Re-solves the posed props on the real male_08 skeleton with pac's bone3 math.

- cane: right-arm angles so the cane tip rests on the floor beside the right foot
- cigarette: drag pose so the filter reaches the lips at the peak of the drag
- aim: how the hunch interacts with the aim layers, giving the neck/upper-back
  compensation used while aiming up/down and while a weapon is out
"""
import os

import numpy as np

import meshkit as mk
from fit_male08 import IDLE, HEAD, load_obj, posture_bone3

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


def cane_geometry():
    v, _ = load_obj(os.path.join(REPO, "props/cane.obj"))
    shaft = v[v[:, 2] < -6]
    axis = shaft[:, :2].mean(axis=0)
    tip = np.array([axis[0], axis[1], v[:, 2].min()])
    top = np.array([axis[0], axis[1], -6.0])
    return tip, top


def solve_cane(fit, model):
    tip_l, top_l = cane_geometry()
    names = ["right upperarm", "right forearm", "right hand"]
    x0 = np.array([*fit.cane_arm["right upperarm"], fit.cane_arm["right forearm"][1], *fit.cane_arm["right hand"]])
    foot = model.bone("ValveBiped.Bip01_R_Foot")
    hand = model.bone("ValveBiped.Bip01_R_Hand")
    cane_m = part_matrix(*fit.cane_hand)
    base = posture_bone3(fit, cane=True)

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
        return np.array([
            (tip[2] - 0.2) * 3.0,                 # resting on the floor
            (tip[0] - (f[0] + 7.0)) * 0.6,        # a step ahead of the right foot
            (tip[1] - (f[1] - 3.5)) * 0.6,        # a little outside it
            max(0.0, lean - 10.0) * 0.5,          # close to vertical
            *((x - x0) * 0.02),                   # stay near the old look
        ])

    x, cost = lm(residual, x0)
    fit.cane_arm = {"right upperarm": tuple(np.round(x[0:3], 2)), "right forearm": (0, round(float(x[3]), 2), 0),
                    "right hand": tuple(np.round(x[4:7], 2))}
    mats = pose_with(x)
    w = mats[hand] @ cane_m
    fit.cane_report = {"tip": w[:3, :3] @ tip_l + w[:3, 3], "foot": mats[foot][:3, 3], "cost": cost}


def solve_cig(fit, model):
    hand = model.bone("ValveBiped.Bip01_L_Hand")
    head = model.bone(HEAD)
    cig_m = part_matrix(*fit.cig_hand)
    filt_l = np.array([-0.45, 0, 0])
    tip_l = np.array([2.85, 0, 0])
    base = posture_bone3(fit, smoke=True)
    names = ["left upperarm", "left forearm", "left hand"]
    old = {n: fit.cig_arm[n] for n in names}
    d0 = np.array([*old["left upperarm"][1], old["left forearm"][1][1], *old["left hand"][1]])
    hd = fit.head
    lips_head = hd.surface(0.22, hd.mouth_v) + hd.F * 0.05

    def pose_with(d, env=1.0):
        b = dict(base)
        b["left upperarm"] = ((0, 0, 0), tuple(np.array(old["left upperarm"][0]) + env * d[0:3]))
        b["left forearm"] = ((0, 0, 0), (0, old["left forearm"][0][1] + env * d[3], 0))
        b["left hand"] = ((0, 0, 0), tuple(np.array(old["left hand"][0]) + env * d[4:7]))
        return model.pose(IDLE, b)

    def residual(d):
        mats = pose_with(d)
        w = mats[hand] @ cig_m
        filt = w[:3, :3] @ filt_l + w[:3, 3]
        tip = w[:3, :3] @ tip_l + w[:3, 3]
        hw = mats[head]
        lips = hw[:3, :3] @ lips_head + hw[:3, 3]
        fwd = hw[:3, :3] @ hd.F
        out = (tip - filt) / np.linalg.norm(tip - filt)
        return np.array([*((filt - lips) * 3.0),
                         max(0.0, 0.55 - out @ fwd) * 4.0,     # cigarette points away from the face
                         *((d - d0) * 0.02)])

    d, cost = lm(residual, d0)
    fit.cig_arm = {
        "left upperarm": (old["left upperarm"][0], tuple(np.round(d[0:3], 2))),
        "left forearm": (old["left forearm"][0], (0, round(float(d[3]), 2), 0)),
        "left hand": (old["left hand"][0], tuple(np.round(d[4:7], 2))),
    }
    mats = pose_with(d)
    w = mats[hand] @ cig_m
    hw = mats[head]
    fit.cig_report = {"filter": w[:3, :3] @ filt_l + w[:3, 3], "lips": hw[:3, :3] @ lips_head + hw[:3, 3], "cost": cost}
    # tip of the cigarette in the hand frame, for the ember and smoke
    fit.cig_hand_tip = tuple(np.round(cig_m[:3, :3] @ tip_l + cig_m[:3, 3], 4))


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


def solve(fit, model):
    solve_cane(fit, model)
    solve_cig(fit, model)
