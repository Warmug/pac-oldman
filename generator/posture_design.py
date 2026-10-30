"""Evaluate posture variants on the real poses without going through an outfit file."""
import numpy as np

from ingame import InGame


def pitch(v):
    return float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def angle(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


TWO_HANDED = ("smg", "ar2", "shotgun", "crossbow", "rpg", "physgun")


class Evaluator:
    def __init__(self, model, face_dir):
        self.m, self.g, self.face = model, InGame(model), face_dir
        b = model.bone
        self.i = {k: b("ValveBiped.Bip01_" + k) for k in ("Head1", "Spine4", "Pelvis", "R_UpperArm", "R_Hand", "L_Hand")}

    def aim_vec(self, mats, h):
        rh, lh = mats[self.i["R_Hand"]][:3, 3], mats[self.i["L_Hand"]][:3, 3]
        return lh - rh if h in TWO_HANDED else rh - mats[self.i["R_UpperArm"]][:3, 3]

    def run(self, h, p, bone3):
        bare = self.g.pose(h, p)
        pose = self.g.pose(h, p, bone3)
        i = self.i
        fb = bare[i["Head1"]][:3, :3] @ self.face
        fp = pose[i["Head1"]][:3, :3] @ self.face
        cb = bare[i["Spine4"]][:3, :3] @ np.array([0.0, 1, 0])
        cp = pose[i["Spine4"]][:3, :3] @ np.array([0.0, 1, 0])
        return {
            "head_err": pitch(fp) - pitch(fb),                 # face pitch vs the animation's
            "neck_add": (pitch(fp) - pitch(cp)) - (pitch(fb) - pitch(cb)),   # extra neck bend
            "neck": pitch(fp) - pitch(cp),
            "aim_dev": angle(self.aim_vec(pose, h), self.aim_vec(bare, h)),  # weapon direction change
            "hand_move": float(np.linalg.norm((pose[i["R_Hand"]][:3, 3] - pose[i["Pelvis"]][:3, 3]) - (bare[i["R_Hand"]][:3, 3] - bare[i["Pelvis"]][:3, 3]))),
        }


def posture(hunch=(3, 5, 9, 13), neck=36.0, legs=True, extra=()):
    b = [("pelvis", (0, -0.8, 0), (0, 0, 0)), ("spine", (0, 0, 0), (0, hunch[0], 0)),
         ("spine 1", (0, 0, 0), (0, hunch[1], 0)), ("spine 2", (0, 0, 0), (0, hunch[2], 0)),
         ("spine 4", (0, 0, 0), (0, hunch[3], 0)), ("neck", (0, 0, 0), (0, neck, 0))]
    if legs:
        for s in ("left", "right"):
            b += [(f"{s} thigh", (0, 0, 0), (0, -8, 0)), (f"{s} calf", (0, 0, 0), (0, 16, 0)), (f"{s} foot", (0, 0, 0), (0, -8, 0))]
    return b + list(extra)
