"""Measures what an outfit's bone3 parts do to the head, neck and weapon aim in game."""
import numpy as np

import sim
from ingame import InGame


def pitch(v):
    return float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


class Metrics:
    def __init__(self, model, face_dir):
        self.m = model
        self.g = InGame(model)
        self.face = face_dir
        b = model.bone
        self.ids = {k: b("ValveBiped.Bip01_" + k) for k in ("Head1", "Spine4", "R_UpperArm", "R_Hand", "L_Hand", "R_Forearm")}

    def measure(self, mats, holdtype):
        i = self.ids
        face = mats[i["Head1"]][:3, :3] @ self.face
        chest = mats[i["Spine4"]][:3, :3] @ np.array([0.0, 1.0, 0.0])
        out = {"head": pitch(face), "neck": pitch(face) - pitch(chest), "head_z": mats[i["Head1"]][2, 3]}
        rh, lh = mats[i["R_Hand"]][:3, 3], mats[i["L_Hand"]][:3, 3]
        if holdtype in ("smg", "ar2", "shotgun", "crossbow", "rpg", "physgun"):
            out["aim"] = pitch(lh - rh)
        else:
            out["aim"] = pitch(rh - mats[i["R_UpperArm"]][:3, 3])
        out["hand_above_head"] = float(rh[2] - mats[i["Head1"]][2, 3])
        return out

    def outfit_bone3(self, outfit, holdtype, eye_pitch, commands=(), speed=0.0):
        ev = outfit.evaluate(sim.State(commands, holdtype, pitch=eye_pitch, speed=speed, time=6.0))
        return ev["bone3"]

    def compare(self, outfit, holdtype, eye_pitch, commands=()):
        bare = self.measure(self.g.pose(holdtype, eye_pitch), holdtype)
        b3 = self.outfit_bone3(outfit, holdtype, eye_pitch, commands)
        posed = self.measure(self.g.pose(holdtype, eye_pitch, b3), holdtype)
        return bare, posed
