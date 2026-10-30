"""In-game poses: aim layer pose + the engine's head_pitch, then pac's bone3 on top.

The game drives aim_pitch (weapon aim layers) and head_pitch (the head follows the view,
in every holdtype) from the eye pitch. The export has aim poses per holdtype/pitch and
head_pitch samples on idle_all_01; head_pitch only turns Head1, so its local rotation is
lifted from those samples and layered onto any pose.
"""
import numpy as np

import meshkit as mk

HEAD = "ValveBiped.Bip01_Head1"
HEAD_PITCH_SAMPLES = (-60, -30, 0, 30, 60)


def local(model, mats, i):
    p = model.parents[i]
    return np.linalg.inv(mats[p]) @ mats[i] if p >= 0 else mats[i]


class InGame:
    def __init__(self, model):
        self.m = model
        h = model.bone(HEAD)
        base = model.pose("aim_normal_0")
        self.head_delta = {}
        for hp in HEAD_PITCH_SAMPLES:
            mats = base if hp == 0 else model.pose(f"head_normal_{hp}")
            d = np.linalg.inv(local(model, base, h)) @ local(model, mats, h)
            self.head_delta[hp] = d[:3, :3]

    def head_pitch_angles(self, eye_pitch):
        hp = float(np.clip(eye_pitch, -60, 60))
        xs = np.array(HEAD_PITCH_SAMPLES, float)
        k = int(np.clip(np.searchsorted(xs, hp) - 1, 0, len(xs) - 2))
        t = (hp - xs[k]) / (xs[k + 1] - xs[k])
        a = np.array(mk.matrix_to_source_angle(self.head_delta[HEAD_PITCH_SAMPLES[k]]))
        b = np.array(mk.matrix_to_source_angle(self.head_delta[HEAD_PITCH_SAMPLES[k + 1]]))
        return tuple(a * (1 - t) + b * t)

    def label(self, holdtype, eye_pitch):
        name = f"aim_{holdtype}_{eye_pitch}"
        if name in self.m.poses:
            return name
        avail = sorted(int(k.rsplit("_", 1)[1]) for k in self.m.poses if k.startswith(f"aim_{holdtype}_"))
        return f"aim_{holdtype}_{min(avail, key=lambda a: abs(a - eye_pitch))}"

    def pose(self, holdtype, eye_pitch, bone3=(), label=None):
        """World matrices as the game would show them, with bone3 [(bone, pos, ang)] applied by pac."""
        items = [(HEAD, (0, 0, 0), self.head_pitch_angles(eye_pitch))] + list(bone3)
        return self.m.pose(label or self.label(holdtype, eye_pitch), items)
