#!/usr/bin/env python3
"""Numbers behind the posture fix, from the real male_08 animation poses.

For each holdtype and aim pitch it evaluates an outfit with sim.py (so events and
proxies are exactly what the .txt does) and reports, against the bare animation:
  head   - pitch of the face direction (positive = looking up)
  neck   - how far the head is bent back relative to the chest (spine4)
  weapon - pitch of the right-hand attachment, i.e. where a held weapon points
"""
import argparse
import os

import numpy as np

import sim
from male08 import Model

def pitch_of(v):
    return float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def measures(model, mats, face_dir_head):
    head = mats[model.bone("ValveBiped.Bip01_Head1")]
    chest = mats[model.bone("ValveBiped.Bip01_Spine4")]
    rh = mats[model.bone("ValveBiped.Anim_Attachment_RH")]
    face = head[:3, :3] @ face_dir_head
    # the chest's own "forward": the face direction expressed in the chest frame of the idle pose
    return {"head": pitch_of(face), "weapon": pitch_of(rh[:3, 0]), "head_z": float(head[2, 3]),
            "chest": pitch_of(chest[:3, :3] @ CHEST_FWD) if CHEST_FWD is not None else 0.0}


CHEST_FWD = None


def table(model, outfit, face_dir, holdtypes, pitches, commands=()):
    """aim pitch p follows gmod: positive = looking down."""
    rows = []
    for h in holdtypes:
        for p in pitches:
            label = f"aim_{h}_{p}"
            if label not in model.poses:
                continue
            bare = measures(model, model.pose(label), face_dir)
            ev = outfit.evaluate(sim.State(commands, h, pitch=p, speed=0, time=6.0))
            withp = measures(model, model.pose(label, ev["bone3"]), face_dir)
            rows.append((h, p, bare, withp))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anims", default=None)
    ap.add_argument("--outfit", required=True)
    args = ap.parse_args()
    import fit_male08 as fm
    model = Model(args.dump, anims=args.anims)
    hd = fm.HeadData(model, fm.REF)
    outfit = sim.Outfit(args.outfit)
    global CHEST_FWD
    idle = model.pose("aim_normal_0")
    chest = idle[model.bone("ValveBiped.Bip01_Spine4")]
    CHEST_FWD = chest[:3, :3].T @ np.array([1.0, 0, 0])
    pitches = [-80, -45, -15, 0, 15, 45, 80]
    print("aim: gmod eye pitch, + = looking down. head/gun/chest: world pitch, + = up")
    print(f"{'holdtype':9s} {'aim':>4s} | {'head bare':>9s} {'head pac':>9s} | {'chest bare':>10s} {'chest pac':>9s} | {'gun bare':>8s} {'gun pac':>8s} | drop")
    for h, p, bare, withp in table(model, outfit, hd.F, ["normal", "passive", "fist", "pistol", "smg", "ar2", "melee"], pitches):
        print(f"{h:9s} {p:4d} | {bare['head']:9.1f} {withp['head']:9.1f} | {bare['chest']:10.1f} {withp['chest']:9.1f} | "
              f"{bare['weapon']:8.1f} {withp['weapon']:8.1f} | {bare['head_z'] - withp['head_z']:5.1f}")


if __name__ == "__main__":
    main()
