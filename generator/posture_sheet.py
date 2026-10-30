#!/usr/bin/env python3
"""Contact sheet of the posture across holdtypes and aim pitches, old outfit vs new.

    python3 generator/posture_sheet.py --dump build/male08/male08_dump.txt --old OLD.txt --new outfit/old_man_2_male08.txt

Every cell is posed by evaluating the outfit file with sim.py. Weapon holdtypes get a
plain stand-in gun on the right-hand attachment so you can see where it points.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import meshkit as mk
import preview
import sim
from male08 import Model
from render_previews import BodyTextures, add_outfit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GUNS = {"pistol": (7.0, 1.0, 1.5), "smg": (13.0, 1.4, 2.6), "ar2": (22.0, 1.5, 3.0), "melee": (20.0, 0.8, 0.8)}


def gun_mesh(holdtype):
    if holdtype not in GUNS:
        return None
    length, width, height = GUNS[holdtype]
    body = mk.box((length, width, height), (length / 2 - 1.5, 0, 0.5))
    grip = mk.box((1.2, width * 0.9, 3.0), (0.2, 0, -1.3))
    return mk.concat([body, grip])


def cell(model, outfit, textures, holdtype, pitch, commands, out_dir, name, models=True):
    label = f"aim_{holdtype}_{pitch}"
    st = sim.State(commands, holdtype, pitch=pitch, speed=0, time=6.0)
    ev = outfit.evaluate(st)
    mats = model.pose(label, ev["bone3"])
    scene = preview.Scene(420, 520, ground_z=0.0)
    for m in model.skinned(mats):
        mat = m.material.lower()
        if "error" in mat or "mouth" in mat:
            continue
        if "eyeball" in mat:
            scene.add(m, color=[0.3, 0.24, 0.2], roughness=0.3)
            continue
        tex = textures.get(m.material)
        scene.add(m, roughness=0.75, **({"map": tex} if tex else {"color": [0.7, 0.7, 0.7]}))
    if models:
        add_outfit(scene, outfit, model, mats, ev, sim.Placer(outfit, model, mats, ev))
    gun = gun_mesh(holdtype)
    if gun is not None:
        scene.add(gun, mats[model.bone("ValveBiped.Anim_Attachment_RH")], color=[0.12, 0.12, 0.13], roughness=0.4, metalness=0.3)
    scene.view(name, (14, 78, 40), (6, 0, 36), 40)
    return scene.render(out_dir)[0]


def sheet(paths, rows, cols, title, out):
    w, h = 420, 520
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    img = Image.new("RGB", (cols * w + 150, rows * h + 60), (28, 30, 34))
    d = ImageDraw.Draw(img)
    d.text((14, 16), title, font=font, fill=(235, 235, 235))
    for (r, c, label_row, label_col), p in paths.items():
        tile = Image.open(p).convert("RGB")
        img.paste(tile, (150 + c * w, 60 + r * h))
        if c == 0:
            d.text((14, 60 + r * h + h // 2), label_row, font=small, fill=(220, 220, 220))
        if r == 0:
            d.text((150 + c * w + 12, 64), label_col, font=small, fill=(255, 255, 255))
    img.save(out)
    return out


def run(model, outfit, textures, holdtypes, pitches, out_dir, title, out_file, commands=(), models=True):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for r, h in enumerate(holdtypes):
        for c, p in enumerate(pitches):
            look = "looking up" if p < 0 else ("looking down" if p > 0 else "level")
            paths[(r, c, h, f"aim {-p:+d} ({look})")] = cell(model, outfit, textures, h, p, commands, out_dir,
                                                               f"{title.split()[0]}_{h}_{p}", models)
    return sheet(paths, len(holdtypes), len(pitches), title, out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="build/previews")
    args = ap.parse_args()
    model = Model(args.dump)
    textures = BodyTextures(model, os.path.dirname(os.path.abspath(args.dump)))
    holdtypes = ["normal", "pistol", "ar2", "melee"]
    pitches = [-45, 0, 45]
    for tag, path in (("OLD", args.old), ("NEW", args.new)):
        outfit = sim.Outfit(path)
        print(run(model, outfit, textures, holdtypes, pitches, os.path.join(args.out, tag.lower()),
                  f"{tag} posture", os.path.join(args.out, f"posture_{tag.lower()}.png"), models=(tag == "NEW")))


if __name__ == "__main__":
    main()
