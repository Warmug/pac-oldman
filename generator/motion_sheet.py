#!/usr/bin/env python3
"""Contact sheet of the outfit in motion: walk/run cycle phases, crouch and crouch-walk.

    python3 generator/motion_sheet.py [--dump build/male08/male08_dump.txt] [--out build/previews]

Each cell is the dumped animation pose with the outfit evaluated by sim.py at that speed
(crouch poses with the is_crouching event on), seen from his right side and front-right.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sim
from male08 import Model
from render_previews import OUTFIT, REPO, BodyTextures, posed_scene

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
POSES = [("aim_normal_0", 0, "standing"), ("walk_0", 90, "walk 0%"), ("walk_0.25", 90, "walk 25%"),
         ("walk_0.5", 90, "walk 50%"), ("walk_0.75", 90, "walk 75%"), ("run_0", 200, "run 0%"),
         ("run_0.5", 200, "run 50%"), ("crouch_idle", 0, "crouch"), ("cwalk_0", 60, "crouch walk")]
VIEWS = [("right side", np.array([4, -125, 4])), ("front right", np.array([95, -80, 10]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(REPO, "build", "male08", "male08_dump.txt"))
    ap.add_argument("--out", default=os.path.join(REPO, "build", "previews"))
    ap.add_argument("--outfit", default=OUTFIT)
    ap.add_argument("--props", default="cane,smoke,glasses,mask,tank,cast")
    args = ap.parse_args()
    model = Model(args.dump)
    outfit = sim.Outfit(args.outfit)
    textures = BodyTextures(model, os.path.dirname(os.path.abspath(args.dump)))
    commands = set(filter(None, args.props.split(",")))
    cells = os.path.join(args.out, "motion")
    os.makedirs(cells, exist_ok=True)

    w, h = 360, 460
    font = ImageFont.truetype(FONT, 20)
    sheet = Image.new("RGB", (len(POSES) * w, len(VIEWS) * h + 34), (28, 30, 34))
    d = ImageDraw.Draw(sheet)
    for c, (label, speed, title) in enumerate(POSES):
        st = sim.State(commands, "normal", speed=speed, time=6.0, crouching=label.startswith("c"))
        scene, mats, _, _ = posed_scene(model, outfit, textures, label, st, w, h)
        pelvis = mats[model.bone("ValveBiped.Bip01_Pelvis")][:3, 3]
        centre = np.array([pelvis[0] + 2, 4.0, max(pelvis[2] - 6, 14)])
        names = []
        for v, (vname, offset) in enumerate(VIEWS):
            name = f"{label.replace('.', '_')}_{vname.replace(' ', '_')}"
            scene.view(name, centre + offset, centre, 40)
            names.append(name)
        paths = scene.render(cells)
        for v, p in enumerate(paths):
            sheet.paste(Image.open(p).convert("RGB"), (c * w, 34 + v * h))
        d.text((c * w + 10, 6), title, font=font, fill=(235, 235, 235))
    out = os.path.join(args.out, "motion_sheet.png")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
