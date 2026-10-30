#!/usr/bin/env python3
"""Preview renders of the generated outfit on male_08 (needs the dump).

    python3 generator/render_previews.py [--dump build/male08/male08_dump.txt] [--out build/previews]

The outfit file is read back and evaluated with sim.py (events, proxies, bone3), so the
pictures show what the .txt does, not what the generator meant to do.
"""
import argparse
import os

import numpy as np

import meshkit as mk
import preview
import sim
from fit_male08 import load_obj
from male08 import Model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTFIT = os.path.join(REPO, "outfit", "old_man_2_male08.txt")


class BodyTextures:
    """Textures decoded from the exporter (tex_<material>.png next to the dump)."""

    def __init__(self, model, dump_dir):
        self.map = {}
        for m in model.meshes:
            name = os.path.basename(m["material"]).lower()
            png = os.path.join(dump_dir, f"tex_{name}.png")
            if os.path.exists(png):
                self.map[name] = png

    def get(self, material):
        return self.map.get(os.path.basename(material).lower())


_obj_cache = {}


def obj_mesh(path):
    if path not in _obj_cache:
        v, vt, vn, faces = [], [], [], []
        for line in open(path):
            if line.startswith("v "):
                v.append(list(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                vt.append(list(map(float, line.split()[1:3])))
            elif line.startswith("vn "):
                vn.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        uv = np.array(vt)
        uv[:, 1] = 1 - uv[:, 1]
        _obj_cache[path] = mk.Mesh(np.array(v), np.array(faces), uv, np.array(vn))
    return _obj_cache[path]


def material_style(me):
    style = {"roughness": 0.5}
    if me is None:
        return style
    tex = me.get("basetexture", "")
    lp = sim.local_path(tex) if tex.startswith("http") else None
    if lp and os.path.exists(lp):
        style["map"] = lp
    if me.get("translucent"):
        style["opacity"] = 0.999          # blend with the texture's alpha; the part's Alpha scales it
        style["roughness"] = 0.12
        style["renderOrder"] = 2
    if me.get("phong"):
        style["roughness"] = 0.35
    return style


def add_outfit(scene, outfit, model, mats, ev, place):
    for uid, me in ev["nodes"].items():
        cls = me["ClassName"]
        if cls == "model2":
            path = sim.local_path(me.get("Model", ""))
            if not path or not os.path.exists(path):
                continue
            mesh = obj_mesh(path)
            style = material_style(outfit.material(me.get("Material")))
            alpha = float(me.get("Alpha", 1.0))
            if alpha < 1:
                # pac draws the model with render.SetBlend(Alpha), which replaces the material's $alpha
                style["opacity"] = alpha
                style["renderOrder"] = 2
            scale = np.array(sim.vec(me.get("Scale")) if me.get("Scale") else (1.0, 1.0, 1.0)) * float(me.get("Size", 1.0))
            scene.add(mesh, place.world(uid) @ np.diag([*scale, 1.0]), name=me.get("Name", ""), **style)
        elif cls == "beam":
            end_uid = me.get("EndPointUID")
            if end_uid not in ev["nodes"]:
                continue
            a, b = place.world(uid), place.world(end_uid)
            pts = sim.beam_points(a[:3, 3], b[:3, 3], a[:3, 0], b[:3, 0], float(me.get("Bend", 10)))
            width = float(me.get("Width", 1))
            tube = mk.tube(pts, width / 2, 10, (0.1, 0.1, 0.9, 0.9), up_hint=(0, 0, 1), caps=False)
            scene.add(tube, name=me.get("Name", ""), color=[0.9, 0.95, 0.97], opacity=0.8, roughness=0.15, renderOrder=3)
        elif cls == "sprite" and me.get("Size", 1) > 0 and "ember" in me.get("Name", ""):
            w = place.world(uid)
            glow = mk.lathe([(0, -0.12), (0.12, 0), (0, 0.12)], 8, (0, 0, 1, 1)).transformed(w)
            scene.add(glow, color=[1, 0.45, 0.1], emissive=[1, 0.35, 0.05])


def posed_scene(model, outfit, textures, label, state, width=900, height=900, ground=True):
    ev = outfit.evaluate(state)
    mats = model.pose(label, ev["bone3"])
    feet_z = min(mats[model.bone(b)][2, 3] for b in ("ValveBiped.Bip01_L_Toe0", "ValveBiped.Bip01_R_Toe0")) - 1.2
    scene = preview.Scene(width, height, ground_z=0.0 if ground else None)
    for m in model.skinned(mats):
        tex = textures.get(m.material)
        name = m.material.lower()
        style = {"map": tex} if tex else {"color": [0.75, 0.72, 0.7]}
        if "eyeball" in name:
            # Source draws irises with the eye shader, not these uvs; a dark iris colour reads right
            style = {"color": [0.3, 0.24, 0.2], "doubleSided": True}
        if "mouth" in name or "error" in name:
            continue
        scene.add(m, name=m.material, roughness=0.75, **style)
    place = sim.Placer(outfit, model, mats, ev, ground_z=0.0)
    add_outfit(scene, outfit, model, mats, ev, place)
    return scene, mats, ev, feet_z


def contact_sheet(paths, titles, cols, out, cell=(450, 450)):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 30)), (28, 30, 34))
    d = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip(paths, titles)):
        x, y = (i % cols) * cell[0], (i // cols) * (cell[1] + 30)
        sheet.paste(Image.open(p).convert("RGB").resize(cell), (x, y + 30))
        d.text((x + 10, y + 5), t, font=font, fill=(235, 235, 235))
    sheet.save(out)
    return out


def face_views(scene, mats, model, prefix=""):
    hf = mats[model.bone("ValveBiped.Bip01_Head1")]
    face_c = hf[:3, :3] @ np.array([3.5, -4.5, 0]) + hf[:3, 3]
    fwd = hf[:3, :3] @ np.array([0, -1.0, 0])
    left = hf[:3, :3] @ np.array([0, 0, -1.0])
    scene.view(prefix + "mask_front", face_c + fwd * 20 + left * 7 + np.array([0, 0, 2]), face_c, 30)
    scene.view(prefix + "mask_side", face_c + left * 21 + fwd * 3, face_c, 30)
    scene.view(prefix + "mask_back3q", face_c - fwd * 14 + left * 15 + np.array([0, 0, 4]), face_c, 30)
    return face_c, fwd, left


def render_main(model, outfit, textures, out):
    """Everything on (mask, tank, cane, glasses), standing: full body and the mask close up."""
    everything = {"cane", "glasses", "mask", "tank"}
    scene, mats, _, _ = posed_scene(model, outfit, textures, "aim_normal_0", sim.State(everything, "normal", time=6.0))
    c = np.array([-1.0, 9.0, 35.0])          # between him and the cart
    scene.view("full_front_left", c + np.array([92, 62, 6]), c, 36)
    scene.view("full_left_side", c + np.array([10, 118, 4]), c + np.array([0, -6, 0]), 36)
    scene.view("full_front_right", c + np.array([100, -70, 6]), c + np.array([2, -6, 0]), 36)
    scene.view("full_back_left", c + np.array([-95, 60, 14]), c, 36)
    face_views(scene, mats, model)
    paths = scene.render(os.path.join(out, "main"))
    titles = ["front left", "left side (cart)", "front right (cane)", "back left",
              "mask: front", "mask: side", "mask: strap, back 3/4"]
    return contact_sheet(paths, titles, 4, os.path.join(out, "sheet_mask_and_tank.png"))


def render_smoking(model, outfit, textures, out):
    """Mask off, cigarette on: the arm resting, on its way up, and at the drag (filter at the lips)."""
    paths, titles = [], []
    for t, tag in ((8.0, "resting"), (0.45, "raising"), (0.75, "almost"), (1.8, "drag")):
        st = sim.State({"smoke", "glasses", "cane"}, "normal", time=t)
        scene, mats, _, _ = posed_scene(model, outfit, textures, "aim_normal_0", st, 700, 700)
        face_c, fwd, left = face_views(scene, mats, model, f"{tag}_")
        del scene.desc["views"]
        scene.desc["views"] = []
        chest = face_c - np.array([0, 0, 9.0])
        scene.view(f"smoke_{tag}_front_left", chest + fwd * 40 + left * 26 + np.array([0, 0, 2]), chest, 36)
        scene.view(f"smoke_{tag}_left_side", chest + left * 46 + fwd * 4 + np.array([0, 0, 2]), chest, 36)
        paths += scene.render(os.path.join(out, "smoking"))
        titles += [f"smoking, {tag}: front left", f"smoking, {tag}: left side"]
    return paths, titles


def render_armed(model, outfit, textures, out):
    """Weapon out: cane on the belt, cigarette in the mouth, plain weapon posture."""
    paths, titles = [], []
    for hold, props, tag in (("pistol", {"cane", "glasses", "mask", "tank"}, "pistol, mask on"),
                             ("ar2", {"cane", "glasses", "smoke", "tank"}, "ar2, smoking")):
        st = sim.State(props, hold, time=6.0)
        scene, mats, _, _ = posed_scene(model, outfit, textures, f"aim_{hold}_0", st, 700, 700)
        c = np.array([0.0, 4.0, 35.0])
        scene.view(f"{hold}_front_right", c + np.array([95, -75, 8]), c, 36)
        scene.view(f"{hold}_back_right", c + np.array([-90, -80, 10]), c, 36)
        paths += scene.render(os.path.join(out, "armed"))
        titles += [f"{tag}: front right", f"{tag}: back right (belt cane)"]
    return paths, titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(REPO, "build", "male08", "male08_dump.txt"))
    ap.add_argument("--out", default=os.path.join(REPO, "build", "previews"))
    ap.add_argument("--outfit", default=OUTFIT)
    args = ap.parse_args()
    model = Model(args.dump)
    outfit = sim.Outfit(args.outfit)
    textures = BodyTextures(model, os.path.dirname(os.path.abspath(args.dump)))
    print(render_main(model, outfit, textures, args.out))
    sp, st = render_smoking(model, outfit, textures, args.out)
    print(contact_sheet(sp, st, 4, os.path.join(args.out, "sheet_smoking.png")))
    ap_, at = render_armed(model, outfit, textures, args.out)
    print(contact_sheet(ap_, at, 4, os.path.join(args.out, "sheet_armed.png")))


if __name__ == "__main__":
    main()
