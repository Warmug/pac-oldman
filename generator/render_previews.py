#!/usr/bin/env python3
"""Preview renders of the generated outfit on male_08 (needs the dump).

    python3 generator/render_previews.py --dump build/male08/male08_dump.txt [--out build/previews]

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
        style["opacity"] = max(0.22, float(me.get("alpha", 1.0)) * 0.9)
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
            scene.add(mesh, place.world(uid), name=me.get("Name", ""), **material_style(outfit.material(me.get("Material"))))
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
            style = {"color": [0.95, 0.95, 0.95], "roughness": 0.2}
        scene.add(m, name=m.material, roughness=0.75, **style)
    place = sim.Placer(outfit, model, mats, ev, ground_z=0.0)
    add_outfit(scene, outfit, model, mats, ev, place)
    return scene, mats, ev, feet_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "build", "previews"))
    ap.add_argument("--outfit", default=OUTFIT)
    args = ap.parse_args()
    model = Model(args.dump)
    outfit = sim.Outfit(args.outfit)
    textures = BodyTextures(model, os.path.dirname(os.path.abspath(args.dump)))
    head = model.bone("ValveBiped.Bip01_Head1")

    everything = {"cane", "smoke", "glasses", "mask", "tank"}
    st = sim.State(everything - {"smoke"}, "normal", 0, 0, time=6.0)
    scene, mats, ev, _ = posed_scene(model, outfit, textures, "aim_normal_0", st)
    hp = mats[head][:3, 3]
    scene.view("full_front3q", hp + np.array([58, -44, -18]), hp + np.array([0, 6, -26]), 36)
    scene.view("full_side", hp + np.array([4, 85, -22]), hp + np.array([-2, 8, -26]), 36)
    scene.view("full_back3q", hp + np.array([-55, 50, -14]), hp + np.array([0, 8, -26]), 36)
    hf = mats[head]
    face_c = hf[:3, :3] @ np.array([3.5, -4.5, 0]) + hf[:3, 3]
    fwd = hf[:3, :3] @ np.array([0, -1.0, 0])
    left = hf[:3, :3] @ np.array([0, 0, -1.0])
    scene.view("mask_front", face_c + fwd * 20 + left * 7 + np.array([0, 0, 2]), face_c, 30)
    scene.view("mask_side", face_c + left * 21 + fwd * 3, face_c, 30)
    scene.view("mask_back3q", face_c - fwd * 14 + left * 15 + np.array([0, 0, 4]), face_c, 30)
    print(scene.render(args.out))


if __name__ == "__main__":
    main()
