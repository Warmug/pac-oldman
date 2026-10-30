"""The job playermodel as exported by tools/gmod/dump_male08.lua.

Gives bone names/parents, the bind-pose mesh with weights, attachments and the
bone matrices of real player animations, plus pac's bone3 math and skinning so
the outfit can be posed and rendered offline exactly the way pac would.
"""
import json
import os
import re

import numpy as np

import meshkit as mk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DUMP = os.environ.get("MALE08_DUMP", os.path.join(REPO, "build", "male08", "male08_dump.txt"))

# pac.BoneNameReplacements (lua/pac3/core/client/bones.lua), applied in order
_REPL = [("Anim_Attachment", "attach"), ("RH", "right hand"), ("LH", "left hand"), ("_L_", " left "),
         ("_R_", " right "), ("%p", " "), ("ValveBiped", ""), ("Bip01", ""), ("Neck1", "neck"),
         ("Head1", "head"), ("Toe0", "toe"), ("lowerarm", "lower arm"), ("Bip", ""), (" R", " right"),
         (" L", " left")]
_PUNCT = re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def pac_bone_name(real):
    s = real
    for pat, rep in _REPL:
        s = re.sub(f"[{_PUNCT}]", rep, s) if pat == "%p" else s.replace(pat, rep)
    return re.sub(r"(.*?)(\d+)", r"\1 \2", s.strip().lower())


def mat12(vals):
    m = np.eye(4)
    m[:3, :] = np.asarray(vals, float).reshape(3, 4)
    return m


def angle_matrix(p, y, r):
    m = np.eye(4)
    m[:3, :3] = mk.source_angle_matrix(p, y, r)
    return m


class Model:
    def __init__(self, path=DEFAULT_DUMP):
        self.path = path
        self.names, self.parents = [], []
        self.bind = {}
        self.poses, self.pose_info = {}, {}
        self.attachments = {}
        self.meshes = []
        self.materials = []
        self.pose_params = {}
        self.flexes = []
        self.sequences = []
        self._parse(path)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.pac_index = {pac_bone_name(n): i for i, n in enumerate(self.names)}

    def _parse(self, path):
        mesh = None
        pose = None
        for line in open(path, encoding="utf-8", errors="replace"):
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "b":
                self.names.append(" ".join(parts[3:]))
                self.parents.append(int(parts[2]))
            elif tag == "bind":
                self.bind[int(parts[1])] = mat12(parts[3:15])
            elif tag == "mesh":
                mesh = {"material": " ".join(parts[3:]), "v": [], "n": [], "uv": [], "w": [], "f": []}
                self.meshes.append(mesh)
            elif tag == "v":
                vals = list(map(float, parts[1:]))
                mesh["v"].append(vals[0:3])
                mesh["n"].append(vals[3:6])
                mesh["uv"].append(vals[6:8])
                ws = vals[8:]
                mesh["w"].append([(int(ws[i]), ws[i + 1]) for i in range(0, len(ws), 2)])
            elif tag == "t":
                mesh["f"].append([int(parts[1]), int(parts[2]), int(parts[3])])
            elif tag == "pose":
                pose = parts[1]
                self.poses[pose] = {}
                self.pose_info[pose] = {"sequence": parts[2], "cycle": float(parts[3]), "params": parts[4]}
            elif tag == "pm":
                self.poses[pose][int(parts[1])] = mat12(parts[2:14])
            elif tag == "att":
                label, name, bone = parts[1], parts[2], int(parts[3])
                pos = np.array(list(map(float, parts[4:7])))
                ang = tuple(map(float, parts[7:10]))
                self.attachments.setdefault(label, {})[name] = (bone, pos, ang)
            elif tag == "mat":
                self.materials.append({"path": parts[2], "base": parts[3], "w": int(parts[4]), "h": int(parts[5])})
            elif tag == "pp":
                self.pose_params[parts[1]] = (float(parts[2]), float(parts[3]))
            elif tag == "flex":
                self.flexes.append(parts[2])
            elif tag == "seqnames":
                self.sequences += parts[1:]
        for m in self.meshes:
            for k in ("v", "n", "uv"):
                m[k] = np.array(m[k], float)
            m["f"] = np.array(m["f"], int)

    # --- skeleton ------------------------------------------------------------

    def bone(self, name):
        """Index from a real ValveBiped name or a pac friendly name."""
        if name in self.index:
            return self.index[name]
        return self.pac_index[name]

    def children_order(self):
        order, seen = [], set()

        def visit(i):
            if i in seen:
                return
            if self.parents[i] >= 0:
                visit(self.parents[i])
            seen.add(i)
            order.append(i)

        for i in range(len(self.names)):
            visit(i)
        return order

    def pose(self, label, bone3=None):
        """World bone matrices for a dumped pose, with pac bone3 manipulations applied.
        bone3: {bone name: (position, angles)} or a list of (bone name, position, angles);
        several parts on one bone compose in order, like pac's BuildBonePositions loop."""
        base = self.poses[label]
        if not bone3:
            return dict(base)
        items = bone3.items() if isinstance(bone3, dict) else [(b, (p, a)) for b, p, a in bone3]
        manip = {}
        for name, (pos, ang) in items:
            m = np.eye(4)
            m[:3, 3] = pos
            i = self.bone(name)
            manip[i] = manip.get(i, np.eye(4)) @ m @ angle_matrix(*ang)
        out = {}
        for i in self.children_order():
            if i not in base:
                continue
            p = self.parents[i]
            if p >= 0 and p in out:
                world = out[p] @ np.linalg.inv(base[p]) @ base[i]
            else:
                world = base[i].copy()
            if i in manip:
                world = world @ manip[i]
            out[i] = world
        return out

    # --- mesh ------------------------------------------------------------------

    def bind_world(self, i):
        """Bind-pose world matrix of bone i (util.GetModelMeshes gives pose-to-bone, i.e. its inverse)."""
        return np.linalg.inv(self.bind[i])

    def skinned(self, mats, mesh_filter=None):
        """meshkit.Mesh per dumped mesh, linear-blend skinned to the bone matrices in `mats`."""
        skin = {i: mats[i] @ self.bind[i] for i in mats if i in self.bind}
        out = []
        for mi, m in enumerate(self.meshes):
            if mesh_filter and not mesh_filter(m):
                continue
            v = np.zeros_like(m["v"])
            n = np.zeros_like(m["n"])
            vh = np.c_[m["v"], np.ones(len(m["v"]))]
            for k, ws in enumerate(m["w"]):
                acc = np.zeros((4, 4))
                for b, w in ws:
                    acc += w * skin[b]
                v[k] = (acc @ vh[k])[:3]
                n[k] = acc[:3, :3] @ m["n"][k]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            mesh = mk.Mesh(v, m["f"], m["uv"], n)
            mesh.material = m["material"]
            out.append(mesh)
        return out

    def skeleton_json(self, path):
        atts = sorted({name for label in self.attachments.values() for name in label})
        with open(path, "w") as fh:
            json.dump({"model": "models/player/Suits/male_08_shirt_tie.mdl", "bones": self.names,
                       "parents": self.parents, "attachments": atts}, fh, indent=1)
