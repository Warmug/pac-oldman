#!/usr/bin/env python3
"""Validate a PAC3 outfit .txt against the PAC3 source and the files it points at.

    python3 tools/verify.py outfit/old_man_2_male08.txt [--pac3 ~/pac3] [--skeleton generator/data/male08_skeleton.json]

Checks
- the file decodes with pac's own luadata.lua (run in LuaJIT, bytecode whitelist on),
- every ClassName exists and every property is a storable var of that class, with the right type,
- events: known event, operator, argument count; proxies: target property/axis exist,
  expression passes pac's word blacklist, compiles, and evaluates to numbers,
- bone names exist on the model (pac's friendly-name rules) or are pac special bones,
- referenced materials and part UIDs exist,
- every repo URL maps to a local file; OBJs are under pac's 21845 triangle limit, have
  normals and no uv exactly on a wrap edge.
Prints the list of repo files the outfit needs on GitHub.
"""
import argparse
import json
import math
import os
import re
import sys

from lupa.luajit21 import lua_type
from lupa.luajit21 import LuaRuntime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PREFIX = "https://raw.githubusercontent.com/Warmug/pac-oldman/main/"
MAX_TRIS = 21845

SPECIAL_BONES = {"hitpos", "footstep", "skirt", "skirt2", "hitpos_world_props", "hitpos_world",
                 "hitpos_world_noang", "hitpos_ent_ang", "hitpos_ent_ang_zero_pitch", "pos_ang",
                 "pos_eyeang", "eyepos_eyeang", "eyepos_ang", "pos_noang", "camera", "player_eyes",
                 "physgun_beam_endpos", "hit position"}

# Not a bone: pac.GetBonePosAng falls back to the player's origin turned to his eye yaw (pitch
# zeroed) for any name that isn't a bone, attachment or special. The outfit uses this one name
# for things that must stay level and face where he faces (the oxygen cart); any other unknown
# bone is still an error, so typos are caught.
ORIGIN_BONE = "none"
ORIGIN_FALLBACK = re.compile(r"if not pos then\s+pos = ent:GetPos\(\)\s+end\s+if not ang then\s+"
                             r"if ent:IsPlayer\(\) then\s+ang = ent:EyeAngles\(\)\s+ang\.p = 0")

HOLDTYPES = {"normal", "melee", "melee2", "fist", "knife", "smg", "ar2", "pistol", "rpg", "physgun",
             "grenade", "shotgun", "crossbow", "slam", "passive", "magic", "duel", "revolver", "camera"}

# pac.BoneNameReplacements (lua/pac3/core/client/bones.lua), applied in order with string.gsub
BONE_REPLACEMENTS = [("Anim_Attachment", "attach"), ("RH", "right hand"), ("LH", "left hand"),
                     ("_L_", " left "), ("_R_", " right "), ("%p", " "), ("ValveBiped", ""),
                     ("Bip01", ""), ("Neck1", "neck"), ("Head1", "head"), ("Toe0", "toe"),
                     ("lowerarm", "lower arm"), ("Bip", ""), (" R", " right"), (" L", " left")]
LUA_PUNCT = re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def pac_bone_name(real):
    s = real
    for pat, rep in BONE_REPLACEMENTS:
        s = re.sub(f"[{LUA_PUNCT}]", rep, s) if pat == "%p" else s.replace(pat, rep)
    s = s.strip().lower()
    return re.sub(r"(.*?)(\d+)", r"\1 \2", s)


def pac_attachment_name(name):
    return re.sub(r"(.*?)(\d+)", r"\1 \2", name.strip().lower())


# --- schema from the pac3 source --------------------------------------------------

def _default_type(src):
    src = src.strip()
    if src.startswith('"') or src.startswith("'"):
        return "string"
    if src in ("true", "false"):
        return "boolean"
    if src.startswith("Vector"):
        return "Vector"
    if src.startswith("Angle"):
        return "Angle"
    if src.startswith("Color"):
        return "Color"
    if re.match(r"^-?[\d.]+(e-?\d+)?$", src) or src.startswith("math."):
        return "number"
    return "any"


def parse_storables(text):
    """Yield (template, class_name, {prop: type}) for every PartTemplate block in a part file."""
    blocks = re.split(r"(?=local\s+BUILDER\s*,\s*PART\s*=\s*pac\.PartTemplate\()", text)
    for block in blocks:
        m = re.search(r'pac\.PartTemplate\(\s*(?:"([\w ]+)")?\s*\)', block)
        if not m:
            continue
        template = m.group(1) or ""
        cm = re.search(r'PART\.ClassName\s*=\s*"([\w ]+)"', block)
        class_name = cm.group(1) if cm else None
        props, removed, store = {}, set(), False
        for tok in re.finditer(r'StartStorableVars|EndStorableVars|GetSetPart\(\s*"(\w+)"|GetSet\(\s*"(\w+)"\s*,\s*((?:[^(),]|\([^()]*\))*)'
                               r'|RemoveProperty\(\s*"(\w+)"', block):
            word = tok.group(0)
            if word == "StartStorableVars":
                store = True
            elif word == "EndStorableVars":
                store = False
            elif tok.group(1) and store:
                props[tok.group(1) + "UID"] = "string"
            elif tok.group(2) and store:
                props[tok.group(2)] = _default_type(tok.group(3))
            elif tok.group(4):
                removed.add(tok.group(4))
        for r in removed:
            props.pop(r, None)
        yield template, class_name, props


def load_schema(pac3):
    parts_dir = os.path.join(pac3, "lua/pac3/core/client/parts")
    core = os.path.join(pac3, "lua/pac3/core/client")
    templates = {}
    for name in ("base_part.lua", "base_movable.lua", "base_drawable.lua"):
        for template, cls, props in parse_storables(open(os.path.join(core, name)).read()):
            templates[cls] = (template, props)
    classes = {}
    for dirpath, _, fnames in sorted(os.walk(parts_dir)):
        for fname in sorted(fnames):
            if fname.endswith(".lua") and fname != "material.lua":
                for template, cls, props in parse_storables(open(os.path.join(dirpath, fname)).read()):
                    if cls:
                        classes.setdefault(cls, (template, props))

    known = dict(classes)
    known.update(templates)

    def resolve(template, own):
        out = {}
        chain = []
        while template:
            chain.append(template)
            template = known.get(template, ("", {}))[0]
        for t in reversed(chain):
            out.update(known.get(t, ("", {}))[1])
        out.update(own)
        return out

    schema = {cls: resolve(t, p) for cls, (t, p) in classes.items()}
    schema.update(material_classes(pac3, resolve("base", {})))
    return schema


def material_classes(pac3, base_props):
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute("function Vector(x, y, z) return {__vec = true, x or 0, y or 0, z or 0} end")
    params = lua.execute(open(os.path.join(pac3, "lua/pac3/libraries/shader_params.lua")).read())
    translate = {"vertexlitgeneric": "3d", "unlitgeneric": "2d", "eyerefract": "eye refract"}
    base_groups = {g: dict(v.items()) for g, v in params.base.items()}
    out = {}
    for shader, groups in params.shaders.items():
        merged = {g: dict(v.items()) for g, v in groups.items()}
        for g, bg in base_groups.items():
            merged.setdefault(g, {})
            for k, v in bg.items():
                merged[g].setdefault(k, v)
        props = dict(base_props)
        props["LoadVmt"] = "string"
        props["MaterialOverride"] = "string"
        for g, ps in merged.items():
            for key, info in ps.items():
                typ = info["type"]
                if typ == "matrix":
                    props.update({key + "Position": "Vector", key + "Scale": "Vector",
                                  key + "Angle": "number", key + "AngleCenter": "Vector"})
                elif typ == "texture":
                    props[key] = "string"
                    if info["partial_hdr"]:
                        props[key + "NoHDR"] = "boolean"
                else:
                    default = info["default"]
                    if typ in ("vec3", "color", "vec2"):
                        props[key] = "Vector"
                    elif typ == "bool" or isinstance(default, bool):
                        props[key] = "boolean"
                    elif typ in ("float", "integer") or isinstance(default, (int, float)):
                        props[key] = "number"
                    else:
                        props[key] = "any"
        out["material_" + translate.get(shader, shader)] = props
    return out


def load_events(pac3):
    text = open(os.path.join(pac3, "lua/pac3/core/client/parts/event.lua")).read()
    start = text.index("PART.OldEvents = {")
    events = {}
    body = text[start:]
    for m in re.finditer(r"^\t(\w+) = \{\n(.*?)^\t\},", body, re.S | re.M):
        args = re.search(r"arguments\s*=\s*\{(.*?)\}\s*,\s*\n", m.group(2), re.S)
        names = re.findall(r"\{\s*(\w+)\s*=\s*\"(\w+)\"\s*\}", args.group(1)) if args else []
        events[m.group(1)] = names
    events.setdefault("button", [("button", "string")])
    operators = re.search(r"PART\.Operators = \{(.*?)\}", text, re.S).group(1)
    return events, re.findall(r'"([\w ]+)"', operators)


def load_proxy_functions(pac3):
    text = open(os.path.join(pac3, "lua/pac3/core/client/parts/proxy.lua")).read()
    names = set(re.findall(r"PART\.Inputs\.(\w+)\s*=", text))
    names |= set(re.findall(r'PART\.Inputs\["(\w+)"\]', text))
    names |= {f"var{i}" for i in range(1, 6)} | {f"extra{i}" for i in range(1, 6)}
    lib = open(os.path.join(pac3, "lua/pac3/libraries/expression.lua")).read()
    lib_body = lib[lib.index("local lib = {"):lib.index("local blacklist")]
    names |= set(re.findall(r"^\t(\w+)\s*=", lib_body, re.M))
    names |= {"select"}
    return names


# --- decoding with pac's own luadata ----------------------------------------------

DECODE_ENV = r"""
jit.version_num = 20100 -- same opcode table as LuaJIT 2.1.0-beta3, which luadata knows
function isnumber(v) return type(v) == "number" end
function isstring(v) return type(v) == "string" end
function istable(v) return type(v) == "table" end
function isfunction(v) return type(v) == "function" end
function IsEntity() return false end
function ErrorNoHalt(...) print(...) end
function CompileString(code, name, handle)
	local f, err = loadstring(code, name)
	if not f then return err end
	return f
end
function Vector(x, y, z) return {__type = "Vector", x, y, z} end
function Angle(p, y, r) return {__type = "Angle", p, y, r} end
function Color(r, g, b, a) return {__type = "Color", r, g, b, a} end
"""


def _first(res):
    return res[0] if isinstance(res, tuple) else res


def decode_outfit(pac3, text):
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute(DECODE_ENV)
    luadata = lua.execute(open(os.path.join(pac3, "lua/pac3/libraries/luadata.lua")).read())
    probe = lua.eval("function() return {[1] = Vector(1, 2, 3), x = -0.5} end")
    if not _first(luadata.is_func_ok(probe)):
        raise RuntimeError("luadata's opcode whitelist rejects a trivial table; opcode table mismatch")
    res = luadata.Decode(text)
    data, err = (res + (None,))[:2] if isinstance(res, tuple) else (res, None)
    if data is None:
        raise ValueError(f"pac luadata.Decode failed: {err}")

    def conv(v):
        if lua_type(v) == "table":
            t = v["__type"]
            if t:
                return (t, [v[1], v[2], v[3]])
            keys = list(v.keys())
            if all(isinstance(k, (int, float)) for k in keys):
                return [conv(v[k]) for k in sorted(keys)]
            return {k: conv(v[k]) for k in keys}
        return v

    return conv(data)


# --- checks -------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


def py_type(v):
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, tuple):
        return v[0]
    return type(v).__name__


EXPR_STUBS = {
    "time": 13.37, "synced_time": 13.37, "systime": 13.37, "stime": 13.37, "frametime": 0.016,
    "ftime": 0.016, "owner_velocity_length": 60.0, "owner_velocity_forward": 55.0,
    "owner_velocity_right": 3.0, "owner_velocity_up": 0.0, "owner_eye_angle_pitch": 0.3,
    "owner_eye_angle_yaw": 0.4, "voice_volume": 0.08, "random": 0.5,
}


def check_expression(lua, where, expr, n_expected, funcs, rep):
    for word in ("repeat", "until", "function", "end"):
        if re.search(rf"[{LUA_PUNCT}\s]{word}", expr) or re.search(rf"{word}[{LUA_PUNCT}\s]", expr):
            rep.err(where, f"expression contains blacklisted word '{word}' next to punctuation/space; pac refuses it")
    for name in set(re.findall(r"([A-Za-z_]\w*)\s*\(", expr)) - {"and", "or", "not"}:
        if name not in funcs:
            rep.err(where, f"unknown proxy function {name}()")
    env = lua.eval("{}")
    for k in ("abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "deg", "exp", "floor",
              "log", "log10", "max", "min", "rad", "sin", "sinh", "sqrt", "tanh", "tan"):
        env[k] = lua.eval(f"math.{k}")
    env["clamp"] = lua.eval("function(x, a, b) return math.min(math.max(x, a), b) end")
    env["select"] = lua.eval("select")
    for name in funcs:
        if env[name] is None:
            val = EXPR_STUBS.get(name, 0.5)
            env[name] = lua.eval(f"function() return {val} end")
    res = lua.eval("function(src) return loadstring('local IN = select(1, ...) return ' .. src, 'pac_expression') end")(expr)
    f, err = (res + (None,))[:2] if isinstance(res, tuple) else (res, None)
    if f is None:
        rep.err(where, f"expression does not compile: {err}")
        return
    lua.eval("setfenv")(f, env)
    res = lua.eval("function(f) return {pcall(f)} end")(f)
    vals = [res[i] for i in range(1, len(res) + 1)]
    if not vals[0]:
        rep.err(where, f"expression errors at runtime: {vals[1]}")
        return
    outs = vals[1:]
    if n_expected and len(outs) != n_expected:
        rep.err(where, f"expression returns {len(outs)} values, property needs {n_expected}")
    for o in outs:
        if not isinstance(o, (int, float)) or isinstance(o, bool) or math.isnan(o) or math.isinf(o):
            rep.err(where, f"expression returned non-number {o!r}")


def obj_stats(path):
    tris = verts = normals = 0
    bad_uv = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("f "):
                tris += len(line.split()) - 3
            elif line.startswith("v "):
                verts += 1
            elif line.startswith("vn "):
                normals += 1
            elif line.startswith("vt "):
                u, v = map(float, line.split()[1:3])
                if u in (0.0, 1.0) or v in (0.0, 1.0) or not (0 < u < 1 and 0 < v < 1):
                    bad_uv += 1
    return tris, verts, normals, bad_uv


def check(outfit_path, pac3, skeleton_path):
    rep = Report()
    schema = load_schema(pac3)
    events, operators = load_events(pac3)
    funcs = load_proxy_functions(pac3)
    lua = LuaRuntime(unpack_returned_tuples=True)

    text = open(outfit_path, encoding="utf-8").read()
    try:
        data = decode_outfit(pac3, text)
    except Exception as e:
        rep.err(outfit_path, str(e))
        return rep, [], {}

    bones = None
    if skeleton_path and os.path.exists(skeleton_path):
        sk = json.load(open(skeleton_path))
        bones = {pac_bone_name(b) for b in sk["bones"]} | {pac_attachment_name(a) for a in sk.get("attachments", [])}
    bones_lua = open(os.path.join(pac3, "lua/pac3/core/client/bones.lua")).read()
    origin_ok = (bool(ORIGIN_FALLBACK.search(bones_lua)) and ORIGIN_BONE not in SPECIAL_BONES
                 and (bones is None or ORIGIN_BONE not in bones))

    parts = []

    def walk(node, parent, path):
        me = node["self"]
        parts.append((me, parent, path))
        for i, child in enumerate(node.get("children", []), 1):
            walk(child, me, f"{path}/{child['self'].get('Name') or child['self'].get('ClassName')}")

    for i, root in enumerate(data, 1):
        walk(root, None, root["self"].get("Name", f"root{i}"))

    uids = {}
    names = {}
    for me, parent, path in parts:
        uid = me.get("UniqueID", "")
        if not re.fullmatch(r"[0-9a-f]{64}", uid or ""):
            rep.err(path, f"UniqueID {uid!r} is not 64 hex chars")
        if uid in uids:
            rep.err(path, "duplicate UniqueID")
        uids[uid] = me
        names.setdefault(me.get("Name", ""), []).append(me)

    materials = {me["Name"] for me, _, _ in parts if me.get("ClassName", "").startswith("material_")}
    urls = {}

    for me, parent, path in parts:
        cls = me.get("ClassName")
        if cls not in schema:
            rep.err(path, f"unknown part class {cls!r}")
            continue
        props = schema[cls]
        for key, value in me.items():
            if key in ("ClassName", "UniqueID"):
                continue
            if key not in props:
                rep.err(path, f"{cls} has no storable property {key!r} (pac would silently ignore it)")
                continue
            want, got = props[key], py_type(value)
            if want != "any" and want != got and not (want == "Vector" and got == "Color"):
                rep.err(path, f"{key} should be {want}, got {got} ({value!r})")
            if isinstance(value, str) and value.startswith("http"):
                urls.setdefault(value, []).append(path)

        if "Bone" in props and cls != "group":
            bone = me.get("Bone", "head")
            if bone == ORIGIN_BONE:
                if not origin_ok:
                    rep.err(path, f"bone {bone!r}: this pac version has no origin/eye-yaw fallback for unknown bones")
            elif bone not in SPECIAL_BONES and bones is not None and bone not in bones:
                rep.err(path, f"bone {bone!r} does not exist on the model")

        if cls in ("model2", "beam"):
            mat = me.get("Material", "")
            if mat and not mat.startswith("http") and "/" not in mat and mat not in materials:
                rep.err(path, f"material {mat!r} is not defined by any material part")
        if cls == "model2":
            if "Bone" not in me:
                rep.warn(path, "no Bone set; pac defaults to 'head'")

        if cls == "event":
            ev = me.get("Event", "")
            if ev not in events:
                rep.err(path, f"unknown event {ev!r}")
            else:
                args = me.get("Arguments", "").split("@@")
                spec = events[ev]
                if spec and len(args) > len(spec):
                    rep.err(path, f"{ev} takes {len(spec)} arguments, got {len(args)}")
                for (aname, atype), aval in zip(spec, args):
                    if atype == "number" and aval and not re.fullmatch(r"-?[\d.]+", aval):
                        rep.err(path, f"argument {aname} should be a number, got {aval!r}")
                    if atype == "boolean" and aval not in ("0", "1", "", "true", "false"):
                        rep.err(path, f"argument {aname} should be 0/1, got {aval!r}")
                if ev == "holdtype":
                    for h in args[0].split(";"):
                        if h not in HOLDTYPES:
                            rep.err(path, f"{h!r} is not a gmod holdtype")
            if me.get("Operator", "find simple") not in operators:
                rep.err(path, f"unknown operator {me.get('Operator')!r}")
            if "Invert" not in me:
                rep.warn(path, "Invert not set; pac's default is true (show when the condition is met)")

        if cls == "proxy":
            var = me.get("VariableName", "")
            target_cls = parent.get("ClassName") if parent else None
            target_props = schema.get(target_cls, {})
            if var not in target_props:
                rep.err(path, f"proxy targets {var!r} which {target_cls} does not have")
                continue
            vtype = target_props[var]
            axis = me.get("Axis", "")
            if axis:
                valid = {"Angle": "pyr", "Vector": "xyzrgb"}.get(vtype, "")
                if axis not in valid or len(axis) != 1:
                    rep.err(path, f"axis {axis!r} is invalid for {vtype} property {var}")
            n = 3 if vtype in ("Angle", "Vector") and not axis else 1
            if vtype == "boolean":
                n = 1
            check_expression(lua, path, me.get("Expression", ""), n, funcs, rep)

        for key, value in me.items():
            if key.endswith("UID") and key != "UniqueID" and value and value not in uids:
                rep.err(path, f"{key} points at a part that is not in the outfit")

    files = {}
    for url, where in sorted(urls.items()):
        if url.startswith(RAW_PREFIX):
            rel = url[len(RAW_PREFIX):]
            local = os.path.join(REPO, rel)
            files[rel] = local
            if not os.path.exists(local):
                rep.err(where[0], f"{rel} is referenced but missing from the repo")
            elif rel.endswith(".obj"):
                tris, verts, normals, bad_uv = obj_stats(local)
                if tris > MAX_TRIS:
                    rep.err(rel, f"{tris} triangles > pac limit {MAX_TRIS}")
                if normals == 0:
                    rep.err(rel, "no vn lines; pac would light it with a constant normal")
                if bad_uv:
                    rep.err(rel, f"{bad_uv} uvs on/over a wrap edge (pac applies % 1)")
        else:
            rep.warn(where[0], f"external url {url}")
    return rep, parts, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outfit")
    ap.add_argument("--pac3", default=os.path.expanduser("~/pac3"))
    ap.add_argument("--skeleton", default=os.path.join(REPO, "generator/data/male08_skeleton.json"))
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.pac3, "lua/pac3")):
        sys.exit(f"pac3 source not found at {args.pac3} (git clone https://github.com/CapsAdmin/pac3 ~/pac3)")
    rep, parts, files = check(args.outfit, args.pac3, args.skeleton)
    by_class = {}
    for me, _, _ in parts:
        by_class[me.get("ClassName")] = by_class.get(me.get("ClassName"), 0) + 1
    print(f"{args.outfit}: {len(parts)} parts " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
    if not os.path.exists(args.skeleton):
        print("WARNING: no male_08 skeleton file, bone names were not checked")
    for w in rep.warnings:
        print("warning:", w)
    for e in rep.errors:
        print("ERROR:", e)
    print("\nrepo files the outfit loads (upload these to Warmug/pac-oldman, branch main):")
    for rel in sorted(files):
        print("  " + rel)
    print(f"\n{len(rep.errors)} errors, {len(rep.warnings)} warnings")
    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()
