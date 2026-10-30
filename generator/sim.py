"""Offline stand-in for pac's runtime, driven by the generated outfit file itself.

Given pac_event states, the active weapon's holdtype, aim pitch, speed and time it works
out which parts are visible (events, inherited hiding), evaluates proxy expressions
(LuaJIT, same expression rules as pac) and returns the bone3 manipulations plus the
world placement of every visible model, beam and marker, ready to pose and render.
"""
import math
import os
import sys

import numpy as np
from lupa.luajit21 import LuaRuntime

import meshkit as mk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import verify  # noqa: E402

RAW = "https://raw.githubusercontent.com/Warmug/pac-oldman/main/"


class State:
    def __init__(self, commands=(), holdtype="normal", pitch=0.0, speed=0.0, time=0.0, crouching=False):
        self.commands = set(commands)
        self.holdtype = holdtype
        self.pitch = pitch      # eye pitch in degrees, positive = looking down (Source)
        self.speed = speed
        self.time = time
        self.crouching = crouching


class Outfit:
    def __init__(self, path, pac3=os.path.expanduser("~/pac3")):
        text = open(path, encoding="utf-8").read()
        self.tree = verify.decode_outfit(pac3, text)
        self.parts = {}
        self.parent = {}

        def index(node, parent):
            me = node["self"]
            self.parts[me["UniqueID"]] = node
            self.parent[me["UniqueID"]] = parent
            for c in node.get("children", []):
                index(c, me["UniqueID"])

        for root in self.tree:
            index(root, None)
        self.lua = LuaRuntime(unpack_returned_tuples=True)

    # --- events / visibility -------------------------------------------------------

    def event_says_hide(self, ev, st):
        name = ev.get("Event")
        args = ev.get("Arguments", "").split("@@")
        if name == "command":
            cond = args[0] in st.commands
        elif name == "holdtype":
            cond = st.holdtype is not None and st.holdtype in args[0].split(";")
        elif name == "is_crouching":
            cond = st.crouching
        else:
            raise NotImplementedError(name)
        invert = ev.get("Invert", True)
        return (not cond) if invert else cond

    def hidden(self, uid, st):
        node = self.parts[uid]
        if node["self"].get("Hide"):
            return True
        for c in node.get("children", []):
            if c["self"]["ClassName"] == "event" and self.event_says_hide(c["self"], st):
                return True
        p = self.parent[uid]
        return p is not None and self.hidden(p, st)

    # --- proxies ------------------------------------------------------------------------

    def eval_expression(self, expr, st):
        stub = {
            "time": st.time, "owner_velocity_length": st.speed, "owner_velocity_forward": st.speed,
            "owner_eye_angle_pitch": -(1 + max(-89, min(89, st.pitch)) / 89) / 2 + 1,
            "ftime": 1 / 60, "frametime": 1 / 60, "voice_volume": 0.0,
        }
        env = self.lua.eval("{}")
        for k in ("abs", "sin", "cos", "min", "max", "floor", "sqrt", "atan2", "rad", "deg"):
            env[k] = self.lua.eval(f"math.{k}")
        env["clamp"] = self.lua.eval("function(x, a, b) return math.min(math.max(x, a), b) end")
        env["select"] = self.lua.eval("select")
        for k, v in stub.items():
            env[k] = self.lua.eval(f"function() return {v!r} end")
        f = self.lua.eval("function(src) return loadstring('local IN = select(1, ...) return ' .. src) end")(expr)
        self.lua.eval("setfenv")(f, env)
        res = f()
        return list(res) if isinstance(res, tuple) else [res]

    def props_with_proxies(self, node, st):
        me = dict(node["self"])
        for c in node.get("children", []):
            cs = c["self"]
            if cs["ClassName"] != "proxy" or self.hidden(cs["UniqueID"], st):
                continue
            var, axis = cs["VariableName"], cs.get("Axis", "")
            vals = self.eval_expression(cs["Expression"], st)
            if cs.get("Additive"):
                continue  # integrates over time (wheel spin); not meaningful in a still frame
            cur = me.get(var)
            if isinstance(cur, tuple):
                comps = list(cur[1])
                keys = "pyr" if cur[0] == "Angle" else "xyz"
                start = keys.index(axis) if axis else 0
                for i, v in enumerate(vals):
                    if start + i < 3:
                        comps[start + i] = v
                me[var] = (cur[0], comps)
            elif var == "Hide":
                me["Hide"] = bool(vals[0] and vals[0] > 0)
            else:
                me[var] = vals[0]
        return me

    # --- evaluation ---------------------------------------------------------------------

    def evaluate(self, st):
        """-> dict(bone3=[(bone, pos, ang)], models=[...], beams=[...], jiggles=[...], markers={uid: ...})"""
        out = {"bone3": [], "models": [], "beams": [], "jiggles": [], "nodes": {}}
        for uid, node in self.parts.items():
            if self.hidden(uid, st):
                continue
            cls = node["self"]["ClassName"]
            if cls in ("bone3", "model2", "beam", "jiggle", "sprite", "particles"):
                me = self.props_with_proxies(node, st)
                if me.get("Hide"):
                    continue
                out["nodes"][uid] = me
                if cls == "bone3":
                    out["bone3"].append((me["Bone"], vec(me.get("Position")), vec(me.get("Angles"))))
        return out

    def material(self, name):
        for node in self.parts.values():
            me = node["self"]
            if me["ClassName"].startswith("material_") and me.get("Name") == name:
                return me
        return None


def vec(v):
    if v is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(x) for x in v[1])


def local_path(url):
    return os.path.join(REPO, url[len(RAW):]) if url.startswith(RAW) else None


class Placer:
    """World transforms of parts, mirroring pac's base_movable rules."""

    def __init__(self, outfit, model, mats, ev, ground_z=0.0):
        self.o, self.model, self.mats, self.ev = outfit, model, mats, ev
        self.ground_z = ground_z
        self.cache = {}

    def bone_matrix(self, bone_name):
        try:
            return self.mats[self.model.bone(bone_name)]
        except KeyError:
            # not a bone: pac falls back to the player's origin turned to his eye yaw; the dumped
            # model stands at the origin facing +x, the way he faces when his eyes are level ahead
            return np.eye(4)

    def world(self, uid):
        if uid in self.cache:
            return self.cache[uid]
        me = self.ev["nodes"].get(uid) or self.o.parts[uid]["self"]
        parent_uid = self.o.parent[uid]
        parent = self.o.parts[parent_uid]["self"] if parent_uid else None
        local = np.eye(4)
        local[:3, :3] = mk.source_angle_matrix(*vec(me.get("Angles")))
        local[:3, 3] = vec(me.get("Position"))
        if parent is not None and parent["ClassName"] == "jiggle":
            base = self.jiggle_world(parent_uid)
        else:
            base = self.bone_matrix(me.get("Bone", "head"))
        w = base @ local
        aim = me.get("AimPartUID")
        if aim:
            # base_movable CalcAngles: Angles + (aim part position - own position):Angle(), added per component
            d = self.world(aim)[:3, 3] - w[:3, 3]
            pitch = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
            yaw = math.degrees(math.atan2(d[1], d[0]))
            p, y, r = vec(me.get("Angles"))
            w[:3, :3] = mk.source_angle_matrix(p + pitch, y + yaw, r)
        self.cache[uid] = w
        return w

    def jiggle_world(self, uid):
        """Settled jiggle: bone frame + offset, pitch/roll held level, dropped onto the ground."""
        me = self.ev["nodes"].get(uid) or self.o.parts[uid]["self"]
        local = np.eye(4)
        local[:3, :3] = mk.source_angle_matrix(*vec(me.get("Angles")))
        local[:3, 3] = vec(me.get("Position"))
        w = self.bone_matrix(me.get("Bone", "head")) @ local
        p, y, r = mk.matrix_to_source_angle(w[:3, :3])
        out = np.eye(4)
        out[:3, :3] = mk.source_angle_matrix(0 if me.get("ConstrainPitch") else p, y, 0 if me.get("ConstrainRoll") else r)
        out[:3, 3] = w[:3, 3]
        if me.get("Ground"):
            out[2, 3] = self.ground_z
        return out


def beam_points(start, end, dir_a, dir_b, bend, res=28, frequency=1.0):
    """pac.DrawBeam's curve."""
    pts = []
    for i in range(res + 1):
        frac = i / res
        bendmult = math.sin(frac * math.pi * frequency) * bend
        pts.append(start + (end - start) * frac + (dir_a * bendmult) * (1 - frac) + (dir_b * bendmult) * frac)
    return np.array(pts)
