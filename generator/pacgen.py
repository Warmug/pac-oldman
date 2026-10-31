"""Minimal PAC3 outfit builder.

Parts are plain objects (class name, ordered properties, children) that
serialize to the luadata text PAC reads from garrysmod/data/pac3/*.txt.

Only literals and Vector()/Angle() calls are emitted: pac's luadata decoder
runs a bytecode whitelist and rejects anything else (arithmetic, locals, ...).
"""
import hashlib

# Not a bone: pac.GetBonePosAng (lua/pac3/core/client/bones.lua) falls back to the player's origin
# turned to his eye yaw, with no pitch or roll, for any name that isn't a bone, attachment or special.
ORIGIN_BONE = "none"


class Vector(tuple):
    def __new__(cls, x=0.0, y=0.0, z=0.0):
        return super().__new__(cls, (float(x), float(y), float(z)))


class Angle(tuple):
    def __new__(cls, p=0.0, y=0.0, r=0.0):
        return super().__new__(cls, (float(p), float(y), float(r)))


def num(x, places=4):
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        return str(x)
    x = round(float(x), places)
    if x == 0:
        return "0"
    if x == int(x):
        return str(int(x))
    return f"{x:.{places}f}".rstrip("0").rstrip(".")


def lua_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class Part:
    def __init__(self, class_name, name, **props):
        self.class_name = class_name
        self.props = {"Name": name}
        self.props.update(props)
        self.children = []
        self.uid = None

    @property
    def name(self):
        return self.props["Name"]

    def add(self, *parts):
        self.children.extend(p for p in parts if p is not None)
        return self

    def walk(self, depth=0, parent=None):
        yield self, depth, parent
        for child in self.children:
            yield from child.walk(depth + 1, self)


def assign_uids(root, seed):
    """Deterministic 64-hex UIDs from the part's path, so regenerating keeps references stable."""
    seen = set()

    def visit(part, path):
        part.uid = hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()
        if part.uid in seen:
            raise ValueError(f"duplicate uid for {path}")
        seen.add(part.uid)
        for i, child in enumerate(part.children):
            visit(child, f"{path}/{i}:{child.class_name}:{child.name}")

    visit(root, "root")


def lua_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return num(v)
    if isinstance(v, str):
        return lua_string(v)
    if isinstance(v, Vector):
        return "Vector({}, {}, {})".format(*(num(c) for c in v))
    if isinstance(v, Angle):
        return "Angle({}, {}, {})".format(*(num(c, 3) for c in v))
    if isinstance(v, Part):
        if v.uid is None:
            raise ValueError(f"part {v.name!r} referenced before uids were assigned")
        return lua_string(v.uid)
    raise TypeError(f"cannot encode {type(v).__name__}: {v!r}")


def encode_part(part, indent=0):
    t = "\t" * indent
    out = [t + '["self"] = {']
    fields = [("ClassName", part.class_name), ("UniqueID", part.uid)] + list(part.props.items())
    for key, value in fields:
        out.append(f'{t}\t["{key}"] = {lua_value(value)},')
    out.append(t + "},")
    out.append(t + '["children"] = {')
    for i, child in enumerate(part.children, 1):
        out.append(f"{t}\t[{i}] = {{")
        out.append(encode_part(child, indent + 2))
        out.append(t + "\t},")
    out.append(t + "},")
    return "\n".join(out)


def encode_outfit(root, seed):
    assign_uids(root, seed)
    return "[1] = {\n" + encode_part(root, 1) + "\n},\n"


# --- part helpers -------------------------------------------------------------

def group(name, *children, **props):
    return Part("group", name, **props).add(*children)


def bone(name, bone_name, ang=(0, 0, 0), pos=None, *children, **props):
    p = Part("bone3", name, Bone=bone_name, Angles=Angle(*ang), **props)
    if pos is not None:
        p.props["Position"] = Vector(*pos)
    return p.add(*children)


def proxy(name, variable, expression, axis=None, **props):
    p = Part("proxy", name, VariableName=variable, Expression=expression, **props)
    if axis:
        p.props["Axis"] = axis
    return p


def command_event(name, command, show_when_on):
    """pac_event <command> state. show_when_on=True shows the parent only while it is on."""
    return Part("event", name, Event="command", Arguments=f"{command}@@0@@0",
                Operator="equal", Invert=bool(show_when_on))


def holdtype_event(name, holdtypes, show_when_match):
    """Active weapon holdtype is one of `holdtypes` (pac splits the argument on ';')."""
    return Part("event", name, Event="holdtype", Arguments=";".join(holdtypes),
                Operator="equal", Invert=bool(show_when_match))


def crouch_event(name, show_when_crouching):
    return Part("event", name, Event="is_crouching", Operator="equal", Invert=bool(show_when_crouching))


def sit_event(name, show_when_sitting):
    """In a seat or chair (pac's is_sitting: a vehicle other than the prisoner pod, or sit-anywhere)."""
    return Part("event", name, Event="is_sitting", Operator="equal", Invert=bool(show_when_sitting))


def material_3d(name, **params):
    return Part("material_3d", name, **params)


def material_2d(name, **params):
    return Part("material_2d", name, **params)


def model(name, url, bone_name, material, pos=(0, 0, 0), ang=(0, 0, 0), *children, **props):
    p = Part("model2", name, Model=url, Bone=bone_name, Material=material,
             Position=Vector(*pos), Angles=Angle(*ang), **props)
    return p.add(*children)
