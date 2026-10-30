#!/usr/bin/env python3
"""Builds the old man outfit for models/player/Suits/male_08_shirt_tie.mdl.

    python3 generator/make_outfit.py [--dump path/to/male08_dump.txt]

Writes outfit/old_man_2_male08.txt plus the meshes/textures under props/.
Toggles: pac_event cane 2 | smoke 2 | glasses 2 | mask 2 | tank 2
"""
import argparse
import os

from pacgen import (Angle, Part, Vector, bone, command_event, encode_outfit, group, holdtype_event,
                    material_2d, material_3d, model, proxy)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = "https://raw.githubusercontent.com/Warmug/pac-oldman/main/"
OUTFIT = os.path.join(REPO, "outfit", "old_man_2_male08.txt")
SEED = "oldman2-male08"

# holdtypes that count as "no weapon out": the arm posture, the cane in his hand and the
# cigarette in his fingers only apply with these. Anything else gets the plain weapon animation.
UNARMED = ["normal", "passive", "fist"]

SPEED = "owner_velocity_length()"
STILL = f"(1 - clamp({SPEED}/50, 0, 1))"


def url(path):
    return RAW + path


# --- posture ----------------------------------------------------------------------

def posture(fit):
    walk = f"clamp({SPEED}/120, 0, 1)"
    hunch = group(
        "posture: hunch (always on)",
        bone("sink pelvis (keeps feet on ground)", "pelvis", (0, 0, 0), fit.pelvis_sink),
        bone("hunch 1: lower back", "spine", fit.hunch[0], None,
             proxy("idle balance sway", "Angles", f"1.5*sin(time()*0.7)*{STILL}", "p")),
        bone("hunch 2: mid back", "spine 1", fit.hunch[1], None,
             proxy("lean further forward when walking", "Angles",
                   f"{fit.hunch[1][1]:g} + 6*{walk}", "y")),
        bone("hunch 3: upper back", "spine 2", fit.hunch[2]),
        bone("hunch 4: the hump", "spine 4", fit.hunch[3]),
        bone("neck cranes back so he looks ahead", "neck", fit.neck, None,
             proxy("keep eyes up while walking, less crane when aiming up", "Angles",
                   fit.neck_expression(walk), "y")),
        bone("head tremor", "head", (0, 0, 0), None,
             proxy("head tremor (delete this if it bugs you)", "Angles",
                   "1.1*sin(time()*23) + 0.4*sin(time()*37)", "r")),
    )
    for side in ("left", "right"):
        hunch.add(bone(f"bent knees: hip {side[0].upper()}", f"{side} thigh", fit.thigh),
                  bone(f"bent knees: knee {side[0].upper()}", f"{side} calf", fit.calf),
                  bone(f"bent knees: keep foot flat {side[0].upper()}", f"{side} foot", fit.foot))

    arms = group(
        "posture: arms (only with no weapon out)",
        holdtype_event("only when holdtype is normal/passive/fist", UNARMED, True),
        bone("rounded shoulders L", "left clavicle", fit.clavicle_l),
        bone("rounded shoulders R", "right clavicle", fit.clavicle_r),
        bone("arms hang straight down L", "left upperarm", fit.upperarm_hang, None,
             command_event("steps aside while the cigarette is in his hand", "smoke", False)),
        bone("arms hang straight down R", "right upperarm", fit.upperarm_hang, None,
             command_event("steps aside while the cane is in his hand", "cane", False)),
        bone("slight elbow bend L", "left forearm", fit.forearm_bend, None,
             command_event("steps aside while the cigarette is in his hand", "smoke", False)),
        bone("slight elbow bend R", "right forearm", fit.forearm_bend, None,
             command_event("steps aside while the cane is in his hand", "cane", False)),
    )
    return hunch, arms


# --- cane -------------------------------------------------------------------------

CANE_FINGERS = {"right finger 1": -62, "right finger 11": -78, "right finger 12": -40,
                "right finger 2": -66, "right finger 21": -82, "right finger 22": -42,
                "right finger 3": -70, "right finger 31": -84, "right finger 32": -40,
                "right finger 4": -72, "right finger 41": -80, "right finger 42": -38}


def cane_group(fit):
    in_hand = group(
        "cane in his hand (no weapon out)",
        holdtype_event("only when holdtype is normal/passive/fist", UNARMED, True),
        model("cane", url("props/cane.obj"), "right hand", "cane_wood", *fit.cane_hand),
    )
    for b, yaw in CANE_FINGERS.items():
        in_hand.add(bone(f"cane grip: {b}", b, (0, yaw, 0)))
    in_hand.add(bone("cane grip: right finger 0", "right finger 0", (0, 18, -10)),
                bone("cane grip: right finger 01", "right finger 01", (0, 28, 0)),
                bone("cane grip: right finger 02", "right finger 02", (0, 22, 0)))
    for b, ang in fit.cane_arm.items():
        in_hand.add(bone(f"cane grip: {b}", b, ang))

    on_belt = group(
        "cane hooked on his belt (weapon out)",
        holdtype_event("hidden while holdtype is normal/passive/fist", UNARMED, False),
        model("cane on belt", url("props/cane.obj"), "pelvis", "cane_wood", *fit.cane_belt),
    )
    return group(
        "cane  [pac_event cane]",
        command_event("toggle with: pac_event cane 2", "cane", True),
        material_3d("cane_wood", basetexture=url("props/cane.jpg"), bumpmap=url("props/flat_n.png"),
                    phong=True, phongboost=2.5, phongexponent=30, phongfresnelranges=Vector(0.3, 0.7, 1)),
        in_hand, on_belt,
    )


# --- cigarette ----------------------------------------------------------------------

CIG_FINGERS = {"left finger 1": (0, -12, 0), "left finger 11": (0, -22, 0), "left finger 12": (0, -12, 0),
               "left finger 2": (0, -14, 0), "left finger 21": (0, -24, 0), "left finger 22": (0, -12, 0),
               "left finger 3": (0, -48, 0), "left finger 31": (0, -70, 0), "left finger 32": (0, -40, 0),
               "left finger 4": (0, -55, 0), "left finger 41": (0, -72, 0), "left finger 42": (0, -40, 0),
               "left finger 0": (0, 10, 0), "left finger 01": (0, 15, 0)}

DRAG_PERIOD = 13
_T = f"(time()%{DRAG_PERIOD})"
_S = f"clamp(min({_T}/1.0, (3.4-{_T})/1.0), 0, 1)"
DRAG = f"({_S}*{_S}*(3-2*{_S}))*{STILL}"


def drag_expression(base, delta):
    return ", ".join(f"{b:.2f} + ({d:.2f})*{DRAG}" for b, d in zip(base, delta))


def smoke_bits(bone_name, tip, prefix):
    ember = Part("sprite", f"{prefix}ember glow", Bone=bone_name, Position=Vector(*tip),
                 SpritePath="sprites/light_glow02_add", Color=Vector(255, 120, 40), Size=0.6, Alpha=0.9)
    ember.add(proxy("ember flickers and flares up on a drag", "Size",
                    f"0.55 + 0.08*sin(time()*17) + 0.06*sin(time()*29) + 0.9*clamp(min(({_T}-1.0)/0.3, (2.4-{_T})/0.3), 0, 1)"))
    smoke = Part("particles", f"{prefix}smoke from the tip", Bone=bone_name, Position=Vector(*tip),
                 Material="particle/smokesprites_0001", StartSize=0.35, EndSize=5.5, DieTime=3.2,
                 FireDelay=0.14, NumberParticles=1, Velocity=3, Spread=0.4, Gravity=Vector(0, 0, 9),
                 AirResistance=4, StartAlpha=55, EndAlpha=0, Color1=Vector(205, 205, 210),
                 Color2=Vector(170, 170, 178), Lighting=False, Collide=False, RandomRollSpeed=1.5)
    return ember, smoke


def cig_group(fit):
    in_hand = group(
        "cigarette in his fingers (no weapon out)",
        holdtype_event("only when holdtype is normal/passive/fist", UNARMED, True),
        model("cigarette", url("props/cigarette.obj"), "left hand", "cigarette_paper", *fit.cig_hand),
        *smoke_bits("left hand", fit.cig_hand_tip, ""),
    )
    for b, ang in CIG_FINGERS.items():
        in_hand.add(bone(f"cigarette hold: {b}", b, ang))

    drag = group("takes a drag every 13s (mask off)",
                 command_event("not while the oxygen mask is on", "mask", False))
    hold = group("just holds it (mask on)",
                 command_event("only while the oxygen mask is on", "mask", True))
    for b, (base, delta) in fit.cig_arm.items():
        drag.add(bone(f"smoking arm: {b}", b, base, None,
                      proxy("take a drag every 13s while standing still", "Angles", drag_expression(base, delta))))
        hold.add(bone(f"smoking arm: {b}", b, base))
    exhale = Part("particles", "exhaled smoke", Bone="head", Position=Vector(*fit.mouth), Angles=Angle(*fit.exhale_angles),
                  Material="particle/smokesprites_0001", StartSize=0.8, EndSize=7, DieTime=2.6, FireDelay=0.05,
                  NumberParticles=1, Velocity=14, Spread=0.25, Gravity=Vector(0, 0, 5), AirResistance=12,
                  StartAlpha=70, EndAlpha=0, Color1=Vector(210, 210, 215), Color2=Vector(180, 180, 185),
                  Lighting=False, Collide=False)
    exhale.add(proxy("only right after a drag", "Hide",
                     f"((({_T}) > 3.5 and ({_T}) < 5.4) and {STILL} > 0.5) and 0 or 1"))
    drag.add(exhale)
    in_hand.add(drag, hold)

    in_mouth = group(
        "cigarette in his mouth (weapon out, no mask)",
        holdtype_event("hidden while holdtype is normal/passive/fist", UNARMED, False),
        command_event("hidden while the oxygen mask is on", "mask", False),
        model("cigarette in mouth", url("props/cigarette.obj"), "head", "cigarette_paper", *fit.cig_mouth),
        *smoke_bits("head", fit.cig_mouth_tip, "mouth "),
    )
    return group(
        "cigarette  [pac_event smoke]",
        command_event("toggle with: pac_event smoke 2", "smoke", True),
        material_3d("cigarette_paper", basetexture=url("props/cigarette.jpg")),
        in_hand, in_mouth,
    )


# --- glasses ------------------------------------------------------------------------

def glasses_group(fit):
    return group(
        "glasses  [pac_event glasses]",
        command_event("toggle with: pac_event glasses 2", "glasses", True),
        material_3d("oldman_wire", basetexture=url("oldman/oldman_glasses.jpg"), bumpmap=url("props/flat_n.png"),
                    phong=True, phongboost=6, phongexponent=50, envmap="env_cubemap",
                    envmaptint=Vector(0.35, 0.3, 0.2)),
        material_3d("oldman_lens", basetexture=url("props/lens.png"), translucent=True, alpha=0.3,
                    envmap="env_cubemap", envmaptint=Vector(0.25, 0.25, 0.25)),
        model("frames", url(fit.glasses_obj), "head", "oldman_wire"),
        model("lenses", url(fit.lenses_obj), "head", "oldman_lens"),
    )


# --- oxygen -------------------------------------------------------------------------

def tube_beam(name, start_bone, start_pos, start_ang, end_part, bend):
    return Part("beam", name, Bone=start_bone, Position=Vector(*start_pos), Angles=Angle(*start_ang),
                EndPointUID=end_part, Material="o2_tube", Width=0.42, Bend=bend, Resolution=28,
                Frequency=1, StartColor=Vector(232, 240, 240), EndColor=Vector(232, 240, 240),
                StartAlpha=0.85, EndAlpha=0.85, TextureStretch=1, Translucent=True)


def marker(name, bone_name, pos, ang):
    """Invisible anchor for a tube end; its forward axis is the direction the tube leaves in."""
    return Part("sprite", name, Bone=bone_name, Position=Vector(*pos), Angles=Angle(*ang),
                SpritePath="sprites/light_glow02_add", Size=0, Alpha=0)


def oxygen_groups(fit):
    clip = marker("tube clip on his shirt", fit.tube_clip_bone, *fit.tube_clip)
    mask_outlet_to_clip = tube_beam("tube: mask down to his shirt", "head", *fit.mask_outlet, clip, fit.tube_bend_upper)
    mask = group(
        "oxygen mask  [pac_event mask]",
        command_event("toggle with: pac_event mask 2", "mask", True),
        material_3d("o2_mask_plastic", basetexture=url("props/o2_mask.png"), bumpmap=url("props/flat_n.png"),
                    translucent=True, alpha=0.42, phong=True, phongboost=8, phongexponent=60,
                    phongfresnelranges=Vector(0.4, 0.8, 1), envmap="env_cubemap", envmaptint=Vector(0.12, 0.12, 0.12)),
        material_3d("o2_mask_strap", basetexture=url("props/o2_strap.jpg"), bumpmap=url("props/flat_n.png"),
                    phong=True, phongboost=1.5, phongexponent=12, halflambert=True),
        material_2d("o2_tube", basetexture=url("props/o2_tube.png"), vertexcolor=True, vertexalpha=True,
                    translucent=True),
        model("mask shell", url("props/o2_mask.obj"), "head", "o2_mask_plastic"),
        model("elastic strap and nose clip", url("props/o2_strap.obj"), "head", "o2_mask_strap"),
        clip, mask_outlet_to_clip,
    )

    barb = marker("tube connector on the regulator", "pelvis", fit.barb_pos, (90, 0, 0))
    cart = Part("jiggle", "cart rolls along beside him", Bone="pelvis", Position=Vector(*fit.cart_pos),
                Angles=Angle(*fit.cart_ang), Speed=fit.cart_speed, Strain=fit.cart_strain, Ground=True,
                ConstrainPitch=True, ConstrainRoll=True, ResetOnHide=True)
    cart.add(
        model("cylinder and cart", url("props/o2_cart.obj"), "pelvis", "o2_cart"),
        *[model(f"wheel {side}", url("props/o2_wheel.obj"), "pelvis", "o2_cart", pos, (0, 0, 0),
                proxy("wheel rolls when he walks", "Angles", f"owner_velocity_forward()*ftime()*{fit.wheel_deg_per_unit:.3f}",
                      "p", Additive=True, ZeroEyePitch=True))
          for side, pos in fit.wheels],
        barb,
    )
    tank = group(
        "oxygen tank cart  [pac_event tank]",
        command_event("toggle with: pac_event tank 2", "tank", True),
        material_3d("o2_cart", basetexture=url("props/o2_cart.jpg"), bumpmap=url("props/flat_n.png"),
                    phong=True, phongboost=3, phongexponent=24, phongfresnelranges=Vector(0.3, 0.6, 1)),
        cart,
    )
    tube = group(
        "oxygen tube to the tank (mask + tank)",
        command_event("needs the mask on", "mask", True),
        command_event("needs the tank on", "tank", True),
        tube_beam("tube: shirt to the tank", fit.tube_clip_bone, *fit.tube_clip, barb, fit.tube_bend_lower),
    )
    return mask, tank, tube


def build(fit):
    hunch, arms = posture(fit)
    mask, tank, tube = oxygen_groups(fit)
    return group(
        "Old man (male_08) - posture, cane, cigarette, glasses, oxygen",
        hunch, arms, cane_group(fit), cig_group(fit), glasses_group(fit), mask, tank, tube,
        OwnerName="self", EditorExpand=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    import fit as fitmod
    f = fitmod.Fit(args.dump)
    f.write_assets(REPO)
    root = build(f)
    os.makedirs(os.path.dirname(OUTFIT), exist_ok=True)
    with open(OUTFIT, "w", newline="\n") as fh:
        fh.write(encode_outfit(root, SEED))
    print("wrote", OUTFIT)


if __name__ == "__main__":
    main()
