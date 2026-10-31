"""IV drip on a wheeled stand, and a cannula taped to the back of a hand: geometry + textures.

Stand frame: origin on the ground under the pole, +z up. The two hooks at the top point along
+y and -y; the bag hangs on the +y hook (the side facing the patient), its printed face toward
+x. Below the bag: the drip chamber, then the line hangs down to a roller clamp; from there the
tube is a beam in the outfit, so it can reach the hand wherever the arm goes.

Cannula frame: origin where the tube leaves the hub, +x along the tube (toward the wrist),
+z out of the skin.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import meshkit as mk

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _px(x0, y0, x1, y1, size=512, inset=2):
    return ((x0 + inset) / size, (y0 + inset) / size, (x1 - inset) / size, (y1 - inset) / size)


# stand atlas (opaque)
CHROME = _px(0, 0, 256, 256)
RUBBER = _px(256, 0, 384, 128)
GREY = _px(384, 0, 512, 128)
WHITE = _px(256, 128, 384, 256)
BLUE = _px(384, 128, 512, 256)

# bag atlas (RGBA)
BAG = _px(0, 0, 512, 384)          # around x height of the bag
CHAMBER = _px(0, 384, 256, 512)
PORT = _px(256, 384, 384, 512)
SPIKE = _px(384, 384, 512, 512)

# cannula atlas (RGBA)
TAPE = _px(0, 0, 256, 160, 256)
HUB = _px(0, 160, 128, 256, 256)
WINGS = _px(128, 160, 256, 256, 256)

POLE_TOP = 75.5
HOOK_Y = 3.3
BAG_TOP, BAG_BOTTOM, BAG_W, BAG_T = 75.2, 68.4, 3.8, 1.5
CHAMBER_TOP, CHAMBER_BOTTOM, CHAMBER_R = 67.4, 64.9, 0.42
CLAMP_Z = 56.0
LEG_R, LEG_Z0, LEG_Z1 = 7.2, 2.9, 2.0
CASTER_R = 0.85


def drip_point():
    """Where drops form inside the drip chamber (stand frame)."""
    return np.array([0.0, HOOK_Y, CHAMBER_TOP - 0.55])


def chamber_outlet():
    """Bottom of the drip chamber, where the line leaves (stand frame)."""
    return np.array([0.0, HOOK_Y, CHAMBER_BOTTOM - 0.25])


def clamp_point():
    """The roller clamp on the line; the tube to the hand starts here (stand frame)."""
    return np.array([0.0, HOOK_Y, CLAMP_Z])


# --- stand -------------------------------------------------------------------------------

def _caster(angle):
    """A swivel caster at the end of the leg pointing at `angle` (degrees)."""
    a = np.radians(angle)
    d = np.array([np.cos(a), np.sin(a), 0.0])
    tip = d * LEG_R + np.array([0, 0, LEG_Z1])
    parts = [mk.tube([tip, tip - [0, 0, 0.55]], 0.17, 8, GREY)]
    fork = mk.box((0.9, 0.75, 0.35), (0, 0, 0), GREY).transformed(mk.trans(*(tip - [0, 0, 0.6])))
    parts.append(fork)
    # wheel: axle sideways to the leg, trailing 0.35 behind the stem
    side = np.array([-d[1], d[0], 0.0])
    centre = tip - d * 0.35 + np.array([0, 0, CASTER_R - LEG_Z1])
    tire = mk.lathe([(0.45, -0.28), (0.72, -0.28), (CASTER_R, -0.12), (CASTER_R, 0.12), (0.72, 0.28),
                     (0.45, 0.28), (0.45, -0.28)], 16, RUBBER, smooth_angle=50)
    hub = mk.lathe([(0, -0.3), (0.46, -0.3), (0.46, 0.3), (0, 0.3)], 12, GREY)
    wheel = mk.concat([tire, hub])
    # lathe axis is z: turn it onto `side`
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, side)
    ang = np.degrees(np.arccos(np.clip(z @ side, -1, 1)))
    rot = mk.rot(axis / np.linalg.norm(axis), ang) if np.linalg.norm(axis) > 1e-9 else np.eye(4)
    parts.append(wheel.transformed(mk.trans(*centre) @ rot))
    return mk.concat(parts)


def stand_mesh():
    parts = []
    # hub of the base
    parts.append(mk.lathe([(0, 1.7), (0.95, 1.7), (1.15, 1.95), (1.15, 3.35), (0.85, 3.7), (0, 3.7)], 24, CHROME))
    for k in range(5):
        a = np.radians(90 + 72 * k)
        d = np.array([np.cos(a), np.sin(a), 0.0])
        path = [d * 0.7 + [0, 0, LEG_Z0], d * 3.5 + [0, 0, (LEG_Z0 * 1.4 + LEG_Z1) / 2.4], d * LEG_R + [0, 0, LEG_Z1]]
        parts.append(mk.sweep(path, mk.rounded_rect(0.95, 0.42, 0.18, 3), CHROME, up_hint=(0, 0, 1)))
        parts.append(mk.disc_at(path[-1], d, [0, 0, 1], 0.24, 10, CHROME))
        parts.append(_caster(np.degrees(a)))
    # pole: lower tube, height-adjust collar with its knob, upper tube, round cap
    parts.append(mk.tube([(0, 0, 3.4), (0, 0, 42.0)], 0.42, 16, CHROME))
    parts.append(mk.lathe([(0.35, 41.0), (0.62, 41.1), (0.66, 41.6), (0.66, 42.5), (0.5, 42.8), (0.32, 42.8)], 20, CHROME))
    parts.append(mk.tube([(0, 0, 41.8), (0.0, -1.25, 41.8)], 0.16, 8, CHROME))
    parts.append(mk.lathe([(0, -0.25), (0.34, -0.25), (0.38, 0), (0.34, 0.25), (0, 0.25)], 12, GREY)
                 .transformed(mk.trans(0, -1.35, 41.8) @ mk.rot((1, 0, 0), 90)))
    parts.append(mk.tube([(0, 0, 42.0), (0, 0, POLE_TOP)], 0.32, 14, CHROME))
    parts.append(mk.lathe([(0, POLE_TOP - 0.1), (0.36, POLE_TOP - 0.05), (0.3, POLE_TOP + 0.3), (0, POLE_TOP + 0.38)], 12, CHROME))
    # two hooks: out along +-y, curling up at the ends
    for s in (1, -1):
        path = [(0, 0, POLE_TOP - 0.2), (0, s * 1.6, POLE_TOP - 0.2), (0, s * 2.7, POLE_TOP - 0.05),
                (0, s * HOOK_Y, POLE_TOP + 0.35), (0, s * (HOOK_Y + 0.35), POLE_TOP + 1.0),
                (0, s * (HOOK_Y + 0.2), POLE_TOP + 1.6)]
        parts.append(mk.tube(path, 0.17, 8, CHROME, up_hint=(1, 0, 0)))
    # roller clamp on the line under the bag: white body, blue wheel
    c = clamp_point()
    parts.append(mk.box((0.55, 0.62, 1.7), c + [0.0, 0, 0.0], WHITE))
    parts.append(mk.lathe([(0, -0.16), (0.3, -0.16), (0.32, 0), (0.3, 0.16), (0, 0.16)], 12, BLUE)
                 .transformed(mk.trans(*(c + [0.28, 0, 0.2])) @ mk.rot((0, 1, 0), 90)))
    # the short stretch of line from the chamber down into the clamp
    parts.append(mk.tube([chamber_outlet(), c + [0, 0, 0.85]], 0.11, 8, WHITE, caps=False))
    return mk.concat(parts)


# --- bag and drip chamber ------------------------------------------------------------------

def bag_mesh():
    """Soft pouch hanging from the +y hook, face toward +x, port and drip chamber below."""
    cx, cy = 0.0, HOOK_Y
    H = BAG_TOP - BAG_BOTTOM

    def fn(u, t):
        z = BAG_BOTTOM + H * t
        width = BAG_W * (0.42 + 0.58 * np.clip(t / 0.2, 0, 1) ** 0.6)
        thick = BAG_T * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.45 + 0.04
        th = 2 * np.pi * u
        c, s = np.cos(th), np.sin(th)
        # squarish section: flat broad faces toward +-x, rounded sides
        sx = np.sign(c) * np.abs(c) ** 0.7
        sy = np.sign(s) * np.abs(s) ** 0.35
        return np.stack([cx + thick / 2 * sx, cy + width / 2 * sy, z], -1)

    # surface() puts t = 0 (the bottom of the bag) at the region's first image row; flip the rect
    # so the texture is painted upright (top row = top of the bag)
    bag = mk.surface(fn, 40, 24, (BAG[0], BAG[3], BAG[2], BAG[1]), smooth_angle=70)
    # the hanger tab at the top, on the hook
    tab = mk.box((0.12, 0.9, 0.7), (cx, cy, BAG_TOP + 0.25), PORT)
    port = mk.tube([(cx, cy, BAG_BOTTOM + 0.1), (cx, cy, CHAMBER_TOP + 0.3)], 0.2, 10, PORT)
    # spike + chamber: white cap, clear body with a rounded bottom
    cap = mk.lathe([(0.0, CHAMBER_TOP + 0.35), (0.3, CHAMBER_TOP + 0.35), (0.5, CHAMBER_TOP + 0.1),
                    (0.5, CHAMBER_TOP - 0.25), (0.0, CHAMBER_TOP - 0.25)], 16, SPIKE).transformed(mk.trans(cx, cy, 0))
    prof = [(0.0, CHAMBER_BOTTOM - 0.25), (0.12, CHAMBER_BOTTOM - 0.25), (0.3, CHAMBER_BOTTOM - 0.05),
            (CHAMBER_R, CHAMBER_BOTTOM + 0.3), (CHAMBER_R, CHAMBER_TOP - 0.2), (0.0, CHAMBER_TOP - 0.2)]
    chamber = mk.lathe(prof, 18, CHAMBER).transformed(mk.trans(cx, cy, 0))
    return mk.concat([bag, tab, port, cap, chamber])


# --- cannula -------------------------------------------------------------------------------

def cannula_mesh():
    """Clear dressing over the hub, a winged pink catheter hub; the tube leaves along +x."""
    tape = mk.surface(lambda u, v: np.stack([-1.9 + 2.6 * u, -0.85 + 1.7 * v,
                                             0.03 + 0.06 * np.sin(np.pi * u) * np.sin(np.pi * v)], -1),
                      10, 6, TAPE, smooth_angle=80)
    hub = mk.tube([(-1.25, 0, 0.16), (-0.1, 0, 0.16)], 0.17, 10, HUB)
    port = mk.tube([(-0.1, 0, 0.16), (0.12, 0, 0.16)], 0.11, 8, HUB)
    wings = mk.box((0.55, 1.15, 0.05), (-0.85, 0, 0.07), WINGS)
    return mk.concat([tape, hub, port, wings])


# --- textures -------------------------------------------------------------------------------

def paint_stand(path, size=512, seed=21):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3))
    t = np.linspace(0, 1, 256)[:, None, None]
    bands = 150 + 80 * np.cos(2 * np.pi * t) ** 2 - 40 * np.sin(np.pi * t) ** 10
    img[0:256, 0:256] = np.clip(bands * np.array([1.0, 1.02, 1.05]) + rng.normal(0, 1.5, (256, 256, 1)), 0, 255)
    for (x0, y0, x1, y1), col, n in (((256, 0, 384, 128), (28, 28, 30), 3), ((384, 0, 512, 128), (128, 130, 134), 3),
                                     ((256, 128, 384, 256), (236, 238, 240), 2), ((384, 128, 512, 256), (40, 110, 200), 3)):
        img[y0:y1, x0:x1] = np.clip(np.array(col, float) + rng.normal(0, n, (y1 - y0, x1 - x0, 1)), 0, 255)
    Image.fromarray(img.astype(np.uint8)).save(path, quality=92)


def paint_bag(path, size=512, seed=22):
    """RGBA: frosted saline bag (fluid in the lower part, air above), printed label on its front,
    clear drip chamber with fluid at the bottom, white port and spike."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 4))
    # bag: rows = height from the top of the region (bag top) down; columns = around the bag
    h = 384
    rows = np.arange(h)[:, None]
    t = 1 - rows / h                                      # 1 = top of the bag (painted upright)
    fluid = np.broadcast_to(t < 0.82, (h, size))
    rgb = np.where(fluid[..., None], np.array([226.0, 234, 240]), np.array([244.0, 246, 248]))
    alpha = np.where(fluid, 0.62, 0.32)
    rgb = rgb + rng.normal(0, 2, (h, size, 1))
    img[:h, :, :3] = rgb
    img[:h, :, 3] = alpha * 255
    # seam lines and graduations on the bag
    img[:4, :, 3] = 230
    img[:4, :, :3] = 250
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(pil)
    # label on the front face (u around 0..0.25 and 0.75..1 is the +x face; centre it on u = 0)
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT_REG, 12)
    for x_off in (0, size):                                # draw across the wrap at u = 0
        x0 = x_off - 92
        d.rectangle((x0, 120, x0 + 184, 300), fill=(250, 250, 250, 235), outline=(60, 110, 190, 255), width=3)
        d.text((x0 + 14, 132), "0.9% Sodium", font=font, fill=(30, 70, 150, 255))
        d.text((x0 + 14, 154), "Chloride", font=font, fill=(30, 70, 150, 255))
        d.text((x0 + 14, 180), "Injection USP", font=small, fill=(40, 40, 40, 255))
        d.text((x0 + 14, 198), "1000 mL", font=font, fill=(20, 20, 20, 255))
        for k in range(5):
            d.line((x0 + 14, 228 + k * 13, x0 + 170, 228 + k * 13), fill=(120, 120, 120, 220), width=2)
    # chamber: clear, fluid in its lower third
    ch = np.zeros((128, 256, 4))
    ct = np.arange(128)[:, None] / 128                     # 0 = bottom of the profile (rounded end)
    ch[..., :3] = np.array([232.0, 240, 246])
    ch[..., 3] = np.where(ct < 0.38, 150, 70)
    pil.paste(Image.fromarray(ch.astype(np.uint8), "RGBA"), (0, 384))
    port = np.ones((128, 128, 4)) * np.array([238.0, 240, 242, 255])
    pil.paste(Image.fromarray(port.astype(np.uint8), "RGBA"), (256, 384))
    spike = np.ones((128, 128, 4)) * np.array([245.0, 245, 247, 255])
    spike[64:, :, :3] = (70, 120, 200)
    pil.paste(Image.fromarray(spike.astype(np.uint8), "RGBA"), (384, 384))
    pil.save(path)


def paint_cannula(path, size=256, seed=23):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 4))
    tape = np.ones((160, size, 4)) * np.array([246.0, 248, 250, 200])
    yy, xx = np.mgrid[0:160, 0:size]
    tape[(xx % 16 == 0) | (yy % 16 == 0), 3] = 225
    tape[:6, :, 3] = tape[-6:, :, 3] = 235
    tape[:, :6, 3] = tape[:, -6:, 3] = 235
    tape[..., :3] += rng.normal(0, 2, (160, size, 1))
    img[:160] = tape
    img[160:, :128] = (232, 96, 150, 255)               # pink 20G hub
    img[160:, 128:] = (246, 246, 246, 255)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGBA").save(path)
