"""Portable medical oxygen cylinder on a two-wheeled cart: geometry + texture atlas.

Cart frame (cart-local): origin on the ground under the cylinder axis,
+x = front (the label side), +y = left, +z = up. Frame, handle and wheels are
at the back (-x). The regulator sticks out to REG_SIDE (-1 = the cart's right),
which is the side the old man walks on.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import meshkit as mk

ATLAS = 1024
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def px(x0, y0, x1, y1, inset=3):
    return ((x0 + inset) / ATLAS, (y0 + inset) / ATLAS, (x1 - inset) / ATLAS, (y1 - inset) / ATLAS)


BODY = px(0, 0, 512, 1024)
LABEL = px(512, 0, 768, 256, 1)
GAUGE = px(768, 0, 1024, 256, 1)
CHROME = px(512, 256, 768, 384)
GREEN = px(768, 256, 1024, 384)
RUBBER = px(512, 384, 768, 512)
FRAME = px(768, 384, 1024, 512)
HUB = px(512, 512, 640, 640)
KNOB = px(640, 512, 768, 640)
STRAP = px(768, 512, 1024, 640)

R_BODY = 2.2
Z_BOTTOM, Z_SHOULDER, Z_NECK, Z_TOP = 1.7, 21.8, 25.0, 26.2
R_NECK = 0.72
Z_GREEN = 19.4
LABEL_Z, LABEL_SIZE = 15.4, 5.0
REG_SIDE = -1
REG_Z = 27.2
WHEEL_R, WHEEL_X, WHEEL_Y = 2.6, -3.05, 3.0
RAIL_X, RAIL_Y, RAIL_TOP = -2.75, 1.55, 33.6

WHITE = (233, 235, 231)
GREEN_PAINT = (26, 138, 72)


def body_profile():
    pts = [(0.0, Z_BOTTOM), (R_BODY - 0.25, Z_BOTTOM)]
    for a in np.radians(np.linspace(-90, 0, 4)[1:]):
        pts.append((R_BODY - 0.25 + 0.25 * np.cos(a), Z_BOTTOM + 0.25 + 0.25 * np.sin(a)))
    pts += [(R_BODY, z) for z in np.linspace(Z_BOTTOM + 0.25, Z_SHOULDER, 10)[1:]]
    for t in np.linspace(0, 1, 11)[1:]:
        a = t * np.pi / 2
        pts.append((R_NECK + (R_BODY - R_NECK) * np.cos(a), Z_SHOULDER + (Z_NECK - Z_SHOULDER) * np.sin(a)))
    pts += [(R_NECK, Z_TOP - 0.15), (R_NECK - 0.08, Z_TOP), (0.0, Z_TOP)]
    return np.array(pts)


# --- geometry -------------------------------------------------------------------

def cylinder():
    return mk.lathe(body_profile(), 32, BODY)


def label():
    r, w, h = R_BODY + 0.015, LABEL_SIZE, LABEL_SIZE

    def fn(u, v):
        phi = (u - 0.5) * w / r
        return np.stack([r * np.cos(phi), r * np.sin(phi), LABEL_Z + (0.5 - v) * h], -1)

    return mk.surface(fn, 12, 4, LABEL, flip=True)


def strap():
    prof = [(R_BODY - 0.02, 11.4), (R_BODY + 0.09, 11.4), (R_BODY + 0.09, 12.7), (R_BODY - 0.02, 12.7)]
    return mk.lathe(prof, 32, STRAP)


def valve_and_regulator():
    s = REG_SIDE
    parts = [mk.lathe([(0, Z_TOP), (0.58, Z_TOP), (0.58, 27.9), (0.46, 28.1), (0, 28.1)], 16, CHROME)]
    # toggle wheel on top of the post valve
    parts.append(mk.lathe([(0, 28.1), (0.55, 28.1), (0.6, 28.35), (0.55, 28.6), (0, 28.6)], 16, KNOB))
    parts.append(mk.tube([(0, 0.35 * s, REG_Z), (0, 0.95 * s, REG_Z)], 0.32, 12, CHROME))
    # regulator body, flow knob and the hose barb the tube plugs onto
    parts.append(mk.tube([(0, 0.9 * s, REG_Z), (0, 3.3 * s, REG_Z)], 0.62, 16, GREEN))
    parts.append(mk.tube([(0, 3.3 * s, REG_Z), (0, 3.8 * s, REG_Z)], 0.5, 16, KNOB))
    parts.append(mk.tube([(0, 2.9 * s, REG_Z - 0.4), (0, 2.9 * s, 26.05)], 0.17, 10, KNOB))
    parts.append(mk.tube([(0, 2.9 * s, 26.35), (0, 2.9 * s, 26.15)], 0.24, 10, KNOB))
    # pressure gauge on a stem, dial facing forward (+x)
    parts.append(mk.tube([(0, 1.6 * s, REG_Z + 0.5), (0, 1.6 * s, 28.0)], 0.16, 10, CHROME))
    bezel = mk.lathe([(0, -0.3), (0.9, -0.3), (0.97, 0.0), (0.95, 0.2), (0.86, 0.24), (0.84, 0.12)], 24, CHROME)
    face = mk.disc_at((0, 0, 0.12), (0, 0, 1), (0, 1, 0), 0.84, 24, GAUGE)
    gauge = mk.concat([bezel, face]).transformed(mk.trans(0.1, 1.6 * s, 28.75) @ mk.rot((0, 1, 0), 90))
    parts.append(gauge)
    return mk.concat(parts)


def barb_tip():
    """Where the oxygen tube attaches (cart-local), with the tube leaving straight down."""
    return np.array([0.0, 2.9 * REG_SIDE, 26.05])


def frame():
    parts = []
    arc = [(RAIL_X, RAIL_Y * np.cos(t), RAIL_TOP + RAIL_Y * np.sin(t)) for t in np.linspace(0, np.pi, 17)]
    path = [(RAIL_X, RAIL_Y, 1.5)] + arc + [(RAIL_X, -RAIL_Y, 1.5)]
    parts.append(mk.tube(path, 0.27, 10, FRAME, up_hint=(1, 0, 0)))
    grip = [(RAIL_X, RAIL_Y * np.cos(t), RAIL_TOP + RAIL_Y * np.sin(t)) for t in np.linspace(0.28 * np.pi, 0.72 * np.pi, 9)]
    parts.append(mk.tube(grip, 0.38, 12, RUBBER, up_hint=(1, 0, 0)))
    # holder ring hugging the cylinder, and the base the cylinder stands on
    ring = [(2.42 + 0.2 * np.cos(t), 19.2 + 0.2 * np.sin(t)) for t in np.linspace(-np.pi, np.pi, 11)]
    parts.append(mk.lathe(ring, 32, FRAME, smooth_angle=60))
    parts.append(mk.box((5.7, 5.0, 0.3), (-0.25, 0, 1.55), FRAME))
    lip = [(2.45 * np.cos(t), 2.45 * np.sin(t), 2.35) for t in np.radians(np.linspace(-115, 115, 19))]
    parts.append(mk.tube(lip, 0.2, 8, FRAME))
    for side in (1, -1):
        end = (2.45 * np.cos(np.radians(115)), side * 2.45 * np.sin(np.radians(115)), 2.35)
        parts.append(mk.tube([end, (RAIL_X, side * RAIL_Y, 2.35)], 0.2, 8, FRAME))
        parts.append(mk.tube([(1.9, side * 1.5, 1.45), (2.05, side * 1.5, 0.45)], 0.2, 8, FRAME))
        parts.append(mk.tube([(RAIL_X, side * RAIL_Y, 3.6), (WHEEL_X, side * RAIL_Y, WHEEL_R)], 0.22, 8, FRAME))
    parts.append(mk.tube([(2.05, -1.9, 0.34), (2.05, 1.9, 0.34)], 0.34, 12, RUBBER))
    parts.append(mk.tube([(WHEEL_X, -(WHEEL_Y + 0.35), WHEEL_R), (WHEEL_X, WHEEL_Y + 0.35, WHEEL_R)], 0.2, 10, CHROME))
    return mk.concat(parts)


def cart_mesh():
    return mk.concat([cylinder(), label(), strap(), valve_and_regulator(), frame()])


def wheel_mesh():
    """One wheel centred on its axle, axle along y."""
    tire = mk.lathe([(1.75, -0.5), (2.3, -0.5), (2.5, -0.38), (WHEEL_R, 0.0), (2.5, 0.38),
                     (2.3, 0.5), (1.75, 0.5), (1.75, -0.5)], 28, RUBBER, smooth_angle=50)
    hub = mk.lathe([(0, -0.62), (0.45, -0.62), (0.5, -0.42), (1.2, -0.38), (1.77, -0.3),
                    (1.77, 0.3), (1.2, 0.38), (0.5, 0.42), (0.45, 0.62), (0, 0.62)], 20, HUB)
    return mk.concat([tire, hub]).transformed(mk.rot((1, 0, 0), -90))


# --- texture atlas ----------------------------------------------------------------

def _noise(rng, size, amount):
    return rng.normal(0, amount, size)


def _fill(img, rect_px, color, rng, noise=4.0, streaks=0.0):
    x0, y0, x1, y1 = rect_px
    h, w = y1 - y0, x1 - x0
    base = np.ones((h, w, 3)) * np.array(color, float)
    base += _noise(rng, (h, w, 1), noise)
    if streaks:
        base += rng.normal(0, streaks, (1, w, 1)) + rng.normal(0, streaks * 0.5, (h, 1, 1))
    img[y0:y1, x0:x1] = np.clip(base, 0, 255)


def paint_atlas(path):
    rng = np.random.default_rng(7)
    img = np.zeros((ATLAS, ATLAS, 3))

    # cylinder body: rows follow the lathe profile arclength (top of region = top of cylinder)
    prof = body_profile()
    seg = np.linalg.norm(np.diff(prof, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    rows = np.arange(ATLAS)
    z_of_row = np.interp(1 - (rows + 0.5) / ATLAS, s, prof[:, 1])
    r_of_row = np.interp(1 - (rows + 0.5) / ATLAS, s, prof[:, 0])
    body = np.zeros((ATLAS, 512, 3))
    for y in rows:
        if z_of_row[y] >= Z_GREEN:
            c = np.array(GREEN_PAINT, float)
        elif z_of_row[y] <= Z_BOTTOM + 0.01 and r_of_row[y] < R_BODY - 0.2:
            c = np.array((176, 178, 174), float)
        else:
            c = np.array(WHITE, float)
            # a little grime toward the bottom
            c -= max(0.0, 6.0 - z_of_row[y]) * 3.0
        body[y] = c
    body += _noise(rng, body.shape[:2] + (1,), 3.0)
    body += rng.normal(0, 1.5, (1, 512, 1))
    img[0:ATLAS, 0:512] = np.clip(body, 0, 255)

    # chrome: smooth bright/dark bands across the region fake a reflection on tubes and lathes
    t = np.linspace(0, 1, 128)[:, None, None]
    bands = 150 + 70 * np.cos(2 * np.pi * t) ** 2 - 35 * np.sin(2 * np.pi * t * 0.5) ** 8
    img[256:384, 512:768] = np.clip(bands * np.array([1.0, 1.02, 1.05]) + _noise(rng, (128, 256, 1), 1.5), 0, 255)
    _fill(img, (768, 256, 1024, 384), GREEN_PAINT, rng, 3)
    _fill(img, (512, 384, 768, 512), (30, 30, 32), rng, 3)
    _fill(img, (768, 384, 1024, 512), (214, 216, 213), rng, 3)
    _fill(img, (640, 512, 768, 640), (236, 236, 233), rng, 2)
    hub = np.zeros((128, 128, 3))
    for x in range(128):
        hub[:, x] = (112, 114, 118) if (x // 16) % 2 == 0 else (88, 90, 94)
    img[512:640, 512:640] = np.clip(hub + _noise(rng, (128, 128, 1), 3), 0, 255)
    strap_px = np.ones((128, 256, 3)) * 24
    strap_px += (np.sin(np.arange(256) * 1.3)[None, :, None] * 5) + _noise(rng, (128, 256, 1), 3)
    img[512:640, 768:1024] = np.clip(strap_px, 0, 255)

    pil = Image.fromarray(img.astype(np.uint8))
    pil.paste(_label_image(rng), (512, 0))
    pil.paste(_gauge_image(), (768, 0))
    pil.save(path, quality=92)
    return pil


def _label_image(rng):
    size = 256
    bg = np.ones((size, size, 3)) * np.array(WHITE, float) + _noise(rng, (size, size, 1), 3.0)
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    c = size / 2
    d.polygon([(c, 6), (size - 6, c), (c, size - 6), (6, c)], fill=(248, 205, 28))
    d.polygon([(c, 18), (size - 18, c), (c, size - 18), (18, c)], outline=(20, 20, 20), width=3)
    font = ImageFont.truetype(FONT, 33)
    d.text((c, c + 8), "OXYGEN", font=font, fill=(18, 18, 18), anchor="mm")
    d.text((c, size - 52), "2", font=ImageFont.truetype(FONT, 30), fill=(18, 18, 18), anchor="mm")
    # oxidizer pictogram: flame over a circle
    d.ellipse((c - 13, 82, c + 13, 108), outline=(18, 18, 18), width=5)
    flame = [(c - 18, 80), (c - 20, 64), (c - 12, 50), (c - 9, 62), (c - 3, 40), (c + 3, 58),
             (c + 10, 46), (c + 18, 62), (c + 18, 80)]
    d.polygon(flame, fill=(18, 18, 18))
    return im.filter(ImageFilter.SMOOTH)


def _gauge_image():
    size = 256
    im = Image.new("RGB", (size, size), (150, 154, 158))
    d = ImageDraw.Draw(im)
    c = size / 2
    d.ellipse((4, 4, size - 4, size - 4), fill=(246, 246, 242))
    # dial arc: 0 at lower left, full scale at lower right (270 degrees)
    def at(frac, r):
        a = np.radians(135 + 270 * frac)
        return c + r * np.cos(a), c + r * np.sin(a)
    for i in range(31):
        frac = i / 30
        r0 = 92 if i % 5 == 0 else 102
        d.line([at(frac, r0), at(frac, 112)], fill=(20, 20, 20), width=4 if i % 5 == 0 else 2)
    d.arc((26, 26, size - 26, size - 26), 135 + 270 * 0.62, 135 + 270 * 1.0, fill=(40, 160, 70), width=10)
    d.arc((26, 26, size - 26, size - 26), 135, 135 + 270 * 0.12, fill=(200, 40, 40), width=10)
    small = ImageFont.truetype(FONT, 18)
    for frac, txt in ((0.0, "0"), (1 / 3, "1"), (2 / 3, "2"), (1.0, "3")):
        d.text(at(frac, 72), txt, font=small, fill=(20, 20, 20), anchor="mm")
    d.text((c, c + 46), "PSI x1000", font=ImageFont.truetype(FONT, 14), fill=(30, 30, 30), anchor="mm")
    d.text((c, c - 40), "O2", font=ImageFont.truetype(FONT, 22), fill=(26, 120, 64), anchor="mm")
    d.line([(c, c), at(0.78, 104)], fill=(25, 25, 25), width=5)
    d.ellipse((c - 10, c - 10, c + 10, c + 10), fill=(40, 40, 40))
    return im
