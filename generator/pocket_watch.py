"""Gold pocket watch with a ticking small-seconds hand, and its chain: geometry + textures.

Watch frame: origin at the centre of the case, +x out of the dial (the face looks along +x),
+z toward 12 o'clock (the crown and bow), +y toward 3 o'clock. The second hand is its own mesh,
pivoting at its origin in the plane x = 0, pointing to +z at rest; the outfit turns it about x
from the subdial centre (SUBDIAL).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import meshkit as mk

FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def _px(x0, y0, x1, y1, size=512, inset=2):
    return ((x0 + inset) / size, (y0 + inset) / size, (x1 - inset) / size, (y1 - inset) / size)


DIAL = _px(0, 0, 384, 384)
GOLD = _px(384, 0, 512, 192)
GOLD_DARK = _px(384, 192, 512, 384)
STEEL = _px(0, 384, 256, 512)

R_CASE, R_DIAL = 1.05, 0.9
FRONT, BACK = 0.2, -0.24
DIAL_X = 0.13
SUBDIAL = np.array([DIAL_X + 0.004, 0.0, -0.42])     # small-seconds centre, on the dial
BOW_Z, BOW_R = 1.78, 0.36


def bow_point():
    """Top of the bow ring, where the chain hooks on (watch frame)."""
    return np.array([0.0, 0.0, BOW_Z + BOW_R])


def watch_mesh():
    # case: a lathe around +x (build around z, then turn z onto x)
    prof = [(0.0, BACK - 0.06), (0.55, BACK - 0.05), (0.9, BACK), (R_CASE, BACK + 0.12), (R_CASE, FRONT - 0.04),
            (R_CASE - 0.03, FRONT), (R_DIAL + 0.06, FRONT + 0.02), (R_DIAL + 0.02, DIAL_X + 0.02), (R_DIAL, DIAL_X)]
    case = mk.lathe(prof, 40, GOLD, smooth_angle=50).transformed(mk.rot((0, 1, 0), 90))
    dial = mk.disc_at((DIAL_X, 0, 0), (1, 0, 0), (0, 1, 0), R_DIAL, 40, DIAL)
    # pendant, knurled crown and the bow ring for the chain
    pendant = mk.tube([(0, 0, R_CASE - 0.05), (0, 0, R_CASE + 0.32)], 0.15, 12, GOLD)
    crown = mk.lathe([(0, -0.12), (0.2, -0.12), (0.23, 0), (0.2, 0.12), (0, 0.12)], 16, GOLD_DARK)
    crown = crown.transformed(mk.trans(0, 0, R_CASE + 0.42))
    ring = [(0.0, BOW_R * np.sin(t), BOW_Z + BOW_R * np.cos(t) - 0.05) for t in np.linspace(-np.pi, np.pi, 25)]
    bow = mk.tube(ring, 0.065, 8, GOLD, caps=False, up_hint=(1, 0, 0))
    # hour and minute hands painted on the dial; a little cap on the centre post
    post = mk.tube([(DIAL_X, 0, 0), (DIAL_X + 0.05, 0, 0)], 0.05, 8, STEEL)
    return mk.concat([case, dial, pendant, crown, bow, post])


def second_hand_mesh():
    needle = mk.box((0.012, 0.022, 0.24), (0, 0, 0.09), STEEL)
    tail = mk.box((0.012, 0.03, 0.06), (0, 0, -0.04), STEEL)
    return mk.concat([needle, tail])


def paint_watch(path, size=512, seed=31):
    """Enamel dial (Roman numerals, minute track, blued hands at ten past ten, small seconds at six)
    and brushed gold."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    # dial: the disc maps the region's centre to the dial centre, region edge to R_DIAL
    S = 384
    c = S / 2
    d.rectangle((0, 0, S, S), fill=(246, 242, 230))
    font = ImageFont.truetype(FONT_SERIF, 34)
    small = ImageFont.truetype(FONT_SERIF, 13)
    for k in range(60):
        a = np.radians(90 - 6 * k)
        r0, r1 = (0.84, 0.94) if k % 5 == 0 else (0.88, 0.94)
        d.line((c + r0 * c * np.cos(a), c - r0 * c * np.sin(a), c + r1 * c * np.cos(a), c - r1 * c * np.sin(a)),
               fill=(30, 30, 30), width=3 if k % 5 == 0 else 1)
    d.ellipse((c - 0.95 * c, c - 0.95 * c, c + 0.95 * c, c + 0.95 * c), outline=(40, 40, 40), width=2)
    numerals = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    for k, n in enumerate(numerals):
        if n == "VI":
            continue                                   # the small seconds sits there
        a = np.radians(90 - 30 * k)
        x, y = c + 0.68 * c * np.cos(a), c - 0.68 * c * np.sin(a)
        w = d.textlength(n, font=font)
        d.text((x - w / 2, y - 20), n, font=font, fill=(25, 25, 25))
    # small seconds subdial at six (matches SUBDIAL)
    sc_y = c + (0.42 / R_DIAL) * c
    sr = 0.27 / R_DIAL * c
    d.ellipse((c - sr, sc_y - sr, c + sr, sc_y + sr), outline=(40, 40, 40), width=2)
    for k in range(12):
        a = np.radians(90 - 30 * k)
        d.line((c + 0.75 * sr * np.cos(a), sc_y - 0.75 * sr * np.sin(a), c + sr * np.cos(a), sc_y - sr * np.sin(a)),
               fill=(40, 40, 40), width=1)
    d.text((c - d.textlength("1886", font=small) / 2, c - 0.36 * c), "1886", font=small, fill=(90, 70, 40))
    # blued hands at 10:08
    for ang, length, width in ((90 - 304, 0.48, 7), (90 - 48, 0.74, 5)):
        a = np.radians(ang)
        d.line((c, c, c + length * c * np.cos(a), c - length * c * np.sin(a)), fill=(30, 40, 90), width=width)
    d.ellipse((c - 8, c - 8, c + 8, c + 8), fill=(30, 40, 90))
    arr = np.array(img).astype(float)
    # brushed gold, light and dark
    for (x0, y0, x1, y1), base in (((384, 0, 512, 192), (212, 168, 72)), ((384, 192, 512, 384), (150, 112, 40))):
        h, w = y1 - y0, x1 - x0
        t = np.linspace(0, 1, h)[:, None, None]
        shade = 0.82 + 0.3 * np.cos(2 * np.pi * t) ** 2
        arr[y0:y1, x0:x1] = np.clip(np.array(base, float) * shade + rng.normal(0, 4, (h, w, 1)), 0, 255)
    arr[384:512, 0:256] = (40, 48, 80)
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=93)


def paint_chain(path, links=8):
    """Curb chain strip for a beam (RGBA): oval gold links, see-through between them."""
    w, h = 32 * links, 32
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k in range(links * 2):
        x = k * 16
        if k % 2 == 0:
            d.ellipse((x - 2, 6, x + 20, 26), outline=(214, 170, 70, 255), width=5)
        else:
            d.ellipse((x - 2, 10, x + 20, 22), outline=(170, 128, 48, 255), width=4)
    img.save(path)
