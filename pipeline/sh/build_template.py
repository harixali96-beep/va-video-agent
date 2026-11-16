#!/usr/bin/env python3
"""
Builds the Stream House LA clip template for Content Rewards campaign 5f6b6ac5.

Outputs (all 1080x1920 unless noted):
  overlay.png     transparent PNG: hook bar, CTA pill, hashtag block, logo slot frame
  preview.png     layout preview over the campaign's own 16:9 artwork cropped 9:16
  hook.txt        the hook bar text used in the preview (single source of truth)

The official Stream House LA / CWETV logo is NOT invented here: logo.png is used if
present in this folder, otherwise a neutral "LOGO" slot is drawn so the official file
from the campaign's reference materials drops straight in.
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
ANTON = "/usr/share/fonts/truetype/custom/Anton-Regular.ttf"

INK = (8, 8, 8)
WHITE = (255, 255, 255)
ACCENT = (255, 214, 0)      # yellow highlight
RED = (230, 30, 45)

HOOK = os.environ.get("SH_HOOK") or "SHE THOUGHT NOBODY WAS WATCHING"
HASHTAGS = os.environ.get("SH_HASHTAGS") or "#streamhouselacwe  #streamhousela  #cwetv"
CTA = os.environ.get("SH_CTA") or "WATCH THE FULL EP ON @CHICKEN WRAP ENT"

M = 64                      # side safe margin
TOP_SAFE = 150              # keep clear of platform top UI
BOTTOM_SAFE = 1740          # keep clear of platform bottom UI


def font(size):
    return ImageFont.truetype(ANTON, size)


def text_w(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw, s, f, maxw):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_w(draw, t, f)[0] <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def hook_bar(img, draw, text):
    """Black hook bar pinned near the top, Anton uppercase, yellow keyword band."""
    f = font(74)
    lines = wrap(draw, text.upper(), f, W - 2 * M - 56)[:2]
    lh = int(74 * 1.16)
    pad_x, pad_y = 30, 26
    box_h = lh * len(lines) + pad_y * 2
    y0 = TOP_SAFE + 96
    draw.rectangle([M, y0, W - M, y0 + box_h], fill=INK + (232,))
    y = y0 + pad_y
    for i, ln in enumerate(lines):
        tw, th = text_w(draw, ln, f)
        x = (W - tw) // 2
        # last line gets a yellow highlight band for emphasis
        if i == len(lines) - 1:
            draw.rectangle([x - 12, y + th * 0.18, x + tw + 12, y + th * 1.22],
                           fill=ACCENT + (255,))
            draw.text((x, y), ln, font=f, fill=INK)
        else:
            draw.text((x, y), ln, font=f, fill=WHITE)
        y += lh
    return y0, y0 + box_h


def logo_slot(img, draw):
    """Official logo goes top-right; drop in as logo.png to replace the slot."""
    lp = os.path.join(HERE, "logo.png")
    box = (W - M - 210, TOP_SAFE - 54, W - M, TOP_SAFE + 62)
    if os.path.exists(lp):
        logo = Image.open(lp).convert("RGBA")
        r = box[2] - box[0]
        logo = logo.resize((r, int(logo.height * r / logo.width)), Image.LANCZOS)
        img.alpha_composite(logo, (box[0], box[1]))
    else:
        draw.rounded_rectangle(box, 18, outline=WHITE + (170,), width=3,
                               fill=(0, 0, 0, 110))
        f = font(34)
        tw, _ = text_w(draw, "LOGO", f)
        draw.text(((box[0] + box[2] - tw) // 2, box[1] + 22), "LOGO",
                  font=f, fill=WHITE + (190,))


def cta_pill(draw):
    """Red CTA pill, shrinks the type to stay on one line inside the frame."""
    for size in (38, 34, 30, 26):
        f = font(size)
        tw, th = text_w(draw, CTA, f)
        if tw + 68 <= W - 2 * M:
            break
    pad_x, pad_y = 34, 20
    w, h = tw + pad_x * 2, th + pad_y * 2
    x0 = (W - w) // 2
    y0 = BOTTOM_SAFE - h - 96
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], h // 2, fill=RED + (255,))
    draw.text((x0 + pad_x, y0 + pad_y - 6), CTA, font=f, fill=WHITE)
    return y0, y0 + h


def hashtag_block(draw, cy):
    f = font(34)
    tw, th = text_w(draw, HASHTAGS, f)
    x = (W - tw) // 2
    draw.text((x, cy - th // 2 - 6), HASHTAGS, font=f, fill=WHITE + (225,))
    # subtle shadow pass for legibility on bright frames
    return cy


def build_overlay():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    hook_bar(img, d, HOOK)
    logo_slot(img, d)
    y0, y1 = cta_pill(d)
    hashtag_block(d, y1 + 46)
    # caption guide (never rendered to video, preview only flag)
    img.save(os.path.join(HERE, "overlay.png"))
    return img


def build_preview():
    src = os.path.join(HERE, "banner.webp")
    if os.path.exists(src):
        bg = Image.open(src).convert("RGB")
        # 16:9 -> 9:16 centre crop, then darken slightly for caption contrast
        target = W / H
        sw, sh = bg.size
        cw = int(sh * target)
        if cw <= sw:
            bg = bg.crop(((sw - cw) // 2, 0, (sw + cw) // 2, sh))
        else:
            ch = int(sw / target)
            bg = bg.crop((0, (sh - ch) // 2, sw, (sh + ch) // 2))
        bg = bg.resize((W, H), Image.LANCZOS)
    else:
        bg = Image.new("RGB", (W, H), (24, 24, 28))

    canvas = bg.convert("RGBA")
    # bottom scrim so lower text reads on any frame
    grad = Image.new("RGBA", (W, 700), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for i in range(700):
        gd.rectangle([0, i, W, i + 1], fill=(0, 0, 0, int(150 * (i / 700) ** 1.6)))
    canvas.alpha_composite(grad, (0, H - 700))

    tmpl = build_overlay()
    canvas.alpha_composite(tmpl, (0, 0))

    # sample caption line, same slot the renderer uses
    d = ImageDraw.Draw(canvas)
    f = font(88)
    s = "CAUGHT ON CAMERA"
    tw, _ = text_w(d, s, f)
    d.text(((W - tw) // 2, 1180), s, font=f, fill=WHITE, stroke_width=10, stroke_fill=INK)
    canvas.convert("RGB").save(os.path.join(HERE, "preview.png"))
    with open(os.path.join(HERE, "hook.txt"), "w") as fh:
        fh.write(HOOK + "\n" + CTA + "\n" + HASHTAGS + "\n")


if __name__ == "__main__":
    build_overlay()
    if os.path.exists(os.path.join(HERE, "banner.webp")):
        build_preview()
    print("wrote overlay.png")
