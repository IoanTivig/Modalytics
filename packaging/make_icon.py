"""
The brand images made from the logo: the mark alone, and the application icon (the white mark on a brand gradient).

Reads ``src/modalytics/resources/brand/modalytics-logo.png`` and ``modalytics-logo_negativ.png`` (the logo as
designed: the mark and the name, purple / white, see ``modalytics.brand``) and writes:

* ``resources/brand/logo.png`` and ``logo_negativ.png``: the logos cropped to their drawing (what the program shows);
* ``resources/brand/mark.png`` and ``mark_negativ.png``: the mark alone, cropped from the logos;
* ``resources/icons/modalytics.png`` (256 px: the window icon) and ``packaging/modalytics.ico`` (16–256 px: the
  program, its shortcuts and the installer);
* ``packaging/installer/wizard-WxH.png``: the installer's side panel (its first and last pages): the white logo on
  the dark brand gradient, with a stripe of the brand colours; one file per screen scale (``WIZARD_SIZES``);
* ``packaging/installer/small-WxH.png``: the application icon, for the corner of the installer's other pages.

Run again after changing the logo::

    python packaging/make_icon.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modalytics import brand  # noqa: E402

SIZES = (16, 24, 32, 48, 64, 128, 256)
ICON = 256                     # the icon's side (px)
SUPER = 4                      # drawn this many times larger, then reduced: smooth edges
MARK_WIDTH = 0.70              # the mark's width in the icon, as a fraction of its side
INSTALLER = os.path.join(ROOT, "packaging", "installer")
# Inno Setup's wizard picture sizes, 100 % to 250 % screen scale (setup picks the closest one for the screen)
WIZARD_SIZES = ((164, 314), (192, 386), (246, 459), (273, 556), (328, 604), (355, 700), (410, 797))
SMALL_SIZES = ((55, 55), (64, 68), (83, 80), (92, 97), (110, 106), (119, 123), (138, 140))
FONT = r"C:\Windows\Fonts\segoeui.ttf"   # the font of Windows' own windows (Pillow's font where it is missing)


def crop_drawing(logo: Image.Image) -> Image.Image:
    """The logo cropped to its drawing, with a 2 px margin (the designed files have a wide transparent one)."""
    left, top, right, bottom = logo.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
    return logo.crop((max(left - 2, 0), max(top - 2, 0), min(right + 2, logo.width), min(bottom + 2, logo.height)))


def crop_mark(logo: Image.Image) -> Image.Image:
    """The mark alone: the logo's drawing left of the widest empty column gap (the gap before the name).

    Args:
        logo: The full logo (RGBA, transparent background).

    Returns:
        Image.Image: The mark, cropped tight with a 2 px margin.
    """
    # 1. Columns with any drawing in them; the widest gap between two of them separates the mark from the name
    alpha = np.asarray(logo.getchannel("A")) > 20
    cols = np.flatnonzero(alpha.any(axis=0))
    gap = int(np.argmax(np.diff(cols)))
    right = int(cols[gap]) + 1
    # 2. Rows of the mark only, then a small margin
    rows = np.flatnonzero(alpha[:, :right].any(axis=1))
    box = (max(int(cols[0]) - 2, 0), max(int(rows[0]) - 2, 0), right + 2, int(rows[-1]) + 3)
    return logo.crop(box)


def upscale_line_art(img: Image.Image, width: int) -> Image.Image:
    """Enlarge thin line art without blurring it: resize, then steepen the alpha edge back to a crisp line.

    The logo is a small bitmap; a plain enlargement leaves soft grey edges. Steepening the alpha around 50 %
    restores a sharp outline (the lines keep their width), which is then reduced with antialiasing by the caller.

    Args:
        img: RGBA line art in one colour.
        width: Width wanted (px); the height follows.

    Returns:
        Image.Image: RGBA, ``width`` wide.
    """
    height = round(img.height * width / img.width)
    big = img.resize((width, height), Image.Resampling.LANCZOS)
    a = np.asarray(big.getchannel("A")).astype(np.float32) / 255.0
    a = np.clip((a - 0.5) * 3.0 + 0.5, 0.0, 1.0)                     # a steeper edge, centred on the old one
    out = np.asarray(big).copy()
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def make_app_icon(mark_white: Image.Image) -> Image.Image:
    """The application icon: a rounded square with a violet-to-indigo gradient, the white mark in its middle.

    Args:
        mark_white: The negative (white) mark.

    Returns:
        Image.Image: ``ICON`` × ``ICON`` RGBA.
    """
    s = ICON * SUPER
    # 1. Diagonal gradient, top-left violet to bottom-right indigo
    top, bottom = (np.array(Image.new("RGB", (1, 1), c).getpixel((0, 0)), np.float32)
                   for c in (brand.VIOLET, brand.INDIGO))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    t = ((xx + yy) / (2 * (s - 1)))[..., None]
    rgb = (top * (1 - t) + bottom * t).astype(np.uint8)
    square = Image.fromarray(rgb, "RGB").convert("RGBA")
    # 2. Rounded corners (the same 8 px margin and 48 px radius as before, at 256 px)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((8 * SUPER, 8 * SUPER, s - 8 * SUPER, s - 8 * SUPER),
                                           radius=48 * SUPER, fill=255)
    square.putalpha(mask)
    # 3. The mark, centred (a little higher: the mark's weight is at its top)
    mark = upscale_line_art(mark_white, int(s * MARK_WIDTH))
    x = (s - mark.width) // 2
    y = (s - mark.height) // 2 - s // 40
    square.alpha_composite(mark, (x, y))
    return square.resize((ICON, ICON), Image.Resampling.LANCZOS)


def rgb(colour: str) -> np.ndarray:
    """'#B964A6' → array([185, 100, 166]) (float, for blending)."""
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Segoe UI at a pixel size, or Pillow's own font where Segoe UI is missing."""
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size)


def wizard_panel(width: int, height: int, logo_white: Image.Image) -> Image.Image:
    """The installer's side panel, in the same look as the loading window (``ui.splash``).

    Args:
        width: Width (px).
        height: Height (px).
        logo_white: The negative (white) logo, cropped.

    Returns:
        Image.Image: RGB, ``width`` × ``height``.
    """
    # 1. The dark brand gradient, top to bottom, with a soft violet glow behind the logo
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = (yy / (height - 1))[..., None]
    img = rgb(brand.DARK_TOP) * (1 - t) + rgb(brand.DARK_BOTTOM) * t
    cy = height * 0.40
    glow = (np.clip(1 - np.hypot(xx - width / 2, (yy - cy) * 0.8) / (width * 0.75), 0, 1) ** 2)[..., None]
    img = img * (1 - 0.30 * glow) + rgb(brand.VIOLET) * 0.30 * glow
    # 2. A stripe of the brand colours down the right edge (the loading window's border, standing still)
    stripe = max(2, round(width / 70))
    stops = [rgb(c) for c in brand.GRADIENT]
    pos = np.linspace(0, len(stops) - 1, height)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    f = (pos - lo)[:, None]
    column = np.array([stops[i] for i in lo]) * (1 - f) + np.array([stops[i + 1] for i in lo]) * f
    img[:, width - stripe:] = column[:, None, :]
    panel = Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")
    # 3. The white logo, centred, the tagline under it, and a line at the foot
    logo_w = round(width * 0.74)
    logo = logo_white.resize((logo_w, round(logo_white.height * logo_w / logo_white.width)),
                             Image.Resampling.LANCZOS)
    middle = (width - stripe) / 2
    y = round(cy - logo.height / 2)
    panel.paste(logo, (round(middle - logo.width / 2), y), logo)
    draw = ImageDraw.Draw(panel)
    text_font = font(max(9, round(width * 0.058)))
    draw.text((middle, y + logo.height + width * 0.09), brand.TAGLINE, font=text_font, fill=brand.DARK_MUTED,
              anchor="mt")
    draw.text((middle, height - width * 0.10), "Free and open source", font=text_font, fill=brand.DARK_MUTED,
              anchor="mb")
    return panel


def small_icon(width: int, height: int, icon: Image.Image) -> Image.Image:
    """The application icon centred in a transparent box (the corner of the installer's other pages)."""
    side = min(width, height)
    box = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    box.alpha_composite(icon.resize((side, side), Image.Resampling.LANCZOS),
                        ((width - side) // 2, (height - side) // 2))
    return box


def main() -> int:
    # 1. The logos cropped to their drawing, and the marks alone, from both designed logos
    for negative in (False, True):
        logo = Image.open(brand.source_path(negative)).convert("RGBA")
        for mark_only, image in ((False, crop_drawing(logo)), (True, crop_mark(logo))):
            image.save(brand.logo_path(negative, mark_only))
            print(f"✅ {brand.logo_path(negative, mark_only)} {image.size}")
    # 2. The application icon and the .ico with every size Windows asks for
    icon = make_app_icon(Image.open(brand.logo_path(True, mark_only=True)).convert("RGBA"))
    png = os.path.join(ROOT, "src", "modalytics", "resources", "icons", "modalytics.png")
    icon.save(png)
    ico = os.path.join(ROOT, "packaging", "modalytics.ico")
    icon.save(ico, sizes=[(s, s) for s in SIZES])
    print(f"✅ {png}\n✅ {ico}")
    # 3. The installer's pictures, one per screen scale
    os.makedirs(INSTALLER, exist_ok=True)
    logo_white = Image.open(brand.logo_path(True)).convert("RGBA")
    for w, h in WIZARD_SIZES:
        wizard_panel(w, h, logo_white).save(os.path.join(INSTALLER, f"wizard-{w}x{h}.png"))
    for w, h in SMALL_SIZES:
        small_icon(w, h, icon).save(os.path.join(INSTALLER, f"small-{w}x{h}.png"))
    print(f"✅ {INSTALLER} ({len(WIZARD_SIZES)} side panels, {len(SMALL_SIZES)} corner icons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
