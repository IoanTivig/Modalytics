"""
The Modalytics brand: its colour palette and its logo files (no Qt here).

The palette has six colours, from light orchid to deep indigo and two reds::

    orchid   #B964A6   violet  #864CA0   purple  #653988
    indigo   #3A2665   red     #B72E3B   crimson #A2142F

The logo is drawn in ``purple`` (on light backgrounds); the *negative* logo is white (on dark ones and on the
brand colours). The UI themes (``ui.theme``), the loading window (``ui.splash``), the application icon
(``packaging/make_icon.py``) and the reports take their colours from here, so the brand changes in one place.

Files (``resources/brand``):

* ``modalytics-logo.png`` / ``modalytics-logo_negativ.png``: the logo as designed (the mark and the name, purple /
  white, with a transparent margin): the sources of the others.
* ``logo.png`` / ``logo_negativ.png``: the same, cropped to the drawing: what the program shows.
* ``mark.png`` / ``mark_negativ.png``: the mark alone, cropped.

The cropped files are made from the sources by ``python packaging/make_icon.py`` (with the application icon).
"""

from __future__ import annotations

import base64
from pathlib import Path

from modalytics import paths

# 1. The palette, light to dark, then the reds
ORCHID = "#B964A6"
VIOLET = "#864CA0"
PURPLE = "#653988"          # the logo's colour
INDIGO = "#3A2665"
RED = "#B72E3B"
CRIMSON = "#A2142F"

PALETTE: tuple[str, ...] = (ORCHID, VIOLET, PURPLE, INDIGO, RED, CRIMSON)

# The colours a moving brand gradient cycles through (the loading window's border): around the colour wheel
# from orchid through the purples to the reds and back, so the loop has no seam
GRADIENT: tuple[str, ...] = (ORCHID, VIOLET, PURPLE, INDIGO, CRIMSON, RED, ORCHID)

# 2. The dark brand surface (the loading window, the installer's side panel): the white logo goes on it
DARK_TOP = "#1f1630"         # a dark indigo, at the top of a vertical gradient
DARK_BOTTOM = "#110c18"      # nearly black, at its bottom
DARK_TEXT = "#f1ecf5"
DARK_MUTED = "#a497b4"

TAGLINE = "Multimodal imaging analysis"

FOLDER = paths.RESOURCES / "brand"


def logo_path(negative: bool = False, mark_only: bool = False) -> Path:
    """The logo file to show.

    Args:
        negative: True for the white logo (dark backgrounds), False for the purple one (light backgrounds).
        mark_only: True for the mark alone (square-ish: icons, small places), False for the mark and the name.

    Returns:
        Path: The PNG file (transparent background).
    """
    name = "mark" if mark_only else "logo"
    return FOLDER / f"{name}{'_negativ' if negative else ''}.png"


def source_path(negative: bool = False) -> Path:
    """The logo as designed (with its margin): what ``packaging/make_icon.py`` crops the shown files from."""
    return FOLDER / f"modalytics-logo{'_negativ' if negative else ''}.png"


def logo_data_uri(negative: bool = False, mark_only: bool = False) -> str:
    """The logo as a ``data:`` URI, for HTML that must stand alone (reports, help pages).

    Args:
        negative: True for the white logo, False for the purple one.
        mark_only: True for the mark alone.

    Returns:
        str: ``data:image/png;base64,…`` (empty if the file is missing, so a page still renders).
    """
    path = logo_path(negative, mark_only)
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def logo_img(height_px: int = 26) -> str:
    """The purple logo as an HTML ``<img>`` for the top of a report (white pages); empty if the file is missing.

    Args:
        height_px: Height on the page (CSS px).

    Returns:
        str: ``<img …>`` with the logo inline (the report stays a single file).
    """
    uri = logo_data_uri()
    return (f"<img class='brand-logo' src='{uri}' alt='Modalytics' "
            f"style='display:block; height:{height_px}px; margin:0 0 18px'>" if uri else "")
