"""
Converting a large image into Modalytics' standard slide format: a pyramidal OME-TIFF (Qt-free).

Slide files that are already tiled pyramids (SVS, pyramidal TIFF…) are read in
place (``io.tiff_source``) and never converted. A large image that is *not*
a pyramid (a stitched mosaic saved as one plain TIFF, PNG or JPEG) cannot be
shown smoothly or kept inside the project file, so it is converted once into
a pyramidal OME-TIFF next to the original and read in place from there:

* one file (easy to copy and share), BigTIFF, 512 px tiles;
* lossless compression (deflate), so measurements are not changed;
* every level half the size of the one before, down to under 1024 px;
* the pixel size in the OME metadata and the resolution tags, so QuPath,
  Fiji / Bio-Formats and OpenSlide-style readers open it calibrated.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from modalytics import fileio

TILE = 512


def converted_path(path: str) -> str:
    """Where the converted copy of ``path`` goes: ``name.pyramid.ome.tif`` next to it."""
    base = os.path.splitext(path)[0]
    if base.lower().endswith(".ome"):
        base = base[:-4]
    return base + ".pyramid.ome.tif"


def write_pyramid(pixels: np.ndarray, out: str, pixel_size_um: float,
                  progress: Callable[[float], bool] | None = None) -> str:
    """Write a pyramidal OME-TIFF.

    Args:
        pixels: (H, W) grey or (H, W, 3) RGB, 8 or 16 bit (or float32 for phase maps).
        out: Target file (written to ``out.tmp`` first, then renamed, so a failed
            conversion never leaves a half-written file under the real name).
        pixel_size_um: Level-0 pixel size.
        progress: Called with 0–1 after each level; returning False cancels.

    Returns:
        str: ``out``.

    Raises:
        InterruptedError: If cancelled.
    """
    import cv2
    import tifffile

    # 1. The levels: halve until the longest side is under 1024 px
    levels = [pixels]
    while max(levels[-1].shape[:2]) >= 2 * 1024 and min(levels[-1].shape[:2]) >= 2 * TILE // 4:
        h, w = levels[-1].shape[:2]
        levels.append(cv2.resize(levels[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    rgb = pixels.ndim == 3
    options = dict(tile=(TILE, TILE), compression="zlib", photometric="rgb" if rgb else "minisblack",
                   resolutionunit="CENTIMETER")
    tmp = out + ".tmp"

    # 2. Level 0 with the OME metadata, then the reduced levels as sub-IFDs
    try:
        with tifffile.TiffWriter(tmp, bigtiff=True, ome=True) as tif:
            res = 1e4 / pixel_size_um                                    # pixels per cm
            tif.write(levels[0], subifds=len(levels) - 1, resolution=(res, res),
                      metadata={"axes": "YXS" if rgb else "YX", "PhysicalSizeX": pixel_size_um,
                                "PhysicalSizeXUnit": "µm", "PhysicalSizeY": pixel_size_um,
                                "PhysicalSizeYUnit": "µm"}, **options)
            if progress is not None and progress(1 / len(levels)) is False:
                raise InterruptedError("cancelled")
            for i, lv in enumerate(levels[1:], start=1):
                f = pixels.shape[0] / lv.shape[0]
                tif.write(lv, subfiletype=1, resolution=(res / f, res / f), **options)
                if progress is not None and progress((i + 1) / len(levels)) is False:
                    raise InterruptedError("cancelled")
        fileio.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out
