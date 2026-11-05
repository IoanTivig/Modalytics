"""
Whole-slide images read in place from tiled, pyramidal TIFF files (Qt-free).

Aperio ``.svs``, LI-COR / generic pyramidal ``.tif``, OME-TIFF, Hamamatsu
``.ndpi`` and Leica ``.scn`` are all tiled image pyramids already, which is
what the canvas draws from, so they are read as they are: no conversion, no
copy. Tiled pages are read **directly**: the stored tiles a canvas tile covers
are read from the file and decoded (JPEG / JPEG 2000 by ``imagecodecs``) in the
calling thread, so the canvas' reader threads decode in parallel (the decoders
release Python's lock). Other layouts (strips, NDPI) go through ``tifffile``'s
zarr view of the level. Measured on the 2.97 GB Aperio test slide (147 407 ×
60 373 px), reading in place is faster than HoloPath's 23 GB zarr conversion of
it and gives the same full-resolution pixels.

Why not zarr for everything: zarr 3 serves every read through one asyncio
thread doing many small Python steps; while other Python threads are busy (the
render loop, processing steps) that thread barely runs. Measured: 6 blocks of
3 × 3 tiles read in 0.7 s alone, 48 s with one busy Python thread beside them.

The canvas tile is a whole number of the file's own tiles, about 1024 px
(an Aperio file's 240 px tiles → 960 px), so a tile read never cuts through a
stored tile.

Also here: the pixel size and the descriptive metadata the various TIFF
flavours carry (``tiff_pixel_size``, ``tiff_metadata``), shared with the
brightfield TIFF importer, and ``is_slide``, which tells the import step that a
file must be opened as a slide rather than loaded into memory.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading

import numpy as np

from modalytics.io.sources import Level, PixelSource

SLIDE_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".btf")
TARGET_TILE = 1024                       # px: canvas tile size aimed at
LARGE_PIXELS = 64_000_000                # more than this (e.g. 8000²) is not loaded whole into memory
PRINT_LIMIT_UM = 20.0                    # a resolution meaning pixels larger than this is a print DPI, not a calibration


class TiffPyramidSource(PixelSource):
    """A tiled, pyramidal TIFF (SVS, OME-TIFF, …) read in place.

    Args:
        path: The file.

    Attributes:
        path: Absolute path.
        pixel_size_um: Level-0 pixel size from the file, or None.
        metadata: Descriptive metadata from the file (scanner, magnification, date…).
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        import tifffile
        import zarr

        self.path = os.path.abspath(path)
        self._tif = tifffile.TiffFile(self.path)
        series = main_series(self._tif)
        if series is None:
            self._tif.close()
            raise ValueError(f"{os.path.basename(path)}: no 2D image (grey or RGB) in this TIFF")

        # 1. One zarr array per level, finest first (the axes as stored: YX or YXS), and the pages read directly
        self._axes = series.axes
        self._arrays = [zarr.open(series.aszarr(level=i), mode="r") for i in range(len(series.levels))]
        self._direct = [_direct_page(lv, self._axes) for lv in series.levels]
        self._lock = threading.Lock()                   # one file position: reads (not decoding) take turns

        # 2. Levels: tile = a whole number of the file's own tiles, about TARGET_TILE
        h0, w0 = _yx(series.levels[0].shape, self._axes)
        for i, lv in enumerate(series.levels):
            h, w = _yx(lv.shape, self._axes)
            page = lv.keyframe
            th = page.tilelength if page.is_tiled else (page.rowsperstrip or TARGET_TILE)
            tw = page.tilewidth if page.is_tiled else TARGET_TILE
            th = max(1, round(TARGET_TILE / th)) * th if th < TARGET_TILE else th
            tw = max(1, round(TARGET_TILE / tw)) * tw if tw < TARGET_TILE else tw
            self.levels.append(Level(shape=(h, w), tile=(min(th, h), min(tw, w)), downsample=(h0 / h, w0 / w)))
        self.dtype = np.dtype(series.dtype)
        self.channels = series.shape[self._axes.index("S")] if "S" in self._axes else 1

        # 3. Calibration and description
        self.pixel_size_um = tiff_pixel_size(self._tif)
        self.metadata = tiff_metadata(self._tif)

    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        lv = self.levels[level]
        th, tw = lv.tile
        y, x = ty * th, tx * tw
        y1, x1 = min(y + th, lv.shape[0]), min(x + tw, lv.shape[1])
        page = self._direct[level]
        if page is None:
            return np.asarray(self._arrays[level][y:y1, x:x1])
        return self._read_direct(page, y, y1, x, x1)

    def _read_direct(self, page, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Rows y0:y1, columns x0:x1 of a tiled page: its stored tiles read, decoded and put together."""
        tl, tw = page.tilelength, page.tilewidth
        across = (page.imagewidth + tw - 1) // tw
        rows, cols = range(y0 // tl, (y1 - 1) // tl + 1), range(x0 // tw, (x1 - 1) // tw + 1)
        index = [r * across + c for r in rows for c in cols]
        # 1. The compressed bytes (the file position is shared: one reader at a time, it is quick)
        fh = self._tif.filehandle
        with self._lock:
            chunks = []
            for i in index:
                fh.seek(page.dataoffsets[i])
                chunks.append(fh.read(page.databytecounts[i]))
        # 2. Decoded outside the lock (imagecodecs releases Python's lock: threads decode in parallel)
        out = np.empty((y1 - y0, x1 - x0) + ((self.channels,) if self.channels > 1 else ()), self.dtype)
        tables = page.jpegtables
        for i, data in zip(index, chunks):
            r, c = divmod(i, across)
            tile = page.decode(data, i, jpegtables=tables)[0]
            tile = tile.reshape(tile.shape[-3:])[..., :self.channels] if tile is not None else None
            ty0, tx0 = r * tl, c * tw
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + tl)
            sx0, sx1 = max(x0, tx0), min(x1, tx0 + tw)
            dst = out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            if tile is None:                                # a tile the file leaves empty
                dst[...] = 0
            else:
                part = tile[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
                dst[...] = part if self.channels > 1 else part[..., 0]
        return out

    def thumbnail(self) -> np.ndarray:
        """The coarsest level that is still about 1000 px across (some files go down to ~100 px)."""
        lv = next((i for i in range(len(self.levels) - 1, -1, -1) if max(self.levels[i].shape) >= 1000), 0)
        return np.asarray(self._arrays[lv][:])

    def value_range(self) -> tuple[float, float]:
        if self.dtype == np.uint16:
            return 0.0, 65535.0
        return super().value_range()

    def close(self) -> None:
        self._tif.close()


# ---------------------------------------------------------------- what kind of file
def _direct_page(level, axes: str):
    """The page of a level when it can be read tile by tile directly (one tiled, interleaved, non-NDPI page)."""
    if len(level.pages) != 1 or axes not in ("YX", "YXS"):
        return None
    page = level.keyframe
    if not page.is_tiled or page.is_ndpi or page.planarconfig != 1 or page.imagedepth != 1:
        return None
    return page


def main_series(tif):
    """The series holding the image (the largest 2D grey or RGB one), or None (stacks, channels…)."""
    best = None
    for s in tif.series:
        if s.axes not in ("YX", "YXS"):
            continue
        if best is None or np.prod(s.shape) > np.prod(best.shape):
            best = s
    return best


def _yx(shape: tuple, axes: str) -> tuple[int, int]:
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def is_slide(path: str) -> bool:
    """Whether a file is a whole-slide image to open in place: a TIFF pyramid, or a tiled TIFF too large
    to load whole (an SVS always is)."""
    if not path.lower().endswith(SLIDE_EXTENSIONS):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            if tif.is_svs or tif.is_ndpi or tif.is_scn:
                return True
            s = main_series(tif)
            if s is None:
                return False
            h, w = _yx(s.levels[0].shape, s.axes)
            return len(s.levels) > 1 or (s.levels[0].keyframe.is_tiled and h * w > LARGE_PIXELS)
    except Exception:
        return False


def image_size(path: str) -> tuple[int, int] | None:
    """(height, width) of an image file without decoding it (None if unknown)."""
    try:
        if path.lower().endswith(SLIDE_EXTENSIONS):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                s = main_series(tif)
                return _yx(s.levels[0].shape, s.axes) if s is not None else None
        from PIL import Image
        with Image.open(path) as im:
            return im.height, im.width
    except Exception:
        return None


# ---------------------------------------------------------------- calibration and description
def tiff_pixel_size(tif) -> float | None:
    """The level-0 pixel size in µm a TIFF states, or None.

    In order: Aperio ``MPP``, OME ``PhysicalSizeX``, ImageJ (resolution with
    ``unit=micron``), then the resolution tags in centimetres or inches. A
    resolution meaning pixels larger than 20 µm is taken as a print DPI (72,
    300…), not a calibration.
    """
    page = tif.pages[0]
    desc = page.description or ""
    # 1. Aperio: "… |MPP = 0.2512|…"
    m = re.search(r"\|\s*MPP\s*=\s*([0-9.eE+-]+)", desc)
    if m:
        return float(m.group(1))
    # 2. OME: PhysicalSizeX (+ unit, default µm)
    if tif.is_ome:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata or "")
        if m:
            unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', tif.ome_metadata or "")
            factor = {"nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}.get(unit.group(1) if unit else "µm", 1.0)
            return float(m.group(1)) * factor
    # 3. ImageJ: resolution in pixels per unit
    xres = page.tags.get("XResolution")
    if xres is None:
        return None
    num, den = xres.value if isinstance(xres.value, tuple) else (xres.value, 1)
    per_unit = num / den if den else 0
    if per_unit <= 0:
        return None
    if tif.is_imagej:
        unit = (tif.imagej_metadata or {}).get("unit", "")
        if unit in ("micron", "µm", "um", "\\u00B5m"):
            return 1.0 / per_unit
    # 4. Resolution tags: 2 = inch, 3 = centimetre
    unit_tag = page.tags.get("ResolutionUnit")
    unit = int(unit_tag.value) if unit_tag is not None else 2
    um = {2: 25400.0, 3: 10000.0}.get(unit)
    if um is None:
        return None
    px = um / per_unit
    return px if px <= PRINT_LIMIT_UM else None


def tiff_metadata(tif) -> dict:
    """What a TIFF says about where it came from (kept with the image in the project)."""
    page = tif.pages[0]
    out = {}
    desc = page.description or ""
    if tif.is_svs:
        out["format"] = "Aperio SVS"
        for key, name in (("AppMag", "magnification"), ("ScanScope ID", "scanner_id"), ("Date", "scan_date"),
                          ("Time", "scan_time"), ("MPP", "mpp")):
            m = re.search(rf"\|\s*{re.escape(key)}\s*=\s*([^|]+)", desc)
            if m:
                out[name] = m.group(1).strip()
        out["scanner"] = desc.split("\n")[0].split("|")[0].strip()
    elif tif.is_ome:
        out["format"] = "OME-TIFF"
    elif tif.is_ndpi:
        out["format"] = "Hamamatsu NDPI"
    elif tif.is_scn:
        out["format"] = "Leica SCN"
    else:
        out["format"] = "TIFF"
    for tag, name in (("Make", "make"), ("Model", "model"), ("Software", "software"), ("DateTime", "date")):
        if tag in page.tags and str(page.tags[tag].value).strip():
            out[name] = str(page.tags[tag].value).strip()
    return out


# ---------------------------------------------------------------- the same file later?
def fingerprint(path: str) -> dict:
    """Size and a hash of the first MiB: enough to notice that another file now sits at a path."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    return {"size": os.path.getsize(path), "head_sha1": hashlib.sha1(head).hexdigest()}
