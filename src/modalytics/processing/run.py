"""
Running processing steps: views of images, computed when needed (Qt-free).

``make_chain_view(image, chain)`` gives an ``ImageView`` whose pixels are a
``ProcessedSource`` reading the previous step's (``make_view`` is a chain of one step):

* **pixel / neighbourhood steps** are computed tile by tile, on the level the
  canvas or a measurement asks for, from the image's own tiles (plus a margin
  of context for neighbourhood steps). A whole slide's OD map is never computed
  whole: only what is looked at or measured. Recent tiles are kept in a small cache.
* **whole-image steps** are computed once on the full image (refused above
  ``MAX_WHOLE_PIXELS``) and kept in memory.

The view reads ``image.source`` at each tile, so it follows the image when its
file is relinked. Only the recipe (the chain, which view is shown) is saved with
a project (``recipes`` / ``restore_views``); the pixels are recomputed.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from modalytics.core.model import Display, ImageView
from modalytics.io.sources import ArraySource, Level, PixelSource
from modalytics.processing.chains import Chain
from modalytics.processing.registry import ProcessingLibrary, StepSpec, Where
from modalytics.processing.types import MADE, check_tile

MAX_WHOLE_PIXELS = 100_000_000        # whole-image steps refuse larger images (a slide has billions)
CACHE_TILES = 96


class ProcessedSource(PixelSource):
    """The pixels of an image seen through a processing step (see the module docstring).

    Args:
        image: The ImageNode processed.
        spec: The step.
        options: Its option values (resolved).

    Raises:
        ValueError: A whole-image step on an image too large for it, or a result that breaks the contract.
    """

    def __init__(self, image, spec: StepSpec, options: dict) -> None:
        super().__init__()
        self.image, self.spec, self.options = image, spec, dict(options)
        self.output_type = spec.output_for(image.modality)
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.state = spec.prepare(image, self.options) if spec.prepare is not None else {}
        src = image.source
        if spec.kind == "whole":
            # 1. Whole-image steps: computed once, kept
            h, w = src.shape
            if h * w > MAX_WHOLE_PIXELS:
                raise ValueError(f"{spec.name} needs the whole image at once; {image.name} is too large "
                                 f"({w:,} × {h:,} px)")
            pixels = _read_level(src, 0, 0, h, 0, w)
            extra = {"where": Where(0, 0, (h, w), 0)} if spec.wants_where else {}
            result = check_tile(spec.process(pixels, self.options, self.state, image.pixel_size_um[0], **extra),
                                pixels, self.output_type, spec.is_filter)
            self._whole = ArraySource(result)
            self.levels = list(self._whole.levels)
            self.dtype, self.channels = self._whole.dtype, self._whole.channels
        else:
            # 2. Tile by tile: the image's own levels and tiles; a first tile gives the result's form
            self._whole = None
            self.levels = list(src.levels)
            probe = self.read_tile(len(self.levels) - 1, 0, 0)
            self.dtype = probe.dtype
            self.channels = probe.shape[2] if probe.ndim == 3 else 1
        self._range = None

    @property
    def missing(self) -> bool:
        """True when the image's file is not found (its pixels are a placeholder): nothing to measure."""
        return bool(getattr(self.image.source, "missing", False))

    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        if self._whole is not None:
            return self._whole.read_tile(level, ty, tx)
        src = self.image.source
        key = (src.uid, level, ty, tx)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        # 1. The tile's pixels and the context around it (neighbourhood steps)
        lv = src.levels[level]
        th, tw = lv.tile
        y0, x0 = ty * th, tx * tw
        y1, x1 = min(y0 + th, lv.shape[0]), min(x0 + tw, lv.shape[1])
        um = self.image.pixel_size_um[0] * lv.downsample[1]
        m = self.spec.margin_px(self.options, um)
        block = _read_level(src, level, y0 - m, y1 + m, x0 - m, x1 + m)
        # 2. The step (told where the block is, if it asks), checked, cropped back to the tile
        extra = {"where": Where(y0 - m, x0 - m, lv.shape, level)} if self.spec.wants_where else {}
        result = check_tile(self.spec.process(block, self.options, self.state, um, **extra), block,
                            self.output_type, self.spec.is_filter)
        tile = np.ascontiguousarray(result[m:m + (y1 - y0), m:m + (x1 - x0)])
        with self._lock:
            self._cache[key] = tile
            if len(self._cache) > CACHE_TILES:
                self._cache.popitem(last=False)
        return tile

    def value_range(self) -> tuple[float, float]:
        """The contract's first contrast range, else the 1–99 % range of the values seen up close.

        For an image with several levels, the values are those of one full-resolution
        tile where the coarsest level shows the most structure: a feature map's values
        depend on the scale (edges are steeper at full resolution), and a slide's
        coarsest level is mostly empty glass.
        """
        made = MADE.get(self.output_type)
        if made is not None and made[3] is not None:
            return made[3]
        if self.spec.is_filter and np.issubdtype(np.dtype(self.dtype), np.integer):
            return self.image.source.value_range()
        if self._range is None:
            # 1. The coarsest level, whole
            lv = len(self.levels) - 1
            gy, gx = self.levels[lv].grid
            data = np.concatenate([np.concatenate([self.read_tile(lv, ty, tx) for tx in range(gx)], 1)
                                   for ty in range(gy)], 0)
            # 2. Several levels: the full-resolution tile under its busiest part
            if lv > 0:
                data = self.read_tile(0, *self._busiest_tile(data))
            finite = data[np.isfinite(data)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            self._range = (float(lo), float(hi if hi > lo else lo + 1.0))
        return self._range

    def _busiest_tile(self, coarse: np.ndarray) -> tuple[int, int]:
        """(row, column) of the full-resolution tile where the coarsest level varies most."""
        from scipy import ndimage
        grey = coarse.mean(axis=2) if coarse.ndim == 3 else coarse
        grey = np.where(np.isfinite(grey), grey, np.nanmedian(grey)).astype(np.float64)
        down = self.levels[-1].downsample
        th, tw = self.levels[0].tile
        size = max(3, int(round(min(th / down[0], tw / down[1]))))        # one full-resolution tile, shrunk
        mean = ndimage.uniform_filter(grey, size)
        spread = ndimage.uniform_filter(grey * grey, size) - mean * mean
        y, x = np.unravel_index(int(np.argmax(spread)), spread.shape)
        gy, gx = self.levels[0].grid
        return min(gy - 1, int(y * down[0] // th)), min(gx - 1, int(x * down[1] // tw))


def _read_level(src: PixelSource, level: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Pixels rows y0:y1, columns x0:x1 of one level; beyond the edges the image is mirrored."""
    lv: Level = src.levels[level]
    h, w = lv.shape
    cy0, cy1, cx0, cx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    th, tw = lv.tile
    rows = [np.concatenate([src.read_tile(level, ty, tx) for tx in range(cx0 // tw, (cx1 - 1) // tw + 1)], axis=1)
            for ty in range(cy0 // th, (cy1 - 1) // th + 1)]
    block = np.concatenate(rows, axis=0)
    oy, ox = (cy0 // th) * th, (cx0 // tw) * tw
    block = block[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    pad = [(cy0 - y0, y1 - cy1), (cx0 - x0, x1 - cx1)] + [(0, 0)] * (block.ndim - 2)
    if any(p for pair in pad for p in pair):
        # Mirrored like scipy.ndimage's default ("reflect" there = numpy's "symmetric"), so a filter computed
        # tile by tile equals the same filter on the whole image, borders included
        block = np.pad(block, pad, mode="symmetric")
    return block


# ---------------------------------------------------------------- views
def chain_sources(image, chain: Chain, library: ProcessingLibrary) -> list:
    """The stand-in images after each step of a chain (the last one is the view's pixels).

    Each step reads the previous one's result, tile by tile; a stand-in shares the
    image's name, pixel size, placement and metadata, with the step's output type.

    Raises:
        ValueError: If a step is missing, does not accept what the previous one makes, or cannot run.
    """
    from modalytics.core.model import ImageNode
    current, out = image, []
    for i, s in enumerate(chain.steps, start=1):
        spec = library.get(s["step"])
        if spec is None:
            raise ValueError(f"step {i} ('{s['step']}') is not available here")
        if not spec.accepts(current.modality):
            raise ValueError(f"step {i} ({spec.name}) does not process {current.modality} images")
        source = ProcessedSource(current, spec, spec.resolve(s.get("options")))
        current = ImageNode(f"{image.name} › {spec.name}", source, image.pixel_size_um, source.output_type,
                            image.placement, dict(image.metadata))
        out.append(current)
    return out


def make_chain_view(image, chain: Chain, library: ProcessingLibrary | None = None, standard: bool = False) -> ImageView:
    """A view of an image through a processing chain (computed lazily; see the module docstring).

    Args:
        image: The image.
        chain: The chain (its ``measure`` says whether parameters use the result).
        library: The processing steps (default: all here).
        standard: Made by the project's profile (its measured parameters are the plain results).

    Raises:
        ValueError: If the chain is empty, or cannot be applied to the image.
    """
    library = library or ProcessingLibrary()
    if not chain.steps:
        raise ValueError(f"“{chain.name}” has no steps")
    last = chain_sources(image, chain, library)[-1]
    source, out = last.source, last.modality
    spec = source.spec
    made = MADE.get(out)
    lo, hi = source.value_range()
    if source.channels >= 3:
        display = image.display                          # colour stays colour (a filtered slide)
    else:
        display = Display(clim=(lo, hi), colormap=made[2] if made else image.display.colormap)
    unit = spec.unit or (made[0] if made else image.metadata.get("unit", ""))
    return ImageView(chain=chain.to_dict(), name=chain.name, image_type=out, source=source, display=display,
                     unit=unit, standard=standard)


def make_view(image, spec: StepSpec, options: dict | None = None, measure: bool = False) -> ImageView:
    """A view through one step (a chain of one, named after the step; only for looking unless ``measure``).

    Raises:
        ValueError: If the step does not accept the image's type, or cannot run on it.
    """
    if not spec.accepts(image.modality):
        raise ValueError(f"{spec.name} does not process {image.modality} images")
    chain = Chain(spec.name, [{"step": spec.uid, "options": spec.resolve(options)}], measure)
    return make_chain_view(image, chain, ProcessingLibrary.of([spec]))


def recipes(image) -> list[dict]:
    """How to make an image's views again (saved with a project instead of their pixels)."""
    return [{"chain": v.chain, "standard": v.standard, "shown": v is image.shown,
             "display": {"clim": list(v.display.clim), "colormap": v.display.colormap}} for v in image.views]


def restore_views(image, saved: list[dict], library: ProcessingLibrary | None = None) -> list[str]:
    """Make an image's views again from their recipes.

    Recipes that cannot be made here (a step missing on this computer) are kept in
    ``image.metadata["views_not_available"]``, so saving the project does not lose them.

    Returns:
        list[str]: Problems, one per view left out.
    """
    library = library or ProcessingLibrary()
    problems, kept = [], []
    for r in saved or []:
        try:
            chain = Chain.from_dict(r["chain"] if "chain" in r else r)      # older recipes: one step
            view = make_chain_view(image, chain, library, standard=bool(r.get("standard")))
        except (ValueError, KeyError) as exc:
            problems.append(f"{image.name}: the view “{(r.get('chain') or r).get('name', r.get('step'))}” "
                            f"could not be made again ({exc})")
            kept.append(r)
            continue
        d = r.get("display")
        if d:
            view.display = Display(clim=tuple(d["clim"]), colormap=d["colormap"])
        image.views.append(view)
        if r.get("shown"):
            image.shown = view
    if kept:
        image.metadata["views_not_available"] = kept
    return problems


def profile_views(profile, image, library: ProcessingLibrary | None = None) -> tuple[list[ImageView], list[str]]:
    """The views a profile's chains make of a newly imported image (standard views).

    Returns:
        tuple: (views, in the profile's order; problems, e.g. a step not available here).
    """
    if profile is None or not profile.chains_for(image.modality):
        return [], []
    library = library or ProcessingLibrary()
    views, problems = [], []
    for chain in profile.chains_for(image.modality):
        try:
            views.append(make_chain_view(image, chain, library, standard=True))
        except Exception as exc:
            problems.append(f"{image.name}: the profile's chain “{chain.name}” could not be applied ({exc})")
    return views, problems
