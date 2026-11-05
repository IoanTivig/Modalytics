"""
The training projects measured again as if acquired a little differently: augmentation for classifiers (Qt-free).

Glands of one project resemble each other more than glands of the same grade from different projects, and part
of that is how the project was acquired. Each **copy** of a training project is its images as another
acquisition might have given them — one draw per project and copy, the same for all its images, as a session
would be:

* **pixel size**: the calibration off by a factor within ±``pixel_size`` (outlines and pixel size scaled
  together: the same pixels, every length, area and density in µm read differently);
* **gain**: every pixel value × a factor within ±``gain``;
* **offset**: + a shift within ±``offset`` × the image's pixel SD (a phase or illumination offset);
* **noise**: + Gaussian noise of SD ``noise`` × the image's pixel SD (new in every pixel);
* **defocus**: a Gaussian blur of σ up to ``blur_um`` µm.

The graded objects of the group are then measured on those pixels with the profile's parameters, as
``dataset.gather`` does (partial objects as the training set says). The rows keep their project and their copy
(``Dataset.variant``): normalised per project, each copy is normalised on its own, as another project would be;
in cross-validation a fold leaves a project's copies out with it. Measurements are cached per project, profile,
settings and copy, like ``gather``'s.

Only parameters of the image measured see the changes; one reading a class registered from another image is left
empty in a copy whose pixel size changes (the other image is not changed with it).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from modalytics import fileio
from modalytics.io.sources import PixelSource


@dataclass
class Acquisition:
    """How much each copy may differ (see the module docstring); ``copies`` 0: off."""

    copies: int = 0
    pixel_size: float = 0.03
    gain: float = 0.05
    offset: float = 0.10
    noise: float = 0.05
    blur_um: float = 0.0
    seed: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Acquisition":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def key(self) -> str:
        """Identifies the settings (for the cache)."""
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def describe(self) -> str:
        parts = [f"pixel size ±{100 * self.pixel_size:g} %" if self.pixel_size else "",
                 f"gain ±{100 * self.gain:g} %" if self.gain else "",
                 f"offset ±{self.offset:g} SD" if self.offset else "",
                 f"noise {self.noise:g} SD" if self.noise else "",
                 f"defocus up to {self.blur_um:g} µm" if self.blur_um else ""]
        what = ", ".join(p for p in parts if p) or "no change"
        return f"{self.copies} acquisition cop{'y' if self.copies == 1 else 'ies'} of each training project ({what})"

    def draw(self, path: str, copy_index: int) -> dict:
        """One copy's changes for one project (repeatable: from the seed, the project and the copy)."""
        h = int(hashlib.sha1(f"{self.seed}|{path}|{copy_index}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        return {"scale": 1.0 + rng.uniform(-self.pixel_size, self.pixel_size),
                "gain": 1.0 + rng.uniform(-self.gain, self.gain),
                "offset": rng.uniform(-self.offset, self.offset), "noise": self.noise,
                "blur_um": rng.uniform(0.0, self.blur_um) if self.blur_um else 0.0, "seed": h}


# ---------------------------------------------------------------- the changed pixels
def _window(src: PixelSource, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Level-0 pixels rows r0:r1, columns c0:c1 of a source (clipped to the image)."""
    h, w = src.levels[0].shape[:2]
    r0, r1, c0, c1 = max(0, r0), min(h, r1), max(0, c0), min(w, c1)
    th, tw = src.levels[0].tile
    rows = [np.concatenate([src.read_tile(0, ty, tx) for tx in range(c0 // tw, (c1 - 1) // tw + 1)], axis=1)
            for ty in range(r0 // th, (r1 - 1) // th + 1)]
    block = np.concatenate(rows, axis=0)
    oy, ox = (r0 // th) * th, (c0 // tw) * tw
    return block[r0 - oy:r1 - oy, c0 - ox:c1 - ox]


def pixel_sd(src: PixelSource) -> float:
    """The spread of an image's pixel values (SD of its central level-0 tile; 1 if it cannot be read)."""
    try:
        gy, gx = src.levels[0].grid
        v = np.asarray(src.read_tile(0, gy // 2, gx // 2), float)
        v = v[np.isfinite(v)]
        sd = float(v.std()) if v.size else 0.0
        return sd if sd > 0 else 1.0
    except Exception:
        return 1.0


class ChangedSource(PixelSource):
    """A source's level-0 pixels with gain, offset, noise and blur applied (see the module docstring).

    Args:
        base: The original source.
        change: From ``Acquisition.draw``.
        um_per_px: The image's pixel size (for the blur's σ in pixels).
    """

    def __init__(self, base: PixelSource, change: dict, um_per_px: float) -> None:
        super().__init__()
        self.base, self.change = base, change
        self.levels, self.dtype, self.channels = base.levels, base.dtype, base.channels
        self.missing = getattr(base, "missing", False)
        self.sd = pixel_sd(base)
        self.sigma_px = change["blur_um"] / um_per_px if um_per_px > 0 else 0.0
        self._tiles: dict = {}

    def value_range(self) -> tuple[float, float]:
        return self.base.value_range()

    def read_tile(self, level: int, ty: int, tx: int) -> np.ndarray:
        if level != 0:
            return self.base.read_tile(level, ty, tx)
        if (ty, tx) in self._tiles:
            return self._tiles[(ty, tx)]
        import cv2
        th, tw = self.levels[0].tile
        h, w = self.levels[0].shape[:2]
        r0, c0 = ty * th, tx * tw
        r1, c1 = min(h, r0 + th), min(w, c0 + tw)
        pad = int(math.ceil(3 * self.sigma_px)) if self.sigma_px > 0 else 0
        block = _window(self.base, r0 - pad, r1 + pad, c0 - pad, c1 + pad).astype(np.float32)
        top, left = min(pad, r0), min(pad, c0)
        c = self.change
        # 1. Defocus, then gain and offset (the same in the whole project), then noise (new in every pixel)
        if self.sigma_px > 0:
            block = cv2.GaussianBlur(block, (0, 0), self.sigma_px, borderType=cv2.BORDER_REFLECT)
        x = block[top:top + (r1 - r0), left:left + (c1 - c0)]
        x = x * c["gain"] + c["offset"] * self.sd
        if c["noise"]:
            rng = np.random.default_rng([c["seed"], ty, tx])
            x = x + rng.normal(0.0, c["noise"] * self.sd, x.shape).astype(np.float32)
        # 2. Back to the file's kind of values
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            x = np.clip(np.rint(x), info.min, info.max)
        out = x.astype(self.dtype)
        self._tiles[(ty, tx)] = out
        return out


def changed_image(image, change: dict):
    """A copy of an ImageNode as acquired with ``change``: its pixels (and its views') changed, its pixel size and
    its masks' outlines scaled together. The original is not touched."""
    from shapely import affinity
    s = change["scale"]
    px = float(image.pixel_size_um[0])
    out = copy.copy(image)
    out.source = ChangedSource(image.source, change, px)
    out.pixel_size_um = (image.pixel_size_um[0] * s, image.pixel_size_um[1] * s)
    out.views = []
    for v in getattr(image, "views", []):
        v2 = copy.copy(v)
        v2.source = ChangedSource(v.source, change, px)
        out.views.append(v2)
    out.children = []
    for m in image.masks:
        m2 = copy.copy(m)
        m2.geometry = affinity.scale(m.geometry, s, s, origin=(0, 0)) if s != 1.0 else m.geometry
        out.children.append(m2)
    return out


# ---------------------------------------------------------------- measuring the copies
def measure(tset, projects: set[int], settings: Acquisition, progress: Callable[[float], None] | None = None,
            cancelled: Callable[[], bool] | None = None, library=None, use_cache: bool = True):
    """The graded objects of ``projects`` (indices into ``tset.projects``) measured on ``settings.copies`` changed
    acquisitions each (see the module docstring).

    Returns:
        A ``dataset.Dataset`` (``variant``: the copy of each row).

    Raises:
        dataset.Cancelled: If ``cancelled`` returned True.
    """
    from shapely import affinity

    from modalytics.parameters import engine
    from modalytics.parameters.classes import ClassIndex
    from modalytics.parameters.registry import ParameterLibrary, distinct_labels
    from modalytics.training import dataset as ds
    library = library or ParameterLibrary()
    grades = tset.grades
    choices, _missing = engine.select(library, tset.profile.parameters, "object")
    labels = distinct_labels(s for s, _c in choices)
    columns = [(s.uid, labels[s.uid], s.unit, s.help_text()) for s, _c in choices]
    uids = [c[0] for c in columns]
    whole = [s.whole_object for s, _c in choices]
    cache_dir = ds.library_folder() / "cache"
    rows, grade, project, variant, where, partial_rows = [], [], [], [], [], []
    todo = [(pi, k) for pi in sorted(projects) for k in range(settings.copies)]
    for n_done, (pi, k) in enumerate(todo):
        if cancelled is not None and cancelled():
            raise ds.Cancelled()
        entry = tset.projects[pi]
        try:
            key = ds._cache_key(entry.path, tset.profile, tset.group)
        except OSError:
            continue
        cached = cache_dir / f"{key}-acq{settings.key()}-{k}.json"
        if use_cache and cached.exists():
            with open(cached, encoding="utf-8") as fh:
                got = json.load(fh)
        else:
            # 1. The project's images as acquired with this copy's changes
            try:
                images = ds._open(entry.path)
            except Exception:
                continue
            change = settings.draw(entry.path, k)
            scaled = change["scale"] != 1.0
            changed = [changed_image(img, change) for img in images]
            got = {"rows": []}
            objects = [(img, new, m, i, poly) for img, new in zip(images, changed) for m in img.masks
                       if m.group == tset.group for i, poly in enumerate(engine.objects_of(m.geometry))]
            for j, (img, new, m, i, poly) in enumerate(objects):
                if cancelled is not None and cancelled():
                    raise ds.Cancelled()
                _g, sub = tset.profile.label_for(m.group, m.subgroup)
                if sub not in grades:
                    continue
                cut = img.partial_reason(poly, tset.profile.edge_share)       # as the original: its outline
                shape = affinity.scale(poly, change["scale"], change["scale"], origin=(0, 0)) if scaled else poly
                others = () if scaled else changed                            # registered classes: same frame only
                values, _p = engine.measure_object(shape, new, choices, others, ClassIndex(new, others),
                                                   extras=False)
                got["rows"].append({"grade": sub, "image": img.name, "class": f"{m.group} › {m.subgroup}",
                                    "index": i, "point": list(poly.representative_point().coords[0]),
                                    "values": [ds._num(values.get(u)) for u in uids], **({"partial": cut} if cut
                                                                                         else {})})
                if progress is not None:
                    progress((n_done + (j + 1) / max(len(objects), 1)) / max(len(todo), 1))
            cache_dir.mkdir(parents=True, exist_ok=True)
            fileio.write_text(cached, json.dumps(got))
        for r in got["rows"]:
            if r["grade"] not in grades:
                continue
            values = [math.nan if v is None else v for v in r["values"]]
            if r.get("partial"):
                if not tset.include_partial:
                    continue
                values = [math.nan if w else v for v, w in zip(values, whole)]
            rows.append(values)
            grade.append(grades.index(r["grade"]))
            project.append(pi)
            variant.append(k + 1)
            partial_rows.append(bool(r.get("partial")))
            where.append((r["image"], r["class"], r["index"] + 1, tuple(r.get("point") or (math.nan, math.nan))))
        if progress is not None:
            progress((n_done + 1) / max(len(todo), 1))
    values = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(uids))
    return ds.Dataset(columns, values, np.asarray(grade, int), grades, np.asarray(project, int),
                      [p.name for p in tset.projects], where, {}, {}, {}, tset.profile.classes.ordered(tset.group),
                      np.asarray(partial_rows, bool), np.asarray(variant, int))
