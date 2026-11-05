"""
Synthetic graded projects (shared by the tests and the benches): phase maps with glands of two grades whose
difference is known.

* Low grade: smooth elliptical outlines, an even phase inside.
* High grade: wavy, irregular outlines (high best-alignment metric), a grainier phase inside.

Some projects use HoloPath's older class names ("Normal", "Dysplasia low grade", "Dysplasia high grade"), which
the Colonic tissue profile reads as its two grades.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

PX = 0.5                       # µm per pixel
SIZE = 400                     # px (200 µm)


def gland(cx, cy, r, irregular: bool, rng) -> Polygon:
    t = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    if irregular:
        k1, k2 = rng.integers(5, 9), rng.integers(9, 14)
        rr = r * (1 + 0.18 * np.sin(k1 * t + rng.uniform(0, 6)) + 0.10 * np.sin(k2 * t + rng.uniform(0, 6)))
        sx, sy = 1.0, 1.0
    else:
        rr = np.full_like(t, r)
        sx, sy = rng.uniform(1.0, 1.25), 1.0
    return Polygon(np.column_stack([cx + sx * rr * np.cos(t), cy + sy * rr * np.sin(t)])).buffer(0)


def graded_project(path: str, seed: int, n_low: int, n_high: int, old_names: bool = False, profile=None) -> str:
    """Write a .modalytics project with ``n_low`` low-grade and ``n_high`` high-grade glands; returns its path."""
    from modalytics.core.model import ImageNode, MaskNode, Project
    from modalytics.io.project_file import save_project
    from modalytics.io.sources import ArraySource
    rng = np.random.default_rng(seed)
    phase = rng.normal(0.0, 0.02, (SIZE, SIZE)).astype(np.float32)
    yy, xx = (np.mgrid[0:SIZE, 0:SIZE] + 0.5) * PX
    low, high = [], []
    centres = [(30 + 45 * (i % 4), 30 + 45 * (i // 4)) for i in range(16)]
    order = rng.permutation(len(centres))
    for k, idx in enumerate(order[:n_low + n_high]):
        cx, cy = centres[idx]
        is_high = k >= n_low
        poly = gland(cx, cy, rng.uniform(12, 17), is_high, rng)
        inside = np.zeros((SIZE, SIZE), bool)
        from modalytics.parameters import measures as ms
        inside |= ms.rasterize(poly, 0, 0, PX, PX, (SIZE, SIZE))
        grain = rng.normal(0, 0.35 if is_high else 0.05, (SIZE, SIZE)).astype(np.float32)
        phase[inside] += 1.0 + grain[inside]
        (high if is_high else low).append(poly)
    image = ImageNode(f"map {seed}", ArraySource(phase), (PX, PX), "phase", metadata={"wavelength_nm": 666.0})
    names = ("Normal", "Dysplasia high grade") if old_names else ("Low grade dysplasia", "High grade dysplasia")
    for sub, polys in zip(names, (low, high)):
        if polys:
            m = MaskNode(geometry=polys, group="Gland", subgroup=sub)
            m.parent = image
            image.children.append(m)
    project = Project()
    project.reset([image], profile=profile)
    save_project(project, path)
    return path if path.endswith(".modalytics") else path + ".modalytics"


def outlined_project(path: str, image, hand: list, name: str = "outlined", ungraded: list = ()) -> str:
    """Write a .modalytics project with a copy of ``image``'s pixels (a phase map) and ``hand`` as its Gland
    outlines (µm, graded Low grade dysplasia; ``ungraded``: Gland outlines without a grade), e.g. HoloPath's real
    map outlined as a model would; returns its path."""
    from modalytics.core.model import ImageNode, MaskNode, Project
    from modalytics.io.project_file import save_project
    from modalytics.io.sources import ArraySource
    from modalytics.services.segmentation import read_region
    w, h = image.size_um
    pixels, _frame = read_region(image, (0, 0, w, h), image.pixel_size_um[0])
    copy = ImageNode(name, ArraySource(np.asarray(pixels, np.float32).squeeze()), image.pixel_size_um, image.modality,
                     metadata=dict(getattr(image, "metadata", {}) or {}))
    for polys, sub in ((hand, "Low grade dysplasia"), (ungraded, "")):
        if polys:
            m = MaskNode(geometry=list(polys), group="Gland", subgroup=sub)
            m.parent = copy
            copy.children.append(m)
    project = Project()
    project.reset([copy])
    save_project(project, path)
    return path if path.endswith(".modalytics") else path + ".modalytics"
