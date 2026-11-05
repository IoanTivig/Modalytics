"""
How well a segmentation model outlines the objects drawn by hand (Qt-free).

A segmentation model outlines the objects of one class group (the segmentation
layer, e.g. Gland). It is judged on projects whose objects were outlined by hand,
by running it on their images and comparing outlines — on the vector outlines
themselves (shapely areas), so the comparison does not depend on a pixel grid.

**Where.** Either the *whole image* (every object of the group was outlined by
hand: what the model finds elsewhere is invented), or only *around the hand
outlines* (their convex hull widened by a margin: for slides where only some
regions were outlined).

**Overlap** of all hand outlines A with all model outlines B, in µm²:

    Dice = 2·|A ∩ B| / (|A| + |B|)            (Dice 1945)
    IoU  = |A ∩ B| / |A ∪ B|                    (Jaccard 1912)

**Objects.** A model object *matches* a hand object when their IoU ≥ 0.5 (each used
once; above 0.5 a match can only be one-to-one). Matched = found (TP), hand objects
left = missed (FN), model objects left = invented (FP):

    precision = TP / (TP + FP),  recall = TP / (TP + FN),  F1 = 2·TP / (2·TP + FP + FN)

with 95 % Wilson intervals for precision and recall. **Matched IoU** (segmentation
quality, SQ) is the mean IoU of the matches — how well found objects are outlined;
**panoptic quality** PQ = SQ · F1 (Kirillov et al. 2019) combines finding and
outlining in one number. A model object covering ≥ 50 % of two or more hand objects
is a **merge**; a hand object of which two or more model objects lie ≥ 50 % inside
is a **split**.

**GlaS F1** (the challenge's own detection rule, Sirinukunwattana et al. 2017): a model
object is found (TP) when it covers ≥ 50 % of the hand object it overlaps most — measured
on the hand object only, so an outline much larger than the gland still counts — and
FP = model objects − TP, FN = hand objects − TP:

    GlaS F1 = 2·TP / (2·TP + FP + FN) = 2·TP / (hand objects + model objects)

It is more lenient than the F1 above (IoU ≥ 0.5 also penalises outlines too large), and
the one to set beside published gland results.

**Object Dice and object Hausdorff** (the GlaS challenge measures, Sirinukunwattana et
al. 2017): each hand object G is compared with the model object S* that overlaps it
most, and each model object S with the hand object G* that overlaps it most, weighted
by area:

    object Dice = ½ [Σ γ_i Dice(G_i, S*(G_i)) + Σ σ_j Dice(G*(S_j), S_j)],
    γ_i = |G_i| / Σ|G|,  σ_j = |S_j| / Σ|S|        (no overlapping object: Dice 0)

and the same with the Hausdorff distance H (no overlapping object: the nearest one).
Here they are pooled over all the objects of the evaluated images.

**Boundaries of the matches** (Metrics Reloaded, Maier-Hein et al. 2024): **HD95**, the
95th percentile of the distances from each outline to the other (the larger of the two
directions), and the **normalised surface distance** NSD at a tolerance τ (Nikolov et
al. 2021): the share of both outlines lying within τ of the other — both averaged over
the matched objects, from points every ``BOUNDARY_STEP_UM`` along the outlines.

References:
    Dice LR. Measures of the Amount of Ecologic Association Between Species. Ecology. 1945;26(3):297-302.
        doi:10.2307/1932409
    Jaccard P. The Distribution of the Flora in the Alpine Zone. New Phytologist. 1912;11(2):37-50.
        doi:10.1111/j.1469-8137.1912.tb05611.x
    Kirillov A, He K, Girshick R, Rother C, Dollár P. Panoptic Segmentation. CVPR 2019:9396-9405.
        doi:10.1109/CVPR.2019.00963
    Sirinukunwattana K, Pluim JPW, Chen H, et al. Gland segmentation in colon histology images: The glas
        challenge contest. Medical Image Analysis. 2017;35:489-502. doi:10.1016/j.media.2016.08.008
    Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation.
        Nature Methods. 2024;21(2):195-212. doi:10.1038/s41592-023-02151-z
    Nikolov S, Blackwell S, Zverovitch A, et al. Clinically Applicable Segmentation of Head and Neck Anatomy
        for Radiotherapy: Deep Learning Algorithm Development and Validation Study. J Med Internet Res.
        2021;23(7):e26151. doi:10.2196/26151
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from modalytics import fileio
from modalytics.training.classifiers import wilson

CACHE_VERSION = 1
POOR_IOU = 0.7                     # a match outlined worse than this is listed for a look
BOUNDARY_STEP_UM = 0.5             # outline points this far apart for HD95 and NSD
NSD_TOLERANCE_UM = 2.0             # τ of the normalised surface distance


class Cancelled(Exception):
    """Raised when an evaluation is cancelled."""


# ---------------------------------------------------------------- comparing outlines
@dataclass
class Counts:
    """Additive counts of a comparison (one image, one project, or all)."""

    hand: int = 0
    found: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    merges: int = 0
    splits: int = 0
    iou_sum: float = 0.0                # sum of the matches' IoU
    inter: float = 0.0                  # |A ∩ B| µm²
    hand_area: float = 0.0              # |A|
    found_area: float = 0.0             # |B|
    images: int = 0
    # GlaS object measures (sums weighted by object area; areas of each side)
    odice_hand: float = 0.0             # Σ |G_i| · Dice(G_i, S*(G_i))
    odice_found: float = 0.0            # Σ |S_j| · Dice(G*(S_j), S_j)
    ohd_hand: float = 0.0               # Σ |G_i| · H(G_i, S*(G_i))
    ohd_found: float = 0.0              # Σ |S_j| · H(G*(S_j), S_j)
    obj_area_hand: float = 0.0          # Σ |G_i|
    obj_area_found: float = 0.0         # Σ |S_j|
    hd_area_hand: float = 0.0           # Σ |G_i| over those with a counterpart (H defined)
    hd_area_found: float = 0.0
    # Boundaries of the matches
    hd95_sum: float = 0.0
    nsd_sum: float = 0.0
    # GlaS detection: model objects covering ≥ 50 % of the hand object they overlap most
    glas_tp: int = 0
    glas_images: int = 0                # images where glas_tp was counted (results saved before it: none)

    def add(self, other: "Counts") -> "Counts":
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else math.nan

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else math.nan

    @property
    def precision_ci(self) -> tuple[float, float]:
        return wilson(self.tp, self.tp + self.fp)

    @property
    def recall_ci(self) -> tuple[float, float]:
        return wilson(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else math.nan

    @property
    def dice(self) -> float:
        d = self.hand_area + self.found_area
        return 2 * self.inter / d if d else math.nan

    @property
    def iou(self) -> float:
        d = self.hand_area + self.found_area - self.inter
        return self.inter / d if d else math.nan

    @property
    def matched_iou(self) -> float:
        return self.iou_sum / self.tp if self.tp else math.nan

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d else math.nan

    @property
    def sq(self) -> float:
        """Segmentation quality: the matches' mean IoU (= matched IoU)."""
        return self.matched_iou

    @property
    def rq(self) -> float:
        """Recognition quality: F1 of the objects (PQ = SQ · RQ)."""
        return self.f1

    @property
    def glas_f1(self) -> float:
        """GlaS F1: 2·TP / (hand objects + model objects), TP by the GlaS rule (unknown for older results)."""
        if self.glas_images < self.images:
            return math.nan
        d = self.hand + self.found
        return 2 * self.glas_tp / d if d else math.nan

    @property
    def object_dice(self) -> float:
        """GlaS object Dice (both sides; a side without objects counts 0)."""
        if not (self.obj_area_hand or self.obj_area_found):
            return math.nan
        h = self.odice_hand / self.obj_area_hand if self.obj_area_hand else 0.0
        f = self.odice_found / self.obj_area_found if self.obj_area_found else 0.0
        return 0.5 * (h + f)

    @property
    def object_hausdorff(self) -> float:
        """GlaS object Hausdorff distance (µm), over the objects that have a counterpart."""
        parts = [self.ohd_hand / self.hd_area_hand if self.hd_area_hand else None,
                 self.ohd_found / self.hd_area_found if self.hd_area_found else None]
        parts = [x for x in parts if x is not None]
        return 0.5 * sum(parts) if len(parts) == 2 else (parts[0] if parts else math.nan)

    @property
    def hd95(self) -> float:
        """Mean HD95 of the matched objects (µm)."""
        return self.hd95_sum / self.tp if self.tp else math.nan

    @property
    def nsd(self) -> float:
        """Mean normalised surface distance of the matched objects at ``NSD_TOLERANCE_UM``."""
        return self.nsd_sum / self.tp if self.tp else math.nan


@dataclass
class Item:
    """One object to look at: missed, invented, merged, split, or a match outlined poorly.

    Attributes:
        kind: "missed", "invented", "merge", "split" or "poor".
        image: The image's name.
        point: A point inside it (x, y), in the image's µm.
        area: Its area (µm²; of the hand object for missed / split, else of the model's).
        iou: Its best IoU with the other side (0 when nothing overlaps).
        outline: The model's outline (exterior (x, y) µm) when there is one, to draw.
        project: Index of its project in the evaluation (set by ``evaluate``).
    """

    kind: str
    image: str
    point: tuple
    area: float
    iou: float
    outline: list = field(default_factory=list)
    project: int = -1


def _iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if inter > 0 else 0.0


def compare(hand: list, found: list, image: str = "", iou_min: float = 0.5) -> tuple[Counts, list[Item]]:
    """Compare model objects with hand objects (see the module docstring).

    Args:
        hand: Hand-drawn objects (Polygons, µm).
        found: Model objects (Polygons, µm), already limited to the evaluated area.
        image: The image's name (for the items).
        iou_min: IoU from which a model object matches a hand object.

    Returns:
        tuple: (counts, items to look at).
    """
    c = Counts(hand=len(hand), found=len(found), images=1, glas_images=1)
    A = unary_union(hand) if hand else Polygon()
    B = unary_union(found) if found else Polygon()
    c.hand_area, c.found_area = A.area, B.area
    c.inter = A.intersection(B).area if hand and found else 0.0
    # 1. Candidate pairs (overlapping bounds), their IoU and cover fractions
    pairs = {}
    if hand and found:
        tree = shapely.STRtree(found)
        for hi, h in enumerate(hand):
            for fi in tree.query(h):
                f = found[int(fi)]
                inter = h.intersection(f).area
                if inter > 0:
                    pairs[(hi, int(fi))] = (inter / (h.area + f.area - inter), inter / h.area, inter / f.area)
    # 2. One-to-one matches, best IoU first
    used_h, used_f, items, matches = set(), set(), [], []
    for (hi, fi), (iou, _ch, _cf) in sorted(pairs.items(), key=lambda kv: -kv[1][0]):
        if iou < iou_min or hi in used_h or fi in used_f:
            continue
        used_h.add(hi)
        used_f.add(fi)
        matches.append((hi, fi))
        c.tp += 1
        c.iou_sum += iou
        if iou < POOR_IOU:
            items.append(_item("poor", image, found[fi], iou, True))
    best_h = {hi: max((v[0] for (h, _f), v in pairs.items() if h == hi), default=0.0) for hi in range(len(hand))}
    best_f = {fi: max((v[0] for (_h, f), v in pairs.items() if f == fi), default=0.0) for fi in range(len(found))}
    for hi, h in enumerate(hand):
        if hi not in used_h:
            c.fn += 1
            items.append(_item("missed", image, h, best_h[hi], False))
    for fi, f in enumerate(found):
        if fi not in used_f:
            c.fp += 1
            items.append(_item("invented", image, f, best_f[fi], True))
    # 3. GlaS object Dice and Hausdorff: each object against the one overlapping it most (else the nearest)
    most = ({}, {})                       # per side: object → (overlap µm², index of the other object)
    for (hi, fi), (_iou, ch, _cf) in pairs.items():
        inter = ch * hand[hi].area
        if inter > most[0].get(hi, (0.0, -1))[0]:
            most[0][hi] = (inter, fi)
        if inter > most[1].get(fi, (0.0, -1))[0]:
            most[1][fi] = (inter, hi)
    credited = set()                      # GlaS F1: a model object covering ≥ 50 % of its hand object (each once)
    for fi, (inter, hi) in most[1].items():
        if inter >= 0.5 * hand[hi].area and hi not in credited:
            credited.add(hi)
            c.glas_tp += 1
    for side, objs, others in ((0, hand, found), (1, found, hand)):
        other_tree = shapely.STRtree(others) if others else None
        for i, obj in enumerate(objs):
            area = obj.area
            best = most[side].get(i, (0.0, -1))
            if best[0] > 0:
                o = others[best[1]]
                dice = 2 * best[0] / (area + o.area)
                hd = shapely.hausdorff_distance(obj, o)
            else:
                dice = 0.0
                near = other_tree.nearest(obj) if other_tree is not None else None
                hd = shapely.hausdorff_distance(obj, others[int(near)]) if near is not None else None
            if side == 0:
                c.obj_area_hand += area
                c.odice_hand += area * dice
                if hd is not None:
                    c.ohd_hand += area * hd
                    c.hd_area_hand += area
            else:
                c.obj_area_found += area
                c.odice_found += area * dice
                if hd is not None:
                    c.ohd_found += area * hd
                    c.hd_area_found += area
    # 4. Boundaries of the matches: HD95 and NSD
    for hi, fi in matches:
        hd95, nsd = boundary_measures(hand[hi], found[fi])
        c.hd95_sum += hd95
        c.nsd_sum += nsd
    # 5. Merges and splits
    for fi, f in enumerate(found):
        if sum(1 for (h, f2), v in pairs.items() if f2 == fi and v[1] >= 0.5) >= 2:
            c.merges += 1
            items.append(_item("merge", image, f, best_f[fi], True))
    for hi, h in enumerate(hand):
        if sum(1 for (h2, f), v in pairs.items() if h2 == hi and v[2] >= 0.5) >= 2:
            c.splits += 1
            items.append(_item("split", image, h, best_h[hi], False))
    return c, items


def boundary_measures(a: Polygon, b: Polygon, step: float = BOUNDARY_STEP_UM,
                      tolerance: float = NSD_TOLERANCE_UM) -> tuple[float, float]:
    """(HD95, NSD) of two outlines: points every ``step`` µm along each (holes included), their distances to the
    other outline; HD95 = the larger of the two 95th percentiles, NSD = the share within ``tolerance``."""
    pa = shapely.points(shapely.get_coordinates(shapely.segmentize(a.boundary, step)))
    pb = shapely.points(shapely.get_coordinates(shapely.segmentize(b.boundary, step)))
    da, db = shapely.distance(pa, b.boundary), shapely.distance(pb, a.boundary)
    hd95 = float(max(np.percentile(da, 95), np.percentile(db, 95)))
    nsd = float(((da <= tolerance).sum() + (db <= tolerance).sum()) / (len(da) + len(db)))
    return hd95, nsd


def _item(kind: str, image: str, poly: Polygon, iou: float, is_model: bool) -> Item:
    outline = [tuple(map(float, xy)) for xy in poly.simplify(0.5).exterior.coords] if is_model else []
    return Item(kind, image, tuple(poly.representative_point().coords[0]), float(poly.area), float(iou), outline)


AREAS = {
    "marked": "the whole image where its annotation is marked complete, else around the outlines",
    "image": "the whole image (everything was outlined)",
    "hull": "around the hand outlines only",
}


def evaluated_area(hand: list, w_um: float, h_um: float, mode: str, margin_um: float, complete: bool = False):
    """The area compared: the whole image, or the hand outlines' convex hull widened by ``margin_um``.

    Args:
        mode: "image", "hull", or "marked" (the whole image when ``complete`` — the image is marked fully
            outlined for the group — else the hull).
    """
    whole = box(0, 0, w_um, h_um)
    if mode == "marked":
        mode = "image" if complete else "hull"
    if mode == "image":
        return whole
    if not hand:
        return None
    return unary_union(hand).convex_hull.buffer(margin_um).intersection(whole)


PENDING_MARGIN_UM = 2.0          # around an unreviewed suggestion: also left out (its outline may be a little off)


def pending(image, group: str) -> list:
    """The unreviewed suggestions of ``group`` on an image that may be objects nobody outlined yet (not the ones
    that only suggest a grade for an object already drawn)."""
    node = getattr(image, "suggestions", None)
    return [it.geometry for it in (node.items if node is not None else []) if it.group == group and not it.regrade]


def without_pending(region, image, group: str):
    """``region`` minus the unreviewed suggestions (what is under them is neither known object nor known
    background); returns (region, how many were left out)."""
    todo = pending(image, group)
    if not todo or region is None:
        return region, 0
    return region.difference(unary_union(todo).buffer(PENDING_MARGIN_UM)), len(todo)


def unreviewed(paths: list[str], group: str) -> dict:
    """{project name: unreviewed suggestions of ``group``} in project files (read quickly, without the images)."""
    import sqlite3
    out = {}
    for path in paths:
        if not path.lower().endswith(".modalytics") or not os.path.exists(path):
            continue
        try:
            con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
            try:
                if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'suggestions'"
                                   ).fetchone():
                    continue
                n = sum(1 for (data,) in con.execute("SELECT data FROM suggestions")
                        for it in json.loads(data).get("items", [])
                        if it.get("group") == group and not it.get("regrade"))
            finally:
                con.close()
        except (sqlite3.Error, ValueError):
            continue
        if n:
            out[os.path.splitext(os.path.basename(path))[0]] = n
    return out


def limit(found: list, area) -> list:
    """The model objects mostly (≥ 50 %) inside the evaluated area."""
    return [f for f in found if f.intersection(area).area >= 0.5 * f.area]


# ---------------------------------------------------------------- evaluating on projects
@dataclass
class Evaluation:
    """A model judged on projects.

    Attributes:
        projects: The projects' names, in order.
        paths: Their files.
        per_project: Counts per project (same order).
        total: All together.
        items: Objects to look at.
        skipped: Why images or projects were left out.
        models: {image type: model id} used.
        per_image: [(project index, image name, Counts)] — for confidence intervals over images.
    """

    projects: list
    paths: list
    per_project: list
    total: Counts
    items: list
    skipped: list
    models: dict
    per_image: list = field(default_factory=list)
    examples: list = field(default_factory=list)


@dataclass
class Example:
    """One evaluated image kept for figures: a small picture of the region, the hand and the model outlines.

    Attributes:
        project, image: Where it comes from.
        pixels: (H, W) or (H, W, 3) uint8 picture of the region.
        frame: (x0, y0, px, py) of ``pixels`` in the image's local µm.
        hand, found: The outlines (Polygons, the image's local µm).
    """

    project: str
    image: str
    pixels: object
    frame: tuple
    hand: list
    found: list


EXAMPLE_PX = 640                  # an example picture's longer side (pixels)


def _example(img, rect: tuple, hand: list, found: list, project: str) -> Example:
    """A small picture of an evaluated region with its outlines (contrast stretched to 1–99 %)."""
    from modalytics.services.segmentation import read_region
    x0, y0, x1, y1 = rect
    um = max(x1 - x0, y1 - y0) / EXAMPLE_PX
    pixels, frame = read_region(img, rect, um)
    p = np.asarray(pixels, dtype=np.float32)
    lo, hi = np.percentile(p, [1, 99])
    p = np.clip((p - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255).astype(np.uint8)
    return Example(project, img.name, p, frame, list(hand), list(found))


def _polys(geom) -> list:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    return []


def _found(image, rect, pkg, path: str, cache_dir, progress, cancelled, device: str) -> list:
    """The model's objects on a region of an image (cached per project file, image, region and model)."""
    from modalytics.services import segmentation as seg
    key = None
    if cache_dir is not None:
        st, ms = os.stat(path), os.stat(pkg.path)
        key = hashlib.sha1(json.dumps([CACHE_VERSION, os.path.abspath(path), st.st_size, st.st_mtime_ns, image.id,
                                       image.name, [round(v, 3) for v in rect], pkg.path, ms.st_size,
                                       ms.st_mtime_ns]).encode()).hexdigest()[:24]
        cached = cache_dir / f"{key}.json"
        if cached.exists():
            with open(cached, encoding="utf-8") as fh:
                return _polys(shapely.from_wkt(json.load(fh)["wkt"]))
    try:
        geom = seg.segment(image, rect, pkg, progress=progress, cancelled=cancelled, device=device)
    except seg.Cancelled:
        raise Cancelled() from None
    if key is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fileio.write_text(cache_dir / f"{key}.json", json.dumps({"wkt": geom.wkt}))
    return _polys(geom)


def evaluate(paths: list[str], group: str, models: dict, area: str = "image", margin_um: float = 50.0,
             iou_min: float = 0.5, progress: Callable[[float], None] | None = None,
             cancelled: Callable[[], bool] | None = None, device: str = "auto", use_cache: bool = True,
             keep_examples: int = 0) -> Evaluation:
    """Run models on the projects' images and compare with the hand outlines of ``group``.

    Args:
        paths: Project files (.modalytics or HoloPath .hpro).
        group: The class group outlined (all its subgroups count).
        models: {image type: ModelPackage}; images of other types are left out.
        area: "image" (whole image), "hull" (around the hand outlines) or "marked" (whole image where the image
            is marked complete for the group, else around the outlines). Unreviewed suggestions are left out.
        margin_um: The hull's margin.
        iou_min: IoU from which objects match.
        device: Where the model runs ("auto": the best device here).
        use_cache: Reuse the model's outlines found before for the same file, image, region and model.
        keep_examples: Keep this many images (their pixels, hand and model outlines) for figures.

    Raises:
        Cancelled: If ``cancelled`` returned True.
    """
    from modalytics.parameters.engine import objects_of
    from modalytics.training.dataset import _open, library_folder
    cache_dir = library_folder() / "cache" / "segmentation" if use_cache else None
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    per_project, items, skipped, per_image, examples = [], [], [], [], []
    total = Counts()
    for pi, path in enumerate(paths):
        counts = Counts()
        try:
            images = _open(path)
        except Exception as exc:                                  # unreadable: reported, the others go on
            skipped.append(f"{names[pi]}: {exc}")
            per_project.append(counts)
            continue
        todo = [img for img in images if img.modality in models]
        for img in images:
            if img.modality not in models:
                skipped.append(f"{names[pi]} › {img.name}: no model for {img.modality} images")
        for k, img in enumerate(todo):
            if cancelled is not None and cancelled():
                raise Cancelled()
            hand = [p for m in img.masks if m.group == group for p in objects_of(m.geometry)]
            w, h = img.size_um
            region = evaluated_area(hand, w, h, area, margin_um, img.fully_outlined(group))
            region, n_pending = without_pending(region, img, group)
            if n_pending:
                skipped.append(f"{names[pi]} › {img.name}: {n_pending} unreviewed suggestion(s), their places "
                               f"left out")
            if region is None or region.is_empty:
                skipped.append(f"{names[pi]} › {img.name}: no {group} outlined by hand")
                continue
            x0, y0, x1, y1 = region.bounds

            def sub(f, k=k):
                if progress is not None:
                    progress((pi + (k + f) / max(len(todo), 1)) / max(len(paths), 1))
            found = limit(_found(img, (x0, y0, x1, y1), models[img.modality], path, cache_dir, sub, cancelled,
                                 device), region)
            c, its = compare(hand, found, img.name, iou_min)
            for it in its:
                it.project = pi
            per_image.append((pi, img.name, c))
            if len(examples) < keep_examples:                # for figures: the image, its hand and model outlines
                examples.append(_example(img, (x0, y0, x1, y1), hand, found, names[pi]))
            counts.add(c)
            items += its
        per_project.append(counts)
        total.add(counts)
        if progress is not None:
            progress((pi + 1) / max(len(paths), 1))
    return Evaluation(names, list(paths), per_project, total, items, skipped,
                      {t: m.id for t, m in models.items()}, per_image, examples)
