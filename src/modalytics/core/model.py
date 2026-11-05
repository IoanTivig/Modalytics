"""
Domain model: project, images, masks and their hierarchy.

This module is the single source of truth for what is loaded and where it
sits in µm. The canvas and the panels *observe* it through signals; they
never store geometry of their own.

Hierarchy:
    Project
     └─ ImageNode            (WSI, phase map, brightfield…; bottom → top draw order)
         ├─ ImageView        (a processed version of it, made by a processing chain: OD map,
         │                    denoised OD…; same pixel grid, placement and masks; one of them,
         │                    or the original, is shown)
         └─ MaskNode         (vector polygons in the parent image's local µm frame)

All changes go through ``Project`` methods so every observer is notified the
same way. Undoable edits wrap these methods in commands (see ``commands.py``).
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field, replace

import numpy as np

from modalytics.core import annotations as ann
from modalytics.core import geometry as geo
from modalytics.core.annotations import LabelSchema, default_schema
from modalytics.core.geometry import Placement
from modalytics.core.signals import Signal

EDGE_PX = 1.0                     # an object this close to the image border (pixels) touches it
EDGE_SHARE = 0.05                 # default: cut by the edge when more than 5 % of its outline lies on it
PARTIAL_EDGE = "cut by the image edge"
PARTIAL_MARKED = "marked partial"


## Display settings ##

@dataclass(frozen=True)
class Display:
    """How an image is drawn (never changes the data).

    Attributes:
        clim: (low, high) contrast limits for single-channel data.
        colormap: Colormap name for single-channel data ("gray", "viridis"…).
    """

    clim: tuple[float, float] = (0.0, 1.0)
    colormap: str = "gray"


## Nodes ##

class Node:
    """Base node of the layer tree.

    Attributes:
        id: Stable unique id (hex string).
        name: Display name.
        visible: Whether the node is drawn.
        opacity: 0–1 opacity used when superposing modalities.
        parent: Parent node (None for top-level images).
        children: Child nodes (masks under an image).
    """

    kind = "node"

    def __init__(self, name: str) -> None:
        self.id: str = uuid.uuid4().hex
        self.name = name
        self.visible = True
        self.opacity = 1.0
        self.parent: Node | None = None
        self.children: list[Node] = []

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


# Image types ("modalities"): key → name shown to users. Profiles declare which they accept,
# models and parameters which they work on. The first three are acquired (imported); the others are
# made from them by processing steps (``modalytics.processing``) and exist as views of an image.
IMAGE_TYPES = {
    "wsi": "Whole-slide image (scanned)",
    "phase": "Phase map (DHM / QPI)",
    "brightfield": "Brightfield (microscope camera)",
    "od": "OD map (optical density)",
    "opd": "OPD map (optical path difference)",
    "dry_mass": "Dry-mass density map",
    "height": "Height map (thickness)",
    "ri": "Refractive-index map",
    "stain": "Stain map (one stain's optical density)",
    "feature": "Feature map (edges, blobs, ridges, texture)",
}
ACQUIRED_TYPES = ("wsi", "phase", "brightfield")      # imported; the rest are processing results


@dataclass
class ImageView:
    """A processed version of an image, made by a processing chain (its OD map, a denoised OD map…).

    It has the image's pixel grid, placement and masks (it *is* the image, seen
    through the chain's steps), so objects drawn on the image are measured on it
    directly. Only the chain is saved with the project; the pixels are computed
    when needed (``modalytics.processing``).

    Attributes:
        chain: {"name", "steps": [{"step": step id, "options": {…}}], "measure": bool}.
        name: Shown in Layers (the chain's name).
        image_type: The image type of its pixels (``IMAGE_TYPES`` key).
        source: Its PixelSource.
        display: How it is drawn (its own contrast and colormap).
        unit: Unit of its values ("OD", "nm"…).
        standard: Made by the project's profile: when measured, its parameters are the plain
            results; other measured views add labelled ones.
        id: Identity.
    """

    chain: dict
    name: str
    image_type: str
    source: object
    display: Display
    unit: str = ""
    standard: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def measure(self) -> bool:
        """Whether its parameters are measured (else it is only for looking)."""
        return bool(self.chain.get("measure"))

    @property
    def steps(self) -> list[dict]:
        return list(self.chain.get("steps", []))


class ImageNode(Node):
    """An image placed in the µm world.

    Args:
        name: Display name.
        source: Tiled pixel data.
        pixel_size_um: Calibrated (sx, sy) µm per level-0 pixel. This is the
            *parametric rescaling*: changing it rescales the image and its masks
            and keeps every µm measurement correct.
        modality: Image type key ("wsi", "phase", "brightfield"…); importers,
            parameters and segmenters declare which modalities they accept.
        placement: Local → world mapping.
        metadata: Free-form importer metadata (units, source file…).
    """

    kind = "image"

    def __init__(self, name: str, source, pixel_size_um: tuple[float, float], modality: str,
                 placement: Placement | None = None, metadata: dict | None = None) -> None:
        super().__init__(name)
        self.source = source
        self.pixel_size_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
        self.modality = modality
        self.placement = placement or Placement()
        self.metadata = dict(metadata or {})
        lo, hi = source.value_range()
        self.display = Display(clim=(lo, hi), colormap="gray")
        self.views: list[ImageView] = []                 # processed versions (``ImageView``)
        self.shown: ImageView | None = None             # the one drawn (None: the original)

    @property
    def view_source(self):
        """The pixels drawn: the shown view's, or the original's."""
        return self.shown.source if self.shown is not None else self.source

    @property
    def view_display(self) -> Display:
        """Display settings of what is drawn."""
        return self.shown.display if self.shown is not None else self.display

    @property
    def view_type(self) -> str:
        """Image type of what is drawn."""
        return self.shown.image_type if self.shown is not None else self.modality

    @property
    def original_measured(self) -> bool:
        """Whether the original is included in the parameters of its type (Layers › Processed, its tick).

        Unset (``metadata["measure_original"]``): yes, unless the profile's own (standard) view of the image's type
        is included — that view then stands for it, as the profile asks.
        """
        on = self.metadata.get("measure_original")
        if on is not None:
            return bool(on)
        return not any(v.standard and v.measure and v.image_type == self.modality for v in self.views)

    def measured_view(self, image_type: str) -> "ImageView | None":
        """The view the plain parameters of ``image_type`` are measured on (None: the original, or nothing).

        Of the included versions: the profile's (standard) view first, else the original (when it is of that type),
        else the first included view. The others give labelled results (``extra_versions``).
        """
        measured = [v for v in self.views if v.image_type == image_type and v.measure]
        standard = next((v for v in measured if v.standard), None)
        if standard is not None:
            return standard
        if image_type == self.modality and self.original_measured:
            return None
        return measured[0] if measured else None

    def version(self, view: "ImageView | None") -> "ImageNode":
        """A stand-in of one version of the image (a view, or None: the original) with no other versions: what
        its parameters are measured on. It shares the name, pixel size, placement and metadata, so an object
        outlined on the image is measured on that version's pixels."""
        if view is None:
            stand_in = ImageNode(f"{self.name} › Original", self.source, self.pixel_size_um, self.modality,
                                 self.placement, dict(self.metadata))
        else:
            stand_in = ImageNode(f"{self.name} › {view.name}", view.source, self.pixel_size_um, view.image_type,
                                 self.placement, dict(self.metadata, view=view.name, unit=view.unit))
        stand_in.id = self.id
        return stand_in

    def extra_versions(self, image_type: str) -> list[tuple[str, "ImageNode"]]:
        """The other included versions of ``image_type`` (not the plain one): [(label, stand-in)], each giving
        results labelled with its name (e.g. "Mean phase · Denoised")."""
        plain = self.measured_view(image_type)
        if plain is None and not (image_type == self.modality and self.original_measured):
            return []                                    # nothing of that type at all
        out = []
        if plain is not None and image_type == self.modality and self.original_measured:
            out.append(("Original", self.version(None)))
        for v in self.views:
            if v.image_type == image_type and v.measure and v is not plain:
                out.append((v.name, self.version(v)))
        seen: dict[str, int] = {}
        for i, (label, node) in enumerate(out):          # two views of one name: told apart
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                out[i] = (f"{label} ({seen[label]})", node)
        return out

    def as_type(self, image_type: str, view: "ImageView | None" = None) -> "ImageNode | None":
        """This image as one of ``image_type``: a stand-in for the view measured for that type, or
        the image itself, or None.

        For the image's own type, a measured view of that type (e.g. a denoising chain in the profile)
        takes the original's place. A view's stand-in shares the name, pixel size, placement and
        metadata, so an object outlined on the image is measured on the view's pixels.

        Args:
            image_type: The parameters' image type ("*": any, the outline only).
            view: Use this view (e.g. a measured chain's own, labelled results) instead of choosing.
        """
        if image_type == "*":
            return self
        view = view if view is not None else self.measured_view(image_type)
        if view is None:
            return self if image_type == self.modality and self.original_measured else None
        return self.version(view)

    @property
    def masks(self) -> list["MaskNode"]:
        """Child masks (the annotation; suggestions are not masks)."""
        return [c for c in self.children if isinstance(c, MaskNode)]

    @property
    def suggestions(self) -> "SuggestionNode | None":
        """The models' suggestions waiting for review on this image, if any."""
        return next((c for c in self.children if isinstance(c, SuggestionNode)), None)

    @property
    def size_um(self) -> tuple[float, float]:
        """(width, height) of the local frame in µm (before placement scale)."""
        h, w = self.source.shape
        return w * self.pixel_size_um[0], h * self.pixel_size_um[1]

    def local_to_world(self) -> np.ndarray:
        """3×3 affine local µm → world µm."""
        return self.placement.local_to_world()

    def pixel_to_world(self) -> np.ndarray:
        """3×3 affine level-0 pixel (x=col, y=row) → world µm."""
        return self.local_to_world() @ geo.scaling(*self.pixel_size_um)

    def world_corners(self) -> np.ndarray:
        """(4, 2) world corners of the image outline, clockwise from top-left."""
        return geo.apply(self.local_to_world(), geo.rect_corners(*self.size_um))

    def fully_outlined(self, group: str) -> bool:
        """True when every object of ``group`` on this image was outlined (marked in Properties): training and
        evaluation may then use the whole image, what is not outlined being surely not of the group."""
        return group in self.metadata.get("fully_outlined", [])

    # ------------------------------------------------------------ partial objects (shape incomplete)
    @property
    def partial_points(self) -> list[tuple[float, float]]:
        """Points (local µm) inside the objects marked partial by hand (``Project.set_partial_points``)."""
        return [tuple(p) for p in self.metadata.get("partial", [])]

    def edge_share(self, poly) -> float:
        """The share of the object's outline lying on the image border (within ``EDGE_PX``): 0 when it does not
        touch it."""
        from shapely.geometry import box
        w, h = self.size_um
        tol = EDGE_PX * max(self.pixel_size_um)
        x0, y0, x1, y1 = poly.bounds
        if not (x0 <= tol or y0 <= tol or x1 >= w - tol or y1 >= h - tol):
            return 0.0
        ring = poly.exterior
        return max(0.0, ring.length - ring.intersection(box(tol, tol, w - tol, h - tol)).length) / ring.length

    def edge_cut(self, poly, max_share: float = 0.0) -> bool:
        """True if the field of view cuts the object: it touches the image border and more than ``max_share`` of
        its outline lies on it (0: any touch). A gland grazing the border keeps nearly all its shape."""
        w, h = self.size_um
        tol = EDGE_PX * max(self.pixel_size_um)
        x0, y0, x1, y1 = poly.bounds
        touches = x0 <= tol or y0 <= tol or x1 >= w - tol or y1 >= h - tol
        return touches and (max_share <= 0 or self.edge_share(poly) > max_share)

    def marked_partial(self, poly) -> bool:
        """True if the object holds one of the points marked partial by hand."""
        from shapely.geometry import Point
        return any(poly.covers(Point(p)) for p in self.partial_points)

    def partial_reason(self, poly, max_share: float = EDGE_SHARE) -> str:
        """Why the object's shape is incomplete ("" if it is not): cut by the image edge (automatic: more than
        ``max_share`` of its outline on the border — the profile's ``edge_share``), or marked partial by hand (cut
        by the section plane, torn, folded, out of focus…). Its shape parameters are then left out of classifier
        training and of the shape statistics; segmentation uses it as any other."""
        if self.edge_cut(poly, max_share):
            return PARTIAL_EDGE
        return PARTIAL_MARKED if self.marked_partial(poly) else ""

    def contains_world(self, x: float, y: float) -> bool:
        """Return True if world point (x, y) falls inside the image."""
        lx, ly = geo.apply(np.linalg.inv(self.local_to_world()), (x, y))
        w, h = self.size_um
        return 0.0 <= lx < w and 0.0 <= ly < h


class MaskNode(Node):
    """Vector annotation of one class (group › subgroup) on one image.

    The geometry is a shapely ``MultiPolygon`` (holes allowed) in the parent
    image's local µm frame, so it follows the image when it moves and stays
    exact at any zoom. Each polygon of the MultiPolygon is one object.

    Args:
        name: Display name (defaults to "Group › Subgroup").
        geometry: shapely geometry or a list of (N, 2) outer rings in local µm.
        color: RGBA tuple (0–1) for the outline; the fill uses a lower alpha.
        group: Annotation group (e.g. "Gland").
        subgroup: Class inside the group (e.g. "Normal").
    """

    kind = "mask"

    def __init__(self, name: str = "", geometry=None, color=(0.2, 0.9, 0.4, 1.0),
                 group: str = "", subgroup: str = "") -> None:
        super().__init__(name or ann.label_text(group, subgroup) or "Mask")
        self.geometry = ann.as_multipolygon(geometry)
        self.color = tuple(color)
        self.group = group
        self.subgroup = subgroup
        self.opacity = 0.8

    @property
    def polygons(self) -> list[np.ndarray]:
        """Outer rings as (N, 2) arrays (read-only convenience)."""
        return [ext for ext, _holes in ann.rings(self.geometry)]

    @property
    def area_um2(self) -> float:
        """Total area in µm² (holes excluded)."""
        return float(self.geometry.area)

    @property
    def object_count(self) -> int:
        """Number of separate polygons (objects)."""
        return len(self.geometry.geoms)


## Suggestions (model-assisted annotation) ##

@dataclass
class Suggestion:
    """One object a model proposes: its outline and suggested class, waiting for the expert.

    Attributes:
        geometry: The outline (shapely Polygon, the image's local µm).
        group: The class group (the segmentation layer, e.g. "Gland").
        subgroup: The suggested grade ("" when no classifier gave one).
        score: The classifier's score or confidence (NaN without one).
        reason: Why that grade (the classifier's reason), shown to the expert.
        regrade: True when the outline is an object already drawn by hand that has no grade: only its grade is
            suggested, and accepting moves it to that grade unchanged.
        id: Stable id (for undo and the selection).
    """

    geometry: object
    group: str
    subgroup: str = ""
    score: float = float("nan")
    reason: str = ""
    regrade: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        import shapely
        return {"wkb": shapely.to_wkb(self.geometry).hex(), "group": self.group, "subgroup": self.subgroup,
                "score": None if self.score != self.score else float(self.score), "reason": self.reason,
                "regrade": self.regrade, "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "Suggestion":
        import shapely
        score = d.get("score")
        return cls(shapely.from_wkb(bytes.fromhex(d["wkb"])), str(d.get("group", "")), str(d.get("subgroup", "")),
                   float("nan") if score is None else float(score), str(d.get("reason", "")),
                   bool(d.get("regrade", False)), str(d.get("id") or uuid.uuid4().hex))


class SuggestionNode(Node):
    """The suggestions of models on one image, waiting for review (Layers › Suggestions).

    Not a mask: measuring, class relations, training and evaluation read only an image's masks (the expert's
    annotation), so unreviewed suggestions never count — unless Measure is asked to include them. Accepting a
    suggestion moves it into the annotation; deleting drops it.

    Attributes:
        items: The suggestions.
        current: Id of the suggestion being reviewed ("" = none).
        source: What made them ({"model", "classifier", "made"}).
        hidden_groups: Groups whose suggestions are not drawn (Layers: each group's eye).
    """

    kind = "suggestions"

    def __init__(self, items=None, source: dict | None = None) -> None:
        super().__init__("Suggestions")
        self.items: list[Suggestion] = list(items or [])
        self.current = ""
        self.source = dict(source or {})
        self.opacity = 0.9
        self.hidden_groups: list[str] = []          # groups whose suggestions are not drawn (Layers, their eye)

    def groups(self) -> list[tuple[str, int]]:
        """[(group, number of suggestions)], in the order they first appear."""
        out: dict[str, int] = {}
        for it in self.items:
            out[it.group] = out.get(it.group, 0) + 1
        return list(out.items())

    def item(self, item_id: str) -> Suggestion | None:
        return next((s for s in self.items if s.id == item_id), None)

    @property
    def current_item(self) -> Suggestion | None:
        return self.item(self.current)


## Project ##

class Project:
    """The loaded images and masks, plus change notification.

    Signals:
        structure_changed(): nodes added, removed, reordered or re-parented.
        node_changed(node, field): a property of one node changed
            (field ∈ "placement", "opacity", "visible", "name", "display",
            "pixel_size").
        selection_changed(node | None): the selected node changed.
        results_changed(): new results (``set_results``), or none.
    """

    def __init__(self) -> None:
        self.images: list[ImageNode] = []     # draw order: index 0 is at the bottom
        self.selected: Node | None = None
        self.selection: list[Node] = []           # every selected layer (Layers: Ctrl/Shift-click); ``selected`` among them
        self.objects: list[tuple] = []            # selected objects: (mask, polygon) — single glands, not whole layers
        self.labels: LabelSchema = default_schema()
        self.profile = None                   # the project's own copy of its profile (profiles.Profile)
        self.profile_id = ""                  # id of the library profile it was made from ("" if none)
        self.path: str | None = None          # project file, once saved
        self.results = None                   # the latest measurement (parameters.results.Results), or None
        self.structure_changed = Signal()
        self.node_changed = Signal()
        self.selection_changed = Signal()
        self.results_changed = Signal()

    # 1. Queries
    def iter_nodes(self):
        """Yield every node, images first then their masks."""
        for img in self.images:
            yield img
            yield from img.children

    def find(self, node_id: str) -> Node | None:
        """Return the node with this id, or None."""
        return next((n for n in self.iter_nodes() if n.id == node_id), None)

    def image_at(self, x: float, y: float) -> ImageNode | None:
        """Return the top-most visible image under world point (x, y)."""
        for img in reversed(self.images):
            if img.visible and img.contains_world(x, y):
                return img
        return None

    def world_bounds(self) -> tuple[float, float, float, float] | None:
        """(xmin, ymin, xmax, ymax) of all visible images, or None if empty."""
        corners = [img.world_corners() for img in self.images if img.visible]
        if not corners:
            return None
        return geo.bounds(np.vstack(corners))

    def reset(self, images: list, labels: LabelSchema | None = None, path: str | None = None,
              profile=None, profile_id: str = "", results=None) -> None:
        """Replace the whole content (new or opened project); observers rebuild once.

        Args:
            images: Images, bottom → top, with their masks.
            labels: Annotation classes; None takes the profile's (or the default).
            path: Project file, or None for an unsaved project.
            profile: The project's copy of its profile (``profiles.Profile``), or None.
            profile_id: Library id of the profile it came from.
            results: The results saved with it (``parameters.results.Results``), or None.
        """
        self.selected = None
        self.selection = []
        self.objects = []
        self.results = results
        self.images = list(images)
        self.profile, self.profile_id = profile, profile_id
        if labels is None and profile is not None:
            labels = copy.deepcopy(profile.classes)
        self.labels = labels or default_schema()
        for img in self.images:
            for m in img.masks:
                self.labels.ensure(m.group, m.subgroup, m.color)
        self.path = path
        self.selection_changed.emit(None)
        self.structure_changed.emit()
        self.results_changed.emit()

    # 2. Structure edits
    def add_image(self, image: ImageNode, index: int | None = None) -> ImageNode:
        """Add an image on top (or at ``index`` in draw order)."""
        self.images.insert(len(self.images) if index is None else index, image)
        for m in image.masks:                      # imported masks may bring new classes
            self.labels.ensure(m.group, m.subgroup, m.color)
        self.structure_changed.emit()
        return image

    def add_mask(self, parent: ImageNode, mask: MaskNode) -> MaskNode:
        """Attach a mask to an image."""
        mask.parent = parent
        parent.children.append(mask)
        self.labels.ensure(mask.group, mask.subgroup, mask.color)
        self.structure_changed.emit()
        return mask

    def remove(self, node: Node) -> None:
        """Remove an image (with its masks) or a mask."""
        if isinstance(node, ImageNode):
            self.images.remove(node)
        elif node.parent is not None:
            node.parent.children.remove(node)
            node.parent = None
        self.selection = [n for n in self.selection if n is not node and n.parent is not node]
        self.objects = [(m, p) for m, p in self.objects if m is not node and m.parent is not None]
        if self.selected is node or (self.selected is not None and self.selected.parent is node):
            self.select(None) if not self.selection else self.select_many(self.selection)
        self.structure_changed.emit()

    def move_image_in_stack(self, image: ImageNode, new_index: int) -> None:
        """Change an image's draw order (0 = bottom)."""
        self.images.remove(image)
        self.images.insert(max(0, min(new_index, len(self.images))), image)
        self.structure_changed.emit()

    def transfer_mask(self, mask: MaskNode, target: ImageNode, duplicate: bool = False) -> MaskNode:
        """Move or copy a mask to another image, keeping its world position.

        The polygons are re-expressed in the target's local frame:
        ``local_target = inv(L_target) · L_source · local_source``.

        Args:
            mask: Mask to transfer.
            target: Destination image.
            duplicate: If True the original stays and a copy is created.

        Returns:
            MaskNode: The mask now owned by ``target`` (the copy if duplicating).
        """
        source = mask.parent
        to_target = np.linalg.inv(target.local_to_world()) @ source.local_to_world()
        geometry = ann.transform(mask.geometry, to_target)
        if duplicate:
            new = copy.copy(mask)
            new.id = uuid.uuid4().hex
            new.name = f"{mask.name} (copy)"
            new.children = []
            new.geometry = geometry
            return self.add_mask(target, new)
        source.children.remove(mask)
        mask.geometry = geometry
        mask.parent = target
        target.children.append(mask)
        self.structure_changed.emit()
        return mask

    def mask_for(self, image: ImageNode, group: str, subgroup: str = "", create: bool = True) -> MaskNode | None:
        """Return the image's mask for a class, creating an empty one if needed.

        Args:
            image: Owning image.
            group, subgroup: The class.
            create: Create the mask when missing (otherwise return None).

        Returns:
            MaskNode | None: The mask.
        """
        for m in image.masks:
            if m.group == group and m.subgroup == subgroup:
                return m
        if not create:
            return None
        return self.add_mask(image, MaskNode(geometry=None, color=self.labels.color(group, subgroup),
                                             group=group, subgroup=subgroup))

    def suggestions_for(self, image: ImageNode, create: bool = True) -> SuggestionNode | None:
        """The image's suggestions layer (created, empty, when missing and ``create``)."""
        node = image.suggestions
        if node is None and create:
            node = SuggestionNode()
            node.parent = image
            image.children.append(node)
            self.structure_changed.emit()
        return node

    def set_suggestions(self, node: SuggestionNode, items: list, source: dict | None = None) -> None:
        """Replace a suggestions layer's items (an empty layer stays, so Layers keeps it until removed)."""
        node.items = list(items)
        if source is not None:
            node.source = dict(source)
        if node.current and node.item(node.current) is None:
            node.current = ""
        self.node_changed.emit(node, "suggestions")

    def set_current_suggestion(self, node: SuggestionNode, item_id: str) -> None:
        """The suggestion under review (selects the layer)."""
        node.current = item_id
        if self.selected is not node:
            self.selected = node
        self.selection, self.objects = [node], []
        self.selection_changed.emit(node)

    def set_mask_geometry(self, mask: MaskNode, geometry) -> None:
        """Replace a mask's geometry (local µm of its image).

        A ``MultiPolygon`` is trusted to be valid (edits produce valid output and
        importers validate through ``as_multipolygon``); re-validating 300 000
        vertices on every brush stroke would cost ~100 ms. Anything else is
        converted and repaired.
        """
        mask.geometry = geometry if isinstance(geometry, ann.MultiPolygon) else ann.as_multipolygon(geometry)
        self.node_changed.emit(mask, "geometry")
        if any(m is mask for m, _p in self.objects):
            # Selected objects of this mask that are gone (edited, moved to another class) leave the selection
            parts = list(mask.geometry.geoms)
            kept = [(m, p) for m, p in self.objects if m is not mask or any(p.equals(q) for q in parts)]
            if len(kept) != len(self.objects):
                self.select_objects(kept) if kept else self.select(None)

    def relabel_mask(self, mask: MaskNode, group: str, subgroup: str, color=None) -> None:
        """Give a whole mask (all its objects) another class; its name follows when it was the class's."""
        if mask.name == (ann.label_text(mask.group, mask.subgroup) or "Mask"):
            mask.name = ann.label_text(group, subgroup) or "Mask"
        mask.group, mask.subgroup = group, subgroup
        mask.color = tuple(color) if color is not None else self.labels.color(group, subgroup)
        self.labels.ensure(group, subgroup, mask.color)
        self.node_changed.emit(mask, "label")
        self.structure_changed.emit()

    # 3. Property edits (each emits node_changed)
    def set_placement(self, image: ImageNode, placement: Placement) -> None:
        """Place an image (its masks follow automatically)."""
        image.placement = placement
        self.node_changed.emit(image, "placement")

    def set_pixel_size(self, image: ImageNode, pixel_size_um: tuple[float, float]) -> None:
        """Change the calibrated pixel size; masks are rescaled with the image."""
        old = image.pixel_size_um
        k = (pixel_size_um[0] / old[0], pixel_size_um[1] / old[1])
        for m in image.masks:
            m.geometry = ann.transform(m.geometry, geo.scaling(*k))
        image.pixel_size_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
        self.node_changed.emit(image, "pixel_size")

    def set_source(self, image: ImageNode, source, metadata: dict | None = None) -> None:
        """Point an image at other pixels of the same image (a relinked file); placement and masks stay.

        Args:
            image: The image.
            source: Its new PixelSource.
            metadata: Replaces the image's metadata (e.g. the new path and fingerprint), if given.
        """
        image.source = source
        if metadata is not None:
            image.metadata = dict(metadata)
        self.node_changed.emit(image, "source")

    def set_opacity(self, node: Node, opacity: float) -> None:
        """Set a node's opacity (clamped to 0–1)."""
        node.opacity = float(min(1.0, max(0.0, opacity)))
        self.node_changed.emit(node, "opacity")

    def set_visible(self, node: Node, visible: bool) -> None:
        """Show or hide a node."""
        node.visible = bool(visible)
        self.node_changed.emit(node, "visible")

    def set_display(self, image: ImageNode, **changes) -> None:
        """Change display settings (``clim``, ``colormap``) of what is drawn (the shown view, or the image)."""
        if image.shown is not None:
            image.shown.display = replace(image.shown.display, **changes)
        else:
            image.display = replace(image.display, **changes)
        self.node_changed.emit(image, "display")

    # 4. Views (processed versions of an image)
    def add_view(self, image: ImageNode, view: ImageView, show: bool = False, index: int | None = None) -> None:
        """Add a processed version of an image (and show it, if asked)."""
        image.views.insert(len(image.views) if index is None else index, view)
        self.node_changed.emit(image, "views")
        if show:
            self.show_view(image, view)

    def remove_view(self, image: ImageNode, view: ImageView) -> None:
        """Remove a view (the original is shown again if it was the shown one)."""
        if view in image.views:
            if image.shown is view:
                self.show_view(image, None)
            image.views.remove(view)
            self.node_changed.emit(image, "views")

    def show_view(self, image: ImageNode, view: ImageView | None) -> None:
        """Draw one of an image's views (None: the original)."""
        if view is image.shown:
            return
        image.shown = view
        self.node_changed.emit(image, "source")         # other pixels: the canvas rebuilds the image

    def set_measured(self, image: ImageNode, view: ImageView | None, on: bool) -> None:
        """Include (or leave out) one version of an image in the parameters: a view, or None: the original."""
        if view is None:
            image.metadata["measure_original"] = bool(on)
        else:
            view.chain = dict(view.chain, measure=bool(on))
        self.node_changed.emit(image, "views")

    def set_suggestions_shown(self, node: SuggestionNode, group: str, shown: bool) -> None:
        """Show or hide one group's suggestions on an image (the others stay as they are)."""
        hidden = set(node.hidden_groups)
        (hidden.discard if shown else hidden.add)(group)
        node.hidden_groups = sorted(hidden)
        self.node_changed.emit(node, "suggestions")

    def set_fully_outlined(self, image: "ImageNode", group: str, on: bool) -> None:
        """Mark (or unmark) every object of ``group`` as outlined on ``image`` (see ``ImageNode.fully_outlined``)."""
        groups = [g for g in image.metadata.get("fully_outlined", []) if g != group] + ([group] if on else [])
        if groups:
            image.metadata["fully_outlined"] = groups
        else:
            image.metadata.pop("fully_outlined", None)
        self.node_changed.emit(image, "fully_outlined")

    def set_partial_points(self, image: "ImageNode", points: list) -> None:
        """Set the points (local µm) of the objects marked partial on ``image`` (see ``ImageNode.partial_reason``)."""
        pts = [[float(x), float(y)] for x, y in points]
        if pts:
            image.metadata["partial"] = pts
        else:
            image.metadata.pop("partial", None)
        self.node_changed.emit(image, "partial")

    def rename(self, node: Node, name: str) -> None:
        """Rename a node."""
        node.name = name
        self.node_changed.emit(node, "name")

    def set_results(self, results) -> None:
        """Replace the latest measurement (``parameters.results.Results``, or None)."""
        self.results = results
        self.results_changed.emit()

    def select(self, node: Node | None) -> None:
        """Select one layer, or none (no-op if unchanged)."""
        nodes = [node] if node is not None else []
        if node is not self.selected or self.selection != nodes or self.objects:
            self.selected, self.selection, self.objects = node, nodes, []
            self.selection_changed.emit(node)

    def select_objects(self, objects: list) -> None:
        """Select single objects: [(mask, polygon)] (their masks become the selected layers; the last one is
        ``selected``). An empty list clears the selection."""
        objects = [(m, p) for i, (m, p) in enumerate(objects)
                   if not any(m is m2 and p.equals(p2) for m2, p2 in objects[:i])]
        masks = list(dict.fromkeys(m for m, _p in objects))
        self.objects, self.selection = objects, masks
        self.selected = objects[-1][0] if objects else None
        self.selection_changed.emit(self.selected)

    def select_many(self, nodes: list, primary: Node | None = None) -> None:
        """Select several layers; ``primary`` (default: the last) is the one Properties shows (``selected``)."""
        nodes = list(dict.fromkeys(n for n in nodes if n is not None))
        primary = primary if primary in nodes else (nodes[-1] if nodes else None)
        if nodes == self.selection and primary is self.selected and not self.objects:
            return
        self.selection, self.selected, self.objects = nodes, primary, []
        self.selection_changed.emit(primary)

    def selected_images(self) -> list["ImageNode"]:
        """The images of the selection (a selected mask or suggestions layer counts as its image), in list order."""
        out = []
        for n in self.selection or ([self.selected] if self.selected is not None else []):
            img = n if isinstance(n, ImageNode) else n.parent if isinstance(n.parent, ImageNode) else None
            if img is not None and img not in out and img in self.images:
                out.append(img)
        return out

    def reorder_images(self, images: list["ImageNode"]) -> None:
        """Set the drawing order (bottom → top) to ``images`` (the same images, another order)."""
        if sorted(map(id, images)) != sorted(map(id, self.images)):
            raise ValueError("reorder_images needs exactly the project's images")
        self.images[:] = list(images)
        self.structure_changed.emit()
