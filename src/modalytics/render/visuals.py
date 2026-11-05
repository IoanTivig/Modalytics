"""
GPU visuals for domain nodes (pygfx scene objects).

One ``ImageVisual`` per ``ImageNode``:

    Group  (matrix = Placement: local µm → world µm; render_order = stack index)
     ├─ tile Images of several levels (finer levels drawn first, closer in z)
     └─ mask fill meshes + outlines in local µm (drawn after the tiles)

Why moving an image is cheap: the tiles live in the image's local frame, so a
move only changes the group's matrix. No tile is re-read or re-uploaded
(in HoloPath/napari, changing ``translate`` re-sliced and re-read tiles).

Draw order and the coarse fallback: every image keeps its coarsest level
loaded as a full-cover fallback. Finer tiles are drawn first with depth
writes; the coarser tiles drawn after fail the depth test where finer data
exists. So no "double exposure" appears when the image is semi-transparent,
and there is never a black hole while detail loads.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pygfx as gfx

from modalytics.core.geometry import to_4x4
from modalytics.core.model import ImageNode, MaskNode, SuggestionNode
from modalytics.render import tiling

# All canvas objects use classic blending in one render queue, so the
# per-object render_order alone decides the draw order.
_ALPHA = "blend"

def _opaque_colormap(name: str) -> gfx.TextureMap:
    """Return a pygfx colormap with alpha forced to 1.

    pygfx's built-in ``gray`` map ramps alpha from 0 (black) to 1 (white), so a
    phase map drawn with it is mostly transparent even at 100 % opacity.
    Transparency must come only from the layer's opacity, so every map is
    rebuilt fully opaque.
    """
    rgba = np.array(getattr(gfx.cm, name).texture.data, dtype=np.float32)
    rgba[:, 3] = 1.0
    # clamp (not repeat): the top of the range must stay the last colour, not wrap to the first
    return gfx.TextureMap(gfx.Texture(rgba, dim=1), filter="linear", wrap="clamp")


_COLORMAPS = {name: _opaque_colormap(name) for name in
              ("gray", "viridis", "magma", "inferno", "plasma", "cividis", "hot", "bone")}
COLORMAP_NAMES = list(_COLORMAPS)
COLORMAPS = _COLORMAPS          # public: also used by the minimap thumbnails


def prepare_tile(data: np.ndarray) -> np.ndarray:
    """Make a tile GPU-ready (runs in a loader thread).

    RGB is padded to RGBA here because WebGPU has no 3-channel texture format;
    doing it in the worker keeps ~6 ms per 1024² tile off the UI thread.

    Args:
        data: (h, w) or (h, w, 3|4) array.

    Returns:
        np.ndarray: C-contiguous uint8 RGBA, or float32 single-channel.
    """
    if data.ndim == 3 and data.shape[2] == 3:
        out = np.empty(data.shape[:2] + (4,), dtype=np.uint8)
        out[..., :3] = data
        out[..., 3] = 255
        return out
    if data.ndim == 2 and data.dtype != np.uint8:
        return np.ascontiguousarray(data, dtype=np.float32)
    return np.ascontiguousarray(data)


class ImageVisual:
    """Draws one image: tiles of the right pyramid level plus its masks.

    Args:
        node: The image node.
    """

    def __init__(self, node: ImageNode) -> None:
        self.node = node
        self.group = gfx.Group()
        self.tiles: dict[tuple[int, int, int], gfx.Image] = {}
        self.tile_bytes: dict[tuple[int, int, int], int] = {}
        self.masks: dict[str, MaskVisual] = {}
        self.suggestions: SuggestionVisual | None = None
        self.wanted_level = len(node.view_source.levels) - 1
        self.stack_index = 0
        self._materials: dict[int, gfx.ImageBasicMaterial] = {}
        self.update_placement()
        self.sync_masks()

    # 1. Level bookkeeping
    @property
    def coarsest(self) -> int:
        return len(self.node.view_source.levels) - 1

    def level_um_per_px(self) -> list[float]:
        """µm per pixel of each level (x axis, including the placement scale)."""
        ps = self.node.pixel_size_um[0] * self.node.placement.scale
        return [ps * lv.downsample[1] for lv in self.node.view_source.levels]

    def material(self, level: int) -> gfx.ImageBasicMaterial:
        """Shared material per level (finer levels write depth, see module docs)."""
        mat = self._materials.get(level)
        if mat is None:
            src = self.node.view_source
            if src.is_rgb:
                mat = gfx.ImageBasicMaterial(clim=(0, 255), interpolation="linear")
            else:
                d = self.node.view_display
                mat = gfx.ImageBasicMaterial(clim=d.clim, map=_COLORMAPS.get(d.colormap, _COLORMAPS["gray"]),
                                             interpolation="linear")
            mat.alpha_mode = _ALPHA
            mat.depth_write = True
            mat.depth_test = True
            mat.opacity = self.node.opacity
            self._materials[level] = mat
        return mat

    # 2. Node property sync
    def update_placement(self) -> None:
        """Copy the node's placement to the group matrix (the only thing a move changes)."""
        # z = stack index: higher images are closer to the camera
        self.group.local.matrix = to_4x4(self.node.local_to_world(), z=float(self.stack_index))

    def update_style(self) -> None:
        """Copy opacity, visibility and display settings to the GPU materials."""
        self.group.visible = self.node.visible
        for mat in self._materials.values():
            mat.opacity = self.node.opacity
            if not self.node.view_source.is_rgb:
                mat.clim = self.node.view_display.clim
                mat.map = _COLORMAPS.get(self.node.view_display.colormap, _COLORMAPS["gray"])

    def update_tile_transforms(self) -> None:
        """Re-place every tile (after a pixel-size change)."""
        for key, obj in self.tiles.items():
            self._place_tile(obj, *key)

    def set_stack_index(self, index: int) -> None:
        """Set the image's position in the draw order (0 = bottom)."""
        self.stack_index = index
        self.group.render_order = index
        self.update_placement()

    # 3. Tiles
    def add_tile(self, key: tuple[int, int, int], data: np.ndarray) -> int:
        """Upload a decoded tile; returns its GPU size in bytes."""
        level = key[0]
        colorspace = "srgb" if data.dtype == np.uint8 else "physical"
        tex = gfx.Texture(data, dim=2, colorspace=colorspace)
        obj = gfx.Image(gfx.Geometry(grid=tex), self.material(level))
        obj.render_order = level            # finer levels first (see module docs)
        self._place_tile(obj, *key)
        self.group.add(obj)
        self.tiles[key] = obj
        self.tile_bytes[key] = data.nbytes
        return data.nbytes

    def remove_tile(self, key: tuple[int, int, int]) -> None:
        """Drop a tile from the scene (GPU memory is freed with the object)."""
        obj = self.tiles.pop(key, None)
        self.tile_bytes.pop(key, None)
        if obj is not None:
            self.group.remove(obj)

    def _place_tile(self, obj: gfx.Image, level: int, ty: int, tx: int) -> None:
        lv = self.node.view_source.levels[level]
        x0, y0, _, _ = tiling.tile_rect_px(lv, ty, tx)
        sx, sy = self.node.pixel_size_um
        px, py = sx * lv.downsample[1], sy * lv.downsample[0]   # µm per pixel of this level
        # pygfx puts texel centres on integer coordinates → shift by half a texel
        obj.local.scale = (px, py, 1.0)
        obj.local.position = (x0 * sx + 0.5 * px, y0 * sy + 0.5 * py, 0.5 - 0.01 * level)

    # 4. Masks
    def sync_masks(self) -> None:
        """Create/remove mask (and suggestion) visuals to match the node's children."""
        current = {m.id: m for m in self.node.masks}
        for mid in list(self.masks):
            if mid not in current:
                self.masks.pop(mid).clear()
        for mid, mask in current.items():
            if mid not in self.masks:
                self.masks[mid] = MaskVisual(mask, self.group)
        node = self.node.suggestions
        if self.suggestions is not None and self.suggestions.node is not node:
            self.suggestions.clear()
            self.suggestions = None
        if node is not None and self.suggestions is None:
            self.suggestions = SuggestionVisual(node, self.group)


class MaskVisual:
    """Fill + outline of a vector mask, in the parent's local µm frame.

    The two objects are added straight into the parent image's group (not a
    sub-group): pygfx sorts by the *nearest* group's render_order, so a
    sub-group would break the image stacking order.

    Args:
        node: The mask node.
        parent_group: The owning image's group.
    """

    def __init__(self, node: MaskNode, parent_group: gfx.Group) -> None:
        self.node = node
        self.group = parent_group
        self.fill: gfx.Mesh | None = None
        self.outline: gfx.Line | None = None
        self.rebuild()

    def clear(self) -> None:
        """Remove the mask's objects from the scene."""
        for obj in (self.fill, self.outline):
            if obj is not None:
                self.group.remove(obj)
        self.fill = self.outline = None

    def rebuild(self) -> None:
        """(Re)build the GPU geometry from the mask's MultiPolygon (holes included).

        Triangulation is cached per polygon: after an edit only new or changed
        polygons are triangulated again, the rest are reused, so a brush stroke
        on an 8 000-object mask rebuilds in tens of ms. The cache key is a
        vectorised fingerprint (bounds, vertex count, area, centroid) — about
        10× cheaper than serialising every polygon.
        """
        import mapbox_earcut as earcut
        import shapely

        self.clear()
        parts = shapely.get_parts(self.node.geometry)
        if len(parts) == 0:
            self._tri_cache = {}
            return
        cache, fresh = getattr(self, "_tri_cache", {}), {}
        verts, tris, outline, offset = [], [], [], 0
        # 1. Per polygon: cached (vertices, triangles, outline rows) or earcut now
        cx, cy = shapely.get_coordinates(shapely.centroid(parts)).T
        finger = np.column_stack([shapely.bounds(parts), shapely.get_num_coordinates(parts),
                                  shapely.area(parts), cx, cy])
        for key, poly in zip(map(bytes, finger), parts):
            item = cache.get(key)
            if item is None:
                ring_list = [np.asarray(poly.exterior.coords)[:-1]] + \
                            [np.asarray(h.coords)[:-1] for h in poly.interiors]
                v = np.vstack(ring_list)
                ends = np.cumsum([len(r) for r in ring_list]).astype(np.uint32)
                t = earcut.triangulate_float64(v.astype(np.float64), ends).reshape(-1, 3).astype(np.int32)
                rows = np.vstack([x for r in ring_list for x in (r, r[:1], np.full((1, 2), np.nan))])
                item = (v, t, rows)
            fresh[key] = item
            v, t, rows = item
            verts.append(v)
            tris.append(t + offset)
            outline.append(rows)
            offset += len(v)
        self._tri_cache = fresh                      # drop polygons that no longer exist
        # 2. One mesh + one line for the whole mask
        pos = np.zeros((offset, 3), np.float32)
        pos[:, :2] = np.concatenate(verts)
        line_rows = np.concatenate(outline)
        line = np.zeros((len(line_rows), 3), np.float32)
        line[:, :2] = line_rows
        r, g, b, _ = self.node.color
        fill_mat = gfx.MeshBasicMaterial(color=(r, g, b, 0.30))
        self.fill = gfx.Mesh(gfx.Geometry(positions=pos, indices=np.concatenate(tris)), fill_mat)
        out_mat = gfx.LineMaterial(thickness=1.5, color=(r, g, b, 1.0))
        self.outline = gfx.Line(gfx.Geometry(positions=line), out_mat)

        for i, (obj, mat) in enumerate(((self.fill, fill_mat), (self.outline, out_mat))):
            mat.alpha_mode = _ALPHA
            mat.depth_test = False
            obj.render_order = 100 + i          # after all tile levels of the parent
            obj.local.z = 0.8
            self.group.add(obj)
        self.update_style()

    def update_style(self) -> None:
        """Copy visibility and opacity to the materials."""
        for obj in (self.fill, self.outline):
            if obj is not None:
                obj.visible = self.node.visible
                obj.material.opacity = self.node.opacity


class SuggestionVisual:
    """The models' suggestions on an image: dashed outlines and a light fill, each in its suggested class's colour
    (one mesh and one line for all, coloured per vertex), drawn above the masks.

    Args:
        node: The SuggestionNode.
        parent_group: The owning image's group.
    """

    DASH = (6.0, 4.0)

    def __init__(self, node: SuggestionNode, parent_group: gfx.Group) -> None:
        self.node = node
        self.group = parent_group
        self.colors: dict = {}                       # (group, subgroup) → rgba, set by the canvas
        self.fill: gfx.Mesh | None = None
        self.outline: gfx.Line | None = None
        self.rebuild()

    def clear(self) -> None:
        for obj in (self.fill, self.outline):
            if obj is not None:
                self.group.remove(obj)
        self.fill = self.outline = None

    def rebuild(self, colors: dict | None = None) -> None:
        """(Re)build from the suggestions (``colors``: {(group, subgroup): rgba}; grey for unknown classes)."""
        import mapbox_earcut as earcut
        if colors is not None:
            self.colors = colors
        self.clear()
        verts, tris, vcol, line, lcol, offset = [], [], [], [], [], 0
        hidden = set(self.node.hidden_groups)
        for it in self.node.items:
            poly = it.geometry
            if poly.is_empty or poly.geom_type != "Polygon" or it.group in hidden:   # a group hidden in Layers
                continue
            rgba = self.colors.get((it.group, it.subgroup)) or self.colors.get((it.group, "")) or (0.8, 0.8, 0.8, 1)
            ring_list = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in poly.interiors]
            v = np.vstack(ring_list)
            ends = np.cumsum([len(r) for r in ring_list]).astype(np.uint32)
            t = earcut.triangulate_float64(v.astype(np.float64), ends).reshape(-1, 3).astype(np.int32)
            verts.append(v)
            tris.append(t + offset)
            vcol.append(np.tile([rgba[0], rgba[1], rgba[2], 0.18], (len(v), 1)))
            offset += len(v)
            for r in ring_list:
                rows = np.vstack([r, r[:1], np.full((1, 2), np.nan)])
                line.append(rows)
                lcol.append(np.tile([rgba[0], rgba[1], rgba[2], 1.0], (len(rows), 1)))
        if not verts:
            return
        pos = np.zeros((offset, 3), np.float32)
        pos[:, :2] = np.concatenate(verts)
        rows = np.concatenate(line)
        lpos = np.zeros((len(rows), 3), np.float32)
        lpos[:, :2] = rows
        fill_mat = gfx.MeshBasicMaterial(color_mode="vertex")
        self.fill = gfx.Mesh(gfx.Geometry(positions=pos, indices=np.concatenate(tris),
                                          colors=np.concatenate(vcol).astype(np.float32)), fill_mat)
        out_mat = gfx.LineMaterial(thickness=2.0, color_mode="vertex", dash_pattern=self.DASH)
        self.outline = gfx.Line(gfx.Geometry(positions=lpos, colors=np.concatenate(lcol).astype(np.float32)),
                                out_mat)
        for i, (obj, mat) in enumerate(((self.fill, fill_mat), (self.outline, out_mat))):
            mat.alpha_mode = _ALPHA
            mat.depth_test = False
            obj.render_order = 110 + i               # above the masks
            obj.local.z = 0.9
            self.group.add(obj)
        self.update_style()

    def update_style(self) -> None:
        for obj in (self.fill, self.outline):
            if obj is not None:
                obj.visible = self.node.visible and self.node.parent is not None and self.node.parent.visible
                obj.material.opacity = self.node.opacity


class TileBudget:
    """Least-recently-used GPU tile budget shared by all images.

    Args:
        max_bytes: GPU memory allowed for tiles.
    """

    def __init__(self, max_bytes: int = 1536 * 1024 ** 2) -> None:
        self.max_bytes = max_bytes
        self.used = 0
        self._lru: OrderedDict[tuple, int] = OrderedDict()

    def touch(self, key: tuple) -> None:
        """Mark a tile as used this frame."""
        if key in self._lru:
            self._lru.move_to_end(key)

    def add(self, key: tuple, nbytes: int) -> None:
        self._lru[key] = nbytes
        self.used += nbytes

    def forget(self, key: tuple) -> None:
        self.used -= self._lru.pop(key, 0)

    def over_budget(self, protected: set) -> list[tuple]:
        """Return least-recently-used keys to evict (never ``protected`` ones)."""
        out, used = [], self.used
        for key, nbytes in self._lru.items():
            if used <= self.max_bytes:
                break
            if key not in protected:
                out.append(key)
                used -= nbytes
        return out
