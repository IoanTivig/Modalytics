"""
Importers: turn files into ``ImageNode`` / ``MaskNode`` objects.

Stage-1 prototype set (each will become a plugin with a manifest later):

* ``import_zarr_wsi``   — multiscale zarr cache of a whole-slide image.
* ``import_tiff_wsi``   — a slide file read in place (SVS, pyramidal TIFF, NDPI, SCN).
* ``import_phase_txt``  — legacy DHM ``Phase.txt``.
* ``import_hpro``       — a HoloPath ``.hpro`` project (images + vector masks),
  so existing work opens in Modalytics.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np

from modalytics.core.annotations import canonical
from modalytics.core.geometry import Placement
from modalytics.core.model import ImageNode, MaskNode
from modalytics.io.sources import ArraySource
from modalytics.io.tiff_source import TiffPyramidSource, fingerprint
from modalytics.io.zarr_source import ZarrPyramidSource


def import_zarr_wsi(path: str, pixel_size_um: float | None = None, name: str | None = None) -> ImageNode:
    """Open a zarr pyramid as a whole-slide image.

    Args:
        path: Zarr group directory.
        pixel_size_um: Override when the metadata has no calibration.
        name: Display name (defaults to the folder name).

    Returns:
        ImageNode: The WSI node (modality "wsi").

    Raises:
        ValueError: If no pixel size is known.
    """
    source = ZarrPyramidSource(path)
    ps = pixel_size_um or source.pixel_size_um
    if ps is None:
        raise ValueError(f"{path}: no pixel size in metadata; pass pixel_size_um")
    name = name or os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
    node = ImageNode(name, source, (ps, ps), "wsi", metadata={"path": source.path})
    print(f"✅ WSI {name}: {source.shape[1]}×{source.shape[0]} px, {ps} µm/px, {len(source.levels)} levels")
    return node


def import_tiff_wsi(path: str, pixel_size_um: float | None = None, name: str | None = None) -> ImageNode:
    """Open a slide file in place (SVS, pyramidal TIFF / OME-TIFF, NDPI, SCN) as a whole-slide image.

    Args:
        path: The file.
        pixel_size_um: Use this instead of the file's own value (or when it has none).
        name: Display name (defaults to the file name).

    Returns:
        ImageNode: The WSI node (modality "wsi"); its metadata holds what the file says about itself
        (scanner, magnification…) and the file's fingerprint (``project_file`` checks it on reopening).

    Raises:
        ValueError: If the file is not a readable image, or no pixel size is known.
    """
    source = TiffPyramidSource(path)
    ps = pixel_size_um or source.pixel_size_um
    if ps is None:
        raise ValueError(f"{path}: the file does not give its pixel size; pass pixel_size_um")
    name = name or os.path.splitext(os.path.basename(path))[0]
    meta = dict(source.metadata, path=source.path, file_fingerprint=fingerprint(source.path))
    if source.pixel_size_um:
        meta["pixel_size_in_file"] = source.pixel_size_um
    node = ImageNode(name, source, (ps, ps), "wsi", metadata=meta)
    print(f"✅ WSI {name}: {source.shape[1]}×{source.shape[0]} px, {ps} µm/px, {len(source.levels)} levels "
          f"({meta.get('format')}, read in place)")
    return node


def import_phase_txt(path: str, name: str | None = None, placement: Placement | None = None) -> ImageNode:
    """Open a legacy DHM ``Phase.txt`` as a phase image.

    Args:
        path: File path.
        name: Display name (defaults to "<area>/<image> Phase").
        placement: Initial placement (defaults to the world origin).

    Returns:
        ImageNode: The phase node (modality "phase", values in radians).

    Goes through the Lyncée importer (``resources/importers/phase/lyncee_phase_txt.py``)
    and the phase contract, like File › Import images.
    """
    from modalytics.image_import import import_file
    from modalytics.image_import.registry import ImporterLibrary
    library = ImporterLibrary()
    node = import_file(path, library, library.get("phase:lyncee_phase_txt"))[0]
    if name is None:
        parent = os.path.basename(os.path.dirname(path))
        area = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = f"{area}/{parent} Phase"
    node.name = name
    if placement is not None:
        node.placement = placement
    return node


def import_hpro(path: str) -> list[ImageNode]:
    """Read a HoloPath ``.hpro`` project: images and their masks.

    HoloPath stored geometry on napari layers (``scale``, ``translate``,
    ``affine``); here it becomes a ``Placement`` plus a calibrated pixel size.
    Masks become ``MaskNode`` children of the image they belong to (by owner id,
    else by name prefix), in two forms:

    * shapes layers (vector outlines): napari vertices are (row, col) in
      pixel-centre coordinates, so local µm = ((col + 0.5)·sx, (row + 0.5)·sy);
    * labels layers ("Manual Mask" painted with HoloPath's brush): a pixel map,
      any non-zero value = the mask's class (its metadata ``group`` /
      ``subgroup``); each 8-connected patch is one object — as HoloPath counted
      them — traced to a vector outline along the pixel edges (areas exact).

    Args:
        path: ``.hpro`` file.

    Returns:
        list[ImageNode]: Images (bottom → top) with their masks attached.
    """
    images: list[ImageNode] = []
    by_id: dict[str, ImageNode] = {}
    with zipfile.ZipFile(path) as z:
        project = json.loads(z.read("project.json"))
        layers = project.get("layers", [])

        # 1. Images
        for layer in layers:
            if layer.get("type") != "image":
                continue
            meta = layer.get("metadata", {})
            tr = layer.get("transform", {})
            ps = float((tr.get("scale") or [1.0])[-1])
            ty, tx = (tr.get("translate") or [0.0, 0.0])[-2:]
            placement = Placement(x_um=float(tx), y_um=float(ty))
            node = None
            if layer.get("is_pyramid") or meta.get("zarr_path"):
                zpath = _resolve_zarr(layer.get("zarr_ref") or meta.get("zarr_path"), path)
                if zpath is None:
                    print(f"⚠️ {layer['name']}: zarr cache not found, skipped")
                    continue
                node = import_zarr_wsi(zpath, pixel_size_um=ps, name=layer["name"])
                node.placement = placement
            elif layer.get("data_file", "").endswith(".npy"):
                with z.open("layers/" + layer["data_file"]) as f:
                    data = np.load(f)
                modality = "phase" if "phase" in meta.get("image_type", "").lower() else "brightfield"
                node = ImageNode(layer["name"], ArraySource(data), (ps, ps), modality, placement,
                                 metadata={"unit": meta.get("value_unit", ""),
                                           "phase_to_height_nm": meta.get("phase_to_height_factor_nm")})
            if node is None:
                continue
            node.visible = bool(layer.get("visible", True))
            node.opacity = float(layer.get("opacity", 1.0))
            images.append(node)
            by_id[layer.get("id", "")] = node
            by_id[meta.get("id", "")] = node

        # 2. Vector masks (napari shapes layers)
        for layer in layers:
            if layer.get("type") != "shapes" or not layer.get("data_file"):
                continue
            meta = layer.get("metadata", {})
            owner = by_id.get(meta.get("owner", "")) or next(
                (img for img in images if layer["name"].startswith(img.name)), None)
            if owner is None:
                print(f"⚠️ {layer['name']}: no owner image, skipped")
                continue
            shapes = json.loads(z.read("layers/" + layer["data_file"]))
            sx, sy = owner.pixel_size_um
            polygons = [np.column_stack([(np.asarray(s)[:, 1] + 0.5) * sx, (np.asarray(s)[:, 0] + 0.5) * sy])
                        for s in shapes.get("shapes", []) if len(s) >= 3]
            edge = (shapes.get("edge_color") or [[0.2, 0.9, 0.4, 1.0]])[0]
            short = layer["name"][len(owner.name):].strip(" -") or layer["name"]
            # HoloPath names masks "<group> - <subgroup>" (e.g. "Glande - Normal")
            group, _, subgroup = short.partition(" - ")
            group = canonical(meta.get("group") or group)
            subgroup = canonical(meta.get("subgroup") or subgroup)
            mask = MaskNode("", polygons, color=tuple(edge), group=group, subgroup=subgroup)
            mask.parent = owner
            owner.children.append(mask)

        # 3. Pixel masks (napari labels layers: masks painted with HoloPath's brush)
        from modalytics.core.annotations import default_schema
        from modalytics.services.segmentation import vectorize
        colours = default_schema()
        for layer in layers:
            if layer.get("type") != "labels" or not layer.get("data_file", "").endswith(".npy"):
                continue
            meta = layer.get("metadata", {})
            owner = by_id.get(meta.get("owner", "")) or next(
                (img for img in images if layer["name"].startswith(img.name)), None)
            if owner is None:
                print(f"⚠️ {layer['name']}: no owner image, skipped")
                continue
            with z.open("layers/" + layer["data_file"]) as f:
                labels = np.load(f)
            if labels.ndim != 2 or not labels.any():
                continue
            # Its own grid (a mask may be stored downsampled): pixel size and origin, relative to the image
            tr = layer.get("transform", {})
            sy, sx = [float(v) for v in (tr.get("scale") or owner.pixel_size_um[::-1])[-2:]]
            ty, tx = [float(v) for v in (tr.get("translate") or [owner.placement.y_um, owner.placement.x_um])[-2:]]
            geometry = vectorize(labels > 0, (tx - owner.placement.x_um, ty - owner.placement.y_um, sx, sy))
            group = canonical(meta.get("group") or "")
            subgroup = canonical(meta.get("subgroup") or "")
            mask = MaskNode("", geometry, color=colours.color(group, subgroup), group=group, subgroup=subgroup)
            mask.parent = owner
            owner.children.append(mask)
    print(f"✅ {os.path.basename(path)}: {len(images)} images, "
          f"{sum(len(i.children) for i in images)} masks")
    return images


def _resolve_zarr(ref: str | None, project_path: str) -> str | None:
    """Find a WSI zarr cache referenced by an old project.

    HoloPath moved its data folder (``resources/input_database`` →
    ``database/input_database``), so absolute paths in old projects may be
    stale. Tries the path as written, the moved location, and next to the project.
    """
    if not ref:
        return None
    candidates = [ref, ref.replace("resources/input_database", "database/input_database"),
                  os.path.join(os.path.dirname(project_path), os.path.basename(ref))]
    return next((c for c in candidates if os.path.isdir(c)), None)
