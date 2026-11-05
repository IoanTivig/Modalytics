"""
Annotation exchange formats (pure parsing + geometry, no Qt).

* **GeoJSON**, QuPath-compatible: coordinates in the image's level-0 *pixel*
  units, one Feature per object, ``properties.classification`` =
  ``{"name": "Group: Subgroup", "color": [r, g, b]}``.
* **Leica / Aperio ImageScope XML** (ported from HoloPath's
  ``conversion/leica_import.py``): ``<Annotation Name=…>`` groups of
  ``<Region>`` polygons with vertices in level-0 slide pixels;
  ``NegativeROA="1"`` regions are exclusions (holes).

Importers return ``{(group, subgroup): (MultiPolygon in local µm, rgba)}``;
the caller merges them into the image's masks (one undo step).
"""

from __future__ import annotations

import json
import math
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import mapping, shape

from modalytics.core import annotations as ann

# Leica/ImageScope annotation names used in our data → (group, subgroup)
LEICA_CLASSES = {
    "nondisplasicgland": ("Gland", "Normal"),
    "nondysplasticgland": ("Gland", "Normal"),
    "lgg": ("Gland", "Dysplasia low grade"),
    "hgg": ("Gland", "Dysplasia high grade"),
}


def _pixels_to_local(geom, image):
    """Level-0 pixel coordinates → the image's local µm."""
    sx, sy = image.pixel_size_um
    return affinity.scale(geom, xfact=sx, yfact=sy, origin=(0, 0))


def _local_to_pixels(geom, image):
    sx, sy = image.pixel_size_um
    return affinity.scale(geom, xfact=1 / sx, yfact=1 / sy, origin=(0, 0))


def _merge(out: dict, label, geom, color) -> None:
    if label in out:
        out[label] = (ann.as_multipolygon(out[label][0].union(geom)), out[label][1])
    else:
        out[label] = (ann.as_multipolygon(geom), color)


# ------------------------------------------------------------------ GeoJSON
def export_geojson(image, path: str) -> int:
    """Write all masks of an image as a QuPath-compatible GeoJSON FeatureCollection.

    Args:
        image: ImageNode whose masks are exported.
        path: Output ``.geojson`` file.

    Returns:
        int: Number of features (objects) written.
    """
    features = []
    for m in image.masks:
        name = f"{m.group}: {m.subgroup}" if m.subgroup else m.group
        color = [int(round(c * 255)) for c in m.color[:3]]
        for poly in _local_to_pixels(m.geometry, image).geoms:
            features.append({
                "type": "Feature", "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": {"objectType": "annotation", "classification": {"name": name, "color": color}},
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    return len(features)


def import_geojson(path: str, image) -> dict:
    """Read a GeoJSON file (QuPath or Modalytics) into class geometries for an image.

    Args:
        path: ``.geojson`` / ``.json`` file (FeatureCollection, Feature list or single Feature).
        image: Target ImageNode (pixel coordinates are converted with its calibration).

    Returns:
        dict: {(group, subgroup): (MultiPolygon in local µm, rgba)}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("features", []) if isinstance(data, dict) and data.get("type") == "FeatureCollection" \
        else data if isinstance(data, list) else [data]
    out: dict = {}
    for feat in feats:
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        cls = (feat.get("properties") or {}).get("classification") or {}
        name = cls.get("name") or "Unclassified"
        group, _, subgroup = name.partition(":")
        rgb = cls.get("color") or [180, 180, 180]
        color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)
        local = _pixels_to_local(ann.as_multipolygon(shape(geom)), image)
        _merge(out, (ann.canonical(group), ann.canonical(subgroup) if subgroup else ""), local, color)
    return out


# ------------------------------------------------------------------ Leica / Aperio XML
def colorref_to_rgba(value) -> tuple[float, float, float, float]:
    """Decode a Windows COLORREF (0x00BBGGRR) as used by ``LineColor``."""
    try:
        v = int(value)
        return ((v & 0xFF) / 255, ((v >> 8) & 0xFF) / 255, ((v >> 16) & 0xFF) / 255, 1.0)
    except (TypeError, ValueError):
        return (0.0, 1.0, 1.0, 1.0)


def import_aperio_xml(path: str, image) -> tuple[dict, dict]:
    """Read a Leica/Aperio ImageScope XML annotation file for a whole-slide image.

    Args:
        path: ``.xml`` file.
        image: The slide's ImageNode (vertices are level-0 pixels of this slide).

    Returns:
        tuple: ({(group, subgroup): (MultiPolygon in local µm, rgba)},
        info {"microns_per_pixel", "regions", "reported_area_um2", "mpp_mismatch"}).

    Raises:
        ValueError: If the file is not an ImageScope annotation file.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Not a valid XML file: {exc}") from exc
    if root.tag != "Annotations":
        raise ValueError(f"Root element is <{root.tag}>, expected <Annotations> (Leica/Aperio ImageScope)")
    try:
        mpp = float(root.get("MicronsPerPixel", "nan"))
    except ValueError:
        mpp = math.nan
    out: dict = {}
    info = {"microns_per_pixel": mpp, "regions": 0, "reported_area_um2": 0.0,
            "mpp_mismatch": bool(math.isfinite(mpp) and abs(mpp / image.pixel_size_um[0] - 1) > 0.01)}
    for anno in root.findall("Annotation"):
        name = (anno.get("Name") or "").strip() or f"Annotation {anno.get('Id', '?')}"
        group, subgroup = LEICA_CLASSES.get(name.lower(), (name, ""))
        color = colorref_to_rgba(anno.get("LineColor", 0))
        positive, negative = [], []
        for region in anno.findall("./Regions/Region"):
            pts = [(float(v.get("X")), float(v.get("Y"))) for v in region.findall("./Vertices/Vertex")
                   if v.get("X") is not None and v.get("Y") is not None]
            if len(pts) < 3:
                continue
            poly = ann.as_multipolygon([np.asarray(pts)])
            (negative if region.get("NegativeROA") == "1" else positive).append(poly)
            info["regions"] += 1
            try:
                info["reported_area_um2"] += float(region.get("AreaMicrons", "0")) * (
                    -1 if region.get("NegativeROA") == "1" else 1)
            except ValueError:
                pass
        if not positive:
            continue
        geom = shapely.union_all(positive)
        if negative:
            geom = geom.difference(shapely.union_all(negative))
        _merge(out, (group, subgroup), _pixels_to_local(ann.as_multipolygon(geom), image), color)
    return out, info
