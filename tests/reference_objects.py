"""
The objects the parameter port is checked on (shared by the test and by
``scripts/make_parameter_reference.py``, which produces HoloPath's values).

Real data: the HoloPath test project's phase map and its gland / lamina
propria outlines. Synthetic shapes on the same phase map add cases the real
ones lack: an exact ellipse, a rotated rectangle, a ring with a hole and an
object cut by the image border.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box


def reference_image_and_objects():
    """(phase ImageNode, [(name, polygon in its local µm)]), or None without the test data."""
    import os

    import testdata
    from modalytics.io import importers
    if not os.path.exists(testdata.HPRO):
        return None
    image = importers.import_hpro(testdata.HPRO)[0]
    objects = []
    for m in image.masks:
        for i, poly in enumerate(m.geometry.geoms):
            objects.append((f"{m.group}/{m.subgroup or '-'}/{i}", poly))
    w, h = image.size_um
    objects += [
        ("synthetic/ellipse", affinity.rotate(affinity.scale(Point(60, 60).buffer(1.0, 256), 30, 14), 25)),
        ("synthetic/rotated_rectangle", affinity.rotate(box(150, 30, 200, 60), -35)),
        ("synthetic/ring", Point(60, 180).buffer(25, 256).difference(Point(60, 180).buffer(10, 256))),
        ("synthetic/cut_by_border", Polygon([(w - 15, 120), (w + 20, 120), (w + 20, 160), (w - 15, 160)])),
    ]
    return image, objects


def label_images(image, objects):
    """Rasterise the objects (as ``ObjectContext.mask`` does) into label images.

    Objects that would overlap go into separate label images, so every object
    keeps exactly its own pixels.

    Returns:
        tuple: (label stack (K, H, W) int32, {name: (layer, label)}).
    """
    from modalytics.parameters.context import ObjectContext
    h, w = image.source.levels[0].shape[:2]
    layers, where = [], {}
    for name, poly in objects:
        ctx = ObjectContext(poly, image)
        y0, y1, x0, x1 = ctx.bbox
        for k, layer in enumerate(layers):
            if not (layer[y0:y1, x0:x1][ctx.mask] > 0).any():
                break
        else:
            layers.append(np.zeros((h, w), np.int32))
            k = len(layers) - 1
        label = int(layers[k].max()) + 1
        layers[k][y0:y1, x0:x1][ctx.mask] = label
        where[name] = (k, label)
    return np.stack(layers), where
