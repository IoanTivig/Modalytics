"""Flat-field correction: removes uneven illumination and vignetting (slides and brightfield images).

Microscope fields are rarely lit evenly: corners darker, one side brighter.
Dividing by the illumination makes the background even, so brightness and
optical density mean the same everywhere in the field.

* **Blank image**: a picture of an empty field taken with the same settings
  (the classic correction); an optional dark image (camera with the light off)
  is subtracted from both first.
* **Estimated from the image**: a smooth surface (2D polynomial) fitted to the
  empty background of the image itself; for images that show enough background.

The correction keeps the image's mean brightness.
"""

import numpy as np
from scipy import ndimage

from modalytics.parameters import optical as op
from modalytics.processing import helpers as hp

KEY = 'flat_field'
NAME = 'Flat-field correction'
SHORT = 'Flat field'
ICON = 'mdi6.brightness-6'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Correction'
DESCRIPTION = 'Divides by the illumination (a blank image of an empty field, or a smooth surface fitted to the background) to remove uneven lighting and vignetting.'
OPTIONS = [
    {'key': 'method',
     'label': 'Illumination from',
     'kind': 'choice',
     'default': 'Blank image',
     'choices': ['Blank image', 'Estimated from the image'],
     'tooltip': 'Blank image: an empty field taken with the same settings. Estimated: a smooth surface '
                "fitted to this image's empty background"},
    {'key': 'blank',
     'label': 'Blank image',
     'kind': 'file',
     'default': '',
     'tooltip': 'An empty field (no sample), same microscope, objective and camera settings'},
    {'key': 'dark',
     'label': 'Dark image (optional)',
     'kind': 'file',
     'default': '',
     'tooltip': "The camera's signal with the light off; subtracted from the image and the blank"},
    {'key': 'order',
     'label': 'Surface order (estimated)',
     'kind': 'int',
     'default': 2,
     'minimum': 1,
     'maximum': 4,
     'step': 1,
     'tooltip': 'Smoothness of the estimated illumination (2 = a bowl, typical vignetting)'},
]


def _as_channels(a, channels):
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[2] == 1 and channels > 1:
        a = np.repeat(a, channels, axis=2)
    return a[..., :channels]


def prepare(image, options):
    """The illumination per channel, normalised to mean 1 (as a small picture sampled at each tile)."""
    channels = min(image.source.channels, 3)
    if options["method"] == "Blank image":
        blank = _as_channels(hp.read_picture(options["blank"]), channels)
        dark = _as_channels(hp.read_picture(options["dark"]), channels) if options["dark"] else 0.0
        if np.ndim(dark) and dark.shape != blank.shape:
            raise ValueError("the dark and blank images differ in size")
        field = ndimage.gaussian_filter(blank - dark, (2, 2, 0))       # camera noise is not illumination
        field = np.clip(field, 1e-3, None)
        return {"field": field / field.mean(axis=(0, 1)), "dark": dark}
    # Estimated: a surface through the empty background of each channel
    view = _as_channels(hp.overview(image), channels)
    white = np.asarray(op.estimate_white(image), float)
    grey = view.mean(axis=2)
    background = grey > np.percentile(grey, 50)
    if channels >= 3:
        background &= ~hp.tissue_mask(view, white)
    order = int(options["order"])
    small = (64, 64)
    ys, xs = (np.arange(small[0]) + 0.5) / small[0], (np.arange(small[1]) + 0.5) / small[1]
    field = np.stack([hp.eval_surface(hp.fit_surface(view[..., c], order, background), order, ys, xs)
                      for c in range(channels)], axis=-1)
    field = np.clip(field, 1e-3, None)
    return {"field": field / field.mean(axis=(0, 1)), "dark": 0.0}


def process(pixels, options, state, um_per_px, where):
    # 1. The illumination at these pixels (the small picture stretched over the whole image)
    field = state["field"]
    ys, xs = where.fraction(pixels.shape[:2])
    fy = np.clip(ys * field.shape[0] - 0.5, 0, field.shape[0] - 1)
    fx = np.clip(xs * field.shape[1] - 0.5, 0, field.shape[1] - 1)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    channels = field.shape[2]
    light = np.stack([ndimage.map_coordinates(field[..., c], [yy, xx], order=1) for c in range(channels)], axis=-1)
    # 2. Divided out (after the dark signal is taken off)
    values = pixels.astype(np.float32)
    values = values[..., None] if values.ndim == 2 else values
    dark = state["dark"]
    if np.ndim(dark):
        dy = np.clip((ys * dark.shape[0]).astype(int), 0, dark.shape[0] - 1)
        dx = np.clip((xs * dark.shape[1]).astype(int), 0, dark.shape[1] - 1)
        dark = dark[np.ix_(dy, dx)]
    out = values.copy()
    out[..., :channels] = (values[..., :channels] - dark) / light
    out = out[..., 0] if pixels.ndim == 2 else out
    return hp.to_like(out, pixels)
