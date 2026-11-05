"""Background subtraction (rolling ball): removes a slowly varying background, keeping objects (any image).

A ball of the given radius is rolled under the image surface; where it reaches
is the background, which is subtracted (Sternberg 1983, ImageJ's "Subtract
Background"). Objects smaller than the ball are kept whole; uneven backgrounds
larger than it are removed. For a light background (slides, brightfield: the
automatic choice for them), the ball rolls from above and the background becomes
the brightest value instead of 0.

As in ImageJ, large balls are rolled on a shrunk copy (fast) and the background
is enlarged back smoothly.
"""

import numpy as np
from scipy import ndimage
from skimage.restoration import rolling_ball

from modalytics.processing import helpers as hp

KEY = 'rolling_ball'
NAME = 'Background subtraction (rolling ball)'
SHORT = 'Rolling ball'
ICON = 'mdi6.circle-slice-8'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Correction'
DESCRIPTION = "Rolls a ball under the image and subtracts where it reaches (Sternberg 1983, ImageJ's Subtract Background): removes uneven background larger than the ball, keeps smaller objects."
OPTIONS = [
    {'key': 'radius_um',
     'label': 'Ball radius',
     'kind': 'float',
     'default': 50.0,
     'minimum': 0.5,
     'maximum': 5000.0,
     'step': 1.0,
     'suffix': ' µm',
     'tooltip': 'At least the radius of the largest object to keep'},
    {'key': 'background',
     'label': 'Background',
     'kind': 'choice',
     'default': 'Automatic',
     'choices': ['Automatic', 'Light', 'Dark'],
     'tooltip': 'Automatic: light for slides and brightfield images (dark objects on light glass), '
                'dark for the others (phase maps, feature maps)'},
]


def radius_px(options, um_per_px):
    return max(1.0, options["radius_um"] / um_per_px)


def shrink(options, um_per_px):
    """How much the image is shrunk before rolling (1 = not; ImageJ-like for large balls)."""
    r = radius_px(options, um_per_px)
    return 1 if r <= 10 else 2 if r <= 30 else 4 if r <= 100 else 8 if r <= 400 else int(r // 50)


def margin(options, um_per_px):
    return int(2 * radius_px(options, um_per_px)) + 2 * shrink(options, um_per_px) + 2


def prepare(image, options):
    light = options["background"] == "Light" or (options["background"] == "Automatic"
                                                  and image.modality in ("wsi", "brightfield"))
    return {"top": (hp.dtype_range(image.source.dtype) or (None, None))[1], "light": light}


def _background(values, radius, s, where):
    """The rolling-ball background of a 2D block (rolled on the block shrunk ``s`` times, aligned to the image)."""
    if s == 1:
        return rolling_ball(values, radius=radius)
    rows, cols = hp.aligned(where, values.shape, s)
    part = values[rows, cols]
    h, w = part.shape[0] // s, part.shape[1] // s
    small = part[:h * s, :w * s].reshape(h, s, w, s).min(axis=(1, 3))       # shrunk keeping the minimum
    bg_small = rolling_ball(small, radius=radius / s)
    # Enlarged back: each pixel reads the shrunk grid at its own position
    yy, xx = np.meshgrid((np.arange(values.shape[0]) - rows.start + 0.5) / s - 0.5,
                         (np.arange(values.shape[1]) - cols.start + 0.5) / s - 0.5, indexing="ij")
    return ndimage.map_coordinates(bg_small, [yy, xx], order=1, mode="nearest")


def process(pixels, options, state, um_per_px, where):
    r = radius_px(options, um_per_px)
    s = shrink(options, um_per_px)
    light = state["light"]
    top = state["top"]

    def one(channel):
        v = channel.astype(np.float32)
        if light:
            peak = top if top is not None else float(v.max())
            v = peak - v                                            # dark background, bright objects
            return peak - (v - _background(v, r, s, where))
        return v - _background(v, r, s, where)
    return hp.to_like(hp.per_channel(one, pixels), pixels)
