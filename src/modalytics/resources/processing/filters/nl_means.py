"""Non-local means denoising: strong denoising that keeps fine structure (any image).

Each pixel becomes a mean of pixels whose *surroundings* look alike, wherever
they are nearby (Buades et al. 2005), so repeated textures and edges are kept
while noise averages out. The strength is relative to the image's noise,
estimated once on a full-resolution crop (Immerkær 1996). Slower than the other
filters: on whole slides, keep it for looking at small regions or for small images.
"""

import numpy as np
from skimage.restoration import denoise_nl_means

from modalytics.processing import helpers as hp

KEY = 'nl_means'
NAME = 'Non-local means denoising'
SHORT = 'NL means'
ICON = 'mdi6.auto-fix'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Denoise'
DESCRIPTION = 'Averages pixels whose surroundings look alike (Buades et al. 2005): strong denoising that keeps edges and textures. Slow on large images.'
OPTIONS = [
    {'key': 'strength',
     'label': 'Strength',
     'kind': 'float',
     'default': 0.8,
     'minimum': 0.1,
     'maximum': 5.0,
     'step': 0.1,
     'tooltip': 'Filtering strength in units of the estimated noise (higher = smoother)'},
    {'key': 'patch',
     'label': 'Patch size',
     'kind': 'int',
     'default': 5,
     'minimum': 3,
     'maximum': 15,
     'step': 2,
     'suffix': ' px',
     'tooltip': 'Side of the patches compared'},
    {'key': 'search',
     'label': 'Search distance',
     'kind': 'int',
     'default': 6,
     'minimum': 1,
     'maximum': 30,
     'step': 1,
     'suffix': ' px',
     'tooltip': 'How far away similar patches are looked for'},
]


def margin(options, um_per_px):
    return int(options["patch"]) // 2 + int(options["search"]) + 1


def prepare(image, options):
    """The noise level, from a full-resolution crop."""
    sigma = hp.noise_sigma(hp.sample(image, 512))
    return {"sigma": float(sigma) if np.isfinite(sigma) and sigma > 0 else 1e-3}


def process(pixels, options, state, um_per_px):
    colour = pixels.ndim == 3
    out = denoise_nl_means(pixels.astype(np.float32), h=float(options["strength"]) * state["sigma"],
                           sigma=state["sigma"], patch_size=int(options["patch"]),
                           patch_distance=int(options["search"]), fast_mode=True,
                           channel_axis=-1 if colour else None)
    return hp.to_like(out, pixels)
