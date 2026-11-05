"""Ridge map (vesselness): highlights thin elongated structures — fibres, vessels, membranes (any image).

Frangi et al. 1998: at each scale, the Hessian's eigenvalues tell whether a
pixel lies on a line-like structure (one strong curvature across, little along);
the strongest response over the widths given is kept. Colour images use their
luminance. The "structure" scale (Frangi's γ) is set once per image, from an
overview, so every tile and zoom level is scaled alike.
"""

import math

import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi

from modalytics.processing import helpers as hp

KEY = 'ridges'
NAME = 'Ridge map (Frangi vesselness)'
SHORT = 'Ridges'
ICON = 'mdi6.vector-curve'
INPUT_TYPES = ['*']
OUTPUT_TYPE = 'feature'
UNIT = ''
KIND = 'neighbourhood'
CATEGORY = 'Feature maps'
DESCRIPTION = 'Frangi vesselness (Frangi et al. 1998): bright on thin elongated structures (fibres, vessels, membranes) of the widths given.'
OPTIONS = [
    {'key': 'min_um',
     'label': 'Thinnest (σ)',
     'kind': 'float',
     'default': 0.5,
     'minimum': 0.05,
     'maximum': 100.0,
     'step': 0.1,
     'suffix': ' µm',
     'tooltip': 'Smallest half-width of the structures'},
    {'key': 'max_um',
     'label': 'Thickest (σ)',
     'kind': 'float',
     'default': 3.0,
     'minimum': 0.05,
     'maximum': 200.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Largest half-width of the structures'},
    {'key': 'ridges',
     'label': 'Structures are',
     'kind': 'choice',
     'default': 'Dark',
     'choices': ['Dark', 'Bright'],
     'tooltip': 'Dark lines on a light background (brightfield), or bright ones (phase, fluorescence)'},
]


def sigmas(options, um_per_px):
    lo = max(0.5, options["min_um"] / um_per_px)
    hi = max(lo, options["max_um"] / um_per_px)
    n = max(1, min(6, int(round(math.log2(hi / lo) * 2)) + 1))      # about two widths per doubling
    return np.geomspace(lo, hi, n)


def margin(options, um_per_px):
    return math.ceil(4 * sigmas(options, um_per_px)[-1]) + 1


def prepare(image, options):
    """Frangi's γ (half the strongest Hessian norm at the thinnest width), in value per µm²."""
    um = hp.overview_scale(image)
    values = hp.luminance(hp.overview(image))
    eig = hessian_matrix_eigvals(hessian_matrix(values, sigmas(options, um)[0], use_gaussian_derivatives=True))
    gamma_px = float(np.sqrt((eig ** 2).sum(0)).max()) / 2 or 1.0
    return {"gamma_um": gamma_px / um ** 2}


def process(pixels, options, state, um_per_px):
    # γ back to this level's pixels: second derivatives grow with the pixel size squared
    return frangi(hp.luminance(pixels), sigmas=sigmas(options, um_per_px), gamma=state["gamma_um"] * um_per_px ** 2,
                  black_ridges=options["ridges"] == "Dark").astype(np.float32)
