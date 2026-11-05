"""Stain normalisation: makes a slide look as if stained and scanned like a reference (colour images).

Slides from different labs, batches or scanners differ in colour; normalising
them before segmentation or comparison removes much of that difference.

* **Macenko** (Macenko et al. 2009): the slide's two stain colours and amounts
  are found (colour deconvolution with estimated vectors) and redrawn with the
  reference's colours and its typical amounts. Keeps the tissue structure exactly.
* **Reinhard** (Reinhard et al. 2001): the mean and spread of each colour channel
  (in the Lab colour space) are matched to the reference's. Simple and fast; the
  statistics are taken on the tissue only.

The reference is a picture file you choose, or, for Macenko, the standard H&E
reference of the paper's implementation.
"""

import numpy as np
from skimage import color

from modalytics.parameters import optical as op
from modalytics.processing import helpers as hp

KEY = 'stain_normalisation'
NAME = 'Stain normalisation'
SHORT = 'Stain norm.'
ICON = 'mdi6.palette-swatch'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'pixel'
CATEGORY = 'Correction'
DESCRIPTION = "Matches the slide's colours to a reference: Macenko (stain colours and amounts; Macenko et al. 2009) or Reinhard (Lab colour statistics; Reinhard et al. 2001)."
OPTIONS = [
    {'key': 'method',
     'label': 'Method',
     'kind': 'choice',
     'default': 'Macenko',
     'choices': ['Macenko', 'Reinhard'],
     'tooltip': "Macenko: redraws the two stains with the reference's colours (H&E or H-DAB). "
                'Reinhard: matches the colour statistics'},
    {'key': 'reference',
     'label': 'Reference image',
     'kind': 'file',
     'default': '',
     'tooltip': 'A picture (PNG, JPEG, TIFF) of a slide stained the way you want; empty = the standard '
                'H&E reference (Macenko only)'},
]

# The standard H&E reference (the Macenko implementation's: stain colours as columns, 99th-percentile
# amounts), in natural-log optical density like the published code
REF_VECTORS = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
REF_MAX = np.array([1.9705, 1.0308])


def _od(rgb, white):
    """Natural-log optical density (the units of the published reference amounts)."""
    return -np.log(np.clip(rgb[..., :3].astype(np.float64), 1, None) / white[:3])


def _macenko(rgb, white):
    """(stain vectors as columns, 99th-percentile amounts) of an image's tissue."""
    mask = hp.tissue_mask(rgb, white)
    if mask.sum() < 500:
        raise ValueError("too little tissue to find the stain colours")
    od = _od(rgb, white)[mask]
    _vals, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, [2, 1]]
    angles = np.arctan2(*(od @ plane).T[::-1])
    lo, hi = np.percentile(angles, [1, 99])
    v1, v2 = (plane @ [np.cos(a), np.sin(a)] for a in (lo, hi))
    v1, v2 = (v if v.sum() >= 0 else -v for v in (v1, v2))
    vectors = np.array([v1, v2] if v1[0] > v2[0] else [v2, v1]).T        # haematoxylin (more red OD) first
    amounts = np.linalg.lstsq(vectors, od.T, rcond=None)[0]
    return vectors, np.percentile(amounts, 99, axis=1)


def _lab_stats(rgb, white):
    mask = hp.tissue_mask(rgb, white)
    if mask.sum() < 500:
        mask = np.ones(rgb.shape[:2], bool)
    lab = color.rgb2lab(rgb[..., :3].astype(np.float64) / 255.0)[mask]
    return lab.mean(axis=0), lab.std(axis=0) + 1e-6


def _white_of(rgb):
    grey = rgb[..., :3].astype(float).mean(axis=-1)
    return np.maximum(np.median(rgb[..., :3][grey >= np.percentile(grey, 99)], axis=0).astype(float), 1.0)


def prepare(image, options):
    """The slide's and the reference's colour statistics (from overviews)."""
    if image.source.channels < 3:
        raise ValueError("stain normalisation needs a colour image")
    if np.dtype(image.source.dtype) != np.uint8:
        raise ValueError("stain normalisation works on 8-bit colour images")
    rgb = hp.overview(image)
    white = np.asarray(op.estimate_white(image), float)
    ref = hp.read_picture(options["reference"]) if options["reference"] else None
    if ref is not None and (ref.ndim != 3 or ref.dtype != np.uint8):
        raise ValueError("the reference must be an 8-bit colour picture")
    if options["method"] == "Macenko":
        vectors, amounts = _macenko(rgb, white)
        ref_vectors, ref_max = _macenko(ref, _white_of(ref)) if ref is not None else (REF_VECTORS, REF_MAX)
        return {"white": white, "unmix": np.linalg.pinv(vectors), "scale": ref_max / np.maximum(amounts, 1e-6),
                "redraw": ref_vectors}
    if ref is None:
        raise ValueError("Reinhard normalisation needs a reference image (choose one in the options)")
    mean, std = _lab_stats(rgb, white)
    ref_mean, ref_std = _lab_stats(ref, _white_of(ref))
    return {"mean": mean, "std": std, "ref_mean": ref_mean, "ref_std": ref_std}


def process(pixels, options, state, um_per_px):
    rgb = pixels[..., :3]
    if options["method"] == "Macenko":
        # 1. Stain amounts, rescaled to the reference's, redrawn with its colours on neutral glass of the
        #    slide's own brightness
        od = _od(rgb, state["white"]).reshape(-1, 3)
        amounts = np.clip(od @ state["unmix"].T, 0, None) * state["scale"]
        out = float(np.mean(state["white"][:3])) * np.exp(-(amounts @ state["redraw"].T))
        out = out.reshape(rgb.shape)
    else:
        # 2. Lab statistics matched to the reference's
        lab = color.rgb2lab(rgb.astype(np.float64) / 255.0)
        lab = (lab - state["mean"]) / state["std"] * state["ref_std"] + state["ref_mean"]
        out = color.lab2rgb(lab) * 255.0
    if pixels.shape[2] > 3:                                       # an alpha channel stays as it was
        out = np.concatenate([out, pixels[..., 3:].astype(float)], axis=-1)
    return hp.to_like(out, pixels)
