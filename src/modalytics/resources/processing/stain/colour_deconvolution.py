"""Colour deconvolution: the amount of one stain per pixel (slides and brightfield colour images).

Each stain absorbs its own mix of red, green and blue. In optical density those
mixes add up, so the amount of each stain is found by unmixing (Ruifrok &
Johnston 2001). The stain colours are either the standard ones of that paper,
or estimated from this slide (Macenko et al. 2009: the two main directions of
the tissue's colours), which follows differences between labs and scanners.
The result is that stain's optical density (0 = none of it).
"""

import numpy as np

from modalytics.parameters import optical as op
from modalytics.processing import helpers as hp

KEY = 'colour_deconvolution'
NAME = 'Colour deconvolution (stain map)'
SHORT = 'Stains'
ICON = 'mdi6.palette'
INPUT_TYPES = ['wsi', 'brightfield']
OUTPUT_TYPE = 'stain'
UNIT = 'OD'
KIND = 'pixel'
CATEGORY = 'Quantitative maps'
DESCRIPTION = 'The optical density of one stain (haematoxylin, eosin or DAB), unmixed from the colours with standard (Ruifrok & Johnston 2001) or slide-estimated (Macenko et al. 2009) stain vectors.'
OPTIONS = [
    {'key': 'stains',
     'label': 'Stains on the slide',
     'kind': 'choice',
     'default': 'H&E',
     'choices': ['H&E', 'H-DAB', 'H&E + DAB'],
     'tooltip': 'Haematoxylin and eosin; haematoxylin and DAB (immunohistochemistry); or all three'},
    {'key': 'stain',
     'label': 'Map of',
     'kind': 'choice',
     'default': 'Haematoxylin',
     'choices': ['Haematoxylin', 'Eosin', 'DAB'],
     'tooltip': 'Which stain this map shows (it must be on the slide)'},
    {'key': 'vectors',
     'label': 'Stain colours',
     'kind': 'choice',
     'default': 'Standard (Ruifrok & Johnston)',
     'choices': ['Standard (Ruifrok & Johnston)', 'Estimated from the slide (Macenko)'],
     'tooltip': "Standard: the published stain colours. Estimated: found in this slide's tissue (two "
                'stains only: H&E or H-DAB)'},
]

# Optical density per unit of stain in red, green, blue (Ruifrok & Johnston 2001)
STANDARD = {"Haematoxylin": [0.650, 0.704, 0.286], "Eosin": [0.072, 0.990, 0.105], "DAB": [0.268, 0.570, 0.776]}
ON_SLIDE = {"H&E": ["Haematoxylin", "Eosin"], "H-DAB": ["Haematoxylin", "DAB"],
            "H&E + DAB": ["Haematoxylin", "Eosin", "DAB"]}


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def macenko(od: np.ndarray, names: list, angle_percentile: float = 1.0) -> np.ndarray:
    """Two stain vectors (rows) estimated from tissue optical densities (N × 3), ordered like ``names``.

    Macenko et al. 2009: the plane of the two largest principal directions of the
    tissue's OD; the stains are the extreme angles in that plane.
    """
    # 1. The plane of the two main directions of the colours
    _vals, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, [2, 1]]
    proj = od @ plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(angles, [angle_percentile, 100 - angle_percentile])
    found = [_unit(plane @ [np.cos(a), np.sin(a)]) for a in (lo, hi)]
    found = [v if v.sum() >= 0 else -v for v in found]
    # 2. Name them: each goes to the standard stain it resembles most
    ref = [_unit(STANDARD[n]) for n in names]
    if abs(found[0] @ ref[0]) + abs(found[1] @ ref[1]) < abs(found[0] @ ref[1]) + abs(found[1] @ ref[0]):
        found = found[::-1]
    return np.array(found)


def stain_matrix(rgb_overview, white, names, estimate: bool) -> np.ndarray:
    """Rows = stain vectors (unit length) of ``names``; a third is added for two stains (the leftover colour)."""
    if estimate:
        if len(names) != 2:
            raise ValueError("estimated stain colours need two stains (H&E or H-DAB)")
        mask = hp.tissue_mask(rgb_overview, white)
        if mask.sum() < 500:
            raise ValueError("too little tissue to estimate the stain colours: use the standard ones")
        od = -np.log10(np.clip(rgb_overview[mask][:, :3].astype(float), 1, None) / white[:3])
        rows = list(macenko(od, names))
    else:
        rows = [_unit(STANDARD[n]) for n in names]
    if len(rows) == 2:
        third = np.cross(rows[0], rows[1])
        rows.append(_unit(np.abs(third)))
    return np.array(rows)


def prepare(image, options):
    """The unmixing matrix and the background brightness (once per image)."""
    if image.source.channels < 3:
        raise ValueError("colour deconvolution needs a colour image")
    names = ON_SLIDE[options["stains"]]
    if options["stain"] not in names:
        raise ValueError(f"{options['stain']} is not among the stains chosen ({options['stains']})")
    white = np.asarray(op.estimate_white(image), float)
    if white.size < 3:
        white = np.repeat(white, 3)
    matrix = stain_matrix(hp.overview(image), white, names,
                          options["vectors"].startswith("Estimated"))
    return {"unmix": np.linalg.inv(matrix).astype(np.float32), "white": white[:3].astype(np.float32),
            "index": names.index(options["stain"])}


def process(pixels, options, state, um_per_px):
    od = -np.log10(np.clip(pixels[..., :3].astype(np.float32), 1, None) / state["white"])
    amounts = od @ state["unmix"]
    return np.clip(amounts[..., state["index"]], 0, None)
