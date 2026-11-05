"""
What the image types made by processing must be (their contracts), and how they are first drawn.

A processing step whose result is another image type hands back a 2D float
array; it is checked here. Filters keep their input's type and shape.

    od         optical density, ≥ 0 (0 = no absorption), unit OD
    opd        optical path difference, nm (any sign)
    dry_mass   dry-mass surface density, pg/µm² (any sign: background noise is around 0)
    height     physical thickness, µm (any sign: background noise is around 0)
    ri         refractive index (dimensionless)
    stain      one stain's optical density (colour deconvolution), ≥ 0, unit OD
    feature    a derived map (edges, blobs, ridges, local texture), dimensionless: for looking
               and segmentation, not a physical quantity
"""

from __future__ import annotations

import numpy as np

# Image type → (unit, lowest allowed value or None, colormap, first contrast range or None = from the data)
MADE = {
    "od": ("OD", 0.0, "magma", (0.0, 1.0)),
    "opd": ("nm", None, "viridis", None),
    "dry_mass": ("pg/µm²", None, "viridis", None),
    "height": ("µm", None, "viridis", None),
    "ri": ("", None, "viridis", None),
    "stain": ("OD", 0.0, "magma", (0.0, 1.0)),
    "feature": ("", None, "gray", None),
}


class ContractError(ValueError):
    """A step's result does not meet its image type's contract (the message says what is wrong)."""


def check_tile(result: np.ndarray, pixels: np.ndarray, output_type: str, is_filter: bool) -> np.ndarray:
    """Check one result of ``process`` and return it in the stored form.

    Args:
        result: What ``process`` returned.
        pixels: What it was given.
        output_type: The image type it makes.
        is_filter: The step keeps the input's type (and shape).

    Raises:
        ContractError: With what is wrong.
    """
    r = np.asarray(result)
    if r.shape[:2] != pixels.shape[:2]:
        raise ContractError(f"the result is {r.shape[1]}×{r.shape[0]} px, the input {pixels.shape[1]}×"
                            f"{pixels.shape[0]}: a step keeps the size (and placement) of the image")
    if is_filter:
        return r.astype(pixels.dtype) if np.issubdtype(pixels.dtype, np.integer) else r.astype(np.float32)
    if r.ndim != 2:
        raise ContractError(f"a {output_type} map has one value per pixel (got shape {r.shape})")
    r = r.astype(np.float32)
    low = MADE.get(output_type, (None, None))[1]
    if low is not None and np.nanmin(r) < low - 1e-6:
        raise ContractError(f"{output_type} values must be ≥ {low} (got {np.nanmin(r):.4g})")
    return r
