"""
Legacy DHM phase text files (``Phase.txt``), ported from HoloPath's
``conversion/dhm_image_management.open_dhm_phase_file``.

File layout::

    h=750 w=750
    pixel size=3.14006e-007 m
    data unit=rad
    height conversion factor (-> m)=2.11517e-007
    1.195219 1.208140 …            (h·w values, row-major)

Changes from HoloPath: returns a dataclass, raises clear errors instead of
printing and returning None, parses the numbers about 10× faster
(``str.split`` + ``np.array`` instead of ``np.loadtxt``), and reads the
``data unit`` line instead of assuming radians: the values are returned as
stored, with their unit, and the importer (``resources/importers/phase/
lyncee_phase_txt.py``) leaves the conversion to radians to the phase contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseMap:
    """A quantitative phase image, as stored in the file.

    Attributes:
        values: (H, W) float32 phase values, in ``unit``.
        unit: The file's ``data unit`` (normally "rad").
        pixel_size_um: Sample-plane pixel size in µm.
        height_factor_nm: Optical height (nm) per unit of the values, or None.
        path: Source file.
    """

    values: np.ndarray
    unit: str
    pixel_size_um: float
    height_factor_nm: float | None
    path: str


_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "nm": 1e-3}


def read_phase_txt(path: str) -> PhaseMap:
    """Read a legacy DHM ``Phase.txt``.

    Args:
        path: File path.

    Returns:
        PhaseMap: The phase map with its calibration.

    Raises:
        ValueError: If the header or the data size is not as expected.
    """
    with open(path, "r", encoding="latin-1") as f:
        # 1. Header line 1: "h=750 w=750"
        parts = dict(p.split("=") for p in f.readline().split() if "=" in p)
        try:
            h, w = int(parts["h"]), int(parts["w"])
        except KeyError as exc:
            raise ValueError(f"{path}: missing 'h=' / 'w=' in the first line") from exc

        # 2. Header line 2: "pixel size=3.14006e-007 m"
        line = f.readline()
        if "pixel size" not in line:
            raise ValueError(f"{path}: missing 'pixel size' line")
        value, *unit = line.split("=", 1)[1].split()
        pixel_size_um = float(value) * _UNIT_TO_UM.get(unit[0] if unit else "m", 1e6)

        # 3. Header line 3: "data unit=rad"
        line = f.readline()
        unit = line.split("=", 1)[1].strip() if "unit" in line and "=" in line else "rad"

        # 4. Header line 4: "height conversion factor (-> m)=2.11517e-007"
        line = f.readline()
        factor_nm = float(line.split("=", 1)[1].split()[0]) * 1e9 if "conversion factor" in line else None

        # 5. Data
        values = np.array(f.read().split(), dtype=np.float32)
    if values.size != h * w:
        raise ValueError(f"{path}: expected {h * w} values, found {values.size}")
    return PhaseMap(values.reshape(h, w), unit, pixel_size_um, factor_nm, os.path.abspath(path))
