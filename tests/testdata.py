"""
Where the tests find real microscope files (optional: tests that need one are skipped when it is absent).

The files are not part of the repository. Put them in a folder (default ``_data/test_data`` in the
repository, or the folder named by the ``MODALYTICS_TEST_DATA`` environment variable) under these names:

    slides/slide.svs                 an Aperio SVS slide (0.2512 µm/px, 40×)
    slides/slide.zarr/               the same slide as a zarr pyramid
    slides/licor_rgb.tif             a LI-COR RGB transmission scan (TIFF)
    annotations/slide_leica.xml      Leica / Aperio XML annotations of slide.svs
    dhm/z1/1/Phase.txt               a Lyncée Tec Koala phase export (750 × 750 px, 0.314 µm/px)
    holopath/project.hpro            a HoloPath project: a phase map with gland and lamina masks
    qpi/{tif,txt,csv}/PHA_ep_capture_0000.*   a Phasics phase image in its three export formats

Tests that check exact numbers (sizes, areas) were written for the authors' files; with other files of the
same kinds they are skipped or need their expected values changed.
"""

from __future__ import annotations

import os

import pytest

# The test-data folder: MODALYTICS_TEST_DATA, else _data/test_data in the repository (tests/ → ../_data/test_data)
ROOT = os.environ.get("MODALYTICS_TEST_DATA",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data", "test_data"))

SLIDE_SVS = os.path.join(ROOT, "slides", "slide.svs")
SLIDE_ZARR = os.path.join(ROOT, "slides", "slide.zarr")
LICOR_TIF = os.path.join(ROOT, "slides", "licor_rgb.tif")
LEICA_XML = os.path.join(ROOT, "annotations", "slide_leica.xml")
PHASE_TXT = os.path.join(ROOT, "dhm", "z1", "1", "Phase.txt")
HPRO = os.path.join(ROOT, "holopath", "project.hpro")
QPI = os.path.join(ROOT, "qpi")


def need(path: str) -> str:
    """Skip the calling test when ``path`` does not exist.

    Args:
        path: A file or folder of the test data.

    Returns:
        str: The same path, when it exists.
    """
    if not os.path.exists(path):
        pytest.skip(f"test data not found: {path}")
    return path
