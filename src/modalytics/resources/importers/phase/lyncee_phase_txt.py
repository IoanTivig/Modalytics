"""Lyncée Tec DHM phase maps exported as text (``Phase.txt``).

Ported from HoloPath (conversion/dhm_image_management.open_dhm_phase_file). The
file header gives the size, the pixel size in the sample, the unit of the
values (``data unit=rad``) and the optical height per unit; nothing needs to
be asked. The values are handed over in the file's unit: the phase contract
converts them to radians.

    h=750 w=750
    pixel size=3.14006e-007 m
    data unit=rad
    height conversion factor (-> m)=2.11517e-007
    1.195219 1.208140 …
"""

import os

from modalytics.image_import import Image
from modalytics.io.dhm_legacy import read_phase_txt

KEY = 'lyncee_phase_txt'
NAME = 'Lyncée Tec DHM (Phase.txt)'
IMAGE_TYPE = 'phase'
EXTENSIONS = ['.txt']
DESCRIPTION = 'Phase maps exported as text by Lyncée Tec DHM software. Everything needed is in the file header.'
OPTIONS = []


def can_read(path):
    with open(path, "r", encoding="latin-1") as f:
        first = f.readline().strip()
    return 1.0 if first.startswith("h=") and "w=" in first else 0.0


def needs(path):
    return []                       # the header says everything


def read(path, options):
    pm = read_phase_txt(path)
    # Every export is called Phase.txt: name it after its folders (<area>/<image> Phase); a renamed file (e.g.
    # gathered by the DHM phase map extractor script: sample01-1-1.txt) keeps its own name
    folder = os.path.dirname(os.path.abspath(path))
    if os.path.basename(path).lower() == "phase.txt":
        name = f"{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)} Phase"
    else:
        name = os.path.splitext(os.path.basename(path))[0]
    return Image(pm.values, pm.pixel_size_um, unit=pm.unit, name=name, height_factor_nm=pm.height_factor_nm,
                 metadata={"instrument": "Lyncée Tec DHM"})
