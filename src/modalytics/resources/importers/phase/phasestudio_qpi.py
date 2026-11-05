"""Phasics PhaseStudio phase maps exported as text (``.txt``, tab-separated) or ``.csv`` (semicolon).

Ported from HoloPath (conversion/dhm_image_management.open_qpi_txt_file /
open_qpi_csv_file). Line 1 is a header of "Key : Value" pairs, e.g.::

    Soft : PhaseStudio v2502.0.4, …, Phase Unit : wv, Pixel Size (µm) : 24.00, …,
    Wavelength (nm) : 1000.00, WaveFront Scale : -1.00, …

then one row of values per image row. The values are in the header's Phase
Unit (normally waves); the phase contract converts them to radians. The
optical height per wave is wavelength × |wavefront scale| (as in HoloPath).

The header's pixel size is the camera's pixel pitch, not the size of a pixel in
the sample (that is the pitch divided by the total magnification of the
microscope, which the file does not record). So ``read`` gives no pixel size
(None) and Modalytics asks for it at import, with the pitch shown as a hint
(``hint``); what you enter is kept as your default (Settings › Importers can
change it). PhaseStudio's .tif export is recognised but refused with an
explanation: it is a 16-bit display rescale (the outside of the pupil is 0)
without the scale needed to recover the phase values, so importing it as a
picture would silently give wrong measurements.
"""

import re

import numpy as np

from modalytics.image_import import Image

KEY = 'phasestudio_qpi'
NAME = 'Phasics PhaseStudio (.txt, .csv)'
IMAGE_TYPE = 'phase'
EXTENSIONS = ['.txt', '.csv', '.tif', '.tiff']
DESCRIPTION = 'Phase maps exported by Phasics PhaseStudio (SID4 cameras). The file gives the camera pitch, not the pixel size in the sample, so that is asked at import and kept as your default.'
OPTIONS = []


def header(path):
    """The "Key : Value" pairs of line 1 (keys without their unit in brackets)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    out = {}
    for part in line.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            out[re.sub(r"\s*\(.*?\)\s*", "", key).strip().lower()] = value.strip()
    return out


def is_tif(path):
    return path.lower().endswith((".tif", ".tiff"))


def can_read(path):
    if is_tif(path):                     # PhaseStudio's display export: claimed, to explain the refusal
        import tifffile
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            soft = str(page.tags["Software"].value) if "Software" in page.tags else ""
        return 0.9 if "PhaseStudio" in soft else 0.0
    h = header(path)
    return 0.95 if "PhaseStudio" in h.get("soft", "") else (0.6 if "pixel size" in h else 0.0)


def hint(path):
    if is_tif(path):
        return ""
    pitch = header(path).get("pixel size", "?")
    return (f"The file gives the camera pitch: {pitch} µm. The pixel size in the sample is the pitch ÷ "
            f"the total magnification (e.g. {pitch} µm ÷ 40× = {float(pitch) / 40:.3g} µm)."
            if pitch != "?" else "")


def read(path, options):
    if is_tif(path):
        raise ValueError("this is PhaseStudio's .tif export: a 16-bit display picture without the scale needed "
                         "to recover the phase. Export the phase as .txt or .csv in PhaseStudio and import that.")
    h = header(path)
    pitch_um = float(h["pixel size"])
    wavelength_nm = float(h.get("wavelength", 1000.0))
    scale = float(h.get("wavefront scale", -1.0))
    # Line 2 onwards: the values (semicolons in .csv, tabs / spaces in .txt)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        second = f.readline()
        f.seek(0)
        f.readline()
        values = np.loadtxt(f, delimiter=";" if ";" in second else None, dtype=np.float32)
    # No pixel size: the file has the camera pitch only (asked at import, see the docstring)
    return Image(values, None, unit=h.get("phase unit", "wv"),
                 height_factor_nm=wavelength_nm * abs(scale),
                 metadata={"instrument": "Phasics " + h.get("snp", ""), "software": h.get("soft", ""),
                           "camera_pitch_um": pitch_um,
                           "wavelength_nm": wavelength_nm, "measurement": h.get("measurement name", "")})
