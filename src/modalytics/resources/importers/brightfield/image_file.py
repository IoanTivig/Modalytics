"""Ordinary microscope images: PNG, JPEG, BMP and TIFF (8- or 16-bit, grey or RGB).

TIFF files from microscope software often state the pixel size (ImageJ, OME,
or the resolution tags in centimetres); it is then proposed at import as read
from the file. PNG, JPEG and BMP record at most a print resolution (DPI), which
says nothing about the sample, so their pixel size is asked (``hint`` says so).

Slide files (SVS, TIFF pyramids, very large tiled TIFFs) are opened in place by
Modalytics' own slide reader, so this importer leaves them (``can_read`` 0). A
TIFF holding several planes (a stack, separate channels) is read as its first
plane, and the metadata says how many there were. JPEG files are turned the way
their EXIF orientation says, as photo viewers do.
"""

import os

import numpy as np

from modalytics.image_import import Image
from modalytics.io.tiff_source import is_slide, main_series, tiff_metadata, tiff_pixel_size

KEY = 'image_file'
NAME = 'Microscope image (PNG, JPEG, BMP, TIFF)'
IMAGE_TYPE = 'brightfield'
EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
DESCRIPTION = 'Brightfield images from microscope cameras and image software: PNG, JPEG, BMP and TIFF, 8 or 16 bit, grey or RGB. TIFF pixel sizes are read when the file states them; slide files are left to the slide reader.'
OPTIONS = []

TIFF = ('.tif', '.tiff')
MAGIC = {'.png': b'\x89PNG', '.jpg': b'\xff\xd8', '.jpeg': b'\xff\xd8', '.bmp': b'BM'}


def can_read(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF:
        if is_slide(path):
            return 0.0                              # a slide: opened in place by Modalytics
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return 0.5 if len(tif.pages) else 0.0
    with open(path, 'rb') as f:
        return 0.5 if f.read(8).startswith(MAGIC[ext]) else 0.0


def hint(path):
    if os.path.splitext(path)[1].lower() in TIFF:
        return 'This TIFF does not state the pixel size in the sample.'
    return ('PNG, JPEG and BMP files do not record the pixel size in the sample (at most a print resolution). '
            'The microscope software shows it, or it is the camera pixel ÷ the total magnification.')


def read(path, options):
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = main_series(tif)
            meta = tiff_metadata(tif)
            if series is not None:
                pixels = series.asarray()
            else:                                   # a stack or separate channels: the first plane
                pixels = tif.pages[0].asarray()
                meta['planes_in_file'] = len(tif.pages)
                if pixels.ndim == 3 and pixels.shape[0] in (3, 4) and pixels.shape[2] not in (3, 4):
                    pixels = np.moveaxis(pixels, 0, -1)     # planar RGB → interleaved
            return Image(pixels, tiff_pixel_size(tif), metadata=meta)

    # PNG / JPEG / BMP with OpenCV (decoded from bytes: cv2.imread fails on non-ASCII paths on Windows)
    import cv2
    data = np.fromfile(path, dtype=np.uint8)
    flags = cv2.IMREAD_COLOR if ext in ('.jpg', '.jpeg') else cv2.IMREAD_UNCHANGED   # JPEG: EXIF orientation
    pixels = cv2.imdecode(data, flags)
    if pixels is None:
        raise ValueError('not a readable image')
    if pixels.ndim == 3:
        pixels = pixels[..., [2, 1, 0]] if pixels.shape[2] >= 3 else pixels[..., 0]   # BGR(A) → RGB
    return Image(np.ascontiguousarray(pixels), None, metadata={'format': ext[1:].upper()})
