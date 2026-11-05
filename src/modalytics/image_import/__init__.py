"""
Importing images with modular importers (Qt-free).

    contracts   what an image of each type must be once imported (phase in radians…)
    registry    the importer scripts: built-in + the user's, chosen per file
    defaults    your choices per importer on this computer (pixel size, options, asking)
    run         read, settle the pixel size, check, make ImageNodes (``import_file``)

Importer scripts write ``from modalytics.image_import import Image``.
"""

from modalytics.image_import.contracts import PHASE_UNITS, ContractError, Image

__all__ = ["PHASE_UNITS", "ContractError", "Image", "import_file"]


def import_file(path, library=None, importer=None, options=None, pixel_size_um=None):
    """Import one file without asking (see ``run.import_file``)."""
    from modalytics.image_import.run import import_file as _import_file
    return _import_file(path, library, importer, options, pixel_size_um)
