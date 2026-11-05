"""
Processing: steps that make new versions of an image (Qt-free).

    registry    the processing scripts: built-in + the user's (like parameters and importers)
    types       the contracts of the image types they make (OD, OPD, dry mass)
    run         views computed lazily (``make_view``), and their recipes in projects

A step's result is a *view* of the image (``core.model.ImageView``): same pixel
grid, placement and masks, so objects outlined on the image are measured on its
OD map or dry-mass map directly. Profiles can apply steps at import.
"""
