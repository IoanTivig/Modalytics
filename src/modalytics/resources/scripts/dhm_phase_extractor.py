"""DHM phase map extractor: gather every Phase.txt of a DHM database folder into one folder, renamed.

Walks an input folder (e.g. ``data/dhm/sample01``), finds every
``Phase.txt``, names it after the three folders above it — slide, area (a leading
``z`` dropped) and image — and copies them all, side by side, into one output folder:

    sample01/z1/1/Phase.txt  →  sample01-1-1.txt

A name already taken in the output folder gets ``_2``, ``_3``… (nothing is
overwritten). At the end the copies can be imported into the open project with the
usual importer. Ported from HoloPath's script of the same name.
"""

import os
import shutil

#: properties (edited on the Properties and Options tabs; written here on save)
KEY = 'dhm_phase_extractor'
NAME = 'DHM phase map extractor'
CATEGORY = 'Data preparation'
DESCRIPTION = ('Gather every Phase.txt of a DHM database folder into one folder, renamed slide-area-image.txt '
               '(e.g. sample01/z1/1/Phase.txt → sample01-1-1.txt), ready to import.')
OPTIONS = [
    {'key': 'input', 'label': 'Input folder', 'kind': 'folder', 'default': '',
     'tooltip': 'The DHM database folder: <slide>/<area>/<image>/Phase.txt below it'},
    {'key': 'output', 'label': 'Output folder', 'kind': 'folder', 'default': '',
     'tooltip': 'Where the renamed copies go (made if missing)'},
    {'key': 'filename', 'label': 'File to look for', 'kind': 'text', 'default': 'Phase.txt',
     'tooltip': 'Matched without regard to capitals'},
    {'key': 'offer_import', 'label': 'Offer to import them', 'kind': 'bool', 'default': True,
     'tooltip': 'At the end, offer to import the copies into the open project'},
]


def slide_area_image(path):
    """(slide, area, image) from a ``…/slide/area/image/Phase.txt`` path (a leading 'z' dropped from the area),
    or None if the path is too shallow."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    slide, area, image = parts[-4], parts[-3], parts[-2]
    return slide, area[1:] if area.startswith("z") else area, image


def find(folder, filename):
    """Every file called ``filename`` (any capitals) below ``folder``, sorted."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        out += [os.path.join(root, f) for f in files if f.lower() == filename.lower()]
    return sorted(out)


def free_name(folder, name):
    """``name`` in ``folder``, or ``stem_2.ext``, ``stem_3.ext``… if taken."""
    dest = os.path.join(folder, name)
    stem, ext = os.path.splitext(name)
    n = 2
    while os.path.exists(dest):
        dest = os.path.join(folder, f"{stem}_{n}{ext}")
        n += 1
    return dest


def run(options, ctx):
    # 1. What to copy
    source, target = options["input"], options["output"]
    if not source or not os.path.isdir(source):
        raise ValueError("Choose the input folder (it must exist)")
    if not target:
        raise ValueError("Choose the output folder")
    files = find(source, options["filename"] or "Phase.txt")
    if not files:
        ctx.log(f"⚠️ No {options['filename']} below {source}")
        return "Nothing found"
    os.makedirs(target, exist_ok=True)
    # 2. Copy, renamed
    copied, skipped = [], 0
    for i, src in enumerate(files, start=1):
        ctx.check()
        parts = slide_area_image(src)
        if parts is None:
            ctx.log(f"⚠️ Skipped (path too shallow): {src}")
            skipped += 1
            continue
        ext = os.path.splitext(src)[1] or ".txt"
        dest = free_name(target, "-".join(parts) + ext)
        try:
            shutil.copy2(src, dest)
            copied.append(dest)
            ctx.log(f"✅ {os.path.relpath(src, source)} → {os.path.basename(dest)}")
        except OSError as exc:
            skipped += 1
            ctx.log(f"❌ {src}: {exc}")
        ctx.progress(i / len(files))
    # 3. Hand the copies over for import
    if options["offer_import"] and copied:
        ctx.import_files(copied)
    return f"{len(copied)} file(s) copied to {target}" + (f", {skipped} skipped" if skipped else "")
