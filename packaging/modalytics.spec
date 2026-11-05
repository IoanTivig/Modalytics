# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build of Modalytics: one folder, two programs sharing it; training's PyTorch comes as the training pack.

* ``Modalytics.exe`` — the application (no console window).
* ``modalytics-cli.exe`` — the same program with a console: ``--version``, ``--gpu-info``, and what the
  application runs in child processes (GPU probes, ``--pip`` for Settings › Libraries; ``paths.helper_command``).

What it carries besides the code (``paths.app_root()`` is the bundle folder, ``_internal``):

1. ``modalytics/resources`` and ``modalytics/help`` as files: built-in profiles, parameters, importers, processing
   steps, registration methods, scripts and architectures are loaded from their ``.py`` files at run time; the
   help's pages and pictures.
2. ``models/``: the shipped segmentation models.
3. Whole libraries scripts may import (parameters, importers, processing steps, users' own scripts): numpy, scipy,
   scikit-image, OpenCV, shapely, tifffile — every submodule, not only those Modalytics itself imports.
4. Libraries found by name at run time: zarr / numcodecs codecs, imagecodecs, ONNX Runtime's providers
   (DirectML); pip (Settings › Libraries, the training pack) as plain files, outside the archive.
5. Every package's metadata (``*.dist-info``): Settings › Libraries lists the included libraries from it and pins
   them when yours are installed; entry points (zarr) need it too.
6. For the training pack (``training.pack``, installed beside the program): the libraries it relies on that the
   program carries (``pack.json``'s "provided": numpy, sympy, networkx…) whole, and the whole standard library — a
   pack loaded at run time may import modules the program itself never does. PyTorch itself is not in the build
   environment, so not in the program.

Built by ``packaging/build.py`` in the build environment (``uv sync --extra training --group build``).
"""

import json
import os
import sys
from importlib import metadata

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = os.path.dirname(SPECPATH)                       # noqa: F821 (PyInstaller defines SPECPATH)
SRC = os.path.join(ROOT, "src")
PKG = os.path.join(SRC, "modalytics")


def tree(folder: str, dest: str) -> list[tuple[str, str]]:
    """Every file under ``folder`` (no ``__pycache__``) as (source, destination folder) pairs."""
    out = []
    for here, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(here, folder)
        for f in files:
            out.append((os.path.join(here, f), os.path.normpath(os.path.join(dest, rel))))
    return out


# 1–2. Files loaded at run time
datas = tree(os.path.join(PKG, "resources"), "modalytics/resources")
datas += tree(os.path.join(PKG, "help"), "modalytics/help")
datas += [(os.path.join(ROOT, "models", f), "models") for f in os.listdir(os.path.join(ROOT, "models"))
          if f.endswith(".mmodel")]

# 3–4. Libraries imported by name, or that scripts may use whole
hiddenimports = collect_submodules("modalytics")
for lib in ("scipy", "skimage", "cv2", "shapely", "tifffile", "imagecodecs", "numcodecs", "zarr", "safetensors",
            "sklearn"):
    hiddenimports += collect_submodules(lib, on_error="ignore")
for lib in ("pygfx", "wgpu", "rendercanvas", "qtawesome", "skimage", "imagecodecs", "onnxruntime"):
    datas += collect_data_files(lib)
# pip as plain files, not in the program's archive: its distlib finds its launcher templates through the ordinary
# file importer (installing anything with a command-line tool fails otherwise: "Unable to locate finder")
datas += collect_data_files("pip", include_py_files=True)
binaries = []
for lib in ("onnxruntime", "wgpu", "imagecodecs"):
    binaries += collect_dynamic_libs(lib)

# 6. What the training pack relies on: those libraries whole, and the whole standard library
pack = json.load(open(os.path.join(PKG, "resources", "training", "pack.json"), encoding="utf-8"))
modules_of = {}
for module, dists in metadata.packages_distributions().items():
    for d in dists:
        modules_of.setdefault(d.lower().replace("_", "-"), set()).add(module)
for dist_name in pack["provided"]:
    for module in modules_of.get(dist_name, ()):
        if not module.startswith("_"):
            hiddenimports += collect_submodules(module, on_error="ignore")
            datas += collect_data_files(module)
LEAVE_OUT = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3", "ensurepip", "venv", "pydoc_data"}
for module in sorted(sys.stdlib_module_names - LEAVE_OUT):
    if not module.startswith("_"):
        hiddenimports += collect_submodules(module, on_error="ignore") or [module]

# 5. Every package's metadata
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if name:
        try:
            datas += copy_metadata(name)
        except Exception:                                  # a distribution PyInstaller cannot place: skipped
            pass

a = Analysis(                                          # noqa: F821
    [os.path.join(SPECPATH, "launcher.py")],           # noqa: F821
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "jupyter", "notebook", "pip",   # pip: files (above)
              "torch", "torchvision", "timm", "segmentation_models_pytorch", "onnx"],      # the training pack's
    noarchive=False,
)
pyz = PYZ(a.pure)                                      # noqa: F821
icon = os.path.join(SPECPATH, "modalytics.ico")        # noqa: F821
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Modalytics", console=False, icon=icon,  # noqa: F821
          upx=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="modalytics-cli", console=True, icon=icon,  # noqa: F821
          upx=False)
COLLECT(gui, cli, a.binaries, a.datas, name="Modalytics", upx=False)  # noqa: F821
