"""
Entry point: ``python -m modalytics [--theme light] [files…]``.

Without files the start screen opens. Files: a ``.modalytics`` project,
``.hpro`` (HoloPath project), a zarr folder (WSI) or ``Phase.txt``.

``--gpu-info`` prints the GPUs found for drawing and for the models, and what
was chosen (see ``render.gpu`` and ``services.compute``). ``--version`` prints the
version; ``--self-test`` checks the installation (``modalytics.selftest``);
``--install-training [cuda|xpu|cpu]`` installs the training pack (``training.pack``; the hardware chooses when no
variant is given). Internal: ``--gpu-probe API`` (``render.gpu`` runs it in a separate process)
``--pip …`` (the installed app's pip, for Settings › Libraries: ``python -m pip …``) and
``--check-training-pack FOLDER`` (a new training pack tried in a separate process).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Start the application.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit code.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if getattr(sys, "frozen", False):
        sys.dont_write_bytecode = True                   # the installation (Program Files) is read-only
    if argv[:1] == ["--pip"]:                            # pip, as ``python -m pip`` (the installed app has no python)
        import runpy
        sys.argv = ["pip", *argv[1:]]
        try:
            runpy.run_module("pip", run_name="__main__", alter_sys=True)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        return 0
    from modalytics.training import pack
    pack.activate()                                      # the training pack (installed program), before any torch
    parser = argparse.ArgumentParser(prog="modalytics")
    parser.add_argument("files", nargs="*", help=".modalytics or .hpro project, slide file (SVS, pyramidal TIFF), "
                        "zarr folder, phase map or microscope image")
    parser.add_argument("--theme", choices=("dark", "light"), default=None,
                        help="override the theme saved in Settings › General")
    parser.add_argument("--gpu-info", action="store_true", help="print the GPUs found and chosen, then exit")
    parser.add_argument("--version", action="store_true", help="print the version, then exit")
    parser.add_argument("--self-test", action="store_true",
                        help="check this installation (measuring, segmentation, GPU, PDF, training…), then exit")
    parser.add_argument("--gpu-probe", metavar="API", help=argparse.SUPPRESS)
    parser.add_argument("--install-training", nargs="?", const="", metavar="VARIANT", choices=("", "cuda", "xpu", "cpu"),
                        help="install the training pack (PyTorch) for this computer's hardware, or for VARIANT")
    parser.add_argument("--check-training-pack", metavar="FOLDER", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # 0. Hardware: a probe process, the --gpu-info report, else start probing in the background
    from modalytics.render import gpu
    if args.gpu_probe:
        return gpu.probe_main(args.gpu_probe)
    if args.check_training_pack:
        return pack.check_main(args.check_training_pack)
    if args.install_training is not None:
        return pack.install(args.install_training or None)
    if args.self_test:
        from modalytics import selftest
        return selftest.run()
    if args.version:
        from modalytics import __version__
        print(f"Modalytics {__version__}")
        return 0
    if args.gpu_info:
        from modalytics.services import compute
        print(gpu.report())
        print(compute.report())
        return 0
    from modalytics import app_settings
    app_settings.apply_to_environment()              # GPU choices from Settings › General
    from modalytics import diagnostics
    diagnostics.start_if_wanted()                    # the diagnostic log, if Settings › General asks for it
    gpu.start_probe()                                # runs while Qt and the window are built

    # 1. Qt first (rendercanvas picks the Qt binding that is already imported)
    from PySide6 import QtCore, QtWidgets
    # Qt WebEngine (the report printed to PDF) wants this set before the application exists
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Modalytics")
    from PySide6 import QtGui
    from modalytics import paths
    app.setWindowIcon(QtGui.QIcon(str(paths.RESOURCES / "icons" / "modalytics.png")))
    diagnostics.install_qt()                         # Qt's own warnings go to the log too (when it is on)

    from modalytics.core.model import Project
    from modalytics.io import importers
    from modalytics.ui.main_window import MainWindow

    # 2. Project: files from the command line, or none (the start screen)
    project = None
    project_to_open = next((p for p in args.files if p.lower().endswith(".modalytics")), None)
    images = [p for p in args.files if not p.lower().endswith(".modalytics")]
    if images and project is None:
        project = Project()                              # the window gives it the default profile
    nodes = []
    for path in images:
        if path.lower().endswith(".hpro"):
            nodes += importers.import_hpro(path)
        elif os.path.isdir(path):
            nodes.append(importers.import_zarr_wsi(path))
        else:                                            # slides in place; phase maps, images: the importers
            from modalytics.image_import import import_file
            from modalytics.image_import.run import ImportFailed
            from modalytics.io.tiff_source import is_slide
            try:
                nodes += [importers.import_tiff_wsi(path)] if is_slide(path) else import_file(path)
            except (ImportFailed, ValueError) as exc:
                print(f"⚠️ {exc}")

    # 3. Window (images from the command line go in like any import: the profile's types and processing)
    win = MainWindow(project, theme=args.theme or app_settings.get("theme"))
    if nodes:
        win.add_images(nodes)
    win.show()
    if project_to_open:
        win.open_path(project_to_open)
    elif project is not None:
        win.canvas.fit_all()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
