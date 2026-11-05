"""
The installation checks itself: ``modalytics-cli --self-test`` (or ``python -m modalytics --self-test``).

Each check does the real work on small made-up data — nothing of the user's is read or written (a temporary user
library is used) — and prints ✅ or ❌ with what it found, so an installation on another computer can be checked
and the output sent back:

1. **Built-in libraries**: profiles, parameters, importers, processing steps, registration methods, scripts and
   architectures load from their scripts, with no errors.
2. **Help**: every page of the guides that come with the program turns into HTML.
3. **Measuring**: a phase map with two glands measured with the shipped profile's parameters.
4. **Segmentation**: a shipped model run by ONNX Runtime on the best device (the GPU when there is one).
5. **Canvas**: a picture drawn by the GPU (wgpu) without a window, its pixels checked.
6. **Classifiers and exports**: a random forest (scikit-learn) fitted; an Excel workbook with a native chart.
7. **PDF**: an HTML page printed to PDF by Qt WebEngine (reports, training reports).
8. **Training** (when the training pack is installed — ``training.pack``; without it the check says so and
   passes): PyTorch on the best device — a U-Net trained a few steps, then the architecture check: exported to
   ONNX and run by ONNX Runtime, the same outputs as PyTorch. And whether the pack still fits the hardware.
9. **pip** (Settings › Libraries): run through the program.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run() -> int:
    """Run every check; returns 0 when all passed."""
    os.environ["MODALYTICS_HOME"] = tempfile.mkdtemp(prefix="modalytics_selftest_")   # never the user's library
    from modalytics import __version__, paths
    print(f"Modalytics {__version__} — self-test ({'installed' if paths.installed() else 'from the source'}: "
          f"{paths.app_root()})\n", flush=True)
    results = []
    for name, check in CHECKS:
        t0 = time.perf_counter()
        try:
            detail = check()
            ok = True
            if isinstance(detail, Skipped):                 # nothing to check here (e.g. no training pack)
                print(f"ℹ️ {name} — {detail}", flush=True)
                continue
        except Exception as exc:                          # the check failed: its last lines say where
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
            tb = traceback.extract_tb(exc.__traceback__)
            if tb:
                detail += f"  ({Path(tb[-1].filename).name}:{tb[-1].lineno})"
        results.append(ok)
        print(f"{'✅' if ok else '❌'} {name} — {detail}  [{time.perf_counter() - t0:.1f} s]", flush=True)
    failed = results.count(False)
    print(f"\n{'✅' if not failed else '❌'} {len(results) - failed}/{len(results)} checks passed", flush=True)
    return 1 if failed else 0


class Skipped(str):
    """A check with nothing to check on this installation (said, not counted)."""


# ---------------------------------------------------------------- the checks
def _libraries() -> str:
    from modalytics.image_import.registry import ImporterLibrary
    from modalytics.parameters.registry import ParameterLibrary
    from modalytics.processing.registry import ProcessingLibrary
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.registration.registry import RegistrationLibrary
    from modalytics.scripts.registry import ScriptLibrary
    from modalytics.training.architectures import ArchitectureLibrary
    profiles = ProfileLibrary()
    counts, errors = {"profiles": len(profiles.builtin_ids())}, {}
    for label, lib in (("parameters", ParameterLibrary()), ("importers", ImporterLibrary()),
                       ("processing steps", ProcessingLibrary()), ("registration methods", RegistrationLibrary()),
                       ("scripts", ScriptLibrary()), ("architectures", ArchitectureLibrary())):
        counts[label] = len(lib.specs)
        errors.update(lib.errors)
    if errors or not all(counts.values()):
        raise RuntimeError(f"{counts}; errors: {errors}")
    return ", ".join(f"{n} {k}" for k, n in counts.items())


def _help() -> str:
    from modalytics import help as hp
    from modalytics.help.markdown import to_html
    pages = [p for g in hp.guides() for p in g.pages]
    for p in pages:
        to_html(p.path.read_text(encoding="utf-8"))
    images = list(hp.IMAGES.glob("*"))
    if len(pages) < 10 or not images:
        raise RuntimeError(f"{len(pages)} pages, {len(images)} pictures")
    return f"{len(pages)} pages in {len(hp.guides())} guides, {len(images)} pictures"


def _phase_image(size: int = 400):
    """A made-up phase map (0.5 µm/px) with two round glands, as an ImageNode."""
    import numpy as np
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:size, :size]
    values = rng.normal(0.2, 0.05, (size, size)).astype(np.float32)
    for cx, cy, r in ((120, 130, 50), (270, 250, 70)):
        values[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] += 1.2
    return ImageNode("self-test", ArraySource(values), (0.5, 0.5), "phase")


def _measuring() -> str:
    from shapely.geometry import Point
    from modalytics.core.model import MaskNode, Project
    from modalytics.parameters.results import measure_project
    from modalytics.profiles.library import default_profile
    project = Project()
    project.profile, project.profile_id = default_profile()
    img = project.add_image(_phase_image())
    glands = [Point(60, 65).buffer(25), Point(135, 125).buffer(35)]       # in µm
    project.add_mask(img, MaskNode(geometry=glands, group="Gland", subgroup="Low grade dysplasia"))
    res = measure_project(project)
    rows = res.of_scope("object")
    if len(rows) != 2:
        raise RuntimeError(f"{len(rows)} object rows")
    return f"2 glands × {len(rows[0].values)} parameters (profile {project.profile.name})"


def _segmentation() -> str:
    from modalytics.services import compute
    from modalytics.services.segmentation import find_models, segment
    models = find_models()
    phase = [m for m in models if m.modality == "phase"]
    if not phase:
        raise RuntimeError(f"no phase-map model among {len(models)}")
    img = _phase_image(512)
    t0 = time.perf_counter()
    found = segment(img, (0, 0, 256, 256), phase[0], device="auto", use_cache=False)
    device = compute.resolve("auto")
    return (f"{len(models)} models; {phase[0].name} ran in {time.perf_counter() - t0:.1f} s on {device.name}, "
            f"{len(getattr(found, 'geoms', []))} object(s) on made-up data")


def _canvas() -> str:
    import numpy as np
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas
    from modalytics.core.model import ImageNode
    from modalytics.io.sources import ArraySource
    from modalytics.render.gpu import init_gpu
    from modalytics.render.visuals import ImageVisual, prepare_tile
    name = init_gpu()
    data = np.zeros((10, 20, 3), np.uint8)
    data[:5, :, 0] = 255
    data[5:, :, 2] = 255
    node = ImageNode("t", ArraySource(data), (1.0, 1.0), "brightfield")
    vis = ImageVisual(node)
    vis.add_tile((0, 0, 0), prepare_tile(node.source.read_tile(0, 0, 0)))
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"), vis.group)
    canvas = RenderCanvas(size=(40, 20))
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)
    cam = gfx.OrthographicCamera(40, 20, maintain_aspect=False, depth_range=(1, 10000))
    cam.local.scale_y = -1
    cam.local.position = (10, 5, 5000)
    canvas.request_draw(lambda: renderer.render(scene, cam))
    img = np.asarray(canvas.draw())
    if tuple(img[7, 20, :3]) != (255, 0, 0) or tuple(img[13, 20, :3]) != (0, 0, 255):
        raise RuntimeError(f"wrong pixels {tuple(img[7, 20, :3])}, {tuple(img[13, 20, :3])}")
    return f"drawn and read back correctly ({name or 'GPU'})"


def _classifiers_and_exports() -> str:
    import numpy as np
    import xlsxwriter
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(2, 1, (40, 3))])
    y = np.repeat([0, 1], 40)
    acc = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y).score(X, y)
    path = Path(tempfile.mkdtemp()) / "check.xlsx"
    wb = xlsxwriter.Workbook(str(path))
    ws = wb.add_worksheet("Data")
    ws.write_column(0, 0, [1, 2, 3])
    chart = wb.add_chart({"type": "column"})
    chart.add_series({"values": "=Data!$A$1:$A$3"})
    ws.insert_chart("C2", chart)
    wb.close()
    return f"random forest fitted (training accuracy {acc:.2f}); workbook with a chart ({path.stat().st_size} bytes)"


def _pdf() -> str:
    from PySide6 import QtCore, QtWidgets
    if QtWidgets.QApplication.instance() is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        _pdf.app = QtWidgets.QApplication(sys.argv[:1])
    from modalytics.ui.pdf_export import print_pdf
    folder = Path(tempfile.mkdtemp())
    html = folder / "page.html"
    html.write_text("<html><body><h1>Modalytics</h1><p>Printed by Qt WebEngine.</p></body></html>",
                    encoding="utf-8")
    outcome = {}
    loop = QtCore.QEventLoop()
    print_pdf(str(html), str(folder / "page.pdf"), lambda p: (outcome.update(ok=p), loop.quit()),
              lambda why: (outcome.update(failed=why), loop.quit()))
    QtCore.QTimer.singleShot(60000, loop.quit)
    loop.exec()
    if "ok" not in outcome:
        raise RuntimeError(outcome.get("failed", "no answer in 60 s"))
    size = Path(outcome["ok"]).stat().st_size
    return f"printed ({size} bytes)"


def _training() -> str:
    from modalytics.training import architectures as arch
    from modalytics.training import pack
    from modalytics.training import seg_train as st
    ok, why = st.available()
    state = pack.status()
    if not ok:
        if state["state"] == "none":
            return Skipped(f"no training pack installed — {state['message']}")
        raise RuntimeError(why)
    if state["state"] == "changed":
        raise RuntimeError(state["message"])
    import torch
    devices = st.devices()
    dev = devices[0]
    model = st.build("unet_resnet18", 1).to(dev.key)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x = torch.randn(2, 1, 128, 128, device=dev.key)
    target = (x > 0).float()
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    spec = arch.ArchitectureLibrary().specs["unet_resnet18"]
    checked = arch.check(spec, 1)
    if not checked["ok"]:
        raise RuntimeError(f"architecture check: {checked['why']}")
    return (f"PyTorch {torch.__version__} on {dev.label}: 5 steps, loss {losses[0]:.2f} → {losses[-1]:.2f}; "
            f"exported to ONNX, same outputs (max difference {checked['max_diff']:.1e})")


def _wheel(folder: Path) -> Path:
    """A tiny package with a command-line tool, as a wheel (offline: nothing is downloaded)."""
    import base64
    import hashlib
    import zipfile
    files = {"selftest_pkg/__init__.py": "def main():\n    return 0\n",
             "selftest_pkg-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: selftest-pkg\nVersion: 1.0\n",
             "selftest_pkg-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: modalytics\nRoot-Is-Purelib: true\n"
                                                 "Tag: py3-none-any\n",
             "selftest_pkg-1.0.dist-info/entry_points.txt": "[console_scripts]\nselftest-pkg = selftest_pkg:main\n"}
    record = []
    for name, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(text.encode())}")
    record.append("selftest_pkg-1.0.dist-info/RECORD,,")
    path = folder / "selftest_pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
        z.writestr("selftest_pkg-1.0.dist-info/RECORD", "\n".join(record) + "\n")
    return path


def _pip() -> str:
    """pip runs, and installs a package with a command-line tool (what Settings › Libraries and the training pack
    do), from a wheel made here: offline."""
    import subprocess
    from modalytics.parameters.libraries import pip
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run([*pip(), "--version"], capture_output=True, text=True, timeout=120, creationflags=flags)
    if res.returncode != 0 or not res.stdout.startswith("pip "):
        raise RuntimeError((res.stdout + res.stderr).strip()[-300:])
    folder = Path(tempfile.mkdtemp())
    target = folder / "target"
    done = subprocess.run([*pip(), "install", "--no-index", "--no-deps", "--target", str(target),
                           str(_wheel(folder))], capture_output=True, text=True, timeout=300, creationflags=flags)
    tools = list((target / "bin").glob("selftest-pkg*")) if (target / "bin").is_dir() else []
    if done.returncode != 0 or not (target / "selftest_pkg").is_dir() or not tools:
        raise RuntimeError("installing a package failed: " + (done.stdout + done.stderr).strip()[-400:])
    return f"{res.stdout.split(' from ')[0]}; installs a package with its command-line tool"


CHECKS = [("Built-in libraries", _libraries), ("Help", _help), ("Measuring", _measuring),
          ("Segmentation", _segmentation), ("Canvas (GPU drawing)", _canvas),
          ("Classifiers and Excel export", _classifiers_and_exports), ("PDF (Qt WebEngine)", _pdf),
          ("Training (PyTorch)", _training), ("pip (Settings › Libraries)", _pip)]
