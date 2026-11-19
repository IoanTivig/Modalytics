"""
What Modalytics loads when it starts, step by step, so the loading window can say what it is doing (no Qt here).

Most of the start-up time is spent importing the large libraries (NumPy, Shapely, wgpu, pygfx…) and waiting for
the graphics probes (``render.gpu``: one child process per graphics API). None of that touches Qt, so a
``Loader`` does it in a background thread while the loading window (``ui.splash``) keeps animating on the main
thread; the window is then built on the main thread, where Qt wants it, and finds everything already loaded.

    Loader(background_steps()).start()        # in a thread: imports, graphics probes
    loader.state()                            # polled by the window: step, label, detail, done, error

Only modules that do not create Qt objects when imported may be loaded here (``core``, ``io``, ``profiles``,
``tools``, ``services``, and the third-party libraries): a Qt object made in this thread would belong to it.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    """One thing the program loads.

    Attributes:
        label: What it is, for the user: "Loading the rendering engine".
        detail: Smaller text under it while it runs: "wgpu · pygfx".
        run: Does the work; may return a new detail to show (e.g. the GPU found), else None.
    """

    label: str
    detail: str
    run: Callable[[], str | None]


@dataclass
class LoadState:
    """Where the loading is, as the window shows it.

    Attributes:
        index: The step running (0-based); ``total`` when every step is done.
        total: Number of steps.
        label: The running step's label.
        detail: The running step's detail (or the last result, such as the GPU found).
        done: True once every step has finished (or one failed).
        error: The exception of a failed step (None if all went well).
        log: (label, seconds) of the steps done, in order.
    """

    index: int = 0
    total: int = 0
    label: str = ""
    detail: str = ""
    done: bool = False
    error: BaseException | None = None
    log: list[tuple[str, float]] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        """Share of the steps done, 0–1 (for a progress bar)."""
        return 1.0 if self.total == 0 else min(self.index / self.total, 1.0)


def import_modules(*names: str) -> Callable[[], None]:
    """A step's work: import these modules (later imports of them are then instant)."""
    def run() -> None:
        for name in names:
            importlib.import_module(name)
    return run


def check_graphics() -> str | None:
    """Wait for the graphics probes, and name the GPU the canvas will most likely use.

    Returns:
        str: E.g. "NVIDIA GeForce RTX 4070 (Vulkan)"; None when a backend is forced (no probes run).
    """
    if os.environ.get("MODALYTICS_WGPU_BACKENDS"):
        return None
    from modalytics.render import gpu
    wanted = os.environ.get("MODALYTICS_GPU", "")
    found, _problems = gpu.find_candidates(wanted)
    ranked = gpu.rank(found, wanted=wanted)
    return ranked[0].label() if ranked else "no GPU found: drawing in software"


def load_processing_steps() -> str:
    """Load the processing steps' scripts once (they import SciPy, scikit-image…): the Process tab, built with the
    window, then finds those libraries loaded.

    Returns:
        str: How many steps were found, e.g. "29 steps".
    """
    from modalytics.processing.registry import ProcessingLibrary
    return f"{len(ProcessingLibrary().specs)} steps"


def find_compute_devices() -> str:
    """List the devices the segmentation models can run on (ONNX Runtime; cached for the whole session).

    Returns:
        str: The best one, e.g. "AMD Radeon RX 7900 XT (DirectML)".
    """
    from modalytics.services import compute
    found = compute.devices()
    return found[0].label() if found else "CPU"


def background_steps() -> list[Step]:
    """The steps done before the window is built (safe outside the main thread, see the module docstring).

    Returns:
        list[Step]: In order: libraries, rendering engine, the program's own parts, the hardware.
    """
    return [
        Step("Loading the numerical libraries", "NumPy · Shapely", import_modules("numpy", "shapely")),
        Step("Loading the rendering engine", "wgpu · pygfx", import_modules("wgpu", "pygfx")),
        Step("Loading the project model", "images · masks · file formats",
             import_modules("modalytics.core.model", "modalytics.io.importers", "modalytics.io.project_file")),
        Step("Loading profiles and tools", "annotation · classification",
             import_modules("modalytics.profiles.library", "modalytics.tools.annotate",
                            "modalytics.services.classes")),
        Step("Loading the processing steps", "filters · features · SciPy · scikit-image", load_processing_steps),
        Step("Finding the compute devices", "ONNX Runtime", find_compute_devices),
        Step("Checking the graphics hardware", "Vulkan · Direct3D 12 · OpenGL", check_graphics),
    ]


class Loader:
    """Runs steps one after the other in a background thread; ``state()`` tells where it is (thread-safe).

    Args:
        steps: The steps to run.
        extra: Steps the caller will run itself afterwards (on the main thread): counted in ``total`` so the
            progress bar does not reach the end before them.
    """

    def __init__(self, steps: list[Step], extra: int = 0) -> None:
        self.steps = steps
        self._lock = threading.Lock()
        self._state = LoadState(total=len(steps) + extra,
                                label=steps[0].label if steps else "", detail=steps[0].detail if steps else "")
        self._thread = threading.Thread(target=self._run, name="startup-loader", daemon=True)

    def start(self) -> "Loader":
        """Start the background thread (returns self, to chain)."""
        self._thread.start()
        return self

    def state(self) -> LoadState:
        """A copy of the current state (safe to read from any thread)."""
        with self._lock:
            s = self._state
            return LoadState(s.index, s.total, s.label, s.detail, s.done, s.error, list(s.log))

    def wait(self, timeout: float | None = None) -> LoadState:
        """Wait for the thread to finish (tests, or a start without a window)."""
        self._thread.join(timeout)
        return self.state()

    def _run(self) -> None:
        # 1. Each step in turn: announce it, run it, log how long it took
        for i, step in enumerate(self.steps):
            with self._lock:
                self._state.index, self._state.label, self._state.detail = i, step.label, step.detail
            t0 = time.perf_counter()
            try:
                result = step.run()
            except BaseException as exc:                 # noqa: BLE001 (reported on the main thread)
                with self._lock:
                    self._state.error, self._state.done = exc, True
                print(f"❌ Start-up step failed: {step.label}: {exc}")
                return
            with self._lock:
                self._state.log.append((step.label, time.perf_counter() - t0))
                if result:
                    self._state.detail = result
        # 2. All done: the index points past the last background step (the caller's steps follow)
        with self._lock:
            self._state.index = len(self.steps)
            self._state.done = True
