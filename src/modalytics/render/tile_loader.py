"""
Background tile loading with "latest view wins" priority.

A plain thread-pool queue is a poor fit for panning: requests for tiles that
already left the screen pile up and delay the ones in view. Here the canvas
calls ``set_wanted`` every frame with the full list of missing tiles
(centre first); the list *replaces* the queue, so stale requests disappear.
Tiles already being read are finished and cached anyway.

Workers decode in parallel. Blosc decompression and NumPy copies release the
GIL, so threads give real parallelism here without multiprocessing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Hashable

import numpy as np


class TileLoader:
    """Prioritised background reader.

    Args:
        workers: Number of reader threads.
        on_ready: Called (from a worker thread) after each tile is ready;
            the canvas uses it to schedule a redraw.
    """

    def __init__(self, workers: int = 6, on_ready: Callable[[], None] | None = None) -> None:
        self._cond = threading.Condition()
        self._queue: list[tuple[Hashable, Callable[[], np.ndarray]]] = []
        self._inflight: set[Hashable] = set()
        self._done: deque[tuple[Hashable, np.ndarray | None]] = deque()
        self._stop = False
        self.on_ready = on_ready
        self.read_ms: deque[float] = deque(maxlen=2000)   # per-tile read+decode time (diagnostics)
        self.reads = 0
        self._threads = [threading.Thread(target=self._run, name=f"tile-loader-{i}", daemon=True)
                         for i in range(workers)]
        for t in self._threads:
            t.start()

    def set_wanted(self, requests: list[tuple[Hashable, Callable[[], np.ndarray]]]) -> None:
        """Replace the queue with these (key, read_function) requests, in priority order."""
        with self._cond:
            self._queue = [r for r in requests if r[0] not in self._inflight]
            self._cond.notify_all()

    def is_pending(self, key: Hashable) -> bool:
        """True if the tile is queued or being read."""
        with self._cond:
            return key in self._inflight or any(k == key for k, _ in self._queue)

    @property
    def busy(self) -> int:
        """Number of queued plus in-flight tiles."""
        with self._cond:
            return len(self._queue) + len(self._inflight)

    def take_ready(self) -> list[tuple[Hashable, np.ndarray | None]]:
        """Return and clear the tiles finished since the last call (main thread)."""
        with self._cond:
            out = list(self._done)
            self._done.clear()
            return out

    def shutdown(self) -> None:
        """Stop the workers (pending requests are dropped)."""
        with self._cond:
            self._stop = True
            self._queue.clear()
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            # 1. Wait for work
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                key, read = self._queue.pop(0)
                self._inflight.add(key)
            # 2. Read outside the lock
            t0 = time.perf_counter()
            try:
                data = read()
            except Exception as exc:  # a bad tile must not kill the worker
                print(f"⚠️ tile {key} failed: {exc}")
                data = None
            self.read_ms.append((time.perf_counter() - t0) * 1e3)
            self.reads += 1
            # 3. Hand over to the main thread
            with self._cond:
                self._inflight.discard(key)
                self._done.append((key, data))
            if self.on_ready is not None:
                self.on_ready()
