"""
Running a registration method on two images (Qt-free).
"""

from __future__ import annotations

import math

import numpy as np

from modalytics.core import geometry as geo
from modalytics.registration.context import Cancelled, RegistrationContext, Search
from modalytics.registration.registry import MethodSpec


def run_method(spec: MethodSpec, moving, reference, search: Search, options: dict | None = None,
               device: str = "auto", profile=None, progress=None, cancelled=None) -> list:
    """Run ``spec`` and return its candidates, best first (distinct positions, at most ``search.top_k``).

    Args:
        spec: The method.
        moving, reference: The image to place and the one it goes on.
        search: The tool's search (rough position, radius, rotations, scales, mirroring, how many).
        options: The method's own option values (missing ones: their defaults).
        device: Where models run.
        profile: The project's profile (its segmentation models).
        progress: Called with 0–1.
        cancelled: Returns True to abort.

    Returns:
        list[services.registration.Candidate]: Ranked (rank 1 = best).

    Raises:
        registration.context.Cancelled: If cancelled.
        ValueError: If the method cannot work on these images (its message says why) or returns nonsense.
    """
    from modalytics.services.segmentation import Cancelled as SegCancelled
    ctx = RegistrationContext(moving, reference, search, device, profile, progress, cancelled)
    try:
        found = spec.register(ctx, spec.resolve(options)) or []
    except (Cancelled, SegCancelled):
        raise Cancelled() from None
    out = []
    mw, mh = moving.size_um
    for c in found:
        if isinstance(c, dict):
            c = ctx.candidate(matrix=c.get("matrix"), score=c.get("score", 0.0), placement=c.get("placement"))
        p = c.placement
        if not all(math.isfinite(v) for v in (p.x_um, p.y_um, p.rotation_deg, p.scale, c.score)) or p.scale <= 0:
            raise ValueError(f"{spec.name} returned an impossible position")
        out.append(c)
    out.sort(key=lambda c: -c.score)
    # One candidate per distinct position (closer than a quarter of the image: the same match)
    kept, centres = [], []
    sep = 0.25 * max(mw, mh)
    for c in out:
        centre = geo.apply(c.placement.local_to_world(), (mw / 2, mh / 2))
        if any(np.hypot(*(centre - o)) < sep for o in centres):
            continue
        centres.append(centre)
        kept.append(c)
        if len(kept) >= search.top_k:
            break
    for i, c in enumerate(kept):
        c.rank = i + 1
    return kept
