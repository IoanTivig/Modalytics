"""
Measuring: run a profile's parameters on objects or images (Qt-free).

    choices = select(library, profile.parameters)        # [(spec, constants)], in library order
    values, problems = measure_object(polygon, image, choices)

One ``ObjectContext`` is built per object and image type and shared by all
their parameters, so expensive intermediates (texture matrix, fluctuation
image) are computed once. A parameter that fails gives NaN and a problem
message; it never stops the others.

Each parameter is measured on the version of the image its type needs
(``ImageNode.as_type``): the image itself, or one of its views (a slide's OD
map, a phase map's dry-mass map: same pixel grid and placement, so the same
outline). Parameters for a type the image has no version of are skipped and
reported (a phase parameter on a slide; an OD parameter on a slide with no OD
view yet).

Several versions of one type can be included (Layers › Processed, each one's
tick): the plain results come from one (the profile's view, else the original,
else the first view), and each other one adds results labelled with its name —
``"<parameter id>@<version>"`` (``version_uid``), shown as "Mean phase · Denoised".
"""

from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Polygon

from modalytics.parameters.classes import ClassIndex
from modalytics.parameters.context import ClassContext, ImageContext, ObjectContext
from modalytics.parameters.registry import ParameterLibrary, ParameterSpec


def select(library: ParameterLibrary, chosen: dict[str, dict], scope: str | None = None
           ) -> tuple[list[tuple[ParameterSpec, dict]], list[str]]:
    """The parameters a profile chose, with their resolved constants.

    Args:
        library: The parameter library.
        chosen: {parameter id: constants} (a profile's ``parameters``).
        scope: Keep only this scope (None = all).

    Returns:
        tuple: ([(spec, constants)] in library order, [ids not in the library]).
    """
    chosen = library.migrate(chosen)                 # ids from before image types still count
    out = [(spec, spec.resolve(chosen[uid])) for uid, spec in library.specs.items()
           if uid in chosen and (scope is None or spec.scope == scope)]
    return out, [uid for uid in chosen if uid not in library.specs]


def objects_of(geometry) -> list[Polygon]:
    """The separate objects of a mask geometry (each polygon of a MultiPolygon)."""
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty else []
    if isinstance(geometry, MultiPolygon):
        return [p for p in geometry.geoms if not p.is_empty]
    return []


def _run(ctx, choices) -> tuple[dict[str, float], dict[str, str]]:
    values, problems = {}, {}
    for spec, constants in choices:
        if not spec.accepts(ctx.image.modality):
            problems[spec.uid] = f"not for {ctx.image.modality} images"
            continue
        ctx.constants = constants
        try:
            v = spec.compute(ctx)
            values[spec.uid] = float(v) if v is not None else math.nan
        except Exception as exc:                        # one bad parameter never stops the others
            values[spec.uid] = math.nan
            problems[spec.uid] = f"{type(exc).__name__}: {exc}"
    return values, problems


VERSION = "@"                                         # "<parameter id>@<version>": a labelled result


def version_uid(uid: str, version: str) -> str:
    """The id of a parameter's result on another included version of the image (e.g. "…:mean@Denoised")."""
    return f"{uid}{VERSION}{version}"


def split_uid(uid: str) -> tuple[str, str]:
    """(parameter id, version: "" for the plain result)."""
    base, _sep, version = uid.partition(VERSION)
    return base, version


def _by_type(image, choices, make_context, extras: bool = True) -> tuple[dict[str, float], dict[str, str]]:
    """Run the choices per image type, each on the version of the image of that type (and, with ``extras``, on
    the other included versions of that type: labelled results). ``make_context(target, version)``."""
    values, problems = {}, {}
    groups: dict[str, list] = {}
    for spec, constants in choices:
        groups.setdefault(spec.image_type, []).append((spec, constants))
    for image_type, group in groups.items():
        target = image.as_type(image_type)
        if target is None:
            from modalytics.core.model import ACQUIRED_TYPES
            why = ("the original is left out of the parameters (Layers › Processed, its tick)"
                   if image_type == image.modality else
                   f"not for {image.modality} images" if image_type in ACQUIRED_TYPES
                   or image.modality not in ACQUIRED_TYPES else           # an imported map has no original
                   f"{image.name} has no {image_type} view (Process › makes one)")
            for spec, _c in group:                       # skipped: a reason, no value
                problems[spec.uid] = why
        else:
            v, p = _run(make_context(target, ""), group)
            values.update(v)
            problems.update(p)
        if not extras or image_type == "*":
            continue
        for version, stand_in in image.extra_versions(image_type):
            v, p = _run(make_context(stand_in, version), group)
            values.update({version_uid(u, version): x for u, x in v.items()})
            problems.update({version_uid(u, version): x for u, x in p.items()})
    return values, problems


def measure_object(polygon: Polygon, image, choices, images=(), classes=None, extras: bool = True
                   ) -> tuple[dict[str, float], dict[str, str]]:
    """Measure one object on one image (and on its views, for parameters of their types).

    Args:
        polygon: Outline in ``image``'s local µm.
        image: ImageNode measured on.
        choices: From ``select`` (object scope).
        images: The project's images, for parameters that look at another one (``ctx.on``).
        extras: Also on the other included versions of a type (labelled results; ``version_uid``). Off where
            only the plain results are read (training, classifying suggestions).

    Returns:
        tuple: ({parameter id: value}, {parameter id: problem}).
    """
    classes = classes or ClassIndex(image, images)             # views have the image's masks too
    return _by_type(image, choices, lambda target, _v: ObjectContext(polygon, target, images=images,
                                                                     classes=classes), extras)


def measure_image(image, choices, images=(), classes=None, extras: bool = True
                  ) -> tuple[dict[str, float], dict[str, str]]:
    """Measure a whole image, and its views (image-scope choices; ``extras`` as in ``measure_object``)."""
    classes = classes or ClassIndex(image, images)
    return _by_type(image, choices, lambda target, _v: ImageContext(target, images=images, classes=classes), extras)


def measure_class(image, label, choices, images=(), classes=None, library=None,
                  object_constants: dict | None = None, value_cache: dict | None = None, extras: bool = True
                  ) -> tuple[dict[str, float], dict[str, str]]:
    """Measure one class of an image as a whole (class-scope choices), e.g. the glands' mean irregularity.

    Args:
        image: ImageNode.
        label: The class, (group, subgroup); subgroup "" = the whole group.
        choices: From ``select`` (class scope).
        images: The project's images (classes registered onto this one).
        classes: A shared ``ClassIndex`` (default: made here).
        library: The parameter library (for summaries of object parameters).
        object_constants: {parameter id: constants} the profile uses for object parameters.
        value_cache: Object values measured already (see ``ClassContext``), shared between class rows (the plain
            results only: another version measures its objects afresh).
        extras: As in ``measure_object``.

    Returns:
        tuple: ({parameter id: value}, {parameter id: problem}).
    """
    classes = classes or ClassIndex(image, images)
    return _by_type(image, choices, lambda target, version: ClassContext(
        target, label, images=images, classes=classes, library=library, object_constants=object_constants,
        value_cache=value_cache if not version else {}), extras)
