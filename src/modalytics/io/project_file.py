"""
Modalytics project file: ``name.modalytics`` = one SQLite database.

Why one file: a project folder is easy to open by mistake and break; a
single file is not. SQLite is a well-tested single-file container that is
readable from any language.

Saving is atomic: the whole project is written to ``name.modalytics.tmp``
and then swapped in with one ``os.replace``, so a crash or power cut during a
save leaves the previous version intact.

What goes inside:
    meta        format version, label schema, view, creation time, and the
                project's own copy of its profile (so it reopens with exactly
                the analysis it was made with, whatever happens to the library)
    images      calibration, placement, display, source reference; small
                in-memory images (e.g. phase maps) are stored as .npy blobs
    masks       class, colour, visibility, geometry as WKB (exact vectors)
    changelog   who saved what and when (kept across saves)

Whole-slide pyramids are *referenced* (absolute + project-relative path),
never copied: they are tens of GB. That covers zarr folders and slide files
read in place (SVS, pyramidal TIFF: ``io.tiff_source``). A slide file's size and
a hash of its first MiB are kept with it (``file_fingerprint`` in its
metadata); reopening warns if the file at that path is no longer the same one,
so masks are never shown on the wrong slide without a word.

A referenced file that is not found when the project opens (a disconnected
drive, a moved folder) does not drop the image: it opens as a placeholder
(``io.sources.MissingSource``) with its size (kept in its metadata as
``source_shape``), placement and masks, and saving writes its reference back
unchanged. Properties › Relink… points it at the file again.

An image's views (processed versions made by processing chains: its OD map…)
are saved as their recipes (the chain, display, which one is shown, whether it
is the profile's standard one) in its metadata (``views``), never as pixels:
they are made again when the project opens (``processing.run``). A chain whose
steps are not available on this computer is reported, and its recipe is kept so
saving does not lose it.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import io
import json
import os
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import shapely

from modalytics import __version__, fileio
from modalytics.core.annotations import LabelSchema
from modalytics.core.geometry import Placement
from modalytics.core.model import Display, ImageNode, MaskNode, Suggestion, SuggestionNode
from modalytics.io.sources import ArraySource, MissingSource, is_missing
from modalytics.io.tiff_source import TiffPyramidSource, fingerprint
from modalytics.io.zarr_source import ZarrPyramidSource
from modalytics.processing.registry import ProcessingLibrary
from modalytics.processing.run import recipes, restore_views

EXTENSION = ".modalytics"
FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE images (
    id TEXT PRIMARY KEY, ord INTEGER, name TEXT, modality TEXT,
    source_kind TEXT, source_path TEXT, source_rel TEXT, data BLOB,
    pixel_size_x REAL, pixel_size_y REAL, placement TEXT, display TEXT,
    visible INTEGER, opacity REAL, metadata TEXT);
CREATE TABLE masks (
    id TEXT PRIMARY KEY, image_id TEXT, ord INTEGER, name TEXT, grp TEXT, subgroup TEXT,
    color TEXT, visible INTEGER, opacity REAL, geometry BLOB);
CREATE TABLE changelog (time TEXT, user TEXT, action TEXT);
CREATE TABLE results (scope TEXT, ord INTEGER, data TEXT);
CREATE TABLE suggestions (image_id TEXT PRIMARY KEY, data TEXT);
"""


def save_project(project, path: str, view: dict | None = None) -> None:
    """Write a project file atomically.

    Args:
        project: The Project.
        path: Target file (``.modalytics`` is appended if missing).
        view: Optional view state to restore on open ({"cx", "cy", "um_per_px"}).

    Raises:
        ValueError: If an image source type cannot be stored.
    """
    if not path.endswith(EXTENSION):
        path += EXTENSION
    path = os.path.abspath(path)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    history = _read_changelog(path) if os.path.exists(path) else []
    now = _dt.datetime.now().isoformat(timespec="seconds")

    con = sqlite3.connect(tmp)
    try:
        con.executescript(_SCHEMA)
        # 1. Meta
        meta = {"format": "modalytics-project", "format_version": str(FORMAT_VERSION),
                "app_version": __version__, "labels": json.dumps(project.labels.to_dict()),
                "view": json.dumps(view or {}), "saved": now,
                "profile": json.dumps(project.profile.to_dict()) if project.profile is not None else "",
                "profile_id": project.profile_id or ""}
        results = getattr(project, "results", None)
        if results is not None:                     # the latest measurement: its description, then its rows
            info = results.to_dict()
            rows = info.pop("rows")
            meta["results"] = json.dumps(info)
        con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
        if results is not None:
            con.executemany("INSERT INTO results VALUES (?, ?, ?)",
                            [(r["scope"], i, json.dumps(r)) for i, r in enumerate(rows)])
        # 2. Images and their masks
        n_masks = 0
        for order, img in enumerate(project.images):
            kind, spath, rel, blob = _source_record(img, os.path.dirname(path))
            img.metadata["source_shape"] = list(img.source.shape)    # so a missing file keeps its size
            meta = dict(img.metadata)
            views = recipes(img) + meta.pop("views_not_available", [])   # recipes, never pixels
            if views:
                meta["views"] = views
            con.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                img.id, order, img.name, img.modality, kind, spath, rel, blob,
                img.pixel_size_um[0], img.pixel_size_um[1], json.dumps(img.placement.to_dict()),
                json.dumps({"clim": list(img.display.clim), "colormap": img.display.colormap}),
                int(img.visible), img.opacity, json.dumps(_jsonable(meta))))
            for morder, m in enumerate(img.masks):
                con.execute("INSERT INTO masks VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    m.id, img.id, morder, m.name, m.group, m.subgroup, json.dumps(list(m.color)),
                    int(m.visible), m.opacity, shapely.to_wkb(m.geometry)))
                n_masks += 1
            sugg = img.suggestions                      # the models' suggestions waiting for review
            if sugg is not None:
                con.execute("INSERT INTO suggestions VALUES (?, ?)", (img.id, json.dumps(
                    {"id": sugg.id, "visible": sugg.visible, "source": sugg.source, "hidden": sugg.hidden_groups,
                     "items": [it.to_dict() for it in sugg.items]})))
        # 3. Change log (previous entries + this save)
        history.append((now, getpass.getuser(), f"saved {len(project.images)} images, {n_masks} masks"))
        con.executemany("INSERT INTO changelog VALUES (?, ?, ?)", history)
        con.commit()
    finally:
        con.close()
    fileio.replace(tmp, path)      # atomic swap: the old file stays valid until this point
    project.path = path


@dataclass
class ProjectData:
    """What ``load_project`` reads.

    Attributes:
        images: Bottom → top, with their masks attached.
        labels: Annotation classes.
        view: View state to restore ({"cx", "cy", "um_per_px"}, may be empty).
        warnings: Problems the user should see (e.g. missing image files).
        profile: The project's copy of its profile, or None (older files).
        profile_id: Library id of the profile it was made from ("" if unknown).
        results: The latest measurement saved with it (``parameters.results.Results``), or None.
    """

    images: list
    labels: LabelSchema
    view: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    profile: object = None
    profile_id: str = ""
    results: object = None


def load_project(path: str) -> ProjectData:
    """Read a project file.

    Args:
        path: ``.modalytics`` file.

    Returns:
        ProjectData

    Raises:
        ValueError: If the file is not a Modalytics project.
    """
    con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
    warnings: list[str] = []
    try:
        meta = dict(con.execute("SELECT key, value FROM meta"))
        if meta.get("format") != "modalytics-project":
            raise ValueError(f"{path} is not a Modalytics project")
        labels = LabelSchema.from_dict(json.loads(meta.get("labels", "{}")))
        view = json.loads(meta.get("view") or "{}")
        profile = None
        if meta.get("profile"):
            from modalytics.profiles.model import Profile, ProfileError
            try:
                profile = Profile.from_dict(json.loads(meta["profile"]))
            except (ProfileError, ValueError) as exc:
                warnings.append(f"The project's profile could not be read ({exc})")
        images, by_id, library = [], {}, None
        rows = con.execute("SELECT id, name, modality, source_kind, source_path, source_rel, data, pixel_size_x, "
                           "pixel_size_y, placement, display, visible, opacity, metadata FROM images ORDER BY ord")
        for (iid, name, modality, kind, spath, rel, blob, psx, psy, placement, display, visible,
             opacity, metadata) in rows:
            source = _open_source(kind, spath, rel, blob, os.path.dirname(os.path.abspath(path)))
            img_meta = json.loads(metadata or "{}")
            if source is None:
                # 1. Not found: a placeholder with its size, so the image and its masks are kept
                shape = img_meta.get("source_shape") or _shape_from_masks(con, iid, psx, psy)
                source = MissingSource(kind, spath, rel, shape)
                warnings.append(f"{name}: file not found ({spath}). It is kept with its masks as a placeholder; "
                                f"Properties › Relink… points it at the file again.")
            saved_views = img_meta.pop("views", None)
            img = ImageNode(name, source, (psx, psy), modality, Placement.from_dict(json.loads(placement)),
                            img_meta)
            known = img.metadata.get("file_fingerprint")
            if known and kind == "tiff" and not is_missing(img) and fingerprint(source.path) != known:
                warnings.append(f"{name}: {source.path} is not the file this project was made with (its size or "
                                f"contents differ). Check that the masks still fit the slide.")
            img.id = iid
            d = json.loads(display)
            img.display = Display(clim=tuple(d["clim"]), colormap=d["colormap"])
            img.visible, img.opacity = bool(visible), float(opacity)
            if saved_views:                                    # 2. Its views, made again from their recipes
                library = library or ProcessingLibrary()            # read the steps once per project
                warnings += restore_views(img, saved_views, library)   # unavailable ones are kept for saving
            images.append(img)
            by_id[iid] = img
        for (mid, image_id, name, grp, sub, color, visible, opacity, wkb) in con.execute(
                "SELECT id, image_id, name, grp, subgroup, color, visible, opacity, geometry FROM masks "
                "ORDER BY image_id, ord"):
            owner = by_id.get(image_id)
            if owner is None:
                continue
            m = MaskNode(name, shapely.from_wkb(wkb), tuple(json.loads(color)), grp, sub)
            m.id, m.visible, m.opacity = mid, bool(visible), float(opacity)
            m.parent = owner
            owner.children.append(m)
        has_suggestions = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'suggestions'").fetchone()
        for (image_id, data) in (con.execute("SELECT image_id, data FROM suggestions") if has_suggestions else []):
            owner = by_id.get(image_id)
            if owner is None:
                continue
            d = json.loads(data)
            node = SuggestionNode([Suggestion.from_dict(x) for x in d.get("items", [])], d.get("source", {}))
            node.id, node.visible = d.get("id", node.id), bool(d.get("visible", True))
            node.hidden_groups = [str(g) for g in d.get("hidden", [])]
            node.parent = owner
            owner.children.append(node)
        results = None
        if meta.get("results"):                       # files from before measuring have none
            from modalytics.parameters.results import Results
            try:
                info = json.loads(meta["results"])
                info["rows"] = [json.loads(d) for (d,) in con.execute("SELECT data FROM results ORDER BY ord")]
                results = Results.from_dict(info)
            except (ValueError, KeyError, sqlite3.Error) as exc:
                warnings.append(f"The saved results could not be read ({exc}); measure again")
    finally:
        con.close()
    return ProjectData(images, labels, view, warnings, profile, meta.get("profile_id", ""), results)


def read_changelog(path: str) -> list[tuple[str, str, str]]:
    """(time, user, action) entries of a project file, oldest first."""
    return _read_changelog(path)


# ---------------------------------------------------------------- helpers
def _read_changelog(path: str) -> list[tuple[str, str, str]]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return list(con.execute("SELECT time, user, action FROM changelog"))
        finally:
            con.close()
    except sqlite3.Error:
        return []


def _source_record(img: ImageNode, project_dir: str):
    """(kind, absolute path, relative path, blob) describing how to reopen an image."""
    src = img.source
    if isinstance(src, MissingSource):                     # not found when opened: its reference, unchanged
        return src.kind, src.path, src.rel, None
    if isinstance(src, (ZarrPyramidSource, TiffPyramidSource)):
        try:
            rel = os.path.relpath(src.path, project_dir)
        except ValueError:            # different drive on Windows
            rel = None
        return ("zarr" if isinstance(src, ZarrPyramidSource) else "tiff"), src.path, rel, None
    if isinstance(src, ArraySource):
        buf = io.BytesIO()
        np.save(buf, src.data, allow_pickle=False)
        return "array", None, None, buf.getvalue()
    raise ValueError(f"Cannot store image source of type {type(src).__name__}")


def _open_source(kind, spath, rel, blob, project_dir):
    """Re-open an image source; tries the project-relative path first, then the absolute one."""
    if kind == "array":
        return ArraySource(np.load(io.BytesIO(blob), allow_pickle=False))
    candidates = ([os.path.normpath(os.path.join(project_dir, rel))] if rel else []) + [spath]
    if kind == "zarr":
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return ZarrPyramidSource(candidate)
    if kind == "tiff":
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return TiffPyramidSource(candidate)
    return None


def _shape_from_masks(con, image_id, psx: float, psy: float) -> tuple[int, int]:
    """A size for a missing image saved before sizes were kept: its masks' extent (else 1000 px)."""
    w = h = 0.0
    for (wkb,) in con.execute("SELECT geometry FROM masks WHERE image_id = ?", (image_id,)):
        g = shapely.from_wkb(wkb)
        if not g.is_empty:
            w, h = max(w, g.bounds[2]), max(h, g.bounds[3])
    return (max(1000, int(h / psy) + 1), max(1000, int(w / psx) + 1))


def _jsonable(d: dict) -> dict:
    """Keep only JSON-serialisable metadata values."""
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out
