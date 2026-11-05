"""
Sharing a profile as one file: ``name.mprofile``.

A profile on its own only *names* its models, parameters and processing steps;
another computer may not have them. Export therefore packs everything the
profile needs into one zip:

    manifest.json        what is inside: the profile, each file, the libraries the
                         scripts need (name and version here), the Modalytics version
    profile.yaml         the profile itself (its classifiers are written in it)
    models/<id>.mmodel   every segmentation model it uses that is installed here — with
                         its training weights (so it can be fine-tuned), unless left out
    parameters/<f>.py    every parameter it measures that is not built in
    steps/<f>.py         every processing step its chains use that is not built in
    importers/<f>.py     your importers for the image types it accepts (so the other lab
                         can read the same instrument's files)
    scripts/<f>.py       the external scripts it offers (Scripts menu) that are not built in

**Before importing**, ``inspect`` says what an import would do — new items, items
already here (identical), items that differ from yours (added under a new name),
what the profile names but neither the bundle nor this computer has, the libraries
the scripts need that are not installed here — so the user decides with all of it
in view.

**Importing** unpacks it into the user library. Nothing already there is ever
overwritten:

* an identical model, parameter, step or importer (same bytes) is simply reused;
* a different one with the same name is added under a new name (``…_2``) and the
  imported profile is pointed at it, so both analyses keep working;
* the profile is added as a new one ("Name (2)" if the name is taken), or — when the
  user chooses — replaces the profile of the same name (e.g. a newer release of it;
  projects keep their own copy, so they are not changed).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from modalytics.profiles.library import ProfileLibrary
from modalytics.profiles.model import Profile, ProfileError

EXTENSION = ".mprofile"
BUNDLE_FORMAT = "modalytics-profile-bundle"
WEIGHTS = "weights.safetensors"
_KEY_LINE = re.compile(r"""^(KEY\s*=\s*)(['"])(.+?)\2""", re.MULTILINE)
KINDS = ("models", "parameters", "steps", "importers", "scripts")


@dataclass
class ImportResult:
    """What an import did.

    Attributes:
        profile_id: Library id of the imported profile.
        name: Its name (may have gained " (2)").
        added: Models, parameters, steps and importers copied into the library.
        renamed: {old id: new id} for items added under a new name.
        reused: Items that were already here, identical.
        warnings: Anything the user should know (e.g. items missing from the bundle).
        requires: Libraries the bundled scripts need that are not installed here.
        replaced: True when it replaced the profile of the same name.
    """

    profile_id: str
    name: str
    added: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    replaced: bool = False


@dataclass
class Plan:
    """What importing a bundle would do (``inspect``); nothing is changed.

    Attributes:
        name: The profile's name.
        existing: Library id of a profile of the same name here ("" = none).
        items: [(kind, id, status)] — status "new", "same" (identical, reused) or "different" (yours differs: the
            bundle's is added under a new name).
        missing: What the profile names that is neither in the bundle nor here (it would not work fully).
        requires: {library: version in the bundle's computer} the scripts need that are not installed here.
        weights: Whether the models carry their training weights.
        app_version: The Modalytics version that exported it.
        created: When.
        size_mb: The bundle's size.
    """

    name: str
    existing: str = ""
    items: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    requires: dict = field(default_factory=dict)
    weights: bool = True
    app_version: str = ""
    created: str = ""
    size_mb: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for _k, _i, s in self.items if s == status)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def steps_used(profile: Profile) -> set[str]:
    """The processing step ids the profile's chains use."""
    return {s.get("step", "") for chains in profile.processing.values() for chain in chains
            for s in (chain.get("steps", []) if isinstance(chain, dict) else []) if s.get("step")}


def without_weights(data: bytes) -> bytes:
    """A model package without its training weights (the analysis edition does not need them)."""
    src = zipfile.ZipFile(io.BytesIO(data))
    if WEIGHTS not in src.namelist():
        return data
    out = io.BytesIO()
    with src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in src.namelist():
            if name != WEIGHTS:
                z.writestr(name, src.read(name))
    return out.getvalue()


def _restricted_weights(data: bytes) -> str:
    """The licence of the encoder weights a model package started from, when they may not be shared freely."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            src = json.loads(z.read("manifest.json")).get("source", {})
    except (KeyError, ValueError, zipfile.BadZipFile):
        return ""
    w = src.get("encoder_weights") or {}
    return "" if w.get("redistributable", True) else f"{w.get('name', '?')}: {w.get('licence') or 'not stated'}"


def export_profile(profile: Profile, dest: str | Path, models: dict[str, Path],
                   parameters: dict[str, Path], importers: dict[str, Path] | None = None,
                   steps: dict[str, Path] | None = None, requires: dict[str, str] | None = None,
                   weights: bool = True, app_version: str = "", scripts: dict[str, Path] | None = None) -> list[str]:
    """Write a ``.mprofile`` bundle.

    Args:
        profile: The profile to share.
        dest: Target file (``.mprofile`` is appended if missing).
        models: Installed models, {id: package file}.
        parameters: Parameters to include, {parameter id: script file}; built-in
            parameters are left out by the caller (every installation has them).
        importers: Your importers to include, {importer id: script file} (the caller
            picks the ones for the profile's image types; built-ins are left out).
        steps: Your processing steps to include, {step id: script file} (the caller picks those the profile's
            chains use; built-ins are left out).
        requires: {library: version here} the included scripts need.
        weights: Keep the models' training weights (others can fine-tune them; about twice the size).
        app_version: This Modalytics version (recorded).
        scripts: Your external scripts the profile offers, {script id: file} (built-ins are left out by the caller).

    Returns:
        list[str]: Warnings (models the profile names that are not installed here).
    """
    dest = Path(dest)
    if dest.suffix.lower() != EXTENSION:
        dest = dest.with_name(dest.name + EXTENSION)
    warnings = []
    manifest = {"format": BUNDLE_FORMAT, "format_version": 2, "profile": profile.name,
                "profile_revision": profile.revision, "app_version": app_version,
                "created": dt.datetime.now().isoformat(timespec="seconds"), "weights": bool(weights),
                "models": {}, "parameters": {}, "steps": {}, "importers": {}, "scripts": {},
                "requires": dict(requires or {})}
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("profile.yaml", profile.to_yaml())
        # 1. Models (already compressed: stored as they are), with or without their training weights
        for mid in sorted(profile.models()):
            if mid in models:
                data = Path(models[mid]).read_bytes()
                z.writestr(f"models/{mid}.mmodel", data if weights else without_weights(data),
                           compress_type=zipfile.ZIP_STORED)
                manifest["models"][mid] = f"models/{mid}.mmodel"
                restricted = _restricted_weights(data)
                if restricted:
                    warnings.append(f"Model '{mid}' was trained from encoder weights with a restricted licence "
                                    f"({restricted}): whoever receives it is bound by that licence")
            else:
                warnings.append(f"Model '{mid}' is not installed here, so it is not in the bundle")
        # 2. Parameters that are not built in
        for uid in profile.parameters:
            if uid in parameters:
                name = f"parameters/{_parameter_file(uid)}"
                z.write(parameters[uid], name)
                manifest["parameters"][uid] = name
        # 3. Processing steps of its chains that are not built in
        for uid in sorted(steps_used(profile)):
            if uid in (steps or {}):
                name = f"steps/{_parameter_file(uid)}"
                z.write(steps[uid], name)
                manifest["steps"][uid] = name
        # 4. Your importers
        for uid, path in sorted((importers or {}).items()):
            name = f"importers/{_parameter_file(uid)}"
            z.write(path, name)
            manifest["importers"][uid] = name
        # 5. Your external scripts it offers
        for uid in profile.scripts:
            if uid in (scripts or {}):
                name = f"scripts/{uid}.py"
                z.write(scripts[uid], name)
                manifest["scripts"][uid] = name
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    tmp.replace(dest)
    return warnings


def _open(src):
    try:
        z = zipfile.ZipFile(src)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProfileError(f"{Path(src).name} is not a profile bundle ({exc})") from exc
    try:
        manifest = json.loads(z.read("manifest.json"))
        profile = Profile.from_yaml(z.read("profile.yaml").decode("utf-8"))
    except KeyError as exc:
        z.close()
        raise ProfileError(f"{Path(src).name} is not a profile bundle (missing {exc})") from exc
    if manifest.get("format") != BUNDLE_FORMAT:
        z.close()
        raise ProfileError(f"{Path(src).name} is not a profile bundle")
    return z, manifest, profile


def inspect(src: str | Path, library: ProfileLibrary, known: dict[str, dict[str, Path]],
            builtin: dict[str, set] | None = None, available_libraries: set[str] | None = None) -> Plan:
    """What importing a bundle would do (see the module docstring); nothing is changed.

    Args:
        src: The bundle.
        library: The profile library here.
        known: {kind: {id: file}} of what is here, kind in ``KINDS``.
        builtin: {kind: ids} that every installation has (parameters, steps) — not reported as missing.
        available_libraries: Canonical names of the libraries importable here (default: ask ``parameters.libraries``).

    Raises:
        ProfileError: If the file is not a profile bundle.
    """
    z, manifest, profile = _open(src)
    with z:
        plan = Plan(profile.name, library.find(profile.name) or "", weights=bool(manifest.get("weights", True)),
                    app_version=str(manifest.get("app_version", "")), created=str(manifest.get("created", "")),
                    size_mb=Path(src).stat().st_size / 1e6)
        for kind in KINDS:
            here = known.get(kind, {})
            for uid, member in manifest.get(kind, {}).items():
                if uid not in here:
                    status = "new"
                else:
                    status = "same" if _digest(Path(here[uid]).read_bytes()) == _digest(z.read(member)) else "different"
                plan.items.append((kind, uid, status))
    bundled = {(k, u) for k, u, _s in plan.items}
    builtin = builtin or {}
    for kind, ids in (("models", profile.models()), ("parameters", profile.parameters), ("steps", steps_used(profile)),
                      ("scripts", profile.scripts)):
        for uid in ids:
            if (kind, uid) not in bundled and uid not in known.get(kind, {}) and uid not in builtin.get(kind, set()):
                plan.missing.append(f"{kind[:-1]} '{uid}'")
    from modalytics.parameters import libraries
    have = available_libraries if available_libraries is not None else libraries.available()
    plan.requires = {n: v for n, v in manifest.get("requires", {}).items() if libraries.canonical(n) not in have}
    return plan


def import_profile(src: str | Path, library: ProfileLibrary, models_dir: Path, parameters_dir: Path,
                   known_models: dict[str, Path], known_parameters: dict[str, Path],
                   importers_dir: Path | None = None, known_importers: dict[str, Path] | None = None,
                   steps_dir: Path | None = None, known_steps: dict[str, Path] | None = None,
                   replace: str = "", available_libraries: set[str] | None = None,
                   scripts_dir: Path | None = None, known_scripts: dict[str, Path] | None = None) -> ImportResult:
    """Unpack a ``.mprofile`` bundle into the library (see the module docstring).

    Args:
        src: The bundle.
        library: Profile library to add the profile to.
        models_dir: Where new models go (the user library's models folder).
        parameters_dir: Where new parameter scripts go.
        known_models: Models installed here, {id: file} (built-in and user).
        known_parameters: Parameters in the library here, {parameter id: file}.
        importers_dir: Where new importer scripts go (None: bundled importers are skipped).
        known_importers: Importers here, {importer id: file}.
        steps_dir: Where new processing steps go (None: bundled steps are skipped).
        known_steps: Processing steps here, {step id: file}.
        replace: Library id of a profile to replace with the imported one ("" = add it as a new profile).
        scripts_dir: Where new external scripts go (None: bundled scripts are skipped).
        known_scripts: External scripts here, {script id: file}.
        available_libraries: As for ``inspect`` (the result lists the libraries still missing).

    Returns:
        ImportResult

    Raises:
        ProfileError: If the file is not a profile bundle.
    """
    z, manifest, profile = _open(src)
    with z:
        result = ImportResult("", profile.name)

        # 1. Models
        for mid, member in manifest.get("models", {}).items():
            data = z.read(member)
            new_id = _place(mid, data, known_models, lambda i: models_dir / f"{i}.mmodel", result)
            if new_id != mid:
                profile.segmentation = {g: {t: (new_id if m == mid else m) for t, m in per.items()}
                                        for g, per in profile.segmentation.items()}
                profile.classification = {g: (new_id if c == mid else c) for g, c in profile.classification.items()}
        missing = [m for m in profile.models() if m not in known_models and m not in result.added]
        result.warnings += [f"Model '{m}' is neither in the bundle nor installed here" for m in missing]

        # 2. Parameters (renaming one means changing the KEY line inside its script, and the classifiers reading it)
        for uid, member in manifest.get("parameters", {}).items():
            data = z.read(member)
            new_uid = _place(uid, data, known_parameters, lambda u: parameters_dir / _parameter_file(u), result,
                             rewrite=_rekey)
            if new_uid != uid:
                profile.parameters = {(new_uid if p == uid else p): s for p, s in profile.parameters.items()}
                profile.classification = {g: _renamed_in_classifier(c, uid, new_uid)
                                          for g, c in profile.classification.items()}

        # 3. Processing steps (renaming one changes the chains that use it)
        if steps_dir is not None:
            known = known_steps if known_steps is not None else {}
            for uid, member in manifest.get("steps", {}).items():
                new_uid = _place(uid, z.read(member), known, lambda u: steps_dir / _parameter_file(u), result,
                                 rewrite=_rekey)
                if new_uid != uid:
                    for chains in profile.processing.values():
                        for chain in chains:
                            for s in chain.get("steps", []) if isinstance(chain, dict) else []:
                                if s.get("step") == uid:
                                    s["step"] = new_uid

        # 4. Importers (nothing in the profile names them: a renamed one is simply a second importer)
        if importers_dir is not None:
            known = dict(known_importers or {})
            for uid, member in manifest.get("importers", {}).items():
                _place(uid, z.read(member), known, lambda u: importers_dir / _parameter_file(u), result,
                       rewrite=_rekey)

        # 5. External scripts (renaming one changes the list of scripts the profile offers)
        if scripts_dir is not None:
            known = known_scripts if known_scripts is not None else {}
            for uid, member in manifest.get("scripts", {}).items():
                new_uid = _place(uid, z.read(member), known, lambda u: scripts_dir / f"{u}.py", result,
                                 rewrite=_rekey)
                if new_uid != uid:
                    profile.scripts = [new_uid if x == uid else x for x in profile.scripts]

    # 6. Libraries the scripts need that are not here (the caller offers to install them)
    from modalytics.parameters import libraries
    have = available_libraries if available_libraries is not None else libraries.available()
    result.requires = [n for n in manifest.get("requires", {}) if libraries.canonical(n) not in have]

    # 7. The profile itself: a new one (never over an existing one), or replacing the one chosen
    if replace:
        old = library.get(replace)
        profile.name = old.name
        profile.revision = old.revision
        result.profile_id = library.save(profile, replace)
        result.replaced = True
    else:
        profile.name = library.unique_name(profile.name)
        result.profile_id = library.save(profile)
    result.name = profile.name
    return result


def _rekey(data: bytes, new_id: str) -> bytes:
    """A script's ``KEY = "…"`` line set to the new id's key."""
    return _KEY_LINE.sub(lambda m: m.group(1) + m.group(2) + new_id.rsplit(":", 1)[-1] + m.group(2),
                         data.decode("utf-8"), count=1).encode("utf-8")


def _renamed_in_classifier(c, old: str, new: str):
    """A classifier (dict) with a parameter id renamed wherever it reads it."""
    if not isinstance(c, dict):
        return c
    text = json.dumps(c)
    return json.loads(text.replace(json.dumps(old), json.dumps(new)))


def _parameter_file(uid: str) -> str:
    """Where a script of this id lives in its folder: ``<scope>/<type>/<key>.py`` (steps, importers: ``<type>/<key>.py``)."""
    parts = uid.split(":")
    return "/".join(parts[:-1] + [parts[-1] + ".py"])


def _place(item_id: str, data: bytes, known: dict[str, Path], target, result: ImportResult, rewrite=None) -> str:
    """Put one bundled file into the library without overwriting anything.

    Args:
        item_id: Its id in the bundle.
        data: Its bytes.
        known: Items already here, {id: file}.
        target: id → path where a new item of that id is written.
        result: Collects what happened.
        rewrite: (bytes, new id) → bytes, for items whose id is written inside them.

    Returns:
        str: The id it has here (unchanged, or a new ``…_2`` id).
    """
    if item_id in known:
        if _digest(Path(known[item_id]).read_bytes()) == _digest(data):
            result.reused.append(item_id)
            return item_id
        n = 2
        while f"{item_id}_{n}" in known or Path(target(f"{item_id}_{n}")).exists():
            n += 1
        new_id = f"{item_id}_{n}"
        data = rewrite(data, new_id) if rewrite else data
        result.renamed[item_id] = new_id
    else:
        new_id = item_id
        result.added.append(item_id)
    path = Path(target(new_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    known[new_id] = path
    return new_id
