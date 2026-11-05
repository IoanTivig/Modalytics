"""
Profiles: everything one kind of analysis needs, in one editable file.

A profile says what is analysed and how:

    kind            tissue | cells | kinetics: where it appears on the start screen
    images          the image types it accepts (a cell profile has no use for slides)
    classes         groups and their subgroups, with colours; they become the
                    project's annotation classes (the group is what is segmented,
                    the subgroup what each object is classified as)
    aliases         per group, older subgroup names and the subgroup they now mean
                    (e.g. Normal → Low grade dysplasia), so annotations made with an
                    earlier version of the profile still count
    segmentation    per group, the model that segments it on each image type
    processing      per image type, the processing chains applied at import (e.g. a slide's OD
                    map): views of the image; measured ones are the standard results
    parameters      the parameters of the library it measures, with the values of their constants
    classification  per group, the classifier that assigns its subgroups: a Ranges or Score
                    classifier written in the profile (``training.classifiers``), made in
                    Training › Build, or the id of a trained model
    scripts         the external scripts its projects offer in the Scripts menu (ids; Settings ›
                    Scripts) — tasks beside the analysis, e.g. gathering an instrument's files
    report          what the report shows: its key parameters (``parameters``: ids, in order),
                    summarised per class and plotted; the full tables stay in Results / CSV

On disk a profile is a YAML file meant to be readable and editable by hand
(see ``resources/profiles/colonic_tissue.yaml``)::

    name: Colonic tissue
    kind: tissue
    images: [wsi, phase]
    classes:
      Gland:
        color: '#33cc66'
        subgroups: {Normal: '#33cc66', Dysplasia low grade: '#facc26'}
    segmentation:
      Gland: {wsi: wsi_effnetb3_L00_gland_01, phase: unet_efficientnet_b0__imagenet_micronet}
    processing:
      wsi:
      - name: OD map
        steps:
        - step: od:od_map
          options: {background: Estimated from the image}
        measure: true
    parameters:
      object:any:circularity: {}
      object:phase:disorder_strength: {degree: 2, window: 3, refractive_index: 1.38}

Models are named by their package file name without ``.mmodel``; parameters
by their library id ``<scope>:<image type>:<key>``. This module is Qt-free and knows
nothing about where profiles are stored (``profiles.library``) or how they
travel (``profiles.bundle``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import yaml

from modalytics.core.annotations import Group, Label, LabelSchema
from modalytics.core.model import IMAGE_TYPES

FORMAT = "modalytics-profile"
FORMAT_VERSION = 1
MIN_CONFIDENCE = 0.80        # a classifier's grade is suggested from this confidence (``Profile.confidence_for``)
KINDS = {"tissue": "Tissue", "cells": "Cells", "kinetics": "Cell kinetics"}


class ProfileError(ValueError):
    """A profile file that cannot be used (the message says why)."""


## Colours: '#rrggbb' in files, RGBA floats in the program ##

def hex_to_rgba(text: str) -> tuple[float, float, float, float]:
    """'#33cc66' (or '#33cc66ff') → (0.2, 0.8, 0.4, 1.0)."""
    t = str(text).strip().lstrip("#")
    if len(t) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in t):
        raise ProfileError(f"'{text}' is not a colour (expected '#rrggbb')")
    vals = [int(t[i:i + 2], 16) / 255 for i in range(0, len(t), 2)]
    return tuple(round(v, 4) for v in vals) + ((1.0,) if len(vals) == 3 else ())


def rgba_to_hex(rgba) -> str:
    """(0.2, 0.8, 0.4, 1.0) → '#33cc66' (alpha is written only when not opaque)."""
    vals = [max(0, min(255, round(float(v) * 255))) for v in rgba]
    return "#" + "".join(f"{v:02x}" for v in (vals if len(vals) == 4 and vals[3] != 255 else vals[:3]))


## The profile ##

@dataclass
class Profile:
    """One analysis profile (see the module docstring for the fields).

    Attributes:
        name: Shown on the start screen, e.g. "Colonic tissue".
        kind: One of ``KINDS``.
        description: One or two sentences for the start screen.
        images: Accepted image types (keys of ``core.model.IMAGE_TYPES``).
        classes: Groups and subgroups with colours.
        aliases: {group: {older subgroup name: current subgroup}}.
        segmentation: {group: {image type: model id}}.
        processing: {image type: [chain dicts]} (``processing.chains.Chain``), applied at import. The
            older form, one step per entry ({"step", "options"}), is read as chains of one.
        parameters: {parameter id: {constant: value}} in display order.
        classification: {group: classifier} — a ``training.classifiers.Classifier`` as a dict, or a model id.
        min_confidence: {group: the confidence (0.5–1) from which its classifier's grade is given to a new object
            (Segment, Classify selected; below it the object stays the plain group). Only classifiers giving a
            probability have one: Ranges' and Score's grades are always suggested.}
        report: {"parameters": [parameter ids], "pdf": [parameter ids]} — the report's key parameters (empty:
            counts and area only), and those of its PDF summary (empty: the classifier's, then key ones, at most
            six).
        registration: The Register tool's methods in its projects: {"method": default method key (empty: the
            library's, ``registration.registry.DEFAULT_METHOD``), "on": [keys switched on], "off": [keys switched
            off]}. A method not named is offered when it is validated, not when it is experimental; the default
            is always offered.
        steps: Processing steps and chains offered in its projects, as {"toolbar": [ids], "hidden": [ids]};
            an id is a step's ("filters:gaussian_blur") or "chain:<name>". In the toolbar: a button on the Process
            tab (and in Quick step); hidden: nowhere; any other: Quick step / Apply chain only. Without
            "toolbar", the default buttons (``processing.registry.DEFAULT_TOOLBAR``).
        revision: Raised by one at every save, so a project can tell that its
            copy is older than the profile in the library.
    """

    name: str
    kind: str = "tissue"
    description: str = ""
    images: list[str] = field(default_factory=list)
    classes: LabelSchema = field(default_factory=LabelSchema)
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    segmentation: dict[str, dict[str, str]] = field(default_factory=dict)
    processing: dict[str, list[dict]] = field(default_factory=dict)
    parameters: dict[str, dict] = field(default_factory=dict)
    classification: dict[str, object] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    registration: dict = field(default_factory=dict)
    scripts: list = field(default_factory=list)
    measuring: dict = field(default_factory=dict)
    min_confidence: dict = field(default_factory=dict)
    revision: int = 1

    # 1. Queries
    def accepts(self, image_type: str) -> bool:
        """Whether images of this type may be added to a project of this profile."""
        return image_type in self.images

    def toolbar(self) -> list[str]:
        """Ids of the steps and chains with a button on the Process tab, in order."""
        if "toolbar" in self.steps:
            return [str(x) for x in self.steps["toolbar"]]
        from modalytics.processing.registry import DEFAULT_TOOLBAR
        return list(DEFAULT_TOOLBAR)

    def step_place(self, item_id: str) -> str:
        """Where a step or chain ("chain:<name>") is offered: "toolbar", "available" or "hidden"."""
        if item_id in self.toolbar():
            return "toolbar"
        return "hidden" if item_id in self.steps.get("hidden", []) else "available"

    def set_step_place(self, item_id: str, place: str) -> None:
        """Offer a step or chain in the toolbar, in Quick step only ("available") or not at all ("hidden")."""
        toolbar = [x for x in self.toolbar() if x != item_id] + ([item_id] if place == "toolbar" else [])
        hidden = [x for x in self.steps.get("hidden", []) if x != item_id] + ([item_id] if place == "hidden" else [])
        self.steps = {"toolbar": toolbar, **({"hidden": hidden} if hidden else {})}

    def registration_default(self) -> str:
        """The Register tool's default method (key) in its projects."""
        from modalytics.registration.registry import DEFAULT_METHOD
        return self.registration.get("method") or DEFAULT_METHOD

    def registration_offered(self, key: str, experimental: bool) -> bool:
        """Whether the Register tool offers a method in its projects (see ``registration``)."""
        if key == self.registration_default():
            return True
        if key in self.registration.get("off", []):
            return False
        return key in self.registration.get("on", []) or not experimental

    def set_registration_offered(self, key: str, on: bool, experimental: bool) -> None:
        """Offer a method or not (only what differs from the rule is written down)."""
        r = dict(self.registration)
        keep_on = [k for k in r.get("on", []) if k != key] + ([key] if on and experimental else [])
        keep_off = [k for k in r.get("off", []) if k != key] + ([key] if not on and not experimental else [])
        r = {k: v for k, v in r.items() if k not in ("on", "off")}
        r.update({**({"on": keep_on} if keep_on else {}), **({"off": keep_off} if keep_off else {})})
        self.registration = r

    def set_registration_default(self, key: str, experimental: bool) -> None:
        """Make a method the default (and so offered); "" goes back to the library's default."""
        r = {k: v for k, v in self.registration.items() if k != "method"}
        if key:
            r["method"] = key
        self.registration = r
        if key:
            self.set_registration_offered(key, True, experimental)

    def label_for(self, group: str, subgroup: str) -> tuple[str, str]:
        """The current class of an annotation: an older subgroup name (``aliases``) gives the one it now means."""
        return group, self.aliases.get(group, {}).get(subgroup, subgroup)

    def chains_for(self, image_type: str) -> list:
        """The processing chains applied at import to images of this type (``processing.chains.Chain``)."""
        from modalytics.processing.chains import Chain
        return [Chain.from_dict(d) for d in self.processing.get(image_type, [])]

    def made_types(self, library=None) -> list[str]:
        """Image types the profile measures through its chains (e.g. "od" from slides).

        Args:
            library: A ``processing.registry.ProcessingLibrary`` (default: the one here).
        """
        if not self.processing:
            return []
        if library is None:
            from modalytics.processing.registry import ProcessingLibrary
            library = ProcessingLibrary()
        out = []
        for image_type in self.processing:
            for chain in self.chains_for(image_type):
                try:
                    made = chain.output_type(image_type, library)
                except ValueError:
                    continue
                if chain.measure and made != image_type and made not in out:
                    out.append(made)
        return out

    def chain_problems(self, library) -> list[str]:
        """What is wrong with the profile's chains (steps missing or not fitting; two measured chains
        making the same image type from one image type: only one can be the standard)."""
        out = []
        for image_type in self.processing:
            made: dict[str, str] = {}
            for chain in self.chains_for(image_type):
                found = chain.problems(image_type, library)
                out += [f"{image_type}: {p}" for p in found]
                if found or not chain.measure:
                    continue
                t = chain.output_type(image_type, library)
                if t in made:
                    out.append(f"{image_type}: “{made[t]}” and “{chain.name}” are both measured and both make "
                               f"{t} images (only one can give the standard results)")
                made[t] = chain.name
        return out

    @property
    def report_parameters(self) -> list[str]:
        """The report's key parameters (those the profile still measures), in order."""
        return [u for u in self.report.get("parameters", []) if u in self.parameters]

    @property
    def pdf_parameters(self) -> list[str]:
        """The parameters ticked for the PDF summary (those the profile still measures), in order; empty: the
        report chooses (``report.html.pdf_keys``)."""
        return [u for u in self.report.get("pdf", []) if u in self.parameters]

    def classifier_for(self, group: str):
        """The Ranges / Score classifier of a group (``training.classifiers.Classifier``), or None."""
        c = self.classification.get(group)
        if not isinstance(c, dict):
            return None
        from modalytics.training.classifiers import Classifier
        return Classifier.from_dict(c)

    def confidence_for(self, group: str) -> float:
        """The confidence from which the group's classifier grades a new object (``MIN_CONFIDENCE`` if not set)."""
        return float(self.min_confidence.get(group, MIN_CONFIDENCE))

    def model_for(self, group: str, image_type: str) -> str | None:
        """Id of the model that segments ``group`` on images of ``image_type`` (None if none)."""
        return self.segmentation.get(group, {}).get(image_type) or None

    def models(self) -> set[str]:
        """Every model id the profile uses."""
        return {m for per_type in self.segmentation.values() for m in per_type.values() if m} | \
            {c for c in self.classification.values() if c and isinstance(c, str)}

    def problems(self, known_models: set[str] | None = None, known_parameters: set[str] | None = None) -> list[str]:
        """What is inconsistent or missing (empty list = usable as it is).

        Args:
            known_models: Model ids installed here (None: not checked).
            known_parameters: Parameter ids in the library (None: not checked).

        Returns:
            list[str]: One sentence per problem.
        """
        out = []
        if not self.name.strip():
            out.append("The profile has no name")
        if self.kind not in KINDS:
            out.append(f"Unknown kind '{self.kind}' (expected {', '.join(KINDS)})")
        if not self.images:
            out.append("No image types are accepted")
        out += [f"Unknown image type '{t}'" for t in self.images if t not in IMAGE_TYPES]
        groups = {g.name for g in self.classes.groups}
        for group, per_type in self.segmentation.items():
            if group not in groups:
                out.append(f"Segmentation for '{group}', which is not one of the classes")
            for image_type, model in per_type.items():
                if image_type not in self.images:
                    out.append(f"'{group}' has a model for {image_type} images, which the profile does not accept")
                if known_models is not None and model and model not in known_models:
                    out.append(f"Model '{model}' ({group}, {image_type}) is not installed")
        for group in self.classification:
            if group not in groups:
                out.append(f"Classification for '{group}', which is not one of the classes")
                continue
            clf = self.classifier_for(group)
            if clf is None:
                continue
            subgroups = [x.name for x in next(g for g in self.classes.groups if g.name == group).subgroups]
            unknown = [x for x in clf.grades if x not in subgroups]
            if unknown:
                out.append(f"The classifier of '{group}' gives grades that are not its subgroups: {', '.join(unknown)}")
            unmeasured = [u for u in clf.parameters if u not in self.parameters]
            if unmeasured:
                out.append(f"The classifier of '{group}' reads parameters the profile does not measure: "
                           f"{', '.join(unmeasured)}")
        gone = [u for u in self.report.get("parameters", []) if u not in self.parameters]
        if gone:
            out.append(f"The report shows parameters the profile does not measure: {', '.join(gone)}")
        for group, names in self.aliases.items():
            g = next((g for g in self.classes.groups if g.name == group), None)
            if g is None:
                out.append(f"Older names for '{group}', which is not one of the classes")
                continue
            for old, new in names.items():
                if new not in {s.name for s in g.subgroups}:
                    out.append(f"'{group} › {old}' is read as '{new}', which is not a subgroup of {group}")
        for image_type in self.processing:
            if image_type not in self.images:
                out.append(f"Processing for {image_type} images, which the profile does not accept")
        if known_parameters is not None:
            out += [f"Parameter '{p}' is not in the library" for p in self.parameters if p not in known_parameters]
        return out

    def copy(self) -> "Profile":
        return copy.deepcopy(self)

    # 2. File form
    @property
    def edge_share(self) -> float:
        """An object touching the image edge is cut by it (shape incomplete) when more than this share of its
        outline lies on the edge (``measuring.edge_percent``, default 5 %; 0: any touch)."""
        from modalytics.core.model import EDGE_SHARE
        return float(self.measuring.get("edge_percent", 100 * EDGE_SHARE)) / 100

    def to_dict(self) -> dict:
        """Plain data in the file's layout (readable YAML once dumped)."""
        classes = {}
        for g in self.classes.groups:
            entry = {"color": rgba_to_hex(g.color)}
            if g.subgroups:
                entry["subgroups"] = {s.name: rgba_to_hex(s.color) for s in g.subgroups}
            if not g.ordered:
                entry["ordered"] = False                     # categories (grades: the default)
            classes[g.name] = entry
        return {"format": FORMAT, "format_version": FORMAT_VERSION, "name": self.name, "kind": self.kind,
                "description": self.description, "revision": self.revision, "images": list(self.images),
                "classes": classes,
                **({"aliases": {g: dict(a) for g, a in self.aliases.items()}} if self.aliases else {}),
                "segmentation": {g: dict(m) for g, m in self.segmentation.items()},
                "processing": {t: [c.to_dict() for c in self.chains_for(t)]
                               for t, chains in self.processing.items() if chains},
                "parameters": {p: dict(s) for p, s in self.parameters.items()},
                "classification": dict(self.classification), "report": self.report,
                **({"steps": {k: list(v) for k, v in self.steps.items()}} if self.steps else {}),
                **({"registration": {k: (list(v) if isinstance(v, list) else v)
                                     for k, v in self.registration.items()}} if self.registration else {}),
                **({"scripts": list(self.scripts)} if self.scripts else {}),
                **({"measuring": dict(self.measuring)} if self.measuring else {}),
                **({"min_confidence": {g: float(v) for g, v in self.min_confidence.items()}}
                   if self.min_confidence else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        """Build a profile from file data.

        Raises:
            ProfileError: If the data is not a profile or a field has the wrong shape.
        """
        if not isinstance(d, dict) or d.get("format", FORMAT) != FORMAT:
            raise ProfileError("Not a Modalytics profile")
        if int(d.get("format_version", 1)) > FORMAT_VERSION:
            raise ProfileError("The profile was made by a newer Modalytics; update Modalytics to open it")
        try:
            groups = []
            for gname, entry in (d.get("classes") or {}).items():
                entry = entry or {}
                color = hex_to_rgba(entry.get("color", "#b3b3b3"))
                subs = [Label(str(s), hex_to_rgba(c)) for s, c in (entry.get("subgroups") or {}).items()]
                groups.append(Group(str(gname), color, subs, bool(entry.get("ordered", True))))
            return cls(
                name=str(d.get("name", "")).strip(),
                kind=str(d.get("kind", "tissue")),
                description=str(d.get("description") or ""),
                images=[str(t) for t in d.get("images") or []],
                classes=LabelSchema(groups),
                aliases={str(g): {str(o): str(n) for o, n in (names or {}).items()}
                         for g, names in (d.get("aliases") or {}).items()},
                segmentation={str(g): {str(t): str(m) for t, m in (per or {}).items()}
                              for g, per in (d.get("segmentation") or {}).items()},
                processing={str(t): [_chain_dict(c) for c in (chains or [])]
                            for t, chains in (d.get("processing") or {}).items()},
                parameters={str(p): dict(s or {}) for p, s in (d.get("parameters") or {}).items()},
                classification={str(g): (dict(c) if isinstance(c, dict) else str(c))
                                for g, c in (d.get("classification") or {}).items()},
                report=dict(d.get("report")) if isinstance(d.get("report"), dict) else {},   # old files: ""
                steps={str(k): [str(x) for x in (v or [])] for k, v in (d.get("steps") or {}).items()
                       if k in ("toolbar", "hidden")},
                registration={str(k): ([str(x) for x in v] if isinstance(v, list) else str(v))
                              for k, v in (d.get("registration") or {}).items() if k in ("method", "on", "off")},
                scripts=[str(x) for x in d.get("scripts") or []],
                measuring={k: float(v) for k, v in (d.get("measuring") or {}).items() if k in ("edge_percent",)},
                min_confidence={str(g): float(v) for g, v in (d.get("min_confidence") or {}).items()},
                revision=int(d.get("revision", 1)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProfileError(f"Malformed profile: {exc}") from exc

    def to_yaml(self) -> str:
        """The profile as YAML text (keys in the documented order, unicode kept)."""
        header = "# Modalytics profile — edit in Settings › Profiles, or here by hand\n"
        return header + yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=110)

    @classmethod
    def from_yaml(cls, text: str) -> "Profile":
        """Parse YAML text.

        Raises:
            ProfileError: If the text is not valid YAML or not a profile.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ProfileError(f"Not valid YAML: {exc}") from exc
        return cls.from_dict(data)


def _chain_dict(d: dict) -> dict:
    """A chain entry of a profile file in today's form (older single-step entries become chains of one)."""
    from modalytics.processing.chains import Chain
    return Chain.from_dict(d).to_dict()
