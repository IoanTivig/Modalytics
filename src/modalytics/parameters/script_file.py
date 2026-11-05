"""
Editing a script's properties without touching its code (parameters, importers, processing).

A parameter script mixes two things: its *properties* (``KEY``, ``LABEL``,
``UNIT``…, plain values) and its *code* (imports, helpers, ``compute``).
Settings › Parameters edits the properties in a form and the code in an
editor, so the file is split into the two and joined back on save:

    "A docstring."                      "A docstring."
    import numpy as np                  import numpy as np
                             split
    KEY = 'mean'            ───────►    #: properties  ← the marker: the properties go here
    LABEL = 'Phase mean'
    ...                                 def compute(ctx):
                            ◄───────        return float(np.mean(ctx.pixels))
    def compute(ctx):         join
        return …

Only top-level assignments of the known property names with literal values
are properties; everything else (comments, helpers, constants such as
HoloPath's ``MARGIN_PX``, a property computed by code) stays in the code
exactly as written. The marker line may be moved; if it is deleted, the
properties go after the imports.
"""

from __future__ import annotations

import ast
import pprint
from dataclasses import dataclass, field
from typing import ClassVar

# Properties of a parameter script, in the order they are written
PROPERTIES = ("KEY", "LABEL", "UNIT", "CATEGORY", "SCOPE", "IMAGE_TYPE", "DEFAULT_ON", "WHOLE_OBJECT", "ORDER",
              "REQUIRES", "DESCRIPTION", "REFERENCES", "CONSTANTS")
OLDER_NAMES = {"SETTINGS": "CONSTANTS", "IMAGE_TYPES": "IMAGE_TYPE"}   # read as the new name, saved under it
MARKER = "#: properties (edited on the Properties and Constants tabs; written here on save)"


@dataclass
class ScriptFile:
    """A script split into its properties and its code (see the module docstring).

    Subclasses say which names are properties (``PROPERTIES``, in writing order),
    which older names are read as which (``OLDER_NAMES``) and which properties are
    lists of dicts written one entry per line (``LISTS``).

    Attributes:
        properties: {name: value} for the property names found in the file.
        code: The rest of the file, with ``MARKER`` where the properties were.
    """

    PROPERTIES: ClassVar[tuple[str, ...]] = ()
    OLDER_NAMES: ClassVar[dict[str, str]] = {}
    LISTS: ClassVar[tuple[str, ...]] = ()
    MARKER: ClassVar[str] = MARKER

    properties: dict = field(default_factory=dict)
    code: str = ""

    @classmethod
    def convert_older(cls, name: str, value):
        """Turn the value of an older property name into the current form (default: unchanged)."""
        return value

    @classmethod
    def block(cls, props: dict) -> str:
        """The property assignments, one per line."""
        return properties_block(props, cls.PROPERTIES, cls.LISTS)

    @classmethod
    def parse(cls, text: str) -> "ScriptFile":
        """Split a script.

        Raises:
            SyntaxError: If the text is not valid Python.
        """
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        props, spans = {}, []
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in (*cls.PROPERTIES, *cls.OLDER_NAMES)):
                written = node.targets[0].id
                name = cls.OLDER_NAMES.get(written, written)
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue                         # computed by code: stays in the code
                props[name] = cls.convert_older(written, value) if written != name else value
                spans.append((node.lineno - 1, node.end_lineno))
        removed = {i for start, end in spans for i in range(start, end)}
        out, marked = [], False
        for i, line in enumerate(lines):
            if i in removed:
                if not marked:
                    out.append(cls.MARKER + "\n")
                    marked = True
                continue
            out.append(line)
        code = "".join(out)
        if not marked:
            code = _insert_after_header(code, cls.MARKER + "\n")
        return cls(props, code)

    def render(self) -> str:
        """Join properties and code back into a script."""
        block = self.block(self.properties)
        lines = self.code.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() == self.MARKER:
                return "".join(lines[:i]) + block + "".join(lines[i + 1:])
        return _insert_after_header(self.code, block)


class ParameterScript(ScriptFile):
    """A parameter script (properties: KEY, LABEL, UNIT… CONSTANTS)."""

    PROPERTIES = PROPERTIES
    OLDER_NAMES = OLDER_NAMES
    LISTS = ("REFERENCES", "CONSTANTS")

    @classmethod
    def convert_older(cls, name: str, value):
        if name == "IMAGE_TYPES":                    # a list of types → one image type
            from modalytics.parameters.registry import image_type_from_list
            return image_type_from_list(value)[0]
        return value


def properties_block(props: dict, order: tuple[str, ...] = PROPERTIES, lists: tuple[str, ...] = ("CONSTANTS",)) -> str:
    """The property assignments, one per line (list-of-dict properties one entry per line)."""
    out = []
    for name in order:
        if name not in props:
            continue
        value = props[name]
        if name in lists:
            if not value:
                out.append(f"{name} = []")
                continue
            rows = ",\n".join("    " + pprint.pformat(s, width=100, sort_dicts=False).replace("\n", "\n    ")
                              for s in value)
            out.append(f"{name} = [\n{rows},\n]")
        else:
            out.append(f"{name} = {value!r}")
    return "\n".join(out) + "\n"


def _insert_after_header(code: str, block: str) -> str:
    """Insert ``block`` after the module docstring and the imports (at the top if there are none)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return block + "\n" + code
    end = 0
    for i, node in enumerate(tree.body):
        is_doc = i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
            and isinstance(node.value.value, str)
        if is_doc or isinstance(node, (ast.Import, ast.ImportFrom)):
            end = node.end_lineno
        else:
            break
    lines = code.splitlines(keepends=True)
    head, tail = "".join(lines[:end]), "".join(lines[end:])
    return head + ("\n" if head else "") + block + ("\n" if not tail.startswith("\n") else "") + tail


class ImporterScript(ScriptFile):
    """An importer script (properties: KEY, NAME, IMAGE_TYPE, EXTENSIONS… OPTIONS; see image_import.registry)."""

    PROPERTIES = ("KEY", "NAME", "IMAGE_TYPE", "EXTENSIONS", "REQUIRES", "DESCRIPTION", "OPTIONS")
    LISTS = ("OPTIONS",)
    MARKER = "#: properties (edited on the Properties and Options tabs; written here on save)"


class ProcessingScript(ScriptFile):
    """A processing script (properties: KEY, NAME, INPUT_TYPES, OUTPUT_TYPE… OPTIONS; see processing.registry)."""

    PROPERTIES = ("KEY", "NAME", "SHORT", "ICON", "INPUT_TYPES", "OUTPUT_TYPE", "UNIT", "KIND", "CATEGORY",
                  "REQUIRES", "DESCRIPTION", "OPTIONS")
    LISTS = ("OPTIONS",)
    MARKER = "#: properties (edited on the Properties and Options tabs; written here on save)"


class RegistrationScript(ScriptFile):
    """A registration method script (properties: KEY, NAME, MOVING_TYPES… OPTIONS; see registration.registry)."""

    PROPERTIES = ("KEY", "NAME", "DESCRIPTION", "MOVING_TYPES", "REFERENCE_TYPES", "FINDS", "EXPERIMENTAL",
                  "REFINES_ONLY", "REFERENCES", "REQUIRES", "OPTIONS")
    LISTS = ("REFERENCES", "OPTIONS")
    MARKER = "#: properties (edited on the Properties and Options tabs; written here on save)"


class ExternalScript(ScriptFile):
    """An external script (properties: KEY, NAME, CATEGORY, REQUIRES, DESCRIPTION, OPTIONS; see scripts.registry)."""

    PROPERTIES = ("KEY", "NAME", "CATEGORY", "REQUIRES", "DESCRIPTION", "OPTIONS")
    LISTS = ("OPTIONS",)
    MARKER = "#: properties (edited on the Properties and Options tabs; written here on save)"


class ArchitectureScript(ScriptFile):
    """A segmentation architecture (properties: KEY, NAME, DESCRIPTION, DECODER, ENCODER…; see training.architectures)."""

    PROPERTIES = ("KEY", "NAME", "DESCRIPTION", "DECODER", "ENCODER", "DECODER_OPTIONS", "RECOMMENDED", "REFERENCES",
                  "REQUIRES")
    LISTS = ("REFERENCES",)
    MARKER = "#: properties (edited on the Properties tab; written here on save)"
