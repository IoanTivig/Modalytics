"""The help: the Markdown converter, the guides, F1 contexts, links and pictures (Qt-free)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from modalytics import help as hp
from modalytics.help.markdown import inline, slug, to_html

SRC = Path(hp.__file__).resolve().parents[1]                 # src/modalytics


def _constant(module: Path, name: str):
    """A module-level constant read from the source (the UI modules are not imported: no Qt needed)."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ---------------------------------------------------------------- the converter
def test_slug_is_githubs():
    assert slug("Processing: making views") == "processing-making-views"
    assert slug("Registering a phase map onto a slide") == "registering-a-phase-map-onto-a-slide"
    assert slug("Roles: training, evaluation, test") == "roles-training-evaluation-test"


def test_inline():
    assert inline("**bold** and *it*") == "<b>bold</b> and <i>it</i>"
    assert inline("`a < b`") == "<code>a &lt; b</code>"
    assert inline("<kbd>Ctrl</kbd>+<kbd>Z</kbd>") == "<kbd>Ctrl</kbd>+<kbd>Z</kbd>"
    assert inline("x <script>") == "x &lt;script&gt;"                       # other tags are text
    assert inline(r"marked **\***") == "marked <b>*</b>"
    assert inline("[a](b.md#c)", lambda u: "X" + u) == "<a href='Xb.md#c'>a</a>"
    assert inline("![p](images/x.webp)") == "<img src='images/x.webp' alt='p'>"


def test_blocks():
    md = """# Title

Text over
two lines.

## A section

- one
  - nested
- two

1. first
2. second

| a | b |
| --- | --- |
| 1 | x \\| y |

```python
x = 1 < 2
```

> **Note** Careful.

---
"""
    html, heads = to_html(md)
    assert heads == [(1, "Title", "title"), (2, "A section", "a-section")]
    assert "<p>Text over two lines.</p>" in html
    assert "<ul><li>one<ul><li>nested</li></ul></li><li>two</li></ul>" in html
    assert "<ol><li>first</li><li>second</li></ol>" in html
    assert "<td>x | y</td>" in html
    assert "<pre><code>x = 1 &lt; 2</code></pre>" in html
    assert "<blockquote class='note'>" in html
    assert "<hr>" in html


def test_repeated_headings_get_unique_anchors():
    _html, heads = to_html("## Same\n\n## Same\n")
    assert [h[2] for h in heads] == ["same", "same-1"]


# ---------------------------------------------------------------- the guides
def test_guides_and_titles():
    guides = {g.key: g for g in hp.guides()}
    assert {"user", "power"} <= set(guides)
    for g in guides.values():
        assert g.pages[0].key.endswith("/index"), g.key            # each guide opens on its index
        for p in g.pages:
            assert p.title and not p.title.startswith(("0", "1")), p.key


def test_find_with_and_without_number():
    assert hp.find("user/03-annotate").title == "Annotating"
    assert hp.find("user/annotate").title == "Annotating"
    assert hp.find("user/nothing-here") is None


def test_search():
    hits = hp.search("minimum confidence")
    assert hits and all("minimum" in (s + h).lower() or "minimum" in p.path.read_text(encoding="utf-8").lower()
                        for p, h, _a, s in hits)
    assert hp.search("x") == []                                    # too short: nothing


def _pages():
    for g in hp.guides():
        yield from g.pages


@pytest.mark.parametrize("page", list(_pages()), ids=lambda p: p.key)
def test_links_and_pictures_resolve(page):
    """Every relative link leads to a page (and its section), every picture exists."""
    text = page.path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)                  # code holds no links
    text = re.sub(r"`[^`]*`", "", text)
    for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text):
        if "://" in target or target.startswith("mailto:"):
            continue
        path, _, anchor = target.partition("#")
        f = (page.path.parent / path).resolve() if path else page.path
        assert f.is_file(), f"{page.key}: {target} does not exist"
        if anchor:
            _h, heads = to_html(f.read_text(encoding="utf-8"))
            assert anchor in [h[2] for h in heads], f"{page.key}: no section #{anchor} in {f.name}"


# ---------------------------------------------------------------- F1
def test_contexts_lead_to_existing_sections():
    for context, target in hp.CONTEXTS.items():
        key, _, anchor = target.partition("#")
        page = hp.find(key)
        assert page is not None, f"{context}: no page {key}"
        if anchor:
            assert anchor in [h[2] for h in page.headings], f"{context}: no section #{anchor} in {key}"
    assert hp.page_for("unknown") == "user/index"


def test_every_tool_tab_and_page_has_a_context():
    from modalytics.tools.annotate import all_tools
    for cls in all_tools():
        if cls.id != "select":                                     # Select: F1 gives the ribbon tab
            assert f"tool:{cls.id}" in hp.CONTEXTS, cls.id
    for tab in _constant(SRC / "ui" / "main_window.py", "TABS"):
        assert f"tab:{tab}" in hp.CONTEXTS, tab
    for page in _constant(SRC / "ui" / "settings" / "__init__.py", "PAGES"):
        assert f"settings:{page}" in hp.CONTEXTS, page
    window = (SRC / "ui" / "training" / "window.py").read_text(encoding="utf-8")
    for tab in re.findall(r"addTab\([^,]+,\s*\"([^\"]+)\"\)", window):
        assert f"training:{tab}" in hp.CONTEXTS, tab


def test_developer_guide_only_from_the_source(monkeypatch):
    """The developers' guide is the repository's docs/dev: never shown by an installed build."""
    import sys
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert hp.dev_folder() is None
    assert "dev" not in {g.key for g in hp.guides()}
