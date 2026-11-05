"""
The help that comes with Modalytics: pages in Markdown, shown by the Help window (``ui.help_window``) — offline.

Three guides, each a folder of pages:

* **Using Modalytics** (``help/user``): projects, images, annotating, segmenting, measuring, reports — for everyone
  who analyses.
* **Profiles, parameters and training** (``help/power``): the people who prepare an analysis — profiles,
  parameters and scripts, processing, importers, training sets, classifiers and segmentation models.
* **Developers** (the repository's ``docs/dev``): the code, its rules, tests, building and packaging. It is not
  part of the package, so an installed Modalytics never carries it; run from the source, the Help window shows it.

A guide's pages are its ``*.md`` files in name order (``01-…``, ``02-…``); ``index.md`` opens it. A page's title is
its first heading. Pictures are in ``help/images`` (made from the benches: ``scripts/make_help_images.py``).

``CONTEXTS`` says which page F1 opens where you are (a tool, a ribbon tab, a Training tab, a Settings page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


@dataclass
class Page:
    """One help page.

    Attributes:
        key: "<guide>/<file stem>", e.g. "user/03-annotate" (what links and ``CONTEXTS`` name).
        path: Its Markdown file.
        title: Its first heading.
        headings: [(level, text, anchor)] of its other headings.
    """

    key: str
    path: Path
    title: str
    headings: list = field(default_factory=list)


@dataclass
class Guide:
    key: str
    title: str
    folder: Path
    pages: list = field(default_factory=list)


GUIDES = (("user", "Using Modalytics"), ("power", "Profiles, parameters and training"), ("dev", "Developers"))


def dev_folder() -> Path | None:
    """The developers' guide: the repository's ``docs/dev`` (run from the source), else None (installed)."""
    from modalytics.paths import installed
    folder = HERE.parents[2] / "docs" / "dev"                # src/modalytics/help → the repository
    return folder if not installed() and folder.is_dir() else None


def guides() -> list[Guide]:
    """The guides there are (the developers' only from the source), each with its pages in order."""
    from modalytics.help.markdown import to_html
    out = []
    for key, title in GUIDES:
        folder = dev_folder() if key == "dev" else HERE / key
        if folder is None or not folder.is_dir():
            continue
        g = Guide(key, title, folder)
        files = sorted(folder.glob("*.md"), key=lambda p: (p.stem != "index", p.stem))
        for f in files:
            text = f.read_text(encoding="utf-8")
            _html, heads = to_html(text)
            title_ = heads[0][1] if heads else f.stem
            g.pages.append(Page(f"{key}/{f.stem}", f, title_, heads[1:]))
        out.append(g)
    return out


def find(key: str) -> Page | None:
    """A page by its key ("user/03-annotate"; "user/annotate" also finds it, without the number)."""
    want = key.split("#")[0]
    for g in guides():
        for p in g.pages:
            if p.key == want or re.sub(r"/\d+-", "/", p.key) == want:
                return p
    return None


def search(query: str, limit: int = 40) -> list[tuple[Page, str, str, str]]:
    """Pages and sections matching every word of ``query``: [(page, heading text, anchor, snippet)], titles first."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    if not words:
        return []
    hits = []
    for g in guides():
        for p in g.pages:
            text = p.path.read_text(encoding="utf-8")
            # split into sections at headings: each its text
            parts = re.split(r"(?m)^(#{1,4}\s+.*)$", text)
            current, anchor = p.title, ""
            sections = [(p.title, "", parts[0])]
            from modalytics.help.markdown import slug
            for k in range(1, len(parts), 2):
                current = re.sub(r"^#+\s+", "", parts[k]).strip()
                anchor = slug(current)
                sections.append((current, anchor, parts[k + 1] if k + 1 < len(parts) else ""))
            for head, anc, body in sections:
                low_head, low = head.lower(), body.lower()
                if all(w in low_head or w in low for w in words):
                    pos = min((low.find(w) for w in words if w in low), default=0)
                    snippet = re.sub(r"[#*`|>\[\]]", "", body[max(0, pos - 60):pos + 120]).replace("\n", " ").strip()
                    rank = sum(w in low_head for w in words) * 10 + sum(low.count(w) for w in words)
                    hits.append((rank, p, head, anc, snippet))
    hits.sort(key=lambda h: -h[0])
    return [(p, head, anc, snip) for _r, p, head, anc, snip in hits[:limit]]


# Where F1 leads: a tool ("tool:<id>"), a ribbon tab ("tab:<name>"), a Training tab ("training:<tab>"), a Settings
# page ("settings:<page>"), the start screen ("start") — to a page, and a section of it
CONTEXTS = {
    "start": "user/01-getting-started",
    "tab:Arrange": "user/02-projects-and-images#arranging-images",
    "tab:Process": "user/02-projects-and-images#processing-making-views",
    "tab:Annotate": "user/03-annotate",
    "tab:Measure": "user/05-measure-and-results",
    "tab:Report": "user/06-reports",
    "tool:select": "user/03-annotate#selecting",
    "tool:edit_points": "user/03-annotate#editing-points",
    "tool:move": "user/02-projects-and-images#arranging-images",
    "tool:rotate": "user/02-projects-and-images#arranging-images",
    "tool:register": "user/02-projects-and-images#registering-a-phase-map-onto-a-slide",
    "tool:polygon": "user/03-annotate#drawing",
    "tool:freehand": "user/03-annotate#drawing",
    "tool:brush": "user/03-annotate#drawing",
    "tool:eraser": "user/03-annotate#drawing",
    "tool:knife": "user/03-annotate#cutting-with-the-knife",
    "tool:segment": "user/04-segment-review-classify#segmenting",
    "tool:review": "user/04-segment-review-classify#reviewing-suggestions",
    "training:Data": "power/06-training-sets",
    "training:Parameters": "power/07-training-parameters",
    "training:Classifiers": "power/08-classifiers",
    "training:Explore": "power/08-classifiers#explore",
    "training:Build": "power/08-classifiers#build",
    "training:Segmentation models": "power/09-segmentation-models",
    "training:Evaluate": "power/09-segmentation-models#evaluate-existing-models",
    "training:Train": "power/09-segmentation-models#train-new-models",
    "settings:General": "user/07-settings-and-hardware",
    "settings:Profiles": "power/01-profiles",
    "settings:Parameters": "power/02-parameters",
    "settings:Importers": "power/04-importers",
    "settings:Processing": "power/03-processing",
    "settings:Registration": "power/03-processing#registration-methods",
    "settings:Architectures": "power/09-segmentation-models#architectures-and-encoder-weights",
    "settings:Scripts": "power/05-scripts",
}


def page_for(context: str) -> str:
    """The page (and section) F1 opens in a context; the start of the user guide when none is known."""
    return CONTEXTS.get(context, "user/index")
