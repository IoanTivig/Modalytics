"""
The Help window: the guides that come with Modalytics (``modalytics.help``), offline.

    ┌ Modalytics Help ────────────────────────────────────────────────────────────┐
    │ ◀ ▶ ⌂   [Search…               ]                                           │
    │ ┌ Contents ──────────────┐ ┌──────────────────────────────────────────────┐ │
    │ │ Using Modalytics       │ │ Annotating                                    │ │
    │ │   Getting started      │ │ …                                             │ │
    │ │   Annotating           │ │                                               │ │
    │ │     Drawing            │ │                                               │ │
    │ │ Profiles, parameters…  │ │                                               │ │
    │ └────────────────────────┘ └──────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────┘

One window, kept: F1 anywhere brings it up on the page for what you are doing (``help.page_for``), the ribbon's
Help button on the contents. Pages are made from their Markdown when shown (``help.markdown``), in the app's theme;
a link to another page opens it here (Back / Forward), a web address in your browser. Search finds the pages and
sections holding every word typed.
"""

from __future__ import annotations

import html as _html

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics import help as hp
from modalytics.help.markdown import to_html

KEY = QtCore.Qt.ItemDataRole.UserRole

CSS = """
:root { --bg: #ffffff; --fg: #1d2430; --muted: #5b6676; --line: #e1e5ea; --accent: #1f8a70; --code: #f3f5f7;
        --note: #e8f1fb; --tip: #e8f6f0; --caution: #fff4e0; }
body.dark { --bg: #1b1f24; --fg: #dde3ea; --muted: #9aa5b1; --line: #343b44; --accent: #4cc39c; --code: #262c33;
            --note: #1f2d3d; --tip: #1d3129; --caution: #3a2f1c; }
body { background: var(--bg); color: var(--fg); font: 14px/1.6 "Segoe UI", system-ui, sans-serif;
       margin: 0; padding: 26px 34px 60px; max-width: 980px; }
h1 { font-size: 26px; margin: 0 0 14px; } h2 { font-size: 20px; margin: 30px 0 8px; padding-bottom: 4px;
     border-bottom: 2px solid var(--accent); } h3 { font-size: 16px; margin: 22px 0 6px; } h4 { margin: 16px 0 4px; }
a { color: var(--accent); } p, li { max-width: 860px; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; font: 12.5px Consolas, monospace; }
pre { background: var(--code); padding: 10px 14px; border-radius: 6px; overflow-x: auto; }
pre code { padding: 0; background: none; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px;
      font: 12px Consolas, monospace; background: var(--code); }
table { border-collapse: collapse; margin: 10px 0 16px; } th, td { border: 1px solid var(--line); padding: 5px 9px;
        text-align: left; vertical-align: top; } th { background: var(--code); }
img { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; margin: 8px 0; }
blockquote { margin: 12px 0; padding: 8px 14px; border-left: 4px solid var(--line); background: var(--code);
             border-radius: 0 6px 6px 0; } blockquote p { margin: 4px 0; }
blockquote.note { border-color: #0072B2; background: var(--note); }
blockquote.tip { border-color: var(--accent); background: var(--tip); }
blockquote.caution { border-color: #E69F00; background: var(--caution); }
hr { border: none; border-top: 1px solid var(--line); margin: 22px 0; }
.found { background: #ffe58a; color: #000; }
"""

_window: "HelpWindow | None" = None


def show_help(parent=None, page: str = "user/index", theme: str = "dark") -> "HelpWindow":
    """Bring up the Help window (made once) on a page ("user/03-annotate#drawing")."""
    global _window
    if _window is None:
        _window = HelpWindow()
    _window.theme = theme
    _window.open(page)
    _window.show()
    _window.raise_()
    _window.activateWindow()
    return _window


def _page_class():
    """The web page of the Help window: a link to another page opens it here, a web address in the browser."""
    from PySide6.QtWebEngineCore import QWebEnginePage

    class HelpPage(QWebEnginePage):
        def __init__(self, window: "HelpWindow", parent=None) -> None:
            super().__init__(parent)
            self.window = window

        def acceptNavigationRequest(self, url, kind, is_main):  # noqa: N802 (Qt)
            return self.window.navigate(url)

    return HelpPage


class HelpWindow(QtWidgets.QMainWindow):
    """The Help window (see the module docstring)."""

    def __init__(self) -> None:
        super().__init__()
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.setWindowTitle("Modalytics Help")
        self.resize(1250, 860)
        self.theme = "dark"
        self.history: list[str] = []
        self.current: hp.Page | None = None
        self.at = -1
        # 1. Toolbar: back, forward, contents, search
        bar = self.addToolBar("Help")
        bar.setMovable(False)
        from modalytics.ui.ribbon import icon
        ink = self.palette().color(QtGui.QPalette.ColorRole.WindowText).name()   # the theme's text colour
        self.back_act = bar.addAction(icon("mdi6.arrow-left", ink), "Back", lambda: self._step(-1))
        self.fwd_act = bar.addAction(icon("mdi6.arrow-right", ink), "Forward", lambda: self._step(1))
        bar.addAction(icon("mdi6.home-outline", ink), "Contents", lambda: self.open("user/index"))
        self.back_act.setShortcut(QtGui.QKeySequence.StandardKey.Back)
        self.fwd_act.setShortcut(QtGui.QKeySequence.StandardKey.Forward)
        self.back_act.setToolTip("Back (Alt+Left)")
        self.fwd_act.setToolTip("Forward (Alt+Right)")
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the help…")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(380)
        self.search.textChanged.connect(self._searched)
        bar.addWidget(self.search)
        # 2. Contents (or search results) and the page
        split = QtWidgets.QSplitter()
        left = QtWidgets.QStackedWidget()
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(lambda it, _c: it.data(0, KEY) and self.open(it.data(0, KEY)))
        self.results = QtWidgets.QListWidget()
        self.results.setWordWrap(True)
        self.results.itemClicked.connect(lambda it: self.open(it.data(KEY), highlight=self.search.text()))
        left.addWidget(self.tree)
        left.addWidget(self.results)
        self.left = left
        split.addWidget(left)
        self.view = QWebEngineView()
        self.view.setPage(_page_class()(self, self.view))
        split.addWidget(self.view)
        split.setSizes([300, 950])
        self.setCentralWidget(split)
        self._fill_contents()
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self, activated=self.search.setFocus)

    # ------------------------------------------------------------ contents and search
    def _fill_contents(self) -> None:
        self.tree.clear()
        for g in hp.guides():
            top = QtWidgets.QTreeWidgetItem(self.tree, [g.title])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for p in g.pages:
                it = QtWidgets.QTreeWidgetItem(top, ["Overview" if p.key.endswith("/index") else p.title])
                it.setData(0, KEY, p.key)
                for level, text, anchor in p.headings:
                    if level == 2:
                        sub = QtWidgets.QTreeWidgetItem(it, [text])
                        sub.setData(0, KEY, f"{p.key}#{anchor}")
            top.setExpanded(True)

    def _searched(self, text: str) -> None:
        if not text.strip():
            self.left.setCurrentWidget(self.tree)
            return
        self.results.clear()
        for page, head, anchor, snippet in hp.search(text):
            it = QtWidgets.QListWidgetItem(f"{page.title} › {head}\n{snippet[:140]}…"
                                           if head != page.title else f"{page.title}\n{snippet[:140]}…")
            it.setData(KEY, f"{page.key}#{anchor}" if anchor else page.key)
            self.results.addItem(it)
        if not self.results.count():
            self.results.addItem("Nothing found.")
        self.left.setCurrentWidget(self.results)

    # ------------------------------------------------------------ showing a page
    def open(self, key: str, highlight: str = "", remember: bool = True) -> None:
        """Show a page ("power/08-classifiers#build"); remembered for Back / Forward."""
        page_key, _, anchor = key.partition("#")
        page = hp.find(page_key) or hp.find("user/index")
        if page is None:
            self.view.setHtml("<p>The help pages are missing from this installation.</p>")
            return
        body, _heads = to_html(page.path.read_text(encoding="utf-8"), lambda u: self._resolve(page, u))
        theme = "dark" if self.theme == "dark" else ""
        jump = (f"<script>window.addEventListener('load', function() {{ var e = document.getElementById("
                f"'{anchor}'); if (e) e.scrollIntoView(); }});</script>") if anchor else ""
        doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(page.title)}</title>"
               f"<style>{CSS}</style>{jump}</head><body class='{theme}'>{body}</body></html>")
        self.current = page
        self.view.setHtml(doc, QtCore.QUrl.fromLocalFile(str(page.path)))
        if highlight.strip():
            QtCore.QTimer.singleShot(300, lambda: self.view.findText(highlight.split()[0]))
        if remember:
            self.history = self.history[:self.at + 1] + [key]
            self.at = len(self.history) - 1
        self._select(page.key)
        self.back_act.setEnabled(self.at > 0)
        self.fwd_act.setEnabled(self.at < len(self.history) - 1)

    def _select(self, page_key: str) -> None:
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            if it.value().data(0, KEY) == page_key:
                self.tree.setCurrentItem(it.value())
                return
            it += 1

    def _step(self, by: int) -> None:
        if 0 <= self.at + by < len(self.history):
            self.at += by
            self.open(self.history[self.at], remember=False)

    def _resolve(self, page: hp.Page, target: str) -> str:
        """A page's link or picture: relative ones to their files (another page: its .md); the web as is."""
        if "://" in target or target.startswith(("mailto:", "#")):
            return target
        path, _, anchor = target.partition("#")
        url = QtCore.QUrl.fromLocalFile(str((page.path.parent / path).resolve()))
        if anchor:
            url.setFragment(anchor)
        return url.toString()

    def key_of(self, url: QtCore.QUrl) -> str | None:
        """The page key ("user/03-annotate#drawing") a link to a .md file leads to, None if not a help page."""
        if not url.isLocalFile() or not url.toLocalFile().endswith(".md"):
            return None
        from pathlib import Path
        f = Path(url.toLocalFile()).resolve()
        for g in hp.guides():
            for p in g.pages:
                if p.path.resolve() == f:
                    return p.key + (f"#{url.fragment()}" if url.fragment() else "")
        return None

    def navigate(self, url: QtCore.QUrl) -> bool:
        """Where a link goes: the same page (its section: let it), another help page, or a web address."""
        if url.scheme() in ("http", "https", "mailto"):
            QtGui.QDesktopServices.openUrl(url)
            return False
        if url.scheme() == "data" or not url.isLocalFile():
            return True
        key = self.key_of(url)
        if key is None or (self.current is not None and key.split("#")[0] == self.current.key):
            return True                                        # a picture, the page itself, or a section of it
        QtCore.QTimer.singleShot(0, lambda: self.open(key))
        return False
