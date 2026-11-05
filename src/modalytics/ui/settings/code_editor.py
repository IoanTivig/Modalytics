"""
A small Python editor for parameter scripts, and the list of what ``ctx`` offers.

``CodeEditor``: line numbers, Python highlighting (``ctx.…`` members in the
accent colour), the current line shaded, auto-indent (one more level after a
``:``), Tab / Shift+Tab indent with 4 spaces, and ``insert_snippet`` for the
helper list (the cursor lands between the parentheses of a call; a missing
import is added at the top).

``HelperPanel``: the helper entries of ``parameters.context.help_entries``
grouped and searchable; a click shows the help, a double-click inserts.

Deliberately small: anything bigger belongs in an external editor, which the
Parameters page opens on request.
"""

from __future__ import annotations

import keyword
import re

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.ui.settings.widgets import clear_tree

MARKER_START = "#: properties ("

INDENT = "    "


## Highlighting ##

class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Keywords, builtins, numbers, strings (incl. triple-quoted), comments, ``ctx.`` members."""

    def __init__(self, document, tokens: dict) -> None:
        super().__init__(document)

        def fmt(color, bold=False, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Weight.DemiBold)
            f.setFontItalic(italic)
            return f

        dark = QtGui.QColor(tokens.get("bg", "#000000")).lightness() < 128
        self.f_kw = fmt("#c792ea" if dark else "#7c3aed", bold=True)
        self.f_builtin = fmt("#82aaff" if dark else "#1d4ed8")
        self.f_num = fmt("#f78c6c" if dark else "#b45309")
        self.f_str = fmt("#c3e88d" if dark else "#15803d")
        self.f_comment = fmt(tokens.get("text_muted", "#888"), italic=True)
        self.f_ctx = fmt(tokens.get("accent", "#2fb3a3"), bold=True)
        self.f_marker = fmt(tokens.get("text_muted", "#888"), bold=True, italic=True)
        builtins = ("abs|all|any|bool|dict|enumerate|float|int|len|list|max|min|print|range|round|"
                    "set|sorted|str|sum|tuple|zip|isinstance|ValueError|None|True|False")
        self.rules = [
            (re.compile(r"\b(" + "|".join(keyword.kwlist) + r")\b"), self.f_kw),
            (re.compile(r"\b(" + builtins + r")\b"), self.f_builtin),
            (re.compile(r"\b\d+(\.\d*)?([eE][-+]?\d+)?\b"), self.f_num),
            (re.compile(r"\bctx(\.\w+)?"), self.f_ctx),
            (re.compile(r"'[^'\\\n]*(\\.[^'\\\n]*)*'|\"[^\"\\\n]*(\\.[^\"\\\n]*)*\""), self.f_str),
        ]
        self.triple = re.compile(r'"""|\'\'\'')

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt)
        if text.strip().startswith(MARKER_START):       # the properties marker of any kind of script
            self.setFormat(0, len(text), self.f_marker)
            self.setCurrentBlockState(0)
            return
        for rx, f in self.rules:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), f)
        # Comments (outside strings, good enough for short scripts)
        hash_at = _comment_start(text)
        if hash_at >= 0:
            self.setFormat(hash_at, len(text) - hash_at, self.f_comment)
        # Triple-quoted strings (may span lines): block state 1 = this line ends inside one
        start, inside = 0, self.previousBlockState() == 1
        while True:
            m = self.triple.search(text, start)
            if inside:                                  # looking for the closing quotes
                end = m.end() if m else len(text)
                self.setFormat(start, end - start, self.f_str)
                if m is None:
                    self.setCurrentBlockState(1)
                    return
                start, inside = end, False
            else:                                       # looking for an opening
                if m is None:
                    self.setCurrentBlockState(0)
                    return
                start, inside = m.start(), True
                closing = self.triple.search(text, m.end())
                end = closing.end() if closing else len(text)
                self.setFormat(start, end - start, self.f_str)
                if closing is None:
                    self.setCurrentBlockState(1)
                    return
                start, inside = end, False


def _comment_start(text: str) -> int:
    """Index of a ``#`` that starts a comment (not inside a string), or -1."""
    quote = None
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "#":
            return i
    return -1


## The editor ##

class _LineNumbers(QtWidgets.QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt)
        return QtCore.QSize(self.editor.number_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        self.editor.paint_numbers(event)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """Python editor with line numbers, highlighting and auto-indent.

    Args:
        tokens: Theme tokens (colours).
    """

    def __init__(self, tokens: dict, parent=None) -> None:
        super().__init__(parent)
        self.tokens = tokens
        # A real code font when there is one (the system "fixed" font is often a wide Courier);
        # set in the style sheet, which would otherwise override setFont
        families = [f for f in ("Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Liberation Mono", "Menlo")
                    if f in QtGui.QFontDatabase.families()]
        font = QtGui.QFont(families[0]) if families else             QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSizeF(10.0)
        self.setFont(font)
        self.setStyleSheet(f"QPlainTextEdit {{ font-family: '{font.family()}'; font-size: 10pt; }}")
        self.setTabStopDistance(QtGui.QFontMetricsF(font).horizontalAdvance(" ") * 4)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.highlighter = PythonHighlighter(self.document(), tokens)
        self.numbers = _LineNumbers(self)
        self.blockCountChanged.connect(lambda _n: self._update_margins())
        self.updateRequest.connect(self._scroll_numbers)
        self.cursorPositionChanged.connect(self._shade_current_line)
        self._update_margins()
        self._shade_current_line()

    # Line numbers
    def number_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 14 + self.fontMetrics().horizontalAdvance("9") * max(3, digits)

    def _update_margins(self) -> None:
        self.setViewportMargins(self.number_width(), 0, 0, 0)

    def _scroll_numbers(self, rect, dy) -> None:
        if dy:
            self.numbers.scroll(0, dy)
        else:
            self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.numbers.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.number_width(), cr.height()))

    def paint_numbers(self, event) -> None:
        p = QtGui.QPainter(self.numbers)
        p.fillRect(event.rect(), QtGui.QColor(self.tokens.get("surface", "#222")))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            bottom = top + round(self.blockBoundingRect(block).height())
            if block.isVisible() and bottom >= event.rect().top():
                n = block.blockNumber()
                p.setPen(QtGui.QColor(self.tokens["text"] if n == current else self.tokens.get("text_muted", "#888")))
                p.drawText(0, top, self.numbers.width() - 6, self.fontMetrics().height(),
                           QtCore.Qt.AlignmentFlag.AlignRight, str(n + 1))
            block, top = block.next(), bottom

    def _shade_current_line(self) -> None:
        sel = QtWidgets.QTextEdit.ExtraSelection()
        sel.format.setBackground(QtGui.QColor(self.tokens.get("surface_alt", "#333")))
        sel.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])
        self.numbers.update()

    # Typing
    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt)
        key, cursor = event.key(), self.textCursor()
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            line = cursor.block().text()[:cursor.positionInBlock()]
            indent = line[:len(line) - len(line.lstrip())]
            if line.rstrip().endswith(":"):
                indent += INDENT
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        if key == QtCore.Qt.Key.Key_Tab and not cursor.hasSelection():
            self.insertPlainText(INDENT)
            return
        if key in (QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab):
            self._indent_selection(key == QtCore.Qt.Key.Key_Tab)
            return
        super().keyPressEvent(event)

    def _indent_selection(self, more: bool) -> None:
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart()).blockNumber()
        last = doc.findBlock(max(cursor.selectionEnd() - 1, cursor.selectionStart())).blockNumber()
        cursor.beginEditBlock()
        for n in range(first, last + 1):
            b = doc.findBlockByNumber(n)
            c = QtGui.QTextCursor(b)
            if more:
                c.insertText(INDENT)
            else:
                strip = len(b.text()) - len(b.text().lstrip(" "))
                for _ in range(min(4, strip)):
                    c.deleteChar()
        cursor.endEditBlock()

    # Inserting from the helper list
    def insert_snippet(self, text: str, needs_import: str = "") -> None:
        """Insert ``text`` at the cursor (inside the parentheses of a call) and add ``needs_import`` if missing."""
        if needs_import and needs_import not in self.toPlainText():
            self._add_import(needs_import)
        cursor = self.textCursor()
        cursor.insertText(text)
        if text.endswith("()"):
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
        elif "(," in text:                                 # e.g. np.percentile(, 90): before the comma
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left, n=len(text) - text.index("(,") - 1)
        self.setTextCursor(cursor)
        self.setFocus()

    def _add_import(self, line: str) -> None:
        """Put an import after the docstring / existing imports."""
        lines = self.toPlainText().split("\n")
        at, in_doc = 0, False
        for i, ln in enumerate(lines):
            s = ln.strip()
            if i == 0 and s.startswith(('"""', "'''")):
                in_doc = not (len(s) > 3 and s.endswith(s[:3]))
                at = i + 1
                continue
            if in_doc:
                at = i + 1
                if s.endswith(('"""', "'''")):
                    in_doc = False
                continue
            if s.startswith(("import ", "from ")):
                at = i + 1
            elif s and not s.startswith("#"):
                break
        cursor = QtGui.QTextCursor(self.document().findBlockByNumber(min(at, self.blockCount() - 1)))
        if at >= self.blockCount():
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cursor.insertText("\n" + line)
        else:
            cursor.insertText(line + "\n")

    def go_to_line(self, line: int) -> None:
        """Put the cursor at the start of a 1-based line and show it."""
        block = self.document().findBlockByNumber(max(0, line - 1))
        if block.isValid():
            self.setTextCursor(QtGui.QTextCursor(block))
            self.centerCursor()
            self.setFocus()


## The helper list ##

class HelperPanel(QtWidgets.QWidget):
    """Searchable list of what a parameter can use.

    Signals:
        insert_requested(str, str): (text to insert, import it needs or "").
    """

    insert_requested = QtCore.Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        col = QtWidgets.QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search…")
        self.search.textChanged.connect(self._filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(lambda cur, _p: self._describe(cur))
        self.tree.itemDoubleClicked.connect(self._insert)
        self.help = QtWidgets.QLabel()
        self.help.setWordWrap(True)
        self.help.setObjectName("Muted")
        self.help.setMinimumHeight(70)
        self.help.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        hint = QtWidgets.QLabel("Double-click to insert at the cursor.")
        hint.setObjectName("Muted")
        col.addWidget(self.search)
        col.addWidget(self.tree, 1)
        col.addWidget(self.help)
        col.addWidget(hint)
        self.scope = None

    def set_scope(self, scope: str) -> None:
        """Show the entries for an object- or image-scope parameter."""
        from modalytics.parameters.context import help_entries
        self.set_entries(scope, help_entries(scope))

    def set_entries(self, name: str, entries: list[tuple[str, str, str, str]]) -> None:
        """Show a helper list: (group, label, text to insert, help ending in "
Needs: <import>" if any).

        Args:
            name: What the list is for (it is rebuilt only when this changes).
            entries: The entries, in display order.
        """
        if name == self.scope:
            return
        self.scope = name
        clear_tree(self.tree)
        groups = {}
        for group, label, insert, doc in entries:
            g = groups.get(group)
            if g is None:
                g = groups[group] = QtWidgets.QTreeWidgetItem(self.tree, [group])
                g.setExpanded(group != "HoloPath pixel shape")
            it = QtWidgets.QTreeWidgetItem(g, [label])
            needs = doc.split("\nNeeds: ", 1)[1] if "\nNeeds: " in doc else ""
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, (insert, needs, doc))
            it.setToolTip(0, doc)
        self._filter(self.search.text())

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            shown = 0
            for j in range(g.childCount()):
                it = g.child(j)
                data = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
                ok = not text or text in it.text(0).lower() or text in data[2].lower()
                it.setHidden(not ok)
                shown += ok
            g.setHidden(shown == 0)
            if text:
                g.setExpanded(True)

    def _describe(self, item) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.help.setText(f"{item.text(0)}\n{data[2]}" if data else "")

    def _insert(self, item, _col=0) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data:
            self.insert_requested.emit(data[0], data[1])
