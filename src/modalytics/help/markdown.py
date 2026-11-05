"""
The help pages' Markdown turned into HTML (Qt-free, no dependency).

The subset the pages use — what GitHub shows the same way, so the pages read well in the repository too:

* ``#`` … ``####`` headings (each gets an ``id`` from its text, for links ``page.md#the-heading``);
* paragraphs; ``**bold**``, ``*italic*``, ``inline code``, ``<kbd>Ctrl</kbd>`` (and other inline HTML, kept);
  a backslash before a character keeps it as it is (a star that is not bold, a bar inside a table cell);
* lists (``-`` or ``1.``, nested by indenting two or more spaces);
* fenced code blocks (three backticks);
* tables (``| a | b |`` with a ``|---|`` line);
* block quotes; one starting with ``**Note**``, ``**Tip**`` or ``**Caution**`` becomes a coloured box;
* images ``![alt](images/x.png)`` and links ``[text](page.md#anchor)``, both handed to ``link`` to resolve;
* a line of three dashes: a rule.
"""

from __future__ import annotations

import html
import re
from typing import Callable

_INLINE_CODE = re.compile(r"`([^`]+)`")
_ESCAPED = re.compile(r"\\([\\`*_\[\]()#|!<>{}+.-])")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_TAG = re.compile(r"</?(kbd|br|sub|sup|span|b|i|em|strong)\b[^>]*>", re.IGNORECASE)
CALLOUTS = {"note": "Note", "tip": "Tip", "caution": "Caution"}


def slug(text: str) -> str:
    """A heading's id, as GitHub makes it: lower case, punctuation dropped, spaces as hyphens."""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def inline(text: str, link: Callable[[str], str] | None = None) -> str:
    """Inline Markdown → HTML (text escaped; the allowed tags kept)."""
    link = link or (lambda u: u)
    keep: list[str] = []

    def hold(s: str) -> str:
        keep.append(s)
        return f"\x00{len(keep) - 1}\x00"

    text = _INLINE_CODE.sub(lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _ESCAPED.sub(lambda m: hold(html.escape(m.group(1))), text)       # \* \| … : the character itself
    text = _IMAGE.sub(lambda m: hold(f"<img src='{html.escape(link(m.group(2)), True)}' "
                                     f"alt='{html.escape(m.group(1), True)}'>"), text)
    text = _LINK.sub(lambda m: hold(f"<a href='{html.escape(link(m.group(2)), True)}'>") + m.group(1) + hold("</a>"),
                     text)
    text = _TAG.sub(lambda m: hold(m.group(0)), text)
    text = html.escape(text, quote=False)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], text)


def _table(lines: list[str], link) -> str:
    def cells(line: str) -> list[str]:
        line = line.strip().strip("|")
        return [c.strip() for c in re.split(r"(?<!\\)\|", line)]
    head, rows = cells(lines[0]), [cells(x) for x in lines[2:]]
    out = ["<table><thead><tr>" + "".join(f"<th>{inline(c, link)}</th>" for c in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{inline(c, link)}</td>" for c in r) + "</tr>")
    return "".join(out) + "</tbody></table>"


def _list(lines: list[str], link) -> str:
    """Nested lists from their lines (indent: the depth)."""
    items: list[tuple[int, bool, str]] = []
    for line in lines:
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            items.append((len(m.group(1)), m.group(2)[0].isdigit(), m.group(3)))
        elif items:                                           # a continuation line of the item before
            depth, ordered, text = items[-1]
            items[-1] = (depth, ordered, text + " " + line.strip())
    out, stack = [], []                                       # stack: (indent, tag)
    for depth, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and depth < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or depth > stack[-1][0]:
            out.append(f"<{tag}><li>")
            stack.append((depth, tag))
        else:
            out.append("</li><li>")
        out.append(inline(text, link))
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def to_html(text: str, link: Callable[[str], str] | None = None) -> tuple[str, list[tuple[int, str, str]]]:
    """A page's Markdown as HTML.

    Args:
        text: The Markdown.
        link: Resolves an image's or a link's target (e.g. to a file URL, or another page).

    Returns:
        (the HTML body, its headings as [(level, text, id)]).
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out, headings, i = [], [], 0
    used: dict[str, int] = {}
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):                        # fenced code
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*?)\s*#*$", stripped)
        if m:                                                 # heading
            level, title = len(m.group(1)), m.group(2)
            anchor = slug(title)
            if anchor in used:
                used[anchor] += 1
                anchor = f"{anchor}-{used[anchor]}"
            else:
                used[anchor] = 0
            headings.append((level, re.sub(r"[`*]", "", title), anchor))
            out.append(f"<h{level} id='{anchor}'>{inline(title, link)}</h{level}>")
            i += 1
            continue
        if re.match(r"^-{3,}$", stripped):
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out.append(_table(lines[i:j], link))
            i = j
            continue
        if stripped.startswith(">"):                          # block quote / callout
            j = i
            body = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                body.append(lines[j].strip()[1:].strip())
                j += 1
            inner, _h = to_html("\n".join(body), link)
            m2 = re.match(r"^<p><b>(Note|Tip|Caution)</b>", inner)
            kind = m2.group(1).lower() if m2 else ""
            out.append(f"<blockquote class='{kind}'>{inner}</blockquote>")
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):            # list (with continuation and nested lines)
            j = i
            block = []
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[j])
                                                            or lines[j].startswith("  ")):
                block.append(lines[j])
                j += 1
            out.append(_list(block, link))
            i = j
            continue
        j = i                                                 # paragraph
        para = []
        while j < len(lines) and lines[j].strip() and not re.match(
                r"^\s*(#{1,4}\s|```|>|\||([-*]|\d+\.)\s+|-{3,}$)", lines[j]):
            para.append(lines[j].strip())
            j += 1
        if not para:                                          # a line no rule took: kept as a paragraph
            para, j = [stripped], i + 1
        out.append(f"<p>{inline(' '.join(para), link)}</p>")
        i = j
    return "\n".join(out), headings
