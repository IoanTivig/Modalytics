# Help and documentation

## Three guides

| Guide | Folder | Ships with the software? |
| --- | --- | --- |
| Using Modalytics | `src/modalytics/help/user/` | yes (package data) |
| Profiles, parameters and training | `src/modalytics/help/power/` | yes |
| Developers (this one) | `docs/dev/` | **no** — the private repository only; shown by the Help window when run from the source |

Pages are Markdown, ordered by file name (`01-…`); `index.md` opens a guide; a page's title is its first heading.
`help/markdown.py` converts the subset the pages use (headings with GitHub's anchors, lists, tables, code, callouts
`> **Note** / **Tip** / **Caution**`, images, links, `<kbd>`), so the pages read the same in the repository.

## The Help window (`ui/help_window.py`)

One window, kept: contents (each page and its sections), search (every word, pages and sections; `help.search`),
Back / Forward, the app's theme. Links to other `.md` pages open inside; web addresses in the browser. Pages are
rendered from Markdown when shown — nothing is generated at build time.

## F1: context-sensitive help

`help.CONTEXTS` maps where the user is to a page and section:

| Context | Who reports it |
| --- | --- |
| `start`, `tool:<id>`, `tab:<ribbon tab>` | `MainWindow.help_context` (a tool other than Select wins over the tab) |
| `training:<tab>` | `TrainingWindow.help_context` (the inner tab when there is one) |
| `settings:<page>` | `SettingsWindow.show_help` |

`tests/test_help.py` checks that every context's page and anchor exist, that every link and picture in the pages
resolves, and that every tool, ribbon tab, Settings page and Training tab has a context. A heading renamed in a page
therefore fails the tests until `CONTEXTS` follows.

## Pictures

Pictures are **never drawn by hand**: the benches save screenshots in `bench/out/`, and
`scripts/make_help_images.py` copies the ones the pages use into `help/images/` — resized to 1100 px, tall pages cut,
as WebP (~3 MB for 50 pictures) — then lists pictures a page names that do not exist. Run it **after the full bench
set**, so the help shows the program as it is. To add a picture: add the screenshot to a bench, add it to `PICTURES`,
reference it as `![What it shows](../images/name.webp)`.

## Keeping the help current

A change a user can see — a control renamed, a tool added, a default changed — updates the guide in the same change,
like a test. The README is the short public overview; the guides are for the people using the software.
