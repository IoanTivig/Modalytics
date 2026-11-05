# Conventions

## Commits

- **Never add `Co-Authored-By` or any AI attribution to commits.**
- Commits are made by the maintainer. Before one: the full bench set (see [Tests and benches](04-tests-and-benches.md)).

## Code

- **One module per feature or panel.** Logic in `core` / `io` / `services` / `training` / `report` with no Qt import;
  UI in `ui`.
- **Google-style docstrings** (Args / Returns / Raises). A module's docstring says what it is for, how it works and,
  for UI, sketches the layout in a small box drawing.
- **Numbered step comments** in algorithms (`# 1. …`, `# 2. …`), and comments for the *why*.
- **Console messages** start with an emoji: ✅ done, ⚠️ warning, ℹ️ information, ❌ error, 🆕 new.
- A human must be able to follow and improve the code easily: plain names, no cleverness, match the surrounding code.
- **No private APIs of dependencies** (no `_underscore` attributes of Qt, pygfx, wgpu, scikit-learn…).

## Invariants

- µm geometry has one source of truth: `core.model` (`ImageNode.pixel_size_um` + `Placement`).
- Masks are vector polygons in their parent image's local µm frame.
- Every edit goes through `Project` methods and, for user edits, one undo `Command`.
- Hardware is chosen at run time, never assumed (`render/gpu.py`, `services/compute.py`).
- Never put Vulkan and GL in one wgpu instance (crashes on some drivers).
- The canvas presents as "bitmap" by default; never force synchronous draws on resize.
- Phase maps (and other moving images) are registered **onto** the slide — the slide never moves.
- Machine grades never become annotations unseen: they are suggestions until reviewed.

## Words and writing

- Say **project** (one sample) in generic wording, not "biopsy".
- User-facing text is plain English, short sentences, the UI's own words in **bold**; tooltips say what a control does
  and why one would change it.
- **References**: a parameter or method cites a publication only after it was checked (PubMed or Crossref: authors,
  year, title, journal, volume, pages). Formulas go into descriptions. Reports keep references to a minimum.

## Statistics

- Projects, not objects, are the unit of evidence: whole projects in roles, resampled for intervals, left out in
  cross-validation.
- Honest numbers first: anything that took part in a choice is not a final check.
