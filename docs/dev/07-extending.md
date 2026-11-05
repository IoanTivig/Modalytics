# Extending

Most extensions need **no change to Modalytics**: they are scripts in a library. Changes to the program itself follow
the patterns below.

## Scripts (no change to the program)

| Kind | Built-in under | Function | User guide |
| --- | --- | --- | --- |
| Parameter | `resources/parameters/<scope>/<image type>/` | `compute(ctx)` | Parameters |
| Importer | `resources/importers/<image type>/` | `can_read`, `read` (`hint`) | Importers |
| Processing step | `resources/processing/<output type>/` | `process` (`prepare`, `margin`) | Processing |
| Registration method | `resources/registration/` | `register(ctx, options)` | Processing › Registration methods |
| External script | `resources/scripts/` | `run(options, ctx)` | External scripts |
| Architecture | `resources/architectures/` | none (smp decoder + encoder), or `build(…)` | Segmentation models |

A new **built-in** is a new file there, with its test (a known answer: `tests/test_parameters.py`,
`test_processing_steps.py`, `test_registration_methods.py`…) and, for parameters, a description with the formula and
checked references. Each registry (`parameters.registry`, `image_import.registry`, `processing.registry`,
`registration.registry`, `scripts.registry`, `training.architectures`) documents its format in its docstring and has a
template for **New…**.

## A tool

1. Subclass `tools.base.Tool` (Qt-free): `id`, `label`, `icon` (qtawesome, e.g. `mdi6.…`), `shortcut`, `tabs`, `group`,
   `hint`, `pointer`; declared `settings` (`Setting`) and `actions` (`ToolAction`); the event methods
   (`on_pointer_down/move/up`, `on_key`, …) returning True when consumed.
2. Add it to `tools.annotate.all_tools()`; the main window registers every one, the ribbon shows it in its tabs and group, Tool
   Settings builds its form.
3. Edits go through `Project` methods and one undo `Command` per gesture.
4. Add its F1 context to `help.CONTEXTS` (`"tool:<id>"`) and describe it in the user guide.
5. A bench that drives it with the scripted mouse (see `check_annotation`, `check_interactions`).

## A ribbon command

In `MainWindow._build_actions`: `a(text, icon, slot, shortcut, tip=…)` makes a `QAction` (also logged for the
diagnostic log, shortcuts working whichever tab is shown); `_build_ribbon` places it in a group of a tab
(`self.ribbon.add_group(tab, title, [actions])`). Menus reuse the same actions.

## A Settings page

`ui/settings/__init__.py`: add the page to `PAGES` and to the stack in the same order; give it a module
`ui/settings/<page>.py`; add `"settings:<Page>"` to `help.CONTEXTS` and a page or section in the power-user guide.

## A Training tab

`ui/training/window.py` builds the tabs (`self.kinds` and the inner tab widgets); a panel is a module in `ui/training/`
with its logic in `training/`. `TrainingWindow.help_context` names the tab shown, so add `"training:<Tab>"` to
`help.CONTEXTS`. Runs go through `training.runs` (a folder per run, date first, relative paths inside).

## Profile fields

`profiles.model.Profile` reads and writes its YAML; a new field needs a default (old profiles lack it), a line in
`describe()`, the editor in `ui/settings/profiles.py`, and — when it names something on disk — `profiles.bundle` must
carry it in `.mprofile`.

## Background work

Write the work as a Qt-free function `work(progress, cancelled)` and run it through the window's `TaskRunner`
(`ui/tasks.py`): progress and *Cancel* in the status bar, the result back on the UI thread, errors shown and logged.
