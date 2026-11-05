# Tests and benches

## pytest

```bash
.venv/Scripts/python -m pytest              # unit tests + a headless GPU render test, a few minutes
.venv/Scripts/python -m pytest tests/test_help.py -q
```

Unit tests cover the Qt-free code (geometry, annotations, project files, importers, processing, parameters against
HoloPath's reference values, classifiers, metrics, reports, the help pages…). `tests/test_render_offscreen.py`
renders through wgpu offscreen (`rendercanvas.offscreen`); offscreen Qt has no OpenGL here, wgpu works.
Tests needing an optional package skip without it (`pytest.importorskip`).

A few tests read real microscope files (a slide, a DHM phase export, a HoloPath project, Phasics exports). They are
not in the repository: `tests/testdata.py` lists the names it expects in `_data/test_data` (or the folder named by
`MODALYTICS_TEST_DATA`); without them those tests are skipped.

pytest does **not** import every UI module: after changing one, at least `python -m py_compile` it (or run its bench).

## Benches

The benches (`bench/check_*.py`) drive the **real windows**: they build projects, click, type, drag (scripted mouse
and keyboard), and assert the effects — and save screenshots in `bench/out/` (which also feed the help's pictures).
Each prints its checks and a summary. **Every bench runs in a temporary library** (`MODALYTICS_HOME`, set before
anything of Modalytics is imported): even one that only reads may save a setting (the theme, a recent project), and
it must never reach your own library, settings or projects. The shipped models are found in the repository's
`models/`. Each applies the dark theme before its first window, so its screenshots look like the application.

| Area | Bench |
| --- | --- |
| Start screen, Settings, profiles | `check_profiles` |
| Importers, File › Import, slides, missing files | `check_importers`, `check_map_import` |
| Processing, chains, the Process tab, every step | `check_processing`, `check_process_tab`, `check_steps` |
| Parameters, descriptions, class relations | `check_parameter_docs`, `check_relations` |
| Measure, Results, partial objects | `check_results`, `check_partial` |
| Annotation tools, interactions, pen, layers, arrange | `check_annotation`, `check_interactions`, `check_pen`, `check_layers`, `check_arrange` |
| Segment, review, suggestions | `check_segment`, `check_review` |
| Registration | `check_register`, `check_register_synthetic`, `check_registration_methods` |
| Report (HTML, PDF) | `check_report` |
| Scripts, diagnostic log | `check_scripts` |
| Training: data, parameters, Explore, Build, Classify, exports | `check_training` |
| Segmentation: Evaluate | `check_seg_eval` |
| Segmentation: Train, several models, architectures, resolution | `check_seg_train`, `check_seg_compare`, `check_architectures`, `check_train_resolution` (full edition, GPU; they train) |
| Hardware choice | `check_hardware` |
| The Help window, F1 | `check_help` |
| Real input (moves the cursor, presses keys) | `check_dock_drag_real`, `check_modifiers_real` |
| Performance | `bench_canvas.py [--vsync] [--present screen\|bitmap]`, `bench_masks.py`, `bench_whole_slide.py` |

## When to run what

- **After each change**: pytest, and the benches of the area changed. Say what was run.
- **Before a commit**: the full bench set (about half an hour; the training benches longer).
- The whole-slide segmentation benchmark (`bench_whole_slide.py`) only when asked: it takes long and loads the GPU.
- Never on the user's own projects or runs: work on copies (a temporary `MODALYTICS_HOME`).

## Writing a bench

Start from a nearby one (the benches are self-contained scripts: each has its small helpers — waiting for background
tasks, clicking a ribbon action, `pdf_made` / `xlsx_made` in `check_training`): a temporary `MODALYTICS_HOME` at the
top, `AA_ShareOpenGLContexts` set before Qt starts (WebEngine needs it), `apply_theme(app, "dark")` right after the
application is made, screenshots into `bench/out/` (taken once the window shows what they are for — lists loaded,
background work done), the main window closed at the end (`win._dirty = False; win.close()`: it shuts the canvas'
threads down), and a summary line with a non-zero exit code on failure.

The benches that move the **real** mouse and press real keys (`check_dock_drag_real`, `check_modifiers_real`) run
only when nobody is using the computer.
