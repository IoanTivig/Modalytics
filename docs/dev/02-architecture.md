# Architecture

## Packages

```
src/modalytics/
  core/          Qt-free domain model: µm geometry (Placement), Project → ImageNode → MaskNode, vector
                 annotations (shapely) and the class schema, signals, undo commands, declared settings
  io/            pixel sources (tiled, multi-level), slides read in place, zarr, OME-TIFF conversion,
                 DHM Phase.txt, HoloPath .hpro, the .modalytics project file, annotation formats
  image_import/  image contracts (phase → radians), the importer library, importing one file
  processing/    processing steps (scripts), chains, output types' contracts, views computed tile by tile
  registration/  registration method scripts, their context, running and testing them
  parameters/    parameter library (scripts), the measuring context, outline and pixel maths, the engine,
                 results of a whole project
  profiles/      the profile format (YAML), the user's profile library, .mprofile export / import
  services/      segmentation (ONNX runner, tiling, vectorising), suggestions, registration search,
                 compute devices, class changes, partial objects
  tools/         Qt-free interactive tools and their declared settings
  training/      data sets, relevance, range finder, classifiers, metrics, validation, prep, acquisition
                 augmentation, segmentation evaluation and training, architectures, encoders, runs on disk
  report/        HTML reports (project, relevance, classifiers, comparison) and XLSX workbooks — Qt-free
  scripts/       external scripts (tasks beside the analysis)
  help/          the user and power-user guides (Markdown) and their converter
  render/        the GPU canvas (pygfx / wgpu): tile choice, background loader, visuals, view maths, GPU choice
  ui/            PySide6: main window, start screen, ribbon, panels, Settings, Training, Help, PDF printing
  resources/     built-in profiles, parameters, importers, processing steps, registration methods, scripts,
                 architectures
  paths.py, app_settings.py, diagnostics.py, fileio.py, icons.py
```

The dependency direction is one way: `ui` → `render`, `services`, `training`, `report`… → `core`. **Nothing below `ui`
imports Qt** (`core`, `io`, `services`, `tools`, `parameters`, `training`, `report`… are all testable without a
window). `render` imports the GPU stack but no Qt widgets beyond the canvas.

## The one rule: geometry lives in `core.model`

Each image has a calibrated pixel size (`ImageNode.pixel_size_um`) and a `Placement` (translation, rotation, scale,
flip) from its local µm frame to the world. Masks are polygons in their **parent image's local µm frame**. The canvas
mirrors this on the GPU: moving an image changes one matrix and never re-reads a tile. Positions are never stored in
render or UI code.

## How the parts talk

- **Signals** (`core.signals`, Qt-free) carry model changes to the UI and the canvas.
- **Every edit** goes through `Project` methods (which emit signals) and, for user edits, an undo `Command`
  (`core.commands`). Live interactions (drags, sliders) update the model continuously and push **one** command at the
  end.
- **Tools** (`tools/`) receive canvas events and act on the model through a context; their settings are declared
  (`Setting`), and the Tool Settings panel builds the form — tools never touch Qt.
- **Background work** goes through `ui.tasks.TaskRunner` (progress and Cancel in the status bar); the work itself is a
  Qt-free function taking `progress` and `cancelled` callbacks.

## Libraries of scripts

Parameters, importers, processing steps, registration methods, external scripts and architectures share one pattern:

- one `.py` per item, metadata as module constants (`KEY`, `NAME`/`LABEL`, …) and one function (`compute`, `read`,
  `process`, `register`, `run`, `build`);
- built-in ones under `resources/<kind>/`, the user's in their library (`paths.user_dir(kind)`); **the same id
  replaces a built-in**, deleting the copy brings it back; built-ins are never written;
- `parameters.script_file` rewrites a script's properties (constants) without touching its code;
- scripts may import the included libraries and the user's (`parameters.libraries`), declared with `REQUIRES`.

## Profiles

`profiles.model.Profile` is everything one analysis needs (image types, classes, models, parameters with their
constants, chains, registration methods, scripts, classifiers, report choices). The library holds one YAML per
profile; a **project keeps its own copy**, a profile keeps its own copies of chains, a view keeps its exact chain — so
results never change behind the user's back. `profiles.bundle` writes and reads `.mprofile` (everything a profile
needs on another computer).
