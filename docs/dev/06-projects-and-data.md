# Projects and data

## The domain model (`core/`)

- `Project` → `ImageNode` → `MaskNode`. An image has its pixel source (`io.sources`: tiled, multi-level, read-only),
  `pixel_size_um`, `Placement`, display settings, metadata (where it came from, its views), its masks and suggestions.
- A `MaskNode` is one class on one image: a shapely (Multi)Polygon in the image's local µm, with holes. An **object**
  is one polygon of it. Partial objects are marked by a point inside them (so they survive edits).
- `core.annotations`: the class schema (groups → subgroups), geometry operations (union, subtract, cut, the "touching
  stays separate" rule).
- `core.commands`: undo commands. A user edit is one `Command` pushed on the undo stack; it calls `Project` methods
  that emit signals.

## The project file (`io/project_file.py`)

`name.modalytics` is **one SQLite database** — a single file is hard to break by mistake, and SQLite is readable from
any language.

| Table | Holds |
| --- | --- |
| `meta` | format version, class schema, view, creation time, the project's **own copy of its profile** |
| `images` | calibration, placement, display, source reference; small images (phase maps) as `.npy` blobs |
| `masks` | class, colour, visibility, geometry as WKB (exact vectors) |
| `results` | the last measurement |
| `suggestions` | the suggestions waiting for review, per image |
| `changelog` | who saved what and when |

- **Atomic saves**: written to `name.modalytics.tmp`, swapped in with one `os.replace`.
- **Slides are referenced** (absolute and project-relative path), never copied; their size and a hash of the first MiB
  are kept, so reopening warns when the file at that path is another one.
- **Missing files** open as a placeholder (`io.sources.MissingSource`) with their size, placement and masks; saving
  writes the reference back unchanged; *Relink…* points it at the file again.
- **Views** are saved as recipes (the chain, display, shown, standard), made again on opening.
- HoloPath `.hpro` projects are read by `io.importers` (outlines and pixel brush masks → vectors).

## Images in

`image_import` settles what an imported image is: the importer (a script) reads the file and returns values, a pixel
size or `None`, and the unit it read; `image_import.run` converts to the **contract** of its type
(`image_import/contracts.py`: phase in radians…), settles the pixel size (asked, or the user's saved choice,
`image_import.defaults`) and records the provenance (`unit_read`, `unit_factor`, `pixel_size_source`, the importer).
Slides are read in place (`io/tiff_source.py`); a large non-pyramidal image is converted once into a pyramidal
OME-TIFF next to it (`io/convert.py`).

## The user library (`paths.py`)

| OS | Library |
| --- | --- |
| Windows | `%APPDATA%\Modalytics\` |
| Linux | `$XDG_DATA_HOME/modalytics/` (default `~/.local/share/modalytics/`) |
| macOS | `~/Library/Application Support/Modalytics/` |

`MODALYTICS_HOME` moves it all (tests and benches use a temporary one). `paths.user_dir(kind)` gives each folder
(`profiles`, `parameters`, `importers`, `processing`, `chains`, `registration`, `scripts`, `architectures`,
`encoders`, `models`, `training`, `training_runs`, `libraries`, `logs`); `paths.RESOURCES` the built-ins.
`paths.installed()` is true in a frozen build (the installer).

Per-computer preferences (theme, GPU choices, the diagnostic log) are in `app_settings` (Settings › General), not in
profiles.

## Writing files

`fileio` writes safely (temporary file, then replace). Every file a user may move (training runs, reports) uses
relative paths inside, so a folder can be moved or copied whole.
