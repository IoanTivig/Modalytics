# Packaging and releases

Modalytics ships as a Windows installer, `Modalytics-<version>-setup.exe`: everything except PyTorch, which each
computer gets as the **training pack** for its hardware (below). End users never run pip themselves: in the
installed program `paths.installed()` is true and messages say "reinstall Modalytics" rather than `pip install …`.

## Building

```bash
python packaging/build.py                 # environment → program → smoke test → installer, in build/
python packaging/build.py --skip-sync     # build/venv is up to date
python packaging/build.py --no-installer  # stop after the smoke test
python packaging/build.py --test-pack     # also: the built program installs the training pack, then trains with it
```

| Step | What | Where |
| --- | --- | --- |
| 1. Environment | `uv sync --frozen --python 3.12 --group build --no-editable` into `build/venv` (no training extra); `resources/training/pack.json` written from `uv.lock` | from `uv.lock`; `.venv` untouched |
| 2. Program | PyInstaller with `packaging/modalytics.spec` | `build/dist/Modalytics` |
| 3. Smoke test | the built `modalytics-cli.exe`: `--version`, `--gpu-info`, `--pip --version`, `--self-test`; models, help, scripts in the bundle; no `docs/dev` | a failure stops the build |
| 4. Installer | Inno Setup 6 (`ISCC.exe`, found on the PATH, in Program Files or the per-user programs folder) with `packaging/modalytics.iss` | `build/installer` |

Needs: [uv](https://docs.astral.sh/uv/), Inno Setup 6 (`winget install JRSoftware.InnoSetup`); for `--test-pack`,
internet (the pack is downloaded from PyTorch's servers and PyPI).

## The training pack (`training.pack`)

| Variant | When | PyTorch | Download |
| --- | --- | --- | --- |
| `cuda` | an NVIDIA GPU (`nvidia-smi`) | `torch==<locked>+cu128` from download.pytorch.org/whl/cu128 | ~2.6 GB |
| `xpu` | an Intel Arc GPU (DXGI adapters) | `+xpu` from download.pytorch.org/whl/xpu | ~1.2 GB |
| `cpu` | anything else (AMD: no Windows PyTorch yet) | `+cpu` from download.pytorch.org/whl/cpu | ~0.3 GB |

- **Contents** — `from_lock`: the packages only the `training` extra needs (for this platform's markers), at their
  locked versions; "provided" lists what the pack needs that the program carries (the spec bundles those whole).
  After `uv lock`, `python packaging/build.py` (or `from_lock`) rewrites `pack.json`; `tests/test_training_pack.py`
  fails while it is stale.
- **Installing** — `install(variant)`: pip run by the program (`--pip`) resolves the pack with every included
  library pinned (`--dry-run --report`, the Libraries page's mechanism), installs only what the program lacks with
  `--no-deps --target` into `.staging`, then `verify` runs `--check-training-pack` in a new process (PyTorch imported,
  one training step). Only then: moved to `<variant>-<version>`, `current.json` updated, the old pack removed.
- **Using** — `activate()` in `__main__` before anything imports PyTorch: the pack's folder goes on `sys.path` *after*
  the program's own libraries. Never when PyTorch is in the environment (from the source).
- **Hardware changes** — `status()` compares `recommended(hardware())` with the installed variant; the Train tab's
  `ui/training/pack_panel.py` offers Install or Switch (QProcess running `--install-training VARIANT`). A switch is
  used from the next start (a loaded PyTorch cannot be replaced).
- **Untested here**: the `xpu` pack (no Intel GPU on the development machine). `cuda` and `cpu` are tested with
  `--test-pack` (and `--install-training cpu`).

## The program folder

- **Two programs, one folder**: `Modalytics.exe` (no console) and `modalytics-cli.exe` (console). The application
  starts the console twin for its child processes — GPU probes (`render.gpu`), pip (`--pip`, Settings › Libraries:
  `parameters.libraries.pip()`) — through `paths.helper_command()`, with `CREATE_NO_WINDOW`, so their output can be
  read and no window flashes.
- **Files loaded at run time** go in as files: `modalytics/resources` (built-in scripts are loaded from their
  `.py`), `modalytics/help` (pages, pictures), `models/*.mmodel` at the bundle's root (`paths.app_root()` is
  PyInstaller's `sys._MEIPASS`, `_internal`).
- **Libraries scripts may import whole** (`collect_submodules`): users' parameter, importer and processing scripts
  may use any part of numpy, scipy, scikit-image, OpenCV, shapely, tifffile — not only what Modalytics imports.
- **Found by name at run time**: timm and segmentation-models-pytorch encoders, zarr / numcodecs codecs, imagecodecs,
  ONNX Runtime's providers (DirectML), pip with its certificates.
- **Every package's metadata** (`copy_metadata`): Settings › Libraries lists the included libraries from it and
  pins them when the user installs one; zarr's entry points need it.
- Nothing is written into the installation: `sys.dont_write_bytecode` is set when frozen; caches, settings and
  everything the user makes are in `%APPDATA%\Modalytics` and `%LOCALAPPDATA%\Modalytics`.

Harmless PyInstaller warnings: tensorboard (not installed), a few imagecodecs codecs whose DLLs the wheel does not
ship (JPEG XS, Jetraw, HEIF), a Linux library path in torch's inductor.

## The installer

`packaging/modalytics.iss`: all users, Program Files (administrator rights); Start-menu shortcut, optional desktop
icon; an unticked *Download training support now* (runs `modalytics-cli --install-training` as the user who started
setup — `runasoriginaluser` — so the pack goes into their `%LOCALAPPDATA%`); `.modalytics` opens with Modalytics;
the LGPL shown; an uninstaller. An update deletes the previous `_internal` first (no stale files). The user library
and the training pack are never touched (the pack stays in `%LOCALAPPDATA%\Modalytics\training` after uninstalling).
Above ~3.5 GB of program (`ONE_FILE_LIMIT`), `build.py` turns on disk spanning: `setup.exe` + `setup-N.bin` slices
of 1.9 GB (GitHub takes release files under 2 GiB; the build warns about any larger file).

## Checking an installation

`modalytics-cli --self-test` (any computer; a temporary library, nothing of the user's touched) runs
`modalytics.selftest`: built-in libraries, help, measuring, segmentation on the best device, the canvas through wgpu,
a random forest and an Excel chart, a PDF through Qt WebEngine, PyTorch training with the ONNX export check (when a
training pack is installed; said and not counted otherwise; a pack that no longer fits the hardware fails it), pip. Add a check there when a feature depends on something the bundle must carry.

## Signing

Unsigned for now: SmartScreen shows "unknown publisher". With a certificate, `signtool sign` goes after step 2 (both
`.exe`) and after step 4 (the setup files).

## Checklist before a release

1. The full bench set passes, including `check_seg_train` and `check_seg_compare`.
2. `python scripts/make_help_images.py` — the help's pictures from that bench run.
3. Version in `pyproject.toml` **and** `modalytics.__version__` (`tests/test_packaging.py` checks they agree);
   `CITATION.cff`; `NOTICE` (third-party notices of everything bundled).
4. `python packaging/build.py` — the smoke test and self-test pass.
5. Install the setup on a clean machine (no Python): start it, open a project, segment, measure, Report › PDF…,
   F1, a Training run; `modalytics-cli --self-test` there.
6. Publish the setup file(s) in the repository's releases.
