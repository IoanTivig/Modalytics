# Setting up

## The environment

Modalytics uses [uv](https://docs.astral.sh/uv/) and a lock file (`uv.lock`) — the installer is built from the same
lock, so what runs in development is what ships.

```bash
uv venv --python 3.12 .venv
uv sync --extra dev                    # the analysis edition + pytest
uv sync --extra dev --extra training   # the full edition: PyTorch (CUDA on Windows/Linux), smp, onnx
```

A new dependency goes into `pyproject.toml` **and** `uv.lock` (`uv lock`), never only into the venv: the installer
is built from the lock. Note why it is needed in a comment next to it, as the existing lines do.

## Running

```bash
.venv/Scripts/python main.py                       # start screen
.venv/Scripts/python -m modalytics project.modalytics slide.svs Phase.txt
.venv/Scripts/python -m modalytics --gpu-info      # the hardware found and chosen
.venv/Scripts/python -m modalytics --theme light
```

`MODALYTICS_HOME` moves the whole user library (benches use a temporary one, so they never touch yours);
`MODALYTICS_MODELS`, `MODALYTICS_CACHE` move the models and the probability-map cache. The hardware overrides
(`MODALYTICS_GPU`, `MODALYTICS_WGPU_BACKENDS`, `MODALYTICS_INFERENCE`, `MODALYTICS_PRESENT`) are in
[Rendering and hardware](05-rendering-and-hardware.md).

## Editions

| Edition | Extra | Can |
| --- | --- | --- |
| Analysis | — | everything except training segmentation models: analysis, Settings, classifiers (scikit-learn), evaluating models |
| Full | `training` | + training and fine-tuning segmentation models (PyTorch, segmentation-models-pytorch, onnx) |

Code that needs the full edition imports it lazily and says what is missing (`paths.installed()` decides whether the
message says "reinstall Modalytics" or names the extra).

## HoloPath

Modalytics is HoloPath's successor. Reused parts are **ported** and re-tested here, never imported; HoloPath's own
projects (`.hpro`) open in Modalytics.
