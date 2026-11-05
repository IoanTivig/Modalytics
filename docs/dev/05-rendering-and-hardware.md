# Rendering and hardware

## The canvas

`render/canvas.py` draws the project with **pygfx** on **wgpu**, inside a Qt widget (`rendercanvas`). It mirrors
`core.model`: each image is a set of tiles under one transform (its `Placement`), masks are triangulated vector
outlines (`mapbox_earcut`) under their image's transform.

- **Which tiles** (`render/tiling.py`, pure functions): the pyramid level and tiles covering the view, unit-tested.
- **Loading** (`render/tile_loader.py`): six reader threads, "latest view wins" priority; slide tiles are decoded in
  the reading thread (the decoders release the GIL). Tiled TIFF pages are read **directly**, not through zarr (one zarr
  thread starves while another Python thread is busy: measured 0.19 s directly vs 0.73 s, and 48 s with a busy thread).
- **Views** (OD maps, filters) are computed tile by tile through every step of their chain, with the neighbourhood
  margin each step needs (`processing/run.py`).
- **Present mode** — "bitmap" by default (Qt composes; smooth dock resizing, screenshots work, ~145 fps on the dev
  machine). "screen" (native swapchain) is ~1.9× faster but each dock-separator step reconfigures the swapchain and
  the separator lags (`bench/check_dock_drag_real.py`). Never force synchronous draws on resize.
- **Pen input** (`render/pen.py`): every pointer point is replayed (the event system merges them between screen
  updates); pressure for Brush / Eraser; the eraser end; Windows Ink.

## Choosing the drawing GPU (`render/gpu.py`)

At start-up each graphics API the OS offers (Vulkan, Direct3D 12, OpenGL, Metal) is probed **in its own short-lived
process**, so a driver that crashes while being enumerated cannot take the app down. The adapters are ranked:
hardware before software, discrete before integrated before virtual, then Vulkan › D3D12 › GL, then the OS order.
With no usable GPU the software renderer (WARP, llvmpipe) still draws, slowly, and the console says so.

> **Caution** Never put Vulkan and GL in one wgpu instance: on the dev machine (AMD 7900 XT + NVIDIA A2000) wgpu
> crashes when both are enumerated together.

## Choosing the model device (`services/compute.py`)

The installed ONNX Runtime package decides what is possible. Each GPU it can use is listed, ranked CUDA › MIGraphX ›
ROCm › DirectML › Core ML › OpenVINO › CPU, then by dedicated memory. After a session is created it is checked to
really run there (ONNX Runtime otherwise falls back to the CPU silently, e.g. without the CUDA libraries).

| Machine | Package | Models run on |
| --- | --- | --- |
| Windows, any GPU | `onnxruntime-directml` (the default dependency) | DirectML |
| Linux + NVIDIA | `onnxruntime-gpu` | CUDA |
| Linux + AMD | AMD's ONNX Runtime build | MIGraphX |
| anything else | `onnxruntime` | CPU |

## Overrides

| Variable | Effect |
| --- | --- |
| `MODALYTICS_GPU=NVIDIA` | draw on the adapter whose name contains this; `software` forces the software renderer |
| `MODALYTICS_WGPU_BACKENDS=DX12` | skip probing, use these backends (`Vulkan`, `DX12`, `GL`, `Metal`) |
| `MODALYTICS_INFERENCE=NVIDIA` | what the models' "Auto" means: a name substring, a device key (`dml:1`, `cuda:0`) or `cpu` |
| `MODALYTICS_PRESENT=screen` | present through a native swapchain |

`python -m modalytics --gpu-info` prints everything found and chosen; `bench/check_hardware.py` checks every override
and simulated broken drivers.

## Segmentation runtime (`services/segmentation.py`)

The runner resamples the image to the model's µm/px, z-scores over the region, tiles 512 px with 256 px overlap and
Gaussian blending (exactly HoloPath's scheme: mask IoU 0.99986 against HoloPath), and vectorises the result into
polygons with holes. Large regions run block by block; while the GPU runs one tile, the CPU blends the previous one.
Bare-glass tiles of RGB images are skipped. Whole-image probability maps are cached (`%LOCALAPPDATA%/Modalytics/cache`).
fp16, less overlap and larger tiles were measured and rejected (not faster on DirectML, or masks changed).
