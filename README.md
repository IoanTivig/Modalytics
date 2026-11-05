# Modalytics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23991084.svg)](https://doi.org/10.5281/zenodo.23991084)

Open-source, modular platform for **multimodal imaging analysis**. Modalytics places every image of a sample —
whole-slide images (WSI), quantitative phase maps from digital holographic microscopy (DHM) and other quantitative
phase imaging (QPI), and ordinary microscope pictures — in one **micrometre-true space**, where they are superposed,
registered, annotated with vector outlines, segmented by neural networks, measured and classified, and the results
written up as reports.

It is the successor of HoloPath. Modalytics is free software under the GNU LGPL, version 3 or later.

## Download

**Windows 10 / 11 (64-bit):** download `Modalytics-<version>-setup.exe` from the
[Releases](https://github.com/IoanTivig/Modalytics/releases) page and run it. No Python is needed.

- Modalytics chooses the graphics card for drawing (Vulkan, DirectX 12 or OpenGL) and the device for the models
  (DirectML on any recent GPU, or the processor) when it starts; *Settings › General* shows and changes the choice.
- Training segmentation models needs PyTorch: Modalytics downloads the right *training pack* for your hardware
  (NVIDIA CUDA, Intel Arc or processor only) when you first train, or during setup if you tick the option.
- The installer is not code-signed yet: Windows SmartScreen may show "unknown publisher" (*More info › Run anyway*).

## What it does

| | |
| --- | --- |
| **Images** | Aperio SVS, pyramidal (OME-)TIFF and other slides read in place; Lyncée Tec DHM (`Phase.txt`), Phasics, and phase, OPD, dry-mass and height maps; PNG / JPEG / TIFF microscope images; HoloPath projects. New formats are added as small importer scripts. |
| **One µm world** | Each image keeps its own calibration; placing, arranging in grids and registering (gland-map correlation and other methods) put them on one another. |
| **Annotation** | Vector outlines in µm: polygon, freehand, brush, eraser, knife, point editing, classes and grades, undo. |
| **Segmentation** | Neural networks run with ONNX Runtime on any GPU; their outlines are suggestions you review. Gland models for DHM phase maps and H&E slides are included. |
| **Measuring** | About 170 parameters (shape, phase, texture, disorder, relations between classes), each a readable Python script you can adapt or extend. |
| **Classification** | Readable rules (ranges, scores, logistic regression) and learned classifiers (random forest, gradient boosting), validated by project. |
| **Training** | Training sets from your projects; parameter relevance, classifier building and validation; segmentation models trained, compared and evaluated. |
| **Reports** | HTML reports with every figure, PDF summaries, CSV and Excel workbooks with native charts. |
| **Profiles** | A profile gathers what one kind of analysis uses (classes, models, processing, parameters, classifier) and is shared as a single file. *Colonic tissue* is included. |

The **Help** inside the program (<kbd>F1</kbd>) is the user guide; it works offline.

## Validation article and data

Modalytics was validated on DHM phase maps of colonic polyps: gland segmentation, measurement and grading of
dysplasia. The data, the trained models and the analysis profile of that study are published on Zenodo:
[doi:10.5281/zenodo.23987906](https://doi.org/10.5281/zenodo.23987906).

> Tivig I, Calin VL, Mihailescu M, Tarba N, Dumitru A, Voinea, Moisescu MG. *Modalytics, an open-source modular
> platform for multimodal imaging analysis: validation on digital holographic microscopy images of colonic
> polyps* (in preparation).

If you use Modalytics in your research, please cite it and the article. Software: Tivig I. *Modalytics* (all
versions: [doi:10.5281/zenodo.23991084](https://doi.org/10.5281/zenodo.23991084); version 0.1.0, used in the
article: [doi:10.5281/zenodo.23991085](https://doi.org/10.5281/zenodo.23991085)); see also [CITATION.cff](CITATION.cff).

## From the source

Python 3.11 or later and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev                     # the analysis edition and pytest
uv sync --extra dev --extra training    # the full edition: + PyTorch, segmentation-models-pytorch, onnx
uv run python main.py                   # the start screen
uv run python -m pytest                 # the tests
uv run python -m modalytics --gpu-info  # the hardware found and chosen
```

The shipped models are not in the git history: download them from the release (`models-<version>.zip`) into
`models/`. The developer guide is in [docs/dev](docs/dev/index.md): architecture, conventions, tests, extending,
building the installer (`python packaging/build.py`).

## Licence

Copyright © 2026 Ioan Tivig. Modalytics is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License, version 3 or (at your option) any later version — see
[COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING). Third-party notices: [NOTICE](NOTICE). The included models
are distributed under CC BY 4.0; their encoders were initialised from ImageNet and MicroNet (MIT) weights.
