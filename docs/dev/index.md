# Developers

This guide is for people working **on** Modalytics: its structure, its rules, how to test it, how to extend it, how
it is built and released. It lives in the repository (`docs/dev/`) and is **not** part of the package — an
installed Modalytics never carries it. Run from the source, the Help window shows it as a third guide.

> **Note** The real-window benches (`bench/`) and the maintenance scripts (`scripts/`: HoloPath conversions, the
> help's pictures) are kept in the maintainers' working repository and are not published; the pages that mention them
> describe how a release is checked. Everything needed to run, test and build Modalytics is here.

The user guides (`src/modalytics/help/user`, `src/modalytics/help/power`) ship with the software; keep them current
with every change a user can see (see [Help and documentation](10-help-and-docs.md)).

## The pages

1. [Setting up](01-setup.md) — the environment, running from the source, the editions.
2. [Architecture](02-architecture.md) — the packages, the one geometry rule, how the parts talk.
3. [Conventions](03-conventions.md) — code style, commits, what must never happen.
4. [Tests and benches](04-tests-and-benches.md) — pytest, the real-window benches, when to run what.
5. [Rendering and hardware](05-rendering-and-hardware.md) — the GPU canvas, choosing devices at run time.
6. [Projects and data](06-projects-and-data.md) — the project file, the user library, paths.
7. [Extending](07-extending.md) — scripts of every kind, tools, ribbon commands, Settings pages.
8. [Training internals](08-training-internals.md) — data sets, classifiers, validation, segmentation training.
9. [Reports and exports](09-reports-and-exports.md) — HTML reports, PDF, XLSX.
10. [Help and documentation](10-help-and-docs.md) — the Help window, F1 contexts, pictures from the benches.
11. [Packaging and releases](11-packaging-and-releases.md) — the installer and what it must carry.

## In one paragraph

Modalytics is a PySide6 desktop application with a Qt-free core. Every image sits in one micrometre world
(`core.model` + `core.geometry`); the canvas (pygfx / wgpu) only mirrors it. Annotations are vector polygons in µm.
Everything users may adapt — parameters, importers, processing steps, registration methods, external scripts,
architectures — is a small Python script in a library (built-in under `resources/`, the user's in their library),
and a **profile** gathers what one kind of analysis uses. Segmentation runs ONNX models through ONNX Runtime;
training (the full edition) uses PyTorch. Every user edit is one undo command.
