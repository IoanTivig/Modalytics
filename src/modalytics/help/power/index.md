# Profiles, parameters and training

This guide is for the people who **prepare an analysis**: they make the profiles, choose and write the parameters,
add importers, processing steps and scripts, gather training sets and train the classifiers and segmentation models
that the analysis then uses. Everything here is done in **Settings** and **Training**, and ends up in a **profile** —
the one thing the people analysing need.

## The pages

1. [Profiles](01-profiles.md) — what a profile holds, editing, sharing.
2. [Parameters](02-parameters.md) — the parameter library, constants, writing your own, libraries.
3. [Processing](03-processing.md) — processing steps, chains, registration methods.
4. [Importers](04-importers.md) — reading another instrument's files.
5. [External scripts](05-scripts.md) — tasks beside the analysis.
6. [Training sets](06-training-sets.md) — the projects, their roles, measuring.
7. [Choosing parameters](07-training-parameters.md) — which parameters separate the classes.
8. [Classifiers](08-classifiers.md) — Explore, Build, validation, the report, Save to profile.
9. [Segmentation models](09-segmentation-models.md) — evaluating existing models, training new ones.
10. [Good practice](10-good-practice.md) — honest numbers from few samples.

## Where things are kept

Your **library** (on Windows `%APPDATA%\Modalytics`; *Settings › General › Your library* opens it) holds everything
you make, as readable files:

| Folder | Holds |
| --- | --- |
| `profiles/` | profiles (YAML) |
| `parameters/`, `importers/`, `processing/`, `registration/`, `scripts/`, `architectures/` | your scripts (one `.py` each); the same id replaces a built-in one |
| `chains/` | your processing chains |
| `models/` | segmentation models (`.mmodel`) |
| `training/` | training sets (`.mtrain`) |
| `training_runs/` | every analysis, classifier report, training and evaluation run, with its report |
| `libraries/` | Python libraries your scripts need |
| `encoders/` | encoder starting weights for training |

Built-in scripts are never changed: saving one writes your own copy with the same id, which replaces it; deleting
your copy brings the built-in back.
