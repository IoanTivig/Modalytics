# Training internals

Everything under `training/` is Qt-free; `ui/training/` only shows it.

## Data (`training/dataset.py`)

- `TrainingSet` (`.mtrain`, YAML): profile, group, project entries (`ProjectEntry`: path, role, test group),
  evaluation mode (marked or an automatic share with a seed), include-partial, choose-by, the segmentation rows.
- `gather()` measures every graded object with the profile's parameters (cached per project and profile) into a
  `Dataset`: a values matrix (objects × parameters), grade index, project index, roles, partial flags. Test projects are
  measured apart (`gather(test=True)`).
- Roles are whole projects; the automatic evaluation draw keeps every grade on both sides.

## Parameters (`training/relevance.py`)

Per parameter on the chosen cases (objects, image medians or project medians): the main AUC (Jonckheere–Terpstra as a
share for grades; one-vs-rest for categories) with a bootstrap interval, a permutation p-value (trend or omnibus),
Benjamini–Hochberg q, Holm-corrected pairs, per-class summaries, the verdict, redundancy groups (|Spearman ρ| ≥ 0.8)
and the proposal. `report/relevance.py` writes its report.

## Classifiers (`training/classifiers.py`, `ranges.py`, `prep.py`, `acquisition.py`)

- `Classifier` — one dataclass for every kind (`ranges`, `score`, `logistic`, `forest`, `boosting`), with its `prep`
  (normalisation, augmentation, weighting) recorded; `classify` / `probabilities` are numpy only (forests and boosting are stored as
  their trees' arrays and walked, giving exactly scikit-learn's probabilities; scikit-learn is needed only to fit).
- `ranges.youden`, `ranges.auc`; `ranges.joint_youden` finds a rule's cut-offs together (coordinate ascent from each
  one's own best). `fit_score` fits points by logistic regression, then thresholds, then moves the cut-offs together on
  the training objects' balanced accuracy.
- `prep` — per-project normalisation (difference, ratio, robust z against the project's complete objects), project
  weights, noisy copies, SMOTE. `acquisition` — the projects measured again as if acquired differently (cached per
  copy).

## Measures and validation (`training/metrics.py`, `validation.py`)

- `metrics.measures` — accuracy, balanced accuracy, κ (quadratic-weighted with more than two grades), MCC, macro-F1,
  ROC AUC, per-class sensitivity / specificity / PPV / NPV / F1, with intervals: a **cluster bootstrap over projects**
  when there are at least 5 projects (`classifiers.MIN_PROJECTS`), else Wilson over objects (flagged). `compare` — paired differences (and exact
  McNemar when over objects). `assess` — the choice (by cross-validation or the evaluation projects) and the final
  checks (each group that took no part in it).
- `validation` — cross-validation by project (leave-one-project-out up to 10 projects, else 10 folds of whole
  projects): each fold refits the classifier **the same way** (prep, augmentation from the training folds only, cut-offs
  found again), keeps out-of-fold scores (so CV has an AUC), and lists the mistakes.
- `report/classifiers.py` writes the comparison, warnings (short coverage, many tried selections), methods.

## Segmentation (`training/segmentation.py`, `compare.py`, `seg_train.py`)

- `segmentation` — model outlines against hand outlines on the vector geometry, in µm²: matching at IoU ≥ 0.5,
  found / missed / invented, merges and splits, GlaS object rules, PQ, Dice / IoU, HD95 and NSD on matched objects, per
  image; the "where" areas (complete images, around the outlines).
- `compare` — several models on the same images: bootstrap over images, paired differences, the proposal by the
  chosen measure; `report/comparison.py` writes the report, `report/xlsx.py` the workbook.
- `seg_train` (full edition) — patches at the working resolution, augmentation, BCE + soft Dice, AdamW with cosine
  learning rate, mixed precision on CUDA, the best epoch by evaluation Dice, export to ONNX (checked against PyTorch),
  packages with `weights.safetensors` for fine-tuning. `architectures` and `encoders` are the libraries it draws from;
  `Check` proves an architecture's ONNX export matches PyTorch.
- `runs` — a folder per run under `training_runs/<set>/<kind>/`, results JSON with relative paths.
