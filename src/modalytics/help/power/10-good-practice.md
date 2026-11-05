# Good practice

Pathology data sets are small in the way that matters: the number of **samples** (projects), not of objects. A
hundred glands from one biopsy say much less than a hundred glands from twenty. Modalytics' statistics are built
around this; these habits keep the numbers honest.

## Count projects, not objects

- Objects of one project resemble each other: same patient, same section, same day's staining and calibration. That
  is why projects are never split between roles, why intervals resample whole projects, and why cross-validation
  leaves out whole projects.
- With fewer than 5 projects an interval cannot be made over projects; Modalytics falls back to objects and marks it
  **\*** — read it as too narrow.
- Prefer more projects with fewer objects each to the reverse. A project holding a third of all objects dominates
  everything: check the Data tab's object counts.
- Projects of one grade only teach little about the difference between grades, and normalising per project removes
  that grade's difference from them.

## Choose little, and choose once

- **Few parameters.** With a dozen training projects, two or three parameters that separate — from different
  redundancy groups — usually beat ten. Every parameter added is another chance to fit these projects' noise.
- **Choosing is fitting.** Each parameter selection tried and compared on the same projects is part of the fit:
  after three or more, cross-validation flatters the one kept, and Build says so. The **final checks** — evaluation
  projects when choosing by cross-validation, and the test groups — are the honest figures.
- **Readable first.** When a readable classifier (Ranges, Score, logistic regression) is not distinguishable from a
  forest or boosting, prefer it: you can see how each parameter counts, and check it against what you know.
- **Augmentation, normalisation and weighting** do not create information. Try them, but keep them only when the
  final checks — not the training figures — improve.

## Keep a real test

- Decide the **test** projects before looking at results, and leave them alone. A test group that differs on purpose
  (another microscope setup, another lab, another time) tells how far the classifier carries over.
- Never move a project into or out of the test group after seeing its numbers.

## Grades and outlines

- **Grade what you see, then let the classifier disagree.** *Mistakes* lists the objects graded differently; a
  double-click shows each. Some are the classifier's errors; some are the hand grade's — correct the annotation in the
  project, then **Reload**.
- **Partial objects**: mark them. Their shapes are not measured, and segmentation still learns from them.
- **Mark images complete** when every object of a group is drawn: segmentation then learns from the whole image,
  including what is *not* a gland.
- **Review all suggestions** before training a segmentation model: what lies under an unreviewed suggestion is
  unknown, so it is left out.

## Reading the measures

| Measure | Read it as |
| --- | --- |
| **Balanced accuracy** | The mean of each grade's sensitivity — unaffected by one grade being more common. The main criterion. |
| **Accuracy** | The share graded right; flattering when one grade dominates. |
| **Cohen's κ** | Agreement beyond chance (quadratic-weighted with more than two grades). |
| **ROC AUC** | How well the classifier's score ranks the grades, at any cut-off. |
| **Sensitivity / specificity** | Per grade: how many of it are found / how many of the others are not called it. |
| **95 % interval** | The range the true value plausibly lies in, given these projects. Wide intervals mean: more projects. |

An interval that spans from "useless" to "excellent" is not a failure of the method — it is the honest answer for
that number of projects.
