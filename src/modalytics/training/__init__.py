"""
Training (Qt-free): learning from annotated projects.

* ``dataset``: a *training set* — a profile, the class group whose objects are
  classified (the segmentation layer, e.g. Gland), its subgroups as the grades
  (the classification layer), and the projects whose objects were graded by
  hand — gathered into one table of parameter values, with the projects kept
  for an independent evaluation.
* ``ranges``: how well each parameter separates the grades (AUC), and the
  cut-off that separates them best (Youden's index).
"""
