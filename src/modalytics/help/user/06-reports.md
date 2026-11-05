# Reports

The **Report** tab writes the project's report from its measured results. It needs results: measure first. At the
top it says when the results are out of date, when suggestions are still waiting for review, or when unreviewed
suggestions were measured.

## The HTML report

**Report…** writes one self-contained HTML file — pictures and plots inside, no internet needed — and opens it in your
browser. It goes from the whole project to each image to each object:

1. **Summary** — objects per class; the profile's **key parameters**, one per row, with mean, SD, median and IQR per
   class.
2. **Classification** (when classified) — predicted against annotated grades (a confusion matrix, agreement with its
   95 % interval, Cohen's κ) and the share of the highest grade per image and in the project.
3. **Distributions** — each key parameter per class: a box plot with every object, and the classifier's cut-off.
4. **Objects** — one row per object: image, number, class, predicted grade and score, key parameters; sortable and
   filterable on screen.
5. **Images** — each image with its outlines in their class colours and every object numbered as in the table (a red
   number: the predicted grade differs from the annotation), with a scale bar.
6. **Methods** — processing, segmentation models (where they come from, how they were trained and evaluated),
   classifiers (with their validation), and each key parameter's description, formula and references.

## The PDF summary

**PDF…** writes a **summary** of the report as a PDF (A4), made by Modalytics itself and opened in your PDF viewer.
Paper cannot scroll or sort, so the PDF keeps what fits:

- the parameters ticked **In PDF** in the profile (none ticked: the classifier's parameters, then the key
  parameters — at most six);
- the objects grouped by image, with their grade, predicted grade, confidence and the classifier's reason written out
  (partial objects left ungraded are said once);
- the images two per row.

Every value stays in the HTML report and in the Results' CSV export.

> **Tip** Which parameters the report puts forward is the profile's choice — *Settings › Profiles › Parameters*,
> columns **Report** and **In PDF**. Ask whoever prepares your profiles if a parameter you need is missing.
