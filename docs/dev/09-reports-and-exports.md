# Reports and exports

## HTML reports (`report/`, Qt-free)

Every report is **one self-contained HTML file**: pictures and plots embedded (PNG as data URIs, SVG inline), no
internet needed, sortable and filterable tables in a few lines of inline JavaScript, print CSS for paper.

| Report | Module | Made by |
| --- | --- | --- |
| A project's report | `report/html.py` (`build`, `write`) | Report › Report… / PDF… |
| Parameter relevance | `report/relevance.py` | Training › Parameters › Analyse |
| Classifier comparison | `report/classifiers.py` | Training › Build › Write report |
| Segmentation comparison / evaluation | `report/comparison.py` | Train, Evaluate |

Conventions: the measures with their formulas in *Methods*, references kept to a minimum (only what a reader needs to
place the measure; the code cites the rest), intervals everywhere a number is estimated, warnings at the top.

## PDF (`ui/pdf_export.py`)

PDFs are printed **inside Modalytics** with Qt WebEngine (Chromium): `print_pdf(…)` loads the HTML in an offscreen
`QWebEnginePage` and calls `printToPdf`. The project's PDF is a **summary** (`build(…, pdf=True)`): the parameters ticked
*In PDF* (at most `PDF_MAX`), a compact objects list with reasons written out, images two per row. Training reports
print landscape: since the reports' own `@page` CSS overrides the layout, landscape CSS is injected before printing.
`AA_ShareOpenGLContexts` must be set before the `QApplication` exists (`__main__`, and every bench).

## XLSX (`report/xlsx.py`)

Workbooks with XlsxWriter: an *About* sheet, every table (intervals as two numeric columns), and **native charts** drawn
from the workbook's own cells — editable in Excel / LibreOffice. Two quirks found by exporting the charts through
Excel: custom error bars must be literal lists (Excel misreads ranges), and helper columns go after the table.
`ui/training/export.py` saves and opens it, with an installed-aware message when XlsxWriter is missing.

## CSV and GeoJSON

The Results panel exports CSV (UTF-8 with BOM, so spreadsheets read µm²); *File › Export* writes QuPath-compatible
GeoJSON (`io/annotation_files.py`).
