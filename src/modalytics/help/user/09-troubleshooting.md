# Troubleshooting

## Objects are ten times too large (or too small)

The image's **pixel size** is wrong — often a decimal slip (3.14 typed for 0.314 µm/px). Check it in **Properties**
(the image selected): it says where the value came from (*read from the file*, *entered at import*). Ask whoever
prepares your profiles to correct the importer's choice (*Settings › Importers › Your choices at import*) and import
the image again.

## A slide shows as a striped placeholder

Its file was not found where the project expects it — a drive not connected, a folder moved or renamed. Connect the
drive, or select the image and use **Properties › Relink…** to point it to the file where it is now. Nothing is lost
meanwhile: the outlines are kept, and saving keeps the old reference.

## Segment finds nothing, or odd shapes

- Is the right **model** chosen for this image type (Tool Settings › Model)? *Auto* uses the profile's.
- Is the image's **pixel size** right? Models work at a fixed resolution in µm/px.
- A **Threshold** too high, or a **Smallest object** too large, drops objects.
- On a colour image, *Ignore empty background* drops objects that are mostly bare glass.

## Segmenting is slow

The status bar's right end says where the models run. *CPU* means no usable graphics card was found: see
**Settings › General › Hardware details…**. A whole slide takes minutes even on a graphics card — Segment image says
how long beforehand.

## A value is empty in Results

Hover the cell (—): the reason is given. The usual ones: the object is **partial** and the parameter needs the
whole shape; the parameter belongs to another image type (an OD parameter on an image with no OD map); the version
it measures is left out (its **Σ** in Layers).

## An object was not graded

It is **partial** and the classifier reads a parameter that needs the whole shape (the reason says which), or, for
suggestions, the classifier was below the **minimum confidence**. Grade it yourself: select it and press the grade's
number.

## The results are out of date

The banner says why — masks edited, images added, the profile's parameters changed. **Measure** again.

## The PDF or report is missing objects or parameters

The PDF shows only the parameters ticked **In PDF** in the profile (at most six); the HTML report shows the key
parameters, and the Results CSV everything. Unreviewed suggestions are not measured unless *Include suggestions* is
on.

## The canvas is blank, or drawing is very slow

Modalytics fell back to software drawing, or a graphics driver failed. **Settings › General › Draw the canvas on**
can choose another graphics card or API; it takes effect at the next start. Updating the graphics driver often helps.

## Checking the installation

Modalytics can check itself: open a command prompt in its folder (usually `C:\Program Files\Modalytics`) and run

```
modalytics-cli --self-test
```

It measures and segments made-up images, draws on the graphics card, prints a PDF and — when the training pack is
installed — runs a short training, each line saying ✅ or ❌ with what it found. Nothing of yours is read or
changed. Send its output with a bug report.

## Reporting a bug

1. **Settings › General › Keep a diagnostic log**.
2. Do again what went wrong.
3. **Report a bug** (the bug icon at the right end of the ribbon) says where the log is; send it with a few words on
   what you did and what you expected.
