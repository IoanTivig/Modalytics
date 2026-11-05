# Glossary

| Word | Meaning |
| --- | --- |
| **Annotation** | The outlines you drew or accepted, per class. Measuring, reports and training use only these. |
| **AUC** | Area under the ROC curve: how well a value separates two classes — 0.5 not at all, 1 perfectly. |
| **Class** | A group and subgroup, e.g. *Gland › High grade dysplasia*. |
| **Classifier** | A rule or model in the profile that grades objects from their parameters: Ranges, Score, logistic regression, random forest, gradient boosting. |
| **Confidence** | The probability a classifier gives its grade; grades are suggested only from the profile's minimum confidence. |
| **Group** | What is outlined — *Gland*. Segmentation models find a group. |
| **Grade / subgroup** | What each object of a group is graded as — *Low grade dysplasia*. |
| **Image type** | Whole-slide image, phase map, brightfield, and the types processing makes (OD map, OPD map, dry-mass map…). Parameters are per image type. |
| **Key parameters** | The parameters the profile puts forward in reports. |
| **Mask** | All objects of one class on one image — one layer in *Annotations*. |
| **Model** | A segmentation model (`.mmodel`): finds the objects of a group on one image type, at one resolution. |
| **Object** | One separate outline of a class: one gland, one cell. |
| **OD map** | Optical density, −log₁₀(I / I_background): how much light a stained slide absorbs, independent of the scanner's brightness. |
| **OPD** | Optical path difference (nm), made from a phase map. |
| **Parameter** | One measurement — area, mean phase, texture… — per object, per class or per whole image. |
| **Partial object** | An object whose shape is incomplete (cut by the image edge, torn, folded…); shape parameters are left empty for it. |
| **Phase map** | A quantitative phase image (DHM / QPI), in radians: how much the sample delays light. |
| **Pixel size** | Micrometres per pixel in the sample (µm/px). Everything in Modalytics is measured in µm. |
| **Profile** | One kind of analysis: image types, classes, models, parameters, classifiers, processing, report. |
| **Project** | One sample (a biopsy, a smear, a culture) with all its images, in one `.modalytics` file. |
| **Registration** | Finding where an image lies on another — a phase map on a slide. |
| **Suggestion** | An object or a grade a model proposed, waiting for your review. |
| **View** | A processed version of an image (an OD map, a filter): same place and masks, listed under *Processed*. |
| **WSI** | Whole-slide image: a scanned slide, read in place from its file. |
