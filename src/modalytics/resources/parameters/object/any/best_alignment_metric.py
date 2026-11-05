"""Irregularity (BAM) (object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'best_alignment_metric'
LABEL = 'Irregularity (BAM)'
UNIT = ''
CATEGORY = 'shape'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = True
WHOLE_OBJECT = True
ORDER = 155
DESCRIPTION = 'Best alignment metric (BAM) of Awan et al. 2017: how far the outline is from its reference ellipse. The minimum-area ellipse enclosing the outline is found and the plane stretched so that ellipse becomes a circle (the same stretch is applied to the outline, removing the effect of the cutting angle); outline u and circle v are scaled to unit length, centred and sampled at N equally spaced points (constant, default 256); then d_BAM = (1/N)·min over shifts r and rotations θ of Σⱼ |vⱼ − e^(iθ)·u₍ⱼ₊ᵣ₎ mod N|². 0 = an ellipse; larger = more irregular (glands in the paper: about 0–0.1). In colorectal adenocarcinoma images, gland BAM separated normal from cancer and low from high grade (per image: mean BAM, BAM entropy, regularity index).'
REFERENCES = [
    ('Awan R, Sirinukunwattana K, Epstein D, Jefferyes S, Qidwai U, Aftab Z, Mujeeb I, Snead D, '
     'Rajpoot N (2017). Glandular morphometrics for objective grading of colorectal adenocarcinoma '
     'histology images. Sci Rep 7:16852. doi:10.1038/s41598-017-16516-w'),
]
CONSTANTS = [
    {'key': 'points',
     'label': 'Points along the outline',
     'kind': 'int',
     'default': 256,
     'minimum': 32,
     'maximum': 4096,
     'step': 32,
     'tooltip': 'N: how many equally spaced points the outline and the reference circle are compared '
                'at'},
]


def compute(ctx):
    return ctx.best_alignment_metric(ctx.constants['points'])
