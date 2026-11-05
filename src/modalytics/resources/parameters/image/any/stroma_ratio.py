"""Stroma ratio (whole image, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'stroma_ratio'
LABEL = 'Stroma ratio'
UNIT = '%'
CATEGORY = 'relations'
SCOPE = 'image'
IMAGE_TYPE = '*'
DEFAULT_ON = False
ORDER = 400
DESCRIPTION = 'Stroma ratio = 100 · A_stroma / (A_stroma + A_tumour) over the whole image (%), A = the area of each class on it: an area-based version of the tumour–stroma ratio (van Pelt et al. 2018). On a slide, prefer the per-object version on a region outline (the image is mostly empty glass and other tissue). The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration.'
REFERENCES = [
    ('van Pelt GW, Kjær-Frifeldt S, van Krieken JHJM, Al Dieri R, Morreau H, Tollenaar RAEM, Sørensen '
     'FB, Mesker WE (2018). Scoring the tumor-stroma ratio in colon cancer: procedure and '
     'recommendations. Virchows Arch 473(4):405–412. doi:10.1007/s00428-018-2408-z'),
]
CONSTANTS = [
    {'key': 'stroma',
     'label': 'Stroma class',
     'kind': 'label',
     'default': ['Lamina propria', ''],
     'tooltip': 'The class outlining stroma (connective tissue)'},
    {'key': 'tumour',
     'label': 'Tumour class',
     'kind': 'label',
     'default': ['Gland', ''],
     'tooltip': 'The class outlining tumour / epithelium (e.g. glands)'},
]


def compute(ctx):
    c = ctx.constants
    stroma, tumour = ctx.area_of(c['stroma']), ctx.area_of(c['tumour'])
    total = stroma + tumour
    return 100.0 * stroma / total if total > 0 else float('nan')
