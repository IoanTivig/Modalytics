"""Stroma ratio (per object, any image).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'stroma_ratio'
LABEL = 'Stroma ratio'
UNIT = '%'
CATEGORY = 'relations'
SCOPE = 'object'
IMAGE_TYPE = '*'
DEFAULT_ON = False
WHOLE_OBJECT = True
ORDER = 400
DESCRIPTION = "Stroma ratio = 100 · A_stroma / (A_stroma + A_tumour) inside the object (%), A = the area of each class within the object's outline. Measure it on a region outline (e.g. a Region class drawn around the tumour area): an area-based version of the tumour–stroma ratio, which in colon cancer is estimated visually on routine H&E sections and links a high stromal content to a poor prognosis (van Pelt et al. 2018). The classes are constants (chosen in the profile); objects outlined on another, registered image of the project are found through the registration."
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
