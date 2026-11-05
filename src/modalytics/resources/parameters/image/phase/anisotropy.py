"""Scattering anisotropy g (image, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'anisotropy'
LABEL = 'Scattering anisotropy g'
UNIT = ''
CATEGORY = 'scattering'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 233
DESCRIPTION = "Anisotropy factor g = 1 − μs′/μs = 1 − ⟨|∇φ|²⟩ / (4k²⟨1 − cos Δφ⟩) ≈ 1 − ⟨|∇φ|²⟩ / (2k²⟨Δφ²⟩), k = 2π·n₀/λ, over the whole map (background included: measure tissue objects for tissue values) (Xu 2011, Eq. 9): the mean cosine of the scattering angle. Near 1 = mostly forward scattering (large, smooth structures); lower = more scattering to wide angles (finer, fragmented structures). Lower g in the stroma next to cancerous prostate glands identified patients at higher risk of recurrence (Sridharan et al. 2015). Independent of the thickness. Valid for a thin slice of forward-scattering tissue (|Δφ| small, structures optically soft); use a background-corrected phase map (Background flattening) and the section's real thickness."
REFERENCES = [
    ('Wang Z, Ding H, Popescu G (2011). Scattering-phase theorem. Opt Lett 36(7):1215–1217. '
     'doi:10.1364/OL.36.001215'),
    ('Xu M (2011). Scattering-phase theorem: anomalous diffraction by forward-peaked scattering media. '
     'Opt Express 19(22):21643–21651. doi:10.1364/OE.19.021643'),
    ('Ding H, Wang Z, Liang X, Boppart SA, Tangella K, Popescu G (2011). Measuring the scattering '
     'parameters of tissues from quantitative phase imaging of thin slices. Opt Lett 36(12):2281–2283. '
     'doi:10.1364/OL.36.002281'),
    ('Sridharan S, Macias V, Tangella K, Kajdacsy-Balla A, Popescu G (2015). Prediction of prostate '
     'cancer recurrence using quantitative phase imaging. Sci Rep 5:9976. doi:10.1038/srep09976'),
]
CONSTANTS = [
    {'key': 'thickness_um',
     'label': 'Section thickness',
     'kind': 'float',
     'default': 4.0,
     'minimum': 0.1,
     'maximum': 1000.0,
     'step': 0.5,
     'suffix': ' µm',
     'tooltip': 'Thickness L of the tissue section (or sample) the light went through'},
    {'key': 'wavelength_nm',
     'label': 'Wavelength',
     'kind': 'float',
     'default': 0.0,
     'minimum': 0.0,
     'maximum': 2000.0,
     'step': 1.0,
     'suffix': ' nm',
     'tooltip': 'Wavelength λ of the light; 0 = the one the importer read from the file'},
    {'key': 'medium_index',
     'label': 'Medium refractive index',
     'kind': 'float',
     'default': 1.0,
     'minimum': 1.0,
     'maximum': 2.0,
     'step': 0.001,
     'tooltip': 'Refractive index n₀ of what surrounds the section (1.0 air, ≈1.33 water, ≈1.5 '
                'mounting media)'},
]


def compute(ctx):
    c = ctx.constants
    return ctx.scattering(c['thickness_um'], c['wavelength_nm'], c['medium_index'])['g']
