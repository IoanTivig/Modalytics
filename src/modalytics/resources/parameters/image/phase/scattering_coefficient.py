"""Scattering coefficient μs (image, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'scattering_coefficient'
LABEL = 'Scattering coefficient μs'
UNIT = '1/mm'
CATEGORY = 'scattering'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 230
DESCRIPTION = "Scattering coefficient μs (1/mm) from the scattering-phase theorem: μs = 2⟨1 − cos Δφ⟩ / L, Δφ = φ − ⟨φ⟩ the phase's deviation from its mean over the whole map (background included: measure tissue objects for tissue values), L the section thickness (constant) — Xu 2011, Eq. 7; for small phase deviations it is ⟨Δφ²⟩ / L, the original form (Wang, Ding & Popescu 2011). How often light is scattered per mm of tissue; it rises with refractive-index heterogeneity (e.g. 38 % higher in precancerous than in normal esophageal epithelium, Su et al. 2015). Valid for a thin slice of forward-scattering tissue (|Δφ| small, structures optically soft); use a background-corrected phase map (Background flattening) and the section's real thickness."
REFERENCES = [
    ('Wang Z, Ding H, Popescu G (2011). Scattering-phase theorem. Opt Lett 36(7):1215–1217. '
     'doi:10.1364/OL.36.001215'),
    ('Xu M (2011). Scattering-phase theorem: anomalous diffraction by forward-peaked scattering media. '
     'Opt Express 19(22):21643–21651. doi:10.1364/OE.19.021643'),
    ('Ding H, Wang Z, Liang X, Boppart SA, Tangella K, Popescu G (2011). Measuring the scattering '
     'parameters of tissues from quantitative phase imaging of thin slices. Opt Lett 36(12):2281–2283. '
     'doi:10.1364/OL.36.002281'),
    ('Su JW, Lin YH, Chiang CP, Lee JM, Hsieh CM, Hsieh MS, Yang PW, Wang CP, Tseng PH, Lee YC, Sung '
     'KB (2015). Precancerous esophageal epithelia are associated with significantly increased '
     'scattering coefficients. Biomed Opt Express 6(10):3795–3805. doi:10.1364/BOE.6.003795'),
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
]


def compute(ctx):
    return ctx.scattering(ctx.constants['thickness_um'])['mu_s'] * 1000.0
