"""Scattering mean free path lₛ (image, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'scattering_mean_free_path'
LABEL = 'Scattering mean free path lₛ'
UNIT = 'µm'
CATEGORY = 'scattering'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 231
DESCRIPTION = "Scattering mean free path lₛ = 1/μs = L / (2⟨1 − cos Δφ⟩) ≈ L / ⟨Δφ²⟩ (µm), Δφ = φ − ⟨φ⟩ over the whole map (background included: measure tissue objects for tissue values), L the section thickness (constant): the average distance light travels between two scattering events (Wang, Ding & Popescu 2011; Xu 2011). Shorter = more scattering, more heterogeneous tissue. Valid for a thin slice of forward-scattering tissue (|Δφ| small, structures optically soft); use a background-corrected phase map (Background flattening) and the section's real thickness."
REFERENCES = [
    ('Wang Z, Ding H, Popescu G (2011). Scattering-phase theorem. Opt Lett 36(7):1215–1217. '
     'doi:10.1364/OL.36.001215'),
    ('Xu M (2011). Scattering-phase theorem: anomalous diffraction by forward-peaked scattering media. '
     'Opt Express 19(22):21643–21651. doi:10.1364/OE.19.021643'),
    ('Ding H, Wang Z, Liang X, Boppart SA, Tangella K, Popescu G (2011). Measuring the scattering '
     'parameters of tissues from quantitative phase imaging of thin slices. Opt Lett 36(12):2281–2283. '
     'doi:10.1364/OL.36.002281'),
    ('Wang Z, Tangella K, Balla A, Popescu G (2011). Tissue refractive index as marker of disease. J '
     'Biomed Opt 16(11):116017. doi:10.1117/1.3656732'),
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
    return ctx.scattering(ctx.constants['thickness_um'])['l_s']
