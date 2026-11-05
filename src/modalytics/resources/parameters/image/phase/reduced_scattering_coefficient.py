"""Reduced scattering coefficient μs′ (image, phase).

Formula, meaning and references: see DESCRIPTION and REFERENCES below.
"""

KEY = 'reduced_scattering_coefficient'
LABEL = 'Reduced scattering coefficient μs′'
UNIT = '1/mm'
CATEGORY = 'scattering'
SCOPE = 'image'
IMAGE_TYPE = 'phase'
DEFAULT_ON = True
ORDER = 232
DESCRIPTION = "Reduced scattering coefficient μs′ = ⟨|∇φ|²⟩ / (2k²L) (1/mm), k = 2π·n₀/λ, ∇φ the phase gradient (rad/µm) over the whole map (background included: measure tissue objects for tissue values) (pixels at its edge left out), L the section thickness, n₀ the medium's refractive index (Xu 2011, Eq. 8, the corrected form of the phase-gradient relation). The scattering that changes the light's direction; it governs light diffusion in thick tissue. Needs a phase map sharp enough to resolve the scattering structures (it is underestimated otherwise). Valid for a thin slice of forward-scattering tissue (|Δφ| small, structures optically soft); use a background-corrected phase map (Background flattening) and the section's real thickness."
REFERENCES = [
    ('Wang Z, Ding H, Popescu G (2011). Scattering-phase theorem. Opt Lett 36(7):1215–1217. '
     'doi:10.1364/OL.36.001215'),
    ('Xu M (2011). Scattering-phase theorem: anomalous diffraction by forward-peaked scattering media. '
     'Opt Express 19(22):21643–21651. doi:10.1364/OE.19.021643'),
    ('Ding H, Wang Z, Liang X, Boppart SA, Tangella K, Popescu G (2011). Measuring the scattering '
     'parameters of tissues from quantitative phase imaging of thin slices. Opt Lett 36(12):2281–2283. '
     'doi:10.1364/OL.36.002281'),
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
    return ctx.scattering(c['thickness_um'], c['wavelength_nm'], c['medium_index'])['mu_s_reduced'] * 1000.0
