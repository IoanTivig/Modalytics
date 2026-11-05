"""Phase unwrapping: removes the 2π jumps of a wrapped phase map (phase maps).

A phase measured modulo 2π jumps by 2π where the true phase passes ±π (thick or
dense samples). Unwrapping adds the right multiple of 2π to each pixel so the map
becomes continuous (reliability-guided unwrapping, Herráez et al. 2002, as in
scikit-image). Maps that are already unwrapped stay the same. The result keeps
the input's overall level (the most common 2π multiple is not changed).
Needs the whole map at once (phase maps are small).
"""

import numpy as np
from skimage.restoration import unwrap_phase

KEY = 'phase_unwrap'
NAME = 'Phase unwrapping'
SHORT = 'Unwrap'
ICON = 'mdi6.sine-wave'
INPUT_TYPES = ['phase']
OUTPUT_TYPE = 'same'
UNIT = ''
KIND = 'whole'
CATEGORY = 'Correction'
DESCRIPTION = 'Removes the 2π jumps of a wrapped phase map, making it continuous (Herráez et al. 2002). Already unwrapped maps stay the same.'
OPTIONS = []


def process(pixels, options, state, um_per_px):
    # 1. Wrap into (−π, π], then unwrap (so already unwrapped maps come back as they were)
    phase = pixels.astype(np.float64)
    wrapped = np.angle(np.exp(1j * phase))
    unwrapped = unwrap_phase(wrapped)
    # 2. Keep the input's level: shift by the 2π multiple that brings the medians together
    k = np.round((np.nanmedian(phase) - np.nanmedian(unwrapped)) / (2 * np.pi))
    return (unwrapped + 2 * np.pi * k).astype(np.float32)
