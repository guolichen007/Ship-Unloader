"""ULP-scale comparison allowance, separate from physical observation noise."""

import numpy as np


def roundoff_m(*values):
    scale = 1.0
    for value in values:
        scale = max(scale, float(np.max(np.abs(np.asarray(value)), initial=0.0)))
    return 64 * np.finfo(float).eps * scale
