import numpy as np


def ensure_dimensionality(array, dim):
    ar = np.array(array) if not isinstance(array, np.ndarray) else array
    ar = ar.squeeze()
    if ar.ndim != dim:
        raise RuntimeError(f"Argument not {dim} dimensional! (ndim = {ar.ndim})")
    return ar
