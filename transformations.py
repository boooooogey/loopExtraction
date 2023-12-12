""" Various transformations for an input matrix.
"""
import numpy as np
from jax.typing import ArrayLike

PATCH_MIN=1e-6

def log_transform(mat: ArrayLike) -> ArrayLike:
    """Log transforms the data and shifts the minimum to 0.

    Args:
        mat (ArrayLike): Patch

    Returns:
        ArrayLike: Transformed patch.
    """
    mat = np.log(mat)
    mat_min = np.min(mat[np.logical_not(np.isinf(mat))])
    mat = mat - mat_min
    mat[np.isinf(mat)] = PATCH_MIN
    return mat
