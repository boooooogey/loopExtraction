"""Fast functions for log-scale
"""
from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

@jax.jit
def _logsumexp(a:ArrayLike, b:ArrayLike) -> ArrayLike:
    return jax.scipy.special.logsumexp(jnp.concatenate(
                        [jnp.expand_dims(i, axis=-1) for i in (a,b)], axis=-1),
                                        axis=-1)

def logsumexp(a:ArrayLike, b:ArrayLike) -> ArrayLike:
    """Add two arrays in log scale

    Args:
        a (ArrayLike): first array
        b (ArrayLike): second array

    Returns:
        ArrayLike: log of sum of two vectors in log scale 
    """
    if len(a.shape) == 0:
        a = jnp.ones_like(b) * a
    if len(b.shape) == 0:
        b = jnp.ones_like(a) * b
    return _logsumexp(a, b)

@jax.jit
def _logabssubexp(a:ArrayLike, b:ArrayLike) -> ArrayLike:
    weights = jnp.concatenate([jnp.expand_dims((a>=b) * 1 + (b > a) * (-1), axis=-1),
                               jnp.expand_dims((a<b) * 1 + (b<=a)* (-1), axis=-1)], axis=-1) 
    return jax.scipy.special.logsumexp(jnp.concatenate(
                        [jnp.expand_dims(i, axis=-1) for i in (a,b)], axis=-1), b = weights,
                                        axis=-1)

def logabssubexp(a:ArrayLike, b:ArrayLike) -> ArrayLike:
    """Add two arrays in log scale

    Args:
        a (ArrayLike): first array
        b (ArrayLike): second array

    Returns:
        ArrayLike: log of sum of two vectors in log scale 
    """
    if len(a.shape) == 0:
        a = jnp.ones_like(b) * a
    if len(b.shape) == 0:
        b = jnp.ones_like(a) * b
    return _logabssubexp(a, b)

@jax.jit
def logsumexp1(a:ArrayLike) -> ArrayLike:
    """Add one to the input array in log scale

    Args:
        a (ArrayLike): input array

    Returns:
        ArrayLike: input array increased by 1 in log scale
    """
    return jax.nn.relu(a) + jnp.log1p(jnp.exp(-jnp.abs(a)))

@jax.jit
def _running_log_sum(carry:ArrayLike, x:ArrayLike) -> Tuple[ArrayLike, None]:
    ii = jnp.argsort(x[:,1])
    val = x[:, 0] - (x[:, 1] >= len(carry)) * jnp.inf
    return logsumexp(carry, val[ii]), None

@jax.jit
def _diag_norm(carry:ArrayLike, x:ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    val = x - carry
    carry = jnp.roll(carry, 1)
    return carry, val

@jax.jit
def _diag_add_log(carry:ArrayLike, x:ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    val = x + carry
    carry = jnp.roll(carry, 1)
    return carry, val

@jax.jit
def diag_mean_log(a:ArrayLike) -> ArrayLike:
    """Calculate the means of the diagonals in log scale

    Args:
        a (ArrayLike): input array

    Returns:
        ArrayLike: means of each diagonal in log scale
    """
    ii = jnp.arange(a.shape[0])
    index = ii - ii.reshape(-1, 1)
    index = index.at[jnp.tril_indices_from(a, -1)].set(a.shape[0])
    a_w_index = jnp.concatenate([jnp.expand_dims(i, axis=-1) for i in (a, index)], axis=-1)
    return jax.lax.scan(_running_log_sum,
                        jnp.ones(a.shape[0]) * (-jnp.inf),
                        a_w_index)[0] - jnp.log1p(ii[::-1])

@jax.jit
def diag_norm_log(a:ArrayLike) -> ArrayLike:
    """Normalize diagonals to 1 in log scale

    Args:
        a (ArrayLike): input array

    Returns:
        ArrayLike: diagonal normalized array
    """
    log_means = diag_mean_log(a)
    return jax.lax.scan(_diag_norm, log_means, a)[1]

@jax.jit
def diag_scale_log(a:ArrayLike, val:ArrayLike) -> ArrayLike:
    """Normalize diagonals and then scale into given array

    Args:
        a (ArrayLike): input 2d array
        val (ArrayLike): array of scalars 

    Returns:
        ArrayLike: scaled version of input a
    """
    normalized = diag_norm_log(a)
    return jax.lax.scan(_diag_add_log, val, normalized)[1]

@jax.jit
def diag_add_log(a:ArrayLike, val:ArrayLike) -> ArrayLike:
    """Normalize diagonals and then scale into given array

    Args:
        a (ArrayLike): input 2d array
        val (ArrayLike): array of scalars 

    Returns:
        ArrayLike: scaled version of input a
    """
    return jax.lax.scan(_diag_add_log, val, a)[1]

@jax.jit
def _diag_map(carry:ArrayLike, x:ArrayLike, operation:callable) -> Tuple[ArrayLike, ArrayLike]:
    val = operation(x, carry)
    carry = jnp.roll(carry, 1)
    return carry, val

@jax.jit
def diag_map(a:ArrayLike, val:ArrayLike, operation:callable) -> ArrayLike:
    """Map a operation to diagonal of the matrix

    Args:
        a (ArrayLike): input 2d array
        val (ArrayLike): array of scalars 
        operation (callable): operation to be mapped

    Returns:
        ArrayLike: scaled version of input a
    """
    op = partial(_diag_map, operation=operation)
    return jax.lax.scan(op, val, a)[1]

@jax.jit
def _diag_reduce(carry:ArrayLike, x:ArrayLike, operation:callable, mask:callable) -> Tuple[ArrayLike, None]:
    ii = jnp.argsort(x[:,1])
    val = mask(x)
    return operation(carry, val[ii]), None

@jax.jit
def diag_reduce(a:ArrayLike) -> ArrayLike:
    """Calculate the means of the diagonals in log scale

    Args:
        a (ArrayLike): input array

    Returns:
        ArrayLike: means of each diagonal in log scale
    """
    ii = jnp.arange(a.shape[0])
    index = ii - ii.reshape(-1, 1)
    index = index.at[jnp.tril_indices_from(a, -1)].set(a.shape[0])
    a_w_index = jnp.concatenate([jnp.expand_dims(i, axis=-1) for i in (a, index)], axis=-1)
    return jax.lax.scan(_running_log_sum,
                        jnp.ones(a.shape[0]) * (-jnp.inf),
                        a_w_index)[0] - jnp.log1p(ii[::-1])

def func_log_diag_trend(x_range:ArrayLike, params:ArrayLike):
    """Calculate diagonal trend from the given parameters.

    Args:
        x_range (ArrayLike): Domain. Starts from 1.
        params (ArrayLike): Parameters for exponential decay and power law.

    Returns:
        ArrayLike: Diagonal trend
    """
    a, b, c, d = params
    return logsumexp(a + b * jnp.log(x_range), c - d * x_range)

def diag_normalize(mat:ArrayLike, params:ArrayLike):
    """Normalize the diagonals of mat using the trend given by params. In log space.

    Args:
        mat (ArrayLike): given matrix.
        params (ArrayLike): parameters for exponential decay and power law.

    Returns:
        ArrayLike: normalized matrix
    """
    trend = -1 * func_log_diag_trend(jnp.arange(mat.shape[0], dtype=float)+1, params)
    return diag_add_log(mat, trend)

def row_normalize(mat:ArrayLike, params:ArrayLike):
    """Normalize the rows of mat using the trend given by params. In log space.

    Args:
        mat (ArrayLike): given matrix.
        params (ArrayLike): parameters for exponential decay and power law.

    Returns:
        ArrayLike: normalized matrix
    """
    trend = func_log_diag_trend(jnp.arange(mat.shape[0], dtype=float)+1, params)
    return jnp.triu(mat - jnp.reshape(trend, (-1, 1)))

def diag_normalize_inv(mat:ArrayLike, params:ArrayLike):
    """Add back the trend given by params back to diagonals.

    Args:
        mat (ArrayLike): input matrix, normalized previously.
        params (ArrayLike): parameters for exponential decay and power law.

    Returns:
        ArrayLike: unnormalized matrix. 
    """
    trend = func_log_diag_trend(jnp.arange(mat.shape[0], dtype=float)+1, params)
    return diag_add_log(mat, trend)

def flip_diag_row(mat:ArrayLike):
    """Swap row and diagonal elements of a given matrix.

    Args:
        mat (ArrayLike): given matrix.

    Returns:
        ArrayLike: row diagonal swapped matrix. 
    """
    n = mat.shape[0]
    ii = np.arange(n)
    iy = ii.reshape(1,-1) * np.ones(n).reshape(-1,1)
    ix = (ii[::-1].reshape(-1,1) - ii[::-1].reshape(1,-1)) % n
    return mat[ix.astype(int), iy.astype(int)]
