"""Fast functions for log-scale
"""
from typing import Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

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
